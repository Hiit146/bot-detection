import argparse
import os
from typing import Optional, Tuple, Dict, Any

import torch
from torch import Tensor
from torch.nn import Module, Linear, ReLU, Dropout
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

def _cuda_tag_from_version(cuda_version: Optional[str]) -> str:
    if not cuda_version:
        return "cpu"
    version = cuda_version.strip()
    if version.startswith("11.8"):
        return "cu118"
    if version.startswith("12.1"):
        return "cu121"
    # Fallback: best-effort mapping (may need manual adjustment)
    major_minor = version.replace(".", "")[:3]
    return f"cu{major_minor}"


def _install_instructions(torch_version: str, cuda_version: Optional[str]) -> str:
    tag = _cuda_tag_from_version(cuda_version)
    # Strip build suffix from torch version (e.g., "2.7.1+cpu" -> "2.7.1")
    base_torch = torch_version.split("+")[0]
    base = f"https://data.pyg.org/whl/torch-{base_torch}+{tag}.html"
    cmds = [
        f"pip install torch-scatter -f {base}",
        f"pip install torch-sparse -f {base}",
        f"pip install torch-cluster -f {base}",
        f"pip install torch-spline-conv -f {base}",
        "pip install torch-geometric",
    ]
    header = (
        "PyTorch Geometric or its extensions failed to import.\n"
        "This is typically a Windows wheel mismatch (PyTorch x CUDA).\n\n"
        f"Detected Torch: {torch.__version__} | CUDA: {cuda_version or 'cpu'}\n"
        "Install matching wheels with:\n"
    )
    return header + "\n".join(cmds)


try:
    from torch_geometric.data import Data
    from torch_geometric.nn import RGCNConv, GCNConv
    from torch_geometric.utils import degree
    # Proactively check low-level extensions and emit guidance if missing
    try:
        import torch_scatter  # type: ignore
        import torch_sparse  # type: ignore
    except Exception:
        print(_install_instructions(torch.__version__, torch.version.cuda))
except Exception as exc:
    raise RuntimeError(_install_instructions(torch.__version__, torch.version.cuda)) from exc


class MultiTaskHead(Module):
    def __init__(self, hidden_dim: int, stance_num_classes: int, dropout: float) -> None:
        super().__init__()
        self.dropout = Dropout(dropout)
        self.activation = ReLU()
        self.bot_head = Linear(hidden_dim, 1)
        self.stance_head = Linear(hidden_dim, stance_num_classes)

    def forward(self, node_embeddings: Tensor) -> Tuple[Tensor, Tensor]:
        hidden = self.activation(node_embeddings)
        hidden = self.dropout(hidden)
        bot_logits = self.bot_head(hidden).squeeze(-1)
        stance_logits = self.stance_head(hidden)
        return bot_logits, stance_logits


class RGCNEncoder(Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_relations: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        assert num_layers >= 1
        self.layers = torch.nn.ModuleList()
        self.activation = ReLU()
        self.dropout = Dropout(dropout)

        self.layers.append(RGCNConv(input_dim, hidden_dim, num_relations=num_relations))
        for _ in range(num_layers - 1):
            self.layers.append(RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations))

    def forward(self, x: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        for conv in self.layers:
            x = conv(x, edge_index, edge_type)
            x = self.activation(x)
            x = self.dropout(x)
        return x


class GCNEncoder(Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        assert num_layers >= 1
        self.layers = torch.nn.ModuleList()
        self.activation = ReLU()
        self.dropout = Dropout(dropout)

        self.layers.append(GCNConv(input_dim, hidden_dim, add_self_loops=True, normalize=True))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True))

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor]) -> Tensor:
        for conv in self.layers:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = self.activation(x)
            x = self.dropout(x)
        return x


class MGTABModel(Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        stance_num_classes: int,
        num_relations: Optional[int],
    ) -> None:
        super().__init__()
        self.uses_relations = num_relations is not None
        if self.uses_relations:
            self.encoder = RGCNEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_relations=int(num_relations),
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            self.encoder = GCNEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        self.head = MultiTaskHead(hidden_dim=hidden_dim, stance_num_classes=stance_num_classes, dropout=dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Optional[Tensor],
        edge_weight: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        if self.uses_relations and edge_type is not None:
            h = self.encoder(x, edge_index, edge_type)
        else:
            h = self.encoder(x, edge_index, edge_weight)
        return self.head(h)


def load_tensor(path: str) -> Optional[Tensor]:
    if not os.path.exists(path):
        return None
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, Tensor):
        return obj
    raise ValueError(f"Expected a Tensor in {path}, found type {type(obj)}")


def augment_behavioral_features(x: Tensor, edge_index: Tensor) -> Tensor:
    num_nodes = x.size(0)
    src, dst = edge_index
    device = x.device
    in_deg = torch.bincount(dst, minlength=num_nodes).to(device).float().unsqueeze(1)
    out_deg = torch.bincount(src, minlength=num_nodes).to(device).float().unsqueeze(1)
    total_deg = in_deg + out_deg
    ratio = torch.zeros((num_nodes, 1), device=device)
    if x.size(1) >= 2:
        following = x[:, 1].abs() + 1e-6
        ratio = (x[:, 0].float() / following).unsqueeze(1)
    return torch.cat([x, in_deg, out_deg, total_deg, ratio], dim=1)


def normalize_features(x: Tensor, eps: float = 1e-6) -> Tensor:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True)
    return (x - mean) / (std + eps)


def infer_num_classes(labels: Tensor) -> int:
    if labels.dtype.is_floating_point:
        unique_vals = torch.unique(labels)
    else:
        unique_vals, _ = torch.sort(torch.unique(labels))
    return int(unique_vals.numel())


def build_data(data_dir: str) -> Data:
    features = load_tensor(os.path.join(data_dir, "features.pt"))
    edge_index = load_tensor(os.path.join(data_dir, "edge_index.pt"))
    edge_type = load_tensor(os.path.join(data_dir, "edge_type.pt"))
    edge_weight = load_tensor(os.path.join(data_dir, "edge_weight.pt"))
    labels_bot = load_tensor(os.path.join(data_dir, "labels_bot.pt"))
    labels_stance = load_tensor(os.path.join(data_dir, "labels_stance.pt"))

    if features is None or edge_index is None or labels_bot is None or labels_stance is None:
        raise FileNotFoundError(
            "Missing required tensors. Expected features.pt, edge_index.pt, labels_bot.pt, labels_stance.pt"
        )

    # Ensure shapes are correct
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index should have shape [2, num_edges]")

    num_nodes = features.size(0)
    if labels_bot.size(0) != num_nodes or labels_stance.size(0) != num_nodes:
        raise ValueError("Label sizes must match number of nodes in features")

    # Sanitize types
    labels_bot = labels_bot.to(torch.float32)
    if labels_stance.dtype.is_floating_point:
        labels_stance = labels_stance.long()
    else:
        labels_stance = labels_stance.long()

    # Feature processing: augmentation + normalization
    x = features.to(torch.float32)
    x = augment_behavioral_features(x, edge_index.long())
    x = normalize_features(x)

    data = Data(
        x=x,
        edge_index=edge_index.long(),
        edge_type=None if edge_type is None else edge_type.long(),
        edge_weight=None if edge_weight is None else edge_weight.to(torch.float32),
        y_bot=labels_bot,
        y_stance=labels_stance,
    )

    return data


def create_masks(num_nodes: int, train_ratio: float, val_ratio: float, seed: int) -> Tuple[Tensor, Tensor, Tensor]:
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_nodes, generator=gen)
    num_train = int(num_nodes * train_ratio)
    num_val = int(num_nodes * val_ratio)
    train_idx = indices[:num_train]
    val_idx = indices[num_train : num_train + num_val]
    test_idx = indices[num_train + num_val :]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def compute_metrics_bot(logits: Tensor, labels: Tensor) -> Tuple[float, float, float, float, float]:
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        labels_i = labels.long()
        correct = (preds == labels_i).sum().item()
        total = labels_i.numel()
        accuracy = correct / max(total, 1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_i.cpu().numpy(), preds.cpu().numpy(), average="binary", zero_division=0
        )
        try:
            roc_auc = roc_auc_score(labels_i.cpu().numpy(), probs.cpu().numpy())
        except Exception:
            roc_auc = 0.0
    return float(accuracy), float(precision), float(recall), float(f1), float(roc_auc)


def compute_metrics_stance(logits: Tensor, labels: Tensor) -> Tuple[float, float, float, float]:
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        correct = (preds == labels).sum().item()
        total = labels.numel()
        accuracy = correct / max(total, 1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels.cpu().numpy(), preds.cpu().numpy(), average="macro", zero_division=0
        )
    return float(accuracy), float(precision), float(recall), float(f1)


def train_one_epoch(
    model: Module,
    data: Data,
    optimizer: torch.optim.Optimizer,
    train_mask: Tensor,
    device: torch.device,
    use_amp: bool,
    scaler: Optional[torch.cuda.amp.GradScaler],
    grad_clip: float,
    pos_weight: Tensor,
    class_weights: Tensor,
) -> Tuple[float, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_type = None if data.edge_type is None else data.edge_type.to(device)
    edge_weight = None if data.edge_weight is None else data.edge_weight.to(device)
    y_bot = data.y_bot.to(device)
    y_stance = data.y_stance.to(device)

    bot_crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    stance_crit = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    if use_amp and device.type == "cuda":
        with torch.cuda.amp.autocast():
            bot_logits, stance_logits = model(x, edge_index, edge_type, edge_weight)
            loss_bot = bot_crit(bot_logits[train_mask], y_bot[train_mask])
            loss_stance = stance_crit(stance_logits[train_mask], y_stance[train_mask])
            loss = loss_bot + loss_stance
        assert scaler is not None
        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        bot_logits, stance_logits = model(x, edge_index, edge_type, edge_weight)
        loss_bot = bot_crit(bot_logits[train_mask], y_bot[train_mask])
        loss_stance = stance_crit(stance_logits[train_mask], y_stance[train_mask])
        loss = loss_bot + loss_stance
        loss.backward()
        if grad_clip and grad_clip > 0:
            clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
    return float(loss_bot.item()), float(loss_stance.item())


@torch.no_grad()
def evaluate(model: Module, data: Data, mask: Tensor, device: torch.device) -> Dict[str, float]:
    model.eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_type = None if data.edge_type is None else data.edge_type.to(device)
    edge_weight = None if data.edge_weight is None else data.edge_weight.to(device)
    y_bot = data.y_bot.to(device)
    y_stance = data.y_stance.to(device)

    bot_logits, stance_logits = model(x, edge_index, edge_type, edge_weight)
    acc_bot, prec_bot, rec_bot, f1_bot, auc_bot = compute_metrics_bot(bot_logits[mask], y_bot[mask])
    acc_st, prec_st, rec_st, f1_st = compute_metrics_stance(stance_logits[mask], y_stance[mask])
    return {
        "bot_acc": acc_bot,
        "bot_precision": prec_bot,
        "bot_recall": rec_bot,
        "bot_f1": f1_bot,
        "bot_auc": auc_bot,
        "stance_acc": acc_st,
        "stance_precision": prec_st,
        "stance_recall": rec_st,
        "stance_f1": f1_st,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GNN on MGTAB tensors")
    parser.add_argument("--data-dir", type=str, default=os.path.join("MGTAB"), help="Directory containing *.pt tensors")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--print-pyg-install", action="store_true", help="Print matching PyG wheels install commands and exit")
    parser.add_argument("--save-best", type=str, default="", help="Path to save best model (state_dict + metadata)")
    parser.add_argument("--use-amp", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max-norm (0 to disable)")
    parser.add_argument("--log-dir", type=str, default="", help="TensorBoard log directory")
    parser.add_argument("--resume", type=str, default="", help="Resume training from checkpoint path")
    parser.add_argument("--use-mini-batch", action="store_true", help="Enable NeighborLoader mini-batching (optional)")
    args = parser.parse_args()

    if args.print_pyg_install:
        print(_install_instructions(torch.__version__, torch.version.cuda))
        return

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data = build_data(args.data_dir)

    stance_num_classes = infer_num_classes(data.y_stance)
    num_relations: Optional[int] = None
    if getattr(data, "edge_type", None) is not None and data.edge_type is not None:
        num_relations = int(torch.max(data.edge_type).item() + 1)

    model = MGTABModel(
        input_dim=data.x.size(1),
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
        stance_num_classes=stance_num_classes,
        num_relations=num_relations,
    )
    device = torch.device(args.device)
    model.to(device)

    train_mask, val_mask, test_mask = create_masks(data.num_nodes, args.train_ratio, args.val_ratio, args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.use_amp and torch.cuda.is_available()))
    writer = SummaryWriter(args.log_dir) if args.log_dir else None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=max(3, args.patience // 2))

    # Class imbalance handling
    num_pos = float(data.y_bot[train_mask].sum().item())
    num_neg = float(train_mask.sum().item()) - num_pos
    pos_weight = torch.tensor([num_neg / max(num_pos, 1.0)], dtype=torch.float32)
    counts = torch.bincount(data.y_stance[train_mask].cpu(), minlength=stance_num_classes).float()
    class_weights = (counts.sum() / (counts + 1e-6))
    class_weights = class_weights * (stance_num_classes / class_weights.sum())

    best_val = -1.0
    best_state = None
    best_payload: Optional[Dict[str, Any]] = None
    epochs_no_improve = 0

    # Optional resume
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])  # type: ignore
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])  # type: ignore
        if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])  # type: ignore
        if "metrics" in ckpt and "composite" in ckpt["metrics"]:
            best_val = float(ckpt["metrics"]["composite"])  # type: ignore

    for epoch in range(1, args.epochs + 1):
        loss_bot, loss_stance = train_one_epoch(
            model, data, optimizer, train_mask, device, args.use_amp, scaler, args.grad_clip, pos_weight, class_weights
        )
        val_metrics = evaluate(model, data, val_mask, device)
        composite = (val_metrics["bot_f1"] + val_metrics["stance_acc"]) / 2.0
        scheduler.step(composite)

        if composite > best_val:
            best_val = composite
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if args.save_best:
                best_payload = {
                    "state_dict": best_state,
                    "optimizer": optimizer.state_dict(),
                    "scaler": (scaler.state_dict() if scaler is not None else None),
                    "metrics": {**val_metrics, "composite": composite},
                    "config": {
                        "input_dim": int(data.x.size(1)),
                        "hidden_dim": int(args.hidden_dim),
                        "layers": int(args.layers),
                        "dropout": float(args.dropout),
                        "stance_num_classes": int(stance_num_classes),
                        "num_relations": None if num_relations is None else int(num_relations),
                        "use_amp": bool(args.use_amp),
                    },
                }
                torch.save(best_payload, args.save_best)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if writer is not None:
            writer.add_scalar("loss/bot", loss_bot, epoch)
            writer.add_scalar("loss/stance", loss_stance, epoch)
            writer.add_scalar("metrics/val_bot_f1", val_metrics["bot_f1"], epoch)
            writer.add_scalar("metrics/val_stance_acc", val_metrics["stance_acc"], epoch)
            writer.add_scalar("metrics/val_composite", composite, epoch)
            writer.add_scalar("opt/lr", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"Epoch {epoch:03d} | loss_bot={loss_bot:.4f} loss_stance={loss_stance:.4f} | "
            f"val_bot_acc={val_metrics['bot_acc']:.4f} val_bot_prec={val_metrics['bot_precision']:.4f} "
            f"val_bot_rec={val_metrics['bot_recall']:.4f} val_bot_f1={val_metrics['bot_f1']:.4f} val_bot_auc={val_metrics['bot_auc']:.4f} | "
            f"val_stance_acc={val_metrics['stance_acc']:.4f} val_stance_prec={val_metrics['stance_precision']:.4f} "
            f"val_stance_rec={val_metrics['stance_recall']:.4f} val_stance_f1={val_metrics['stance_f1']:.4f} | "
            f"val_composite={composite:.4f}"
        )

        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered after {epoch} epochs. Best composite={best_val:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        if args.save_best and not os.path.exists(args.save_best):
            torch.save({"state_dict": best_state}, args.save_best)

    test_metrics = evaluate(model, data, test_mask, device)
    print(
        "TEST | "
        f"bot_acc={test_metrics['bot_acc']:.4f} bot_prec={test_metrics['bot_precision']:.4f} bot_rec={test_metrics['bot_recall']:.4f} "
        f"bot_f1={test_metrics['bot_f1']:.4f} bot_auc={test_metrics['bot_auc']:.4f} | "
        f"stance_acc={test_metrics['stance_acc']:.4f} stance_prec={test_metrics['stance_precision']:.4f} "
        f"stance_rec={test_metrics['stance_recall']:.4f} stance_f1={test_metrics['stance_f1']:.4f}"
    )
    if writer is not None:
        for k, v in test_metrics.items():
            writer.add_scalar(f"test/{k}", v)
        writer.close()


if __name__ == "__main__":
    main()


