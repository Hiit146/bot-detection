# ML-Based Fake Account and Bot Detection (GNN-Enhanced)


---

## Overview
Social media platforms face increasing challenges from **fake accounts and automated bots** used for misinformation, spam, and manipulation.  
This project presents a **machine-learning-based system** that detects fake accounts in real time, enhanced by a **Graph Neural Network (GNN)** for social network analysis.

---

## Problem Statement
Fake and bot accounts closely mimic real user behavior, making them hard to identify using static, rule-based approaches.  
Our objective is to **develop an adaptive and scalable ML system** capable of accurately detecting such accounts based on behavioral, relational, and activity features.

---

## Proposed System

### Approach
1. Evaluate **10+ ML models** on labeled social media account datasets.  
2. Compare based on **accuracy, precision, recall, and F1-score**.  
3. Integrate **Graph Neural Network (GNN)** to capture complex user-interaction graphs.  

**Top Performing Traditional Models:**

| Model | Accuracy |
|-------|-----------|
| XGBoost | 98.49% |
| Random Forest | 98.25% |
| Gradient Boosting | 98.12% |

The GNN layer further improves detection accuracy for accounts with rich or deceptive interaction patterns.

---

## Key Features Used
- **follower_friend_ratio** – Strongest predictor of authenticity  
- **verified** – High-confidence signal for legitimacy  
- **followers_count**, **friends_count**, **status_frequency** – Behavioral metrics  
- **Dynamic activity ratios** – Capture time-based and evolving user patterns  

Behavioral and network-based features outperform static profile metrics.

---

## Graph Neural Network (GNN) Implementation
The GNN analyzes **node attributes** (account features) and **graph connectivity** (follower/following relationships).  
It enables the model to:
- Detect **community-level anomalies**  
- Recognize **bot clusters** that mimic real users  
- Identify subtle **interaction-based similarities** across accounts  

This complements traditional ML models by providing a **relational learning layer**.

---

## 🧱 System Architecture

**Modules:**
1. **Web Scraping** – Collects social-media data (via Selenium, BeautifulSoup, APIs)  
2. **Data Processing Pipeline** – Streams and cleans data using **Apache Kafka & Spark**  
3. **ML Inference Engine** – Uses **XGBoost + GNN + Redis Cache** for real-time prediction  

```text
User Data → Kafka Stream → Spark Processor → ML/GNN Model → Redis → Frontend Display
