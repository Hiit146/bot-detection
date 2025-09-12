"""
Bot Detection Web Application
Flask-based web app for detecting bots and fake accounts using ML models
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.exceptions import RequestEntityTooLarge
import os
import json
from datetime import datetime
import logging

# Import your ML model integration here
# from models.bot_detector_model import BotDetectorModel

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockBotDetector:
    """Mock bot detector for demonstration - replace with your actual ML model"""
    
    def predict(self, username, platform='twitter'):
        """
        Mock prediction function - replace with your actual ML model prediction
        
        Args:
            username (str): The username to analyze
            platform (str): The social media platform
            
        Returns:
            dict: Prediction results with confidence score
        """
        # Replace this section with your actual ML model
        import random
        
        suspicious_patterns = ['bot', 'fake', '123', 'auto', 'spam']
        is_suspicious = any(pattern in username.lower() for pattern in suspicious_patterns)
        
        if is_suspicious:
            confidence = random.uniform(0.7, 0.95)
            is_bot = True
        else:
            confidence = random.uniform(0.1, 0.4)
            is_bot = False
            
        return {
            'is_bot': is_bot,
            'confidence': round(confidence, 3),
            'risk_level': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.4 else 'Low',
            'features_analyzed': ['username_pattern', 'account_age', 'posting_frequency'],
            'platform': platform,
            'timestamp': datetime.now().isoformat()
        }

# Initialize detector (replace with your actual model)
bot_detector = MockBotDetector()

@app.route('/')
def index():
    """Home page with detection form"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_bot():
    """Handle bot detection requests"""
    try:
        username = request.form.get('username', '').strip()
        platform = request.form.get('platform', 'twitter')
        
        if not username:
            flash('Please enter a username', 'error')
            return redirect(url_for('index'))
            
        if len(username) > 100:
            flash('Username too long (max 100 characters)', 'error')
            return redirect(url_for('index'))
            
        logger.info(f"Detection request for username: {username}, platform: {platform}")
        
        # Run ML prediction
        prediction = bot_detector.predict(username, platform)
        
        return render_template('result.html', 
                             username=username, 
                             platform=platform,
                             prediction=prediction)
                             
    except Exception as e:
        logger.error(f"Error during detection: {str(e)}")
        flash('An error occurred during detection. Please try again.', 'error')
        return redirect(url_for('index'))

@app.route('/api/detect', methods=['POST'])
def api_detect_bot():
    """API endpoint for bot detection"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        username = data.get('username', '').strip()
        platform = data.get('platform', 'twitter')
        
        if not username:
            return jsonify({'error': 'Username is required'}), 400
            
        if len(username) > 100:
            return jsonify({'error': 'Username too long (max 100 characters)'}), 400
            
        prediction = bot_detector.predict(username, platform)
        
        return jsonify({
            'success': True,
            'username': username,
            'platform': platform,
            'prediction': prediction
        })
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
