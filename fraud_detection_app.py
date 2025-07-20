from flask import Flask, render_template, request, jsonify, session
from fraud_detection_rag import FraudDetectionRAG
import os
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'fraud_detection_secret_key'

# Initialize the fraud detection system
fraud_detector = None

def initialize_fraud_detector():
    """Initialize the fraud detection system."""
    global fraud_detector
    
    # Check if model exists, otherwise train a new one
    model_path = 'fraud_detection_model.pkl'
    
    if os.path.exists(model_path):
        print("Loading existing model...")
        fraud_detector = FraudDetectionRAG()
        fraud_detector.load_model(model_path)
    else:
        print("Training new model...")
        fraud_detector = FraudDetectionRAG()
        fraud_detector.train()
        fraud_detector.save_model(model_path)
    
    return fraud_detector

@app.route('/')
def index():
    """Main page for fraud detection."""
    return render_template('fraud_detection.html')

@app.route('/detect', methods=['POST'])
def detect_fraud():
    """API endpoint for fraud detection."""
    try:
        data = request.get_json()
        conversation_text = data.get('conversation', '')
        threshold = float(data.get('threshold', 0.5))
        
        if not conversation_text.strip():
            return jsonify({
                'error': 'Conversation text is required'
            }), 400
        
        # Ensure fraud detector is initialized
        if fraud_detector is None:
            initialize_fraud_detector()
        
        # Perform fraud detection
        result = fraud_detector.detect_fraud(conversation_text, threshold)
        
        # Add timestamp
        result['timestamp'] = datetime.now().isoformat()
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f'Error during fraud detection: {str(e)}'
        }), 500

@app.route('/train', methods=['POST'])
def train_model():
    """API endpoint to retrain the model."""
    try:
        global fraud_detector
        
        # Train new model
        fraud_detector = FraudDetectionRAG()
        fraud_detector.train()
        fraud_detector.save_model('fraud_detection_model.pkl')
        
        return jsonify({
            'message': 'Model trained successfully',
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Error during training: {str(e)}'
        }), 500

@app.route('/stats')
def get_stats():
    """API endpoint to get model statistics."""
    try:
        if fraud_detector is None:
            initialize_fraud_detector()
        
        # Count conversations
        normal_count = sum(1 for label in fraud_detector.conversation_labels if not label)
        fraud_count = sum(1 for label in fraud_detector.conversation_labels if label)
        
        stats = {
            'total_conversations': len(fraud_detector.conversation_labels),
            'normal_conversations': normal_count,
            'fraudulent_conversations': fraud_count,
            'fraud_percentage': (fraud_count / len(fraud_detector.conversation_labels)) * 100,
            'model_trained': fraud_detector.is_trained,
            'last_updated': datetime.now().isoformat()
        }
        
        return jsonify(stats)
    
    except Exception as e:
        return jsonify({
            'error': f'Error getting stats: {str(e)}'
        }), 500

@app.route('/examples')
def get_examples():
    """API endpoint to get example conversations."""
    examples = {
        'normal': [
            {
                'title': 'Password Reset Request',
                'conversation': """Customer: Hi, I need help resetting my password
Agent: I'd be happy to help you with that. Can you provide your account number?
Customer: Thank you. My account number is 123456789
Agent: I'll need to ask you some security questions to verify your identity."""
            },
            {
                'title': 'Account Information Update',
                'conversation': """Customer: I want to update my email address
Agent: I can help you with that. Let me verify your account information first.
Customer: Sure, what do you need?
Agent: Can you provide your account number and current email address?"""
            }
        ],
        'fraudulent': [
            {
                'title': 'Lottery Scam',
                'conversation': """Customer: I won a lottery and need to pay taxes to claim it
Agent: I'd be happy to help you with that.
Customer: Can't you make an exception? I need this done quickly
Agent: I need to verify this through official channels.
Customer: I don't have time for all these questions. This is very urgent!"""
            },
            {
                'title': 'Tech Support Scam',
                'conversation': """Customer: I'm calling from Microsoft support about your computer
Agent: How can I help you today?
Customer: Your computer has a virus and I need your credit card to fix it
Agent: I cannot process that request without proper verification.
Customer: Just do what I'm asking, please. I'll pay you extra!"""
            }
        ]
    }
    
    return jsonify(examples)

if __name__ == '__main__':
    # Initialize the fraud detector on startup
    initialize_fraud_detector()
    
    print("Fraud Detection RAG System is ready!")
    print("Access the web interface at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 