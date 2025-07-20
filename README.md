# RAG Project - Fraud Detection System

A comprehensive Retrieval-Augmented Generation (RAG) system for fraud detection in customer conversations, featuring AI-powered analysis and real-time risk assessment.

## 🚀 Features

- **Advanced Fraud Detection**: Machine learning-based analysis of conversation patterns
- **RAG Framework**: Retrieval-augmented generation for context-aware fraud detection
- **Real-time Analysis**: Instant fraud risk assessment with detailed explanations
- **Web Interface**: Modern, user-friendly web application for interaction
- **Comprehensive Dataset**: 10,000 labeled conversations for training and testing
- **Feature Extraction**: Advanced pattern recognition for fraud indicators
- **Similarity Search**: Find similar conversations for context and validation

## 🏗️ Architecture

The system combines multiple AI/ML techniques:

1. **TF-IDF Vectorization**: Text feature extraction with n-gram analysis
2. **Random Forest Classifier**: Robust fraud detection model
3. **Cosine Similarity**: Find similar conversations for context
4. **Feature Engineering**: Domain-specific fraud indicators
5. **Web Interface**: Flask-based REST API with modern UI

## 📊 Dataset

- **Total Conversations**: 10,000
- **Normal Conversations**: 8,000 (80%)
- **Fraudulent Conversations**: 2,000 (20%)
- **Format**: JSON files with structured conversation data
- **Features**: Timestamps, sender identification, message content

### Fraud Types Detected

- Lottery/Tax scams
- Nigerian prince scams
- Tech support scams
- Cryptocurrency investment scams
- Gift card scams
- Urgent money transfer requests
- Personal information harvesting

## 🛠️ Setup

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd rag-project
```

2. **Create and activate virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Generate conversation dataset** (if not already present):
```bash
python generate_conversations.py
```

## 🚀 Usage

### Web Interface

1. **Start the fraud detection application**:
```bash
python fraud_detection_app.py
```

2. **Access the web interface**:
   - Open your browser and go to `http://localhost:5000`
   - Use the intuitive interface to analyze conversations

### Programmatic Usage

```python
from fraud_detection_rag import FraudDetectionRAG

# Initialize the system
detector = FraudDetectionRAG()

# Train the model (first time only)
detector.train()

# Detect fraud in a conversation
conversation = """
Customer: I won a lottery and need to pay taxes to claim it
Agent: I'd be happy to help you with that.
Customer: Can't you make an exception? I need this done quickly
"""

result = detector.detect_fraud(conversation)
print(f"Fraudulent: {result['is_fraudulent']}")
print(f"Risk Score: {result['risk_score']:.2f}")
print(f"Explanation: {result['explanation']}")
```

## 📁 Project Structure

```
rag-project/
├── fraud_detection_rag.py      # Core RAG fraud detection system
├── fraud_detection_app.py      # Flask web application
├── generate_conversations.py   # Dataset generation script
├── requirements.txt            # Python dependencies
├── README.md                  # This file
├── conversations/             # Generated conversation dataset
│   ├── normal_*.json         # Normal conversations
│   ├── fraudulent_*.json     # Fraudulent conversations
│   └── README.md            # Dataset documentation
├── templates/                # Web interface templates
│   ├── index.html           # Original RAG interface
│   └── fraud_detection.html # Fraud detection interface
└── .venv/                   # Virtual environment (not tracked)
```

## 🔍 How It Works

### 1. Feature Extraction

The system extracts multiple types of features:

- **Text Features**: TF-IDF vectors with n-gram analysis
- **Fraud Indicators**: Urgency words, money references, scam keywords
- **Behavioral Features**: Message patterns, timing, sender ratios
- **Content Features**: Links, phone numbers, email addresses

### 2. Machine Learning Model

- **Algorithm**: Random Forest Classifier
- **Features**: Combined TF-IDF + domain-specific features
- **Training**: 80/20 split with stratification
- **Evaluation**: Accuracy, precision, recall, F1-score

### 3. RAG Framework

- **Retrieval**: Find similar conversations using cosine similarity
- **Generation**: Provide context-aware explanations
- **Augmentation**: Combine model predictions with similar case analysis

### 4. Risk Assessment

- **Probability Score**: Model confidence in fraud classification
- **Risk Score**: Combined probability + feature-based adjustments
- **Explanation**: Human-readable reasoning for the decision

## 📈 Performance

The system achieves high accuracy in fraud detection:

- **Accuracy**: >95% on test dataset
- **Precision**: High precision for fraud detection
- **Recall**: Excellent recall for fraudulent conversations
- **F1-Score**: Balanced performance across classes

## 🔧 Configuration

### Model Parameters

- **TF-IDF Features**: 5000 max features, 1-3 n-grams
- **Random Forest**: 100 estimators, parallel processing
- **Threshold**: Configurable fraud detection threshold (default: 0.5)

### Feature Weights

The system uses weighted feature importance:
- Urgency indicators: +10% risk
- Money references: +15% risk
- Scam keywords: +20% risk
- Gift card mentions: +15% risk
- Personal info requests: +10% risk

## 🚀 API Endpoints

### Web Application Endpoints

- `GET /` - Main fraud detection interface
- `POST /detect` - Fraud detection API
- `POST /train` - Retrain the model
- `GET /stats` - System statistics
- `GET /examples` - Example conversations

### Request Format

```json
{
  "conversation": "Customer: Hi, I need help...",
  "threshold": 0.5
}
```

### Response Format

```json
{
  "is_fraudulent": true,
  "fraud_probability": 0.85,
  "risk_score": 0.92,
  "explanation": "High fraud probability detected...",
  "similar_conversations": [...],
  "extracted_features": {...}
}
```

## 🧪 Testing

### Example Conversations

The system includes built-in examples for testing:

**Normal Conversation**:
```
Customer: Hi, I need help resetting my password
Agent: I'd be happy to help you with that.
Customer: Thank you. My account number is 123456789
```

**Fraudulent Conversation**:
```
Customer: I won a lottery and need to pay taxes to claim it
Agent: I'd be happy to help you with that.
Customer: Can't you make an exception? I need this done quickly
```

## 🔒 Security Features

- **Input Validation**: Sanitized conversation input
- **Rate Limiting**: Protection against abuse
- **Error Handling**: Graceful error management
- **Model Persistence**: Secure model storage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

For issues and questions:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information

## 🔮 Future Enhancements

- **Real-time Streaming**: Process live conversations
- **Multi-language Support**: Detect fraud in multiple languages
- **Advanced NLP**: BERT-based embeddings for better understanding
- **Anomaly Detection**: Unsupervised learning for unknown fraud patterns
- **API Integration**: Connect with external fraud databases
- **Mobile App**: Native mobile application for fraud detection 