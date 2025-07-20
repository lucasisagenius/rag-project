import json
import os
import numpy as np
from datetime import datetime
import re
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionRAG:
    def __init__(self, conversations_dir: str = "conversations"):
        """
        Initialize the Fraud Detection RAG system.
        
        Args:
            conversations_dir: Directory containing conversation JSON files
        """
        self.conversations_dir = conversations_dir
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2
        )
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.conversation_embeddings = None
        self.conversation_labels = None
        self.conversation_texts = []
        self.is_trained = False
        
    def load_conversations(self) -> Tuple[List[str], List[bool]]:
        """
        Load all conversations from JSON files and extract features.
        
        Returns:
            Tuple of (conversation_texts, fraud_labels)
        """
        conversation_texts = []
        fraud_labels = []
        
        print("Loading conversations...")
        
        # Load normal conversations
        normal_files = [f for f in os.listdir(self.conversations_dir) 
                       if f.startswith('normal_') and f.endswith('.json')]
        
        for filename in normal_files:
            filepath = os.path.join(self.conversations_dir, filename)
            with open(filepath, 'r') as f:
                conversation = json.load(f)
                text = self._extract_conversation_text(conversation)
                conversation_texts.append(text)
                fraud_labels.append(conversation['fraud_risk'])
        
        # Load fraudulent conversations
        fraud_files = [f for f in os.listdir(self.conversations_dir) 
                      if f.startswith('fraudulent_') and f.endswith('.json')]
        
        for filename in fraud_files:
            filepath = os.path.join(self.conversations_dir, filename)
            with open(filepath, 'r') as f:
                conversation = json.load(f)
                text = self._extract_conversation_text(conversation)
                conversation_texts.append(text)
                fraud_labels.append(conversation['fraud_risk'])
        
        print(f"Loaded {len(conversation_texts)} conversations")
        print(f"Normal conversations: {sum(1 for label in fraud_labels if not label)}")
        print(f"Fraudulent conversations: {sum(1 for label in fraud_labels if label)}")
        
        return conversation_texts, fraud_labels
    
    def _extract_conversation_text(self, conversation: Dict) -> str:
        """
        Extract and preprocess text from a conversation.
        
        Args:
            conversation: Conversation dictionary
            
        Returns:
            Preprocessed conversation text
        """
        messages = conversation.get('messages', [])
        text_parts = []
        
        for message in messages:
            sender = message.get('sender', '')
            content = message.get('message', '')
            
            # Add sender context
            if sender == 'customer':
                text_parts.append(f"CUSTOMER: {content}")
            elif sender == 'agent':
                text_parts.append(f"AGENT: {content}")
            else:
                text_parts.append(content)
        
        return " ".join(text_parts)
    
    def _extract_features(self, text: str) -> Dict:
        """
        Extract fraud-related features from conversation text.
        
        Args:
            text: Conversation text
            
        Returns:
            Dictionary of extracted features
        """
        text_lower = text.lower()
        
        features = {
            'urgency_words': len(re.findall(r'\b(urgent|quickly|hurry|emergency|immediately|asap)\b', text_lower)),
            'money_words': len(re.findall(r'\b(money|cash|transfer|payment|bank|account|credit|card)\b', text_lower)),
            'pressure_words': len(re.findall(r'\b(pressure|insist|demand|must|need|require)\b', text_lower)),
            'scam_keywords': len(re.findall(r'\b(lottery|prince|inheritance|tax|refund|support|microsoft|apple)\b', text_lower)),
            'gift_card_words': len(re.findall(r'\b(gift.?card|itunes|google.?play|amazon)\b', text_lower)),
            'crypto_words': len(re.findall(r'\b(cryptocurrency|bitcoin|ethereum|crypto|investment)\b', text_lower)),
            'personal_info_words': len(re.findall(r'\b(social.?security|ssn|password|pin|account.?number)\b', text_lower)),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'message_count': len(re.findall(r'(CUSTOMER:|AGENT:)', text)),
            'customer_message_ratio': len(re.findall(r'CUSTOMER:', text)) / max(len(re.findall(r'(CUSTOMER:|AGENT:)', text)), 1),
            'avg_message_length': np.mean([len(msg.split()) for msg in text.split('CUSTOMER:') + text.split('AGENT:') if msg.strip()]),
            'contains_links': 1 if re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text) else 0,
            'contains_phone': 1 if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text) else 0,
            'contains_email': 1 if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text) else 0
        }
        
        return features
    
    def train(self):
        """
        Train the fraud detection model using the conversation dataset.
        """
        print("Training fraud detection model...")
        
        # Load conversations
        conversation_texts, fraud_labels = self.load_conversations()
        self.conversation_texts = conversation_texts
        self.conversation_labels = fraud_labels
        
        # Create TF-IDF embeddings
        print("Creating TF-IDF embeddings...")
        self.conversation_embeddings = self.vectorizer.fit_transform(conversation_texts)
        
        # Extract additional features
        print("Extracting fraud-related features...")
        additional_features = []
        for text in conversation_texts:
            features = self._extract_features(text)
            additional_features.append(list(features.values()))
        
        # Combine TF-IDF features with additional features
        additional_features_array = np.array(additional_features)
        combined_features = np.hstack([
            self.conversation_embeddings.toarray(),
            additional_features_array
        ])
        
        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(
            combined_features, fraud_labels, test_size=0.2, random_state=42, stratify=fraud_labels
        )
        
        # Train classifier
        print("Training Random Forest classifier...")
        self.classifier.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraudulent']))
        
        self.is_trained = True
        print("Training completed!")
    
    def detect_fraud(self, conversation_text: str, threshold: float = 0.5) -> Dict:
        """
        Detect fraud risk in a new conversation.
        
        Args:
            conversation_text: Text of the conversation to analyze
            threshold: Probability threshold for fraud classification
            
        Returns:
            Dictionary containing fraud detection results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before detecting fraud")
        
        # Preprocess conversation text
        processed_text = self._preprocess_new_conversation(conversation_text)
        
        # Create TF-IDF embedding
        text_embedding = self.vectorizer.transform([processed_text])
        
        # Extract additional features
        features = self._extract_features(processed_text)
        additional_features = np.array([list(features.values())])
        
        # Combine features
        combined_features = np.hstack([
            text_embedding.toarray(),
            additional_features
        ])
        
        # Get prediction and probability
        fraud_probability = self.classifier.predict_proba(combined_features)[0][1]
        is_fraudulent = fraud_probability >= threshold
        
        # Find similar conversations
        similar_conversations = self._find_similar_conversations(processed_text, top_k=5)
        
        # Generate risk score and explanation
        risk_score = self._calculate_risk_score(features, fraud_probability)
        explanation = self._generate_explanation(features, similar_conversations, fraud_probability)
        
        return {
            'is_fraudulent': is_fraudulent,
            'fraud_probability': fraud_probability,
            'risk_score': risk_score,
            'threshold': threshold,
            'explanation': explanation,
            'similar_conversations': similar_conversations,
            'extracted_features': features
        }
    
    def _preprocess_new_conversation(self, conversation_text: str) -> str:
        """
        Preprocess new conversation text to match training format.
        
        Args:
            conversation_text: Raw conversation text
            
        Returns:
            Preprocessed text
        """
        # Simple preprocessing - in a real system, you'd want more sophisticated parsing
        lines = conversation_text.strip().split('\n')
        processed_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try to identify sender (customer vs agent)
            if any(word in line.lower() for word in ['customer:', 'user:', 'client:']):
                processed_lines.append(f"CUSTOMER: {line.split(':', 1)[1] if ':' in line else line}")
            elif any(word in line.lower() for word in ['agent:', 'support:', 'representative:']):
                processed_lines.append(f"AGENT: {line.split(':', 1)[1] if ':' in line else line}")
            else:
                # Assume it's a customer message if no clear sender
                processed_lines.append(f"CUSTOMER: {line}")
        
        return " ".join(processed_lines)
    
    def _find_similar_conversations(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Find similar conversations from the training dataset.
        
        Args:
            query_text: Query conversation text
            top_k: Number of similar conversations to return
            
        Returns:
            List of similar conversations with similarity scores
        """
        # Create query embedding
        query_embedding = self.vectorizer.transform([query_text])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.conversation_embeddings)[0]
        
        # Get top-k similar conversations
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        similar_conversations = []
        for idx in top_indices:
            similar_conversations.append({
                'conversation_text': self.conversation_texts[idx][:200] + "...",
                'similarity_score': similarities[idx],
                'is_fraudulent': self.conversation_labels[idx],
                'index': idx
            })
        
        return similar_conversations
    
    def _calculate_risk_score(self, features: Dict, fraud_probability: float) -> float:
        """
        Calculate a comprehensive risk score based on features and probability.
        
        Args:
            features: Extracted features
            fraud_probability: Model prediction probability
            
        Returns:
            Risk score between 0 and 1
        """
        # Base risk from model probability
        base_risk = fraud_probability
        
        # Feature-based risk adjustments
        feature_risk = 0.0
        
        # High-risk features
        if features['urgency_words'] > 2:
            feature_risk += 0.1
        if features['money_words'] > 3:
            feature_risk += 0.15
        if features['scam_keywords'] > 1:
            feature_risk += 0.2
        if features['gift_card_words'] > 0:
            feature_risk += 0.15
        if features['crypto_words'] > 0:
            feature_risk += 0.1
        if features['personal_info_words'] > 0:
            feature_risk += 0.1
        if features['exclamation_count'] > 3:
            feature_risk += 0.05
        if features['uppercase_ratio'] > 0.3:
            feature_risk += 0.05
        
        # Combine base risk and feature risk
        total_risk = min(1.0, base_risk + feature_risk)
        
        return total_risk
    
    def _generate_explanation(self, features: Dict, similar_conversations: List[Dict], fraud_probability: float) -> str:
        """
        Generate a human-readable explanation for the fraud detection result.
        
        Args:
            features: Extracted features
            similar_conversations: Similar conversations found
            fraud_probability: Model prediction probability
            
        Returns:
            Explanation string
        """
        explanation_parts = []
        
        # Add probability-based explanation
        if fraud_probability > 0.8:
            explanation_parts.append("High fraud probability detected by the model.")
        elif fraud_probability > 0.6:
            explanation_parts.append("Moderate fraud probability detected by the model.")
        elif fraud_probability > 0.4:
            explanation_parts.append("Low fraud probability detected by the model.")
        else:
            explanation_parts.append("Very low fraud probability detected by the model.")
        
        # Add feature-based explanations
        if features['urgency_words'] > 2:
            explanation_parts.append("Contains multiple urgency indicators.")
        if features['money_words'] > 3:
            explanation_parts.append("Heavy focus on money and financial transactions.")
        if features['scam_keywords'] > 1:
            explanation_parts.append("Contains known scam-related keywords.")
        if features['gift_card_words'] > 0:
            explanation_parts.append("Mentions gift cards (common in scams).")
        if features['crypto_words'] > 0:
            explanation_parts.append("Discusses cryptocurrency investments.")
        if features['personal_info_words'] > 0:
            explanation_parts.append("Requests sensitive personal information.")
        
        # Add similar conversation context
        fraud_similar = sum(1 for conv in similar_conversations if conv['is_fraudulent'])
        if fraud_similar > 0:
            explanation_parts.append(f"Similar to {fraud_similar} known fraudulent conversations.")
        
        return " ".join(explanation_parts)
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'conversation_embeddings': self.conversation_embeddings,
            'conversation_labels': self.conversation_labels,
            'conversation_texts': self.conversation_texts
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.conversation_embeddings = model_data['conversation_embeddings']
        self.conversation_labels = model_data['conversation_labels']
        self.conversation_texts = model_data['conversation_texts']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")


def main():
    """
    Main function to demonstrate the fraud detection RAG system.
    """
    # Initialize the fraud detection system
    fraud_detector = FraudDetectionRAG()
    
    # Train the model
    fraud_detector.train()
    
    # Save the trained model
    fraud_detector.save_model('fraud_detection_model.pkl')
    
    # Test with example conversations
    test_conversations = [
        # Normal conversation
        """Customer: Hi, I need help resetting my password
Agent: I'd be happy to help you with that. Can you provide your account number?
Customer: Thank you. My account number is 123456789
Agent: I'll need to ask you some security questions to verify your identity.""",
        
        # Fraudulent conversation
        """Customer: I won a lottery and need to pay taxes to claim it
Agent: I'd be happy to help you with that.
Customer: Can't you make an exception? I need this done quickly
Agent: I need to verify this through official channels.
Customer: I don't have time for all these questions. This is very urgent!""",
        
        # Another fraudulent conversation
        """Customer: I'm calling from Microsoft support about your computer
Agent: How can I help you today?
Customer: Your computer has a virus and I need your credit card to fix it
Agent: I cannot process that request without proper verification.
Customer: Just do what I'm asking, please. I'll pay you extra!"""
    ]
    
    print("\n" + "="*60)
    print("TESTING FRAUD DETECTION SYSTEM")
    print("="*60)
    
    for i, conversation in enumerate(test_conversations, 1):
        print(f"\nTest Conversation {i}:")
        print("-" * 40)
        print(conversation)
        print("-" * 40)
        
        result = fraud_detector.detect_fraud(conversation)
        
        print(f"Fraud Detection Result:")
        print(f"  Is Fraudulent: {result['is_fraudulent']}")
        print(f"  Fraud Probability: {result['fraud_probability']:.4f}")
        print(f"  Risk Score: {result['risk_score']:.4f}")
        print(f"  Explanation: {result['explanation']}")
        print()


if __name__ == "__main__":
    main() 