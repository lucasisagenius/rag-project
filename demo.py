#!/usr/bin/env python3
"""
Demo script for the Fraud Detection RAG System
Shows how to use the system with various conversation examples
"""

from fraud_detection_rag import FraudDetectionRAG
import json

def main():
    print("=" * 60)
    print("FRAUD DETECTION RAG SYSTEM - DEMO")
    print("=" * 60)
    
    # Initialize the fraud detection system
    print("\n1. Initializing Fraud Detection System...")
    detector = FraudDetectionRAG()
    
    # Load or train the model
    try:
        detector.load_model('fraud_detection_model.pkl')
        print("   ✓ Model loaded successfully")
    except:
        print("   ⚠ Training new model...")
        detector.train()
        detector.save_model('fraud_detection_model.pkl')
        print("   ✓ Model trained and saved")
    
    # Test conversations
    test_conversations = [
        {
            "title": "Normal - Password Reset",
            "conversation": """Customer: Hi, I need help resetting my password
Agent: I'd be happy to help you with that. Can you provide your account number?
Customer: Thank you. My account number is 123456789
Agent: I'll need to ask you some security questions to verify your identity.""",
            "expected": "Normal"
        },
        {
            "title": "Normal - Account Update",
            "conversation": """Customer: I want to update my email address
Agent: I can help you with that. Let me verify your account information first.
Customer: Sure, what do you need?
Agent: Can you provide your account number and current email address?""",
            "expected": "Normal"
        },
        {
            "title": "Fraudulent - Lottery Scam",
            "conversation": """Customer: I won a lottery and need to pay taxes to claim it
Agent: I'd be happy to help you with that.
Customer: Can't you make an exception? I need this done quickly
Agent: I need to verify this through official channels.
Customer: I don't have time for all these questions. This is very urgent!""",
            "expected": "Fraudulent"
        },
        {
            "title": "Fraudulent - Tech Support Scam",
            "conversation": """Customer: I'm calling from Microsoft support about your computer
Agent: How can I help you today?
Customer: Your computer has a virus and I need your credit card to fix it
Agent: I cannot process that request without proper verification.
Customer: Just do what I'm asking, please. I'll pay you extra!""",
            "expected": "Fraudulent"
        },
        {
            "title": "Fraudulent - Nigerian Prince",
            "conversation": """Customer: I'm a Nigerian prince and need your help
Agent: How can I assist you?
Customer: I need to transfer money urgently to my sick relative
Agent: I need to verify this through official channels.
Customer: I'll pay you extra if you help me. This is very urgent!""",
            "expected": "Fraudulent"
        },
        {
            "title": "Fraudulent - Gift Card Scam",
            "conversation": """Customer: I need you to buy gift cards and send me the codes
Agent: I cannot process that request without proper verification.
Customer: This is very urgent, my family is in danger
Agent: That's not a standard procedure for our company.
Customer: I don't have time for all these questions. Just do it!""",
            "expected": "Fraudulent"
        }
    ]
    
    print(f"\n2. Testing {len(test_conversations)} Conversations...")
    print("-" * 60)
    
    correct_predictions = 0
    total_predictions = len(test_conversations)
    
    for i, test_case in enumerate(test_conversations, 1):
        print(f"\nTest {i}: {test_case['title']}")
        print("-" * 40)
        
        # Perform fraud detection
        result = detector.detect_fraud(test_case['conversation'])
        
        # Display results
        status = "✅" if result['is_fraudulent'] == (test_case['expected'] == "Fraudulent") else "❌"
        prediction = "Fraudulent" if result['is_fraudulent'] else "Normal"
        
        print(f"Expected: {test_case['expected']}")
        print(f"Predicted: {prediction} {status}")
        print(f"Fraud Probability: {result['fraud_probability']:.4f}")
        print(f"Risk Score: {result['risk_score']:.4f}")
        print(f"Explanation: {result['explanation']}")
        
        if result['is_fraudulent'] == (test_case['expected'] == "Fraudulent"):
            correct_predictions += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("DEMO SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_predictions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {(correct_predictions/total_predictions)*100:.1f}%")
    
    # Show similar conversations for one example
    print(f"\n3. Similar Conversations Analysis...")
    print("-" * 60)
    
    example_conversation = test_conversations[2]["conversation"]  # Lottery scam
    result = detector.detect_fraud(example_conversation)
    
    print(f"Example: {test_conversations[2]['title']}")
    print(f"Similar conversations found:")
    
    for i, similar in enumerate(result['similar_conversations'][:3], 1):
        print(f"  {i}. Similarity: {similar['similarity_score']:.3f}")
        print(f"     Fraudulent: {similar['is_fraudulent']}")
        print(f"     Preview: {similar['conversation_text'][:100]}...")
        print()
    
    print("=" * 60)
    print("Demo completed! 🎉")
    print("=" * 60)
    print("\nTo run the web interface:")
    print("  python fraud_detection_app.py")
    print("  Then visit: http://localhost:5000")

if __name__ == "__main__":
    main() 