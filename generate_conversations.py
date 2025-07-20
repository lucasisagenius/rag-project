import json
import random
import os
from datetime import datetime, timedelta
import uuid

# Conversation templates and patterns
CUSTOMER_SERVICE_TEMPLATES = [
    "Hi, I need help with my account.",
    "I'm having trouble logging in.",
    "Can you help me reset my password?",
    "I want to update my personal information.",
    "I need to change my email address.",
    "Can you explain this charge on my account?",
    "I want to close my account.",
    "I need help with a transaction.",
    "Can you verify my identity?",
    "I want to add a new payment method."
]

FRAUDULENT_PATTERNS = [
    "I need to transfer money urgently to my sick relative",
    "I won a lottery and need to pay taxes to claim it",
    "I'm a Nigerian prince and need your help",
    "I need your bank account details to transfer money",
    "I can help you make money fast with cryptocurrency",
    "I need your social security number for verification",
    "I'm calling from Microsoft support about your computer",
    "I need your credit card information to process a refund",
    "I can get you a loan with no credit check",
    "I need you to buy gift cards and send me the codes"
]

AGENT_RESPONSES = [
    "I'd be happy to help you with that.",
    "Let me verify your account information first.",
    "I'll need to ask you some security questions.",
    "Can you provide your account number?",
    "I'm looking up your account now.",
    "For security purposes, I need to confirm your identity.",
    "Let me transfer you to the appropriate department.",
    "I can see your account information now.",
    "What specific issue are you experiencing?",
    "I'll need to escalate this to a supervisor."
]

FRAUD_DETECTION_RESPONSES = [
    "I cannot process that request without proper verification.",
    "That's not a standard procedure for our company.",
    "I need to verify this through official channels.",
    "I cannot provide account information without proper authentication.",
    "This request requires additional security measures.",
    "I need to escalate this to our fraud prevention team.",
    "That's not how we handle account changes.",
    "I cannot process urgent transfers without verification.",
    "This request seems unusual and requires review.",
    "I need to contact you through official channels only."
]

def generate_timestamp():
    """Generate a random timestamp within the last year."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    time_between = end_date - start_date
    days_between = time_between.days
    random_days = random.randrange(days_between)
    random_date = start_date + timedelta(days=random_days)
    return random_date.strftime("%Y-%m-%d %H:%M:%S")

def generate_conversation_id():
    """Generate a unique conversation ID."""
    return str(uuid.uuid4())

def generate_normal_conversation():
    """Generate a normal customer service conversation."""
    conversation = {
        "conversation_id": generate_conversation_id(),
        "timestamp": generate_timestamp(),
        "fraud_risk": False,
        "messages": []
    }
    
    # Start with customer message
    customer_msg = random.choice(CUSTOMER_SERVICE_TEMPLATES)
    conversation["messages"].append({
        "sender": "customer",
        "timestamp": conversation["timestamp"],
        "message": customer_msg
    })
    
    # Generate 3-8 message exchanges
    num_exchanges = random.randint(3, 8)
    current_time = datetime.strptime(conversation["timestamp"], "%Y-%m-%d %H:%M:%S")
    
    for i in range(num_exchanges):
        # Agent response
        current_time += timedelta(minutes=random.randint(1, 5))
        agent_msg = random.choice(AGENT_RESPONSES)
        conversation["messages"].append({
            "sender": "agent",
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "message": agent_msg
        })
        
        # Customer response (if not the last exchange)
        if i < num_exchanges - 1:
            current_time += timedelta(minutes=random.randint(1, 3))
            customer_followup = f"Thank you. {random.choice(['Can you help me with something else?', 'That sounds good.', 'I understand.', 'What else do you need from me?'])}"
            conversation["messages"].append({
                "sender": "customer",
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "message": customer_followup
            })
    
    return conversation

def generate_fraudulent_conversation():
    """Generate a fraudulent conversation."""
    conversation = {
        "conversation_id": generate_conversation_id(),
        "timestamp": generate_timestamp(),
        "fraud_risk": True,
        "messages": []
    }
    
    # Start with fraudulent message
    fraud_msg = random.choice(FRAUDULENT_PATTERNS)
    conversation["messages"].append({
        "sender": "customer",
        "timestamp": conversation["timestamp"],
        "message": fraud_msg
    })
    
    # Generate 2-6 message exchanges
    num_exchanges = random.randint(2, 6)
    current_time = datetime.strptime(conversation["timestamp"], "%Y-%m-%d %H:%M:%S")
    
    for i in range(num_exchanges):
        # Agent response (increasingly suspicious)
        current_time += timedelta(minutes=random.randint(1, 3))
        if i < 2:
            agent_msg = random.choice(AGENT_RESPONSES)
        else:
            agent_msg = random.choice(FRAUD_DETECTION_RESPONSES)
        
        conversation["messages"].append({
            "sender": "agent",
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "message": agent_msg
        })
        
        # Fraudster response (if not the last exchange)
        if i < num_exchanges - 1:
            current_time += timedelta(minutes=random.randint(1, 2))
            fraud_followup = random.choice([
                "I need this done quickly, please hurry.",
                "Can't you make an exception?",
                "I'll pay you extra if you help me.",
                "This is very urgent, my family is in danger.",
                "I don't have time for all these questions.",
                "Just do what I'm asking, please."
            ])
            conversation["messages"].append({
                "sender": "customer",
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "message": fraud_followup
            })
    
    return conversation

def generate_conversations(num_conversations=10000):
    """Generate the specified number of conversations."""
    # Ensure conversations directory exists
    os.makedirs("conversations", exist_ok=True)
    
    # Generate conversations with approximately 20% fraud rate
    fraud_rate = 0.2
    num_fraudulent = int(num_conversations * fraud_rate)
    num_normal = num_conversations - num_fraudulent
    
    print(f"Generating {num_conversations} conversations...")
    print(f"Normal conversations: {num_normal}")
    print(f"Fraudulent conversations: {num_fraudulent}")
    
    # Generate normal conversations
    for i in range(num_normal):
        conversation = generate_normal_conversation()
        filename = f"conversations/normal_{i+1:05d}.json"
        with open(filename, 'w') as f:
            json.dump(conversation, f, indent=2)
        
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1} normal conversations...")
    
    # Generate fraudulent conversations
    for i in range(num_fraudulent):
        conversation = generate_fraudulent_conversation()
        filename = f"conversations/fraudulent_{i+1:05d}.json"
        with open(filename, 'w') as f:
            json.dump(conversation, f, indent=2)
        
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1} fraudulent conversations...")
    
    print(f"\nSuccessfully generated {num_conversations} conversations!")
    print(f"Files saved in the 'conversations' directory.")

if __name__ == "__main__":
    generate_conversations(10000) 