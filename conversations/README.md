# Conversation Dataset

This directory contains 10,000 generated conversations for fraud detection training and testing.

## Dataset Overview

- **Total Conversations**: 10,000
- **Normal Conversations**: 8,000 (80%)
- **Fraudulent Conversations**: 2,000 (20%)
- **Format**: JSON files (one conversation per file)

## File Structure

Each conversation is stored in a separate JSON file with the following naming convention:
- Normal conversations: `normal_XXXXX.json` (where XXXXX is a 5-digit number)
- Fraudulent conversations: `fraudulent_XXXXX.json` (where XXXXX is a 5-digit number)

## JSON Structure

Each conversation file contains:

```json
{
  "conversation_id": "unique-uuid",
  "timestamp": "YYYY-MM-DD HH:MM:SS",
  "fraud_risk": true/false,
  "messages": [
    {
      "sender": "customer" | "agent",
      "timestamp": "YYYY-MM-DD HH:MM:SS",
      "message": "message content"
    }
  ]
}
```

## Conversation Types

### Normal Conversations
- Customer service interactions
- Account management requests
- Technical support queries
- Standard business operations

### Fraudulent Conversations
- Urgent money transfer requests
- Lottery/tax scams
- Nigerian prince scams
- Cryptocurrency investment scams
- Tech support scams
- Gift card scams
- Loan scams

## Usage

This dataset can be used for:
- Training fraud detection models
- Testing RAG systems
- Developing conversation analysis tools
- Evaluating security systems

## Generation

The conversations were generated using `generate_conversations.py` with realistic patterns and timestamps from the past year.

## Statistics

- Average conversation length: 3-8 message exchanges
- Time span: Random timestamps within the last 365 days
- Language: English
- Domain: Customer service and fraud scenarios 