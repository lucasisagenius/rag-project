from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

@app.route('/')
def index():
    """Main page of the RAG application."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_document():
    """Handle document upload."""
    # TODO: Implement document processing and vector storage
    return jsonify({'message': 'Document upload endpoint - to be implemented'})

@app.route('/query', methods=['POST'])
def query():
    """Handle user queries."""
    data = request.get_json()
    question = data.get('question', '')
    
    # TODO: Implement RAG query processing
    response = f"Query received: {question}. RAG processing to be implemented."
    
    return jsonify({'answer': response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 