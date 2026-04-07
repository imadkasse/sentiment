from flask import Flask, request, jsonify
from flask_cors import CORS
import model_loader
import re

app = Flask(__name__)
CORS(app)

model_loader_instance = model_loader.SentimentModelLoader()

# Simple rule-based sentiment for demo (when models aren't available)
POSITIVE_WORDS = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'best', 'beautiful', 'awesome', 'nice', 'perfect', 'brilliant', 'enjoy', 'fun', 'like', 'recommended', 'happy', 'impressive', 'outstanding']
NEGATIVE_WORDS = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'boring', 'waste', 'poor', 'disappointing', 'hate', 'stupid', 'dull', 'slow', 'nonsense', 'ridiculous', 'worse', 'annoying', 'waste', 'avoid', 'nothing', 'never']

def rule_based_sentiment(text):
    text = text.lower()
    pos_count = sum(1 for word in POSITIVE_WORDS if word in text)
    neg_count = sum(1 for word in NEGATIVE_WORDS if word in text)
    
    if pos_count > neg_count:
        return 'positive', 0.7 + (pos_count * 0.03)
    elif neg_count > pos_count:
        return 'negative', 0.7 + (neg_count * 0.03)
    else:
        return 'neutral', 0.5

@app.route('/')
def home():
    models_loaded = list(model_loader_instance.models.keys())
    return jsonify({
        'message': 'Sentiment Analysis API',
        'status': 'running',
        'models_loaded': models_loaded if models_loaded else 'demo_mode (rule-based)',
        'endpoints': {
            '/predict': 'POST - Predict sentiment (provide "text" and optional "model": "ann"|"lstm"|"bert"|"demo")',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(model_loader_instance.models.keys()) if model_loader_instance.models else [],
        'mode': 'demo' if not model_loader_instance.models else 'full'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Please provide "text" field in request body'}), 400
        
        text = data['text']
        model_type = data.get('model', 'demo')
        
        # Validate text
        if not isinstance(text, str):
            return jsonify({'error': 'Text must be a string'}), 400
        
        if len(text.strip()) == 0:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        # Demo mode (rule-based) - always available
        if model_type == 'demo':
            sentiment, confidence = rule_based_sentiment(text)
            return jsonify({
                'text': text,
                'sentiment': sentiment,
                'confidence': round(confidence, 4),
                'model_used': 'demo (rule-based)'
            })
        
        # Check if model is loaded
        if model_type not in model_loader_instance.models:
            available = list(model_loader_instance.models.keys())
            if available:
                return jsonify({
                    'error': f'Model "{model_type}" not loaded. Available: {available}',
                    'hint': 'Use "demo" mode for testing without trained models'
                }), 400
            else:
                return jsonify({
                    'error': 'No models loaded. Available: demo',
                    'hint': 'Use "demo" mode for testing without trained models'
                }), 400
        
        # Make prediction with loaded model
        if model_type == 'ann':
            sentiment, confidence = model_loader_instance.predict_ann(text)
        elif model_type == 'lstm':
            sentiment, confidence = model_loader_instance.predict_lstm(text)
        elif model_type == 'bert':
            sentiment, confidence = model_loader_instance.predict_bert(text)
        
        return jsonify({
            'text': text,
            'sentiment': sentiment,
            'confidence': round(confidence, 4),
            'model_used': model_type
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("Starting Sentiment Analysis API...")
    print("=" * 50)
    
    # Try to load models if they exist
    try:
        print("Loading models...")
        model_loader_instance.load_ann_model('ann_model.pth', 'tfidf_vectorizer.pkl')
        print("? ANN model loaded")
    except Exception as e:
        print(f"? ANN model not available: {e}")
    
    try:
        model_loader_instance.load_lstm_model('lstm_model.pth', 'tokenizer.pkl')
        print("? LSTM model loaded")
    except Exception as e:
        print(f"? LSTM model not available: {e}")
    
    try:
        model_loader_instance.load_bert_model('bert_model.bin')
        print("? BERT model loaded")
    except Exception as e:
        print(f"? BERT model not available: {e}")
    
    if not model_loader_instance.models:
        print("\n? Running in DEMO MODE (rule-based sentiment)")
        print("   Train models in notebook and save to test with ML models")
    
    print("=" * 50)
    print("Starting Flask server on http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
