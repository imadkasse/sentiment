# ============================================
# Model Loader - Pure PyTorch
# ============================================

import torch
import torch.nn as nn
import pickle
import re
import numpy as np

# ========================================
# Simple PyTorch Tokenizer (same as train_models.py)
# ========================================
class SimpleTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_counts = {}
        
    def fit(self, texts):
        for text in texts:
            words = text.split()
            for word in words:
                self.word_counts[word] = self.word_counts.get(word, 0) + 1
        
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        for idx, (word, count) in enumerate(sorted_words[:self.vocab_size-2]):
            self.word2idx[word] = idx + 2
            self.idx2word[idx + 2] = word
    
    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            words = text.split()
            seq = [self.word2idx.get(word, 1) for word in words]
            sequences.append(seq)
        return sequences
    
    def pad_sequences(self, sequences, maxlen=200, padding='post', truncating='post'):
        padded = []
        for seq in sequences:
            if len(seq) < maxlen:
                pad_len = maxlen - len(seq)
                if padding == 'pre':
                    seq = [0] * pad_len + seq
                else:
                    seq = seq + [0] * pad_len
            else:
                if truncating == 'pre':
                    seq = seq[-maxlen:]
                else:
                    seq = seq[:maxlen]
            padded.append(seq)
        return padded

# ========================================
# Text Preprocessing
# ========================================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ========================================
# ANN Model Definition
# ========================================
class ANN(nn.Module):
    def __init__(self, input_dim):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

# ========================================
# LSTM Model Definition
# ========================================
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            batch_first=True, bidirectional=True, 
                            dropout=dropout if n_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        output = self.relu(self.fc1(hidden))
        output = self.dropout(output)
        output = self.fc2(output)
        return output

# ========================================
# Sentiment Model Loader
# ========================================
class SentimentModelLoader:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.tfidf_vectorizer = None
        self.tokenizer = None
        self.bert_tokenizer = None
        self.bert_model = None
        
    def load_ann_model(self, model_path, tfidf_path):
        # Load TF-IDF vectorizer
        with open(tfidf_path, 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        
        input_dim = len(self.tfidf_vectorizer.vocabulary_)
        model = ANN(input_dim)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        self.models['ann'] = model
        
    def load_lstm_model(self, model_path, tokenizer_path):
        # Load PyTorch tokenizer
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        model = LSTMClassifier(vocab_size=10000, embedding_dim=128, 
                               hidden_dim=128, n_layers=2, dropout=0.3)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        self.models['lstm'] = model
        
    def load_bert_model(self, model_path):
        from transformers import BertTokenizer, BertForSequenceClassification
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.bert_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.bert_model.to(self.device)
        self.bert_model.eval()
        self.models['bert'] = self.bert_model
        
    def predict_ann(self, text):
        clean_text = preprocess_text(text)
        features = self.tfidf_vectorizer.transform([clean_text])
        features_tensor = torch.FloatTensor(features.toarray()).to(self.device)
        
        with torch.no_grad():
            output = self.models['ann'](features_tensor)
            prob = torch.sigmoid(output).item()
        
        return 'positive' if prob > 0.5 else 'negative', prob
    
    def predict_lstm(self, text):
        clean_text = preprocess_text(text)
        seq = self.tokenizer.texts_to_sequences([clean_text])
        padded = self.tokenizer.pad_sequences(seq, maxlen=200, padding='post', truncating='post')
        tensor = torch.LongTensor(padded).to(self.device)
        
        with torch.no_grad():
            output = self.models['lstm'](tensor)
            prob = torch.sigmoid(output).item()
        
        return 'positive' if prob > 0.5 else 'negative', prob
    
    def predict_bert(self, text):
        inputs = self.bert_tokenizer(text, return_tensors='pt', 
                                      truncation=True, padding=True, max_length=200)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            prob = torch.softmax(outputs.logits, dim=1)
            confidence = prob[0][1].item()
        
        return 'positive' if confidence > 0.5 else 'negative', confidence
