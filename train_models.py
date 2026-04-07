"""
========================================
Sentiment Analysis Model Training Script
========================================
FAST VERSION - Optimized for quick training

Usage:
    python train_models.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import time
import re

# Configuration - OPTIMIZED FOR SPEED
DATA_PATH = 'imdb.csv'
MAX_FEATURES = 10000
MAX_LEN = 200
BATCH_SIZE = 128  # Larger batch = faster
EPOCHS = {'ann': 5, 'lstm': 3, 'bert': 1}  # Fewer epochs
LEARNING_RATE = 0.002  # Slightly higher LR
VOCAB_SIZE = 10000

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
# Simple PyTorch Tokenizer
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
                seq = seq + [0] * pad_len
            else:
                seq = seq[:maxlen]
            padded.append(seq)
        return padded

# ========================================
# Data Loading
# ========================================
print("\n" + "="*50)
print("LOADING DATA")
print("="*50)

df = pd.read_csv(DATA_PATH)
print(f"Dataset: {len(df)} samples")
df['clean_text'] = df['sentences'].apply(preprocess_text)

# ========================================
# Prepare TF-IDF Data for ANN
# ========================================
print("\n" + "="*50)
print("PREPARING TF-IDF DATA")
print("="*50)

X = df['sentences'].values
y = df['labels'].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Convert to tensors
X_train_tfidf_tensor = torch.FloatTensor(X_train_tfidf.toarray())
X_test_tfidf_tensor = torch.FloatTensor(X_test_tfidf.toarray())
y_train_tensor = torch.FloatTensor(y_train)
y_test_tensor = torch.FloatTensor(y_test)

# Save TF-IDF
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# DataLoaders
train_dataset_ann = TensorDataset(X_train_tfidf_tensor, y_train_tensor)
train_loader_ann = DataLoader(train_dataset_ann, batch_size=BATCH_SIZE, shuffle=True)

# ========================================
# Prepare LSTM Data
# ========================================
print("\n" + "="*50)
print("PREPARING LSTM DATA")
print("="*50)

X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
    df['clean_text'].values, df['labels'].values, test_size=0.2, random_state=42
)

tokenizer = SimpleTokenizer(vocab_size=VOCAB_SIZE)
tokenizer.fit(X_train_lstm)
print(f"Vocab size: {len(tokenizer.word2idx)}")

X_train_seq = tokenizer.texts_to_sequences(X_train_lstm)
X_test_seq = tokenizer.texts_to_sequences(X_test_lstm)

X_train_pad = tokenizer.pad_sequences(X_train_seq, maxlen=MAX_LEN)
X_test_pad = tokenizer.pad_sequences(X_test_seq, maxlen=MAX_LEN)

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

X_train_lstm_tensor = torch.LongTensor(X_train_pad)
X_test_lstm_tensor = torch.LongTensor(X_test_pad)
y_train_lstm_tensor = torch.FloatTensor(y_train_lstm)
y_test_lstm_tensor = torch.FloatTensor(y_test_lstm)

train_dataset_lstm = TensorDataset(X_train_lstm_tensor, y_train_lstm_tensor)
train_loader_lstm = DataLoader(train_dataset_lstm, batch_size=BATCH_SIZE, shuffle=True)

# ========================================
# Model Definitions
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
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.dropout3(x)
        return self.fc4(x)

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
        return self.fc2(output)

def evaluate_model(model, X_test, y_test, model_name):
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        outputs = model(X_test).squeeze()
        probs = torch.sigmoid(outputs).cpu().numpy()
        predictions = (probs > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test.numpy(), predictions)
    precision = precision_score(y_test.numpy(), predictions)
    recall = recall_score(y_test.numpy(), predictions)
    f1 = f1_score(y_test.numpy(), predictions)
    
    print(f"\n{model_name} Results:")
    print(f"  Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return accuracy, precision, recall, f1

# ========================================
# Train ANN
# ========================================
print("\n" + "="*50)
print("TRAINING ANN (5 epochs)")
print("="*50)

ann_model = ANN(input_dim=MAX_FEATURES).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(ann_model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS['ann']):
    ann_model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader_ann:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = ann_model(batch_x).squeeze()
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"  Epoch {epoch+1}/{EPOCHS['ann']} - Loss: {total_loss/len(train_loader_ann):.4f}")

torch.save(ann_model.state_dict(), 'ann_model.pth')
ann_results = evaluate_model(ann_model, X_test_tfidf_tensor, y_test_tensor, 'ANN')

# ========================================
# Train LSTM
# ========================================
print("\n" + "="*50)
print("TRAINING LSTM (3 epochs)")
print("="*50)

lstm_model = LSTMClassifier(vocab_size=VOCAB_SIZE, embedding_dim=128, 
                            hidden_dim=128, n_layers=2, dropout=0.3).to(device)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS['lstm']):
    lstm_model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader_lstm:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = lstm_model(batch_x).squeeze()
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"  Epoch {epoch+1}/{EPOCHS['lstm']} - Loss: {total_loss/len(train_loader_lstm):.4f}")

torch.save(lstm_model.state_dict(), 'lstm_model.pth')
lstm_results = evaluate_model(lstm_model, X_test_lstm_tensor, y_test_lstm_tensor, 'LSTM')

# ========================================
# Train BERT (small subset)
# ========================================
print("\n" + "="*50)
print("TRAINING BERT (1 epoch, 2000 samples)")
print("="*50)

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Small subset for BERT
subset_size = 2000
X_train_bert = df['clean_text'].values[:subset_size]
y_train_bert = df['labels'].values[:subset_size]

train_dataset = Dataset.from_dict({'text': X_train_bert, 'label': y_train_bert})

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

def tokenize_function(examples):
    return bert_tokenizer(examples['text'], truncation=True, padding='max_length', max_length=MAX_LEN)

train_dataset = train_dataset.map(tokenize_function, batched=True)
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

training_args = TrainingArguments(
    output_dir='./bert_results',
    num_train_epochs=1,
    per_device_train_batch_size=32,
    warmup_steps=50,
    logging_steps=200,
    save_strategy='no',
    report_to='none'
)

trainer = Trainer(model=bert_model, args=training_args, train_dataset=train_dataset)
print("Training BERT...")
trainer.train()

torch.save(bert_model.state_dict(), 'bert_model.pth')
print("BERT model saved")

# ========================================
# Summary
# ========================================
print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
print("\nSaved files:")
print("  - ann_model.pth")
print("  - lstm_model.pth")
print("  - bert_model.pth")
print("  - tfidf_vectorizer.pkl")
print("  - tokenizer.pkl")

print("\nModel Performance:")
print(f"  ANN:   Acc={ann_results[0]:.4f}, F1={ann_results[3]:.4f}")
print(f"  LSTM:  Acc={lstm_results[0]:.4f}, F1={lstm_results[3]:.4f}")
print(f"  BERT:  Trained")

print("\n" + "="*50)
print("START API: python app.py")
print("="*50)
