# Sentiment Analysis Project - Tasks

## Phase 1: Data Loading & EDA
- [x] Load IMDB dataset from CSV
- [x] Explore dataset structure (shape, columns, data types)
- [x] Check for missing values and duplicates
- [x] Visualize sentiment distribution (countplot, pie chart)
- [x] Analyze text length distribution
- [x] Generate WordClouds for positive vs negative reviews

## Phase 2: Data Preprocessing
- [x] Lowercase all text
- [x] Remove HTML tags and special characters
- [x] Create TF-IDF vectorization for ANN model
- [x] Implement tokenization for LSTM/BERT models
- [x] Add padding/truncation for sequence models
- [x] Split data into train/validation/test sets

## Phase 3: ANN + TF-IDF Model (PyTorch)
- [x] Build ANN architecture with dense layers
- [x] Implement training loop
- [x] Add evaluation metrics (accuracy, precision, recall, F1)
- [x] Train and validate model
- [x] Generate confusion matrix

## Phase 4: LSTM Model (PyTorch)
- [x] Build LSTM architecture (Embedding ? LSTM ? Dense)
- [x] Implement data loaders with padding
- [x] Train LSTM model
- [x] Evaluate with same metrics as ANN
- [x] Compare results

## Phase 5: BERT Fine-Tuning (PyTorch + HuggingFace)
- [x] Setup BERT tokenizer and model
- [x] Create BERT-specific data preprocessing
- [x] Fine-tune pretrained BERT
- [x] Evaluate BERT performance
- [x] Generate confusion matrix

## Phase 6: Model Comparison
- [x] Compile all metrics into comparison table
- [x] Measure training time for each model
- [x] Measure inference time for each model
- [x] Create visualization of results

## Phase 7: Deployment - Simple API (Flask) ??
- [x] Save the best trained model (e.g., BERT or LSTM)
- [x] Load model for inference
- [x] Create Flask app structure
- [x] Build /predict endpoint
- [x] Accept user input (text review)
- [x] Preprocess input text (same as training)
- [x] Run model inference
- [x] Return prediction (positive / negative) as JSON
- [x] Test API using Postman or curl
- [x] Handle errors and invalid input

## Phase 8: Final Deliverables
- [x] Complete Jupyter Notebook with all code
- [x] Add explanations and documentation
- [x] Final report with conclusions
- [x] Include API usage instructions
