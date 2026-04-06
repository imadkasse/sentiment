# Sentiment Analysis Project - Tasks

## Phase 1: Data Loading & EDA
- [ ] Load IMDB dataset from CSV
- [ ] Explore dataset structure (shape, columns, data types)
- [ ] Check for missing values and duplicates
- [ ] Visualize sentiment distribution (countplot, pie chart)
- [ ] Analyze text length distribution
- [ ] Generate WordClouds for positive vs negative reviews

## Phase 2: Data Preprocessing
- [ ] Lowercase all text
- [ ] Remove HTML tags and special characters
- [ ] Create TF-IDF vectorization for ANN model
- [ ] Implement tokenization for LSTM/BERT models
- [ ] Add padding/truncation for sequence models
- [ ] Split data into train/validation/test sets

## Phase 3: ANN + TF-IDF Model (PyTorch)
- [ ] Build ANN architecture with dense layers
- [ ] Implement training loop
- [ ] Add evaluation metrics (accuracy, precision, recall, F1)
- [ ] Train and validate model
- [ ] Generate confusion matrix

## Phase 4: LSTM Model (PyTorch)
- [ ] Build LSTM architecture (Embedding → LSTM → Dense)
- [ ] Implement data loaders with padding
- [ ] Train LSTM model
- [ ] Evaluate with same metrics as ANN
- [ ] Compare results

## Phase 5: BERT Fine-Tuning (PyTorch + HuggingFace)
- [ ] Setup BERT tokenizer and model
- [ ] Create BERT-specific data preprocessing
- [ ] Fine-tune pretrained BERT
- [ ] Evaluate BERT performance
- [ ] Generate confusion matrix

## Phase 6: Model Comparison
- [ ] Compile all metrics into comparison table
- [ ] Measure training time for each model
- [ ] Measure inference time for each model
- [ ] Create visualization of results

## Phase 7: Final Deliverables
- [ ] Complete Jupyter Notebook with all code
- [ ] Add explanations and documentation
- [ ] Final report with conclusions
