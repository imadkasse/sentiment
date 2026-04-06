# Sentiment Analysis Project – PRD

## 1. Project Overview
This project aims to build and compare multiple models for **Sentiment Analysis** on movie reviews from the IMDB dataset. The goal is to classify reviews as **positive** or **negative** and evaluate the performance of different approaches: traditional ML (ANN with TF-IDF), Deep Learning (LSTM), and Transformer-based (BERT), all implemented in **PyTorch**.

---

## 2. Objectives
- Build baseline model using **ANN + TF-IDF** in PyTorch.  
- Implement **LSTM** model to leverage sequence information in PyTorch.  
- Fine-tune **BERT** for state-of-the-art performance in PyTorch using HuggingFace Transformers.  
- Compare models on **accuracy, precision, recall, F1-score, and inference time**.  
- Perform **EDA (Exploratory Data Analysis)** including visualization of data distribution, text length, and word clouds.  

---

## 3. Dataset
- **Source:** `imdb.csv` (local CSV file or URL)  
- **Size:** 50,000 reviews (25k positive, 25k negative)  
- **Columns:** 
  - `sentences` – Movie review text
  - `labels` – Sentiment label (0=negative, 1=positive)

---

## 4. Features / Requirements

### 4.1 Functional Requirements
1. Load and preprocess the dataset.  
2. Explore data with visualizations:
   - Sentiment distribution (countplot, pie chart)
   - Text length distribution
   - WordClouds for positive vs negative reviews
3. Implement models (all in PyTorch):
   - **ANN** with TF-IDF input
   - **LSTM** with Embedding layer
   - **BERT** fine-tuning
4. Evaluate models:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion matrix
5. Compare performance and training/inference time.

### 4.2 Non-Functional Requirements
- Code must be modular, readable, and reproducible.  
- Use **Python 3.10+**, **PyTorch**, **HuggingFace Transformers**.  
- Environment: Jupyter Notebook or Google Colab.  
- Visualization: Matplotlib, Seaborn, Plotly.

---

## 5. Framework Choice
- This project will use **PyTorch only** for all models.  
- Advantages:
  - Consistent framework for ANN, LSTM, and BERT
  - Easy debugging and flexible experimentation
  - HuggingFace Transformers native support for BERT
  - Pythonic syntax for all deep learning implementations

---

## 6. Preprocessing Steps
- Lowercase all text  
- Remove HTML tags and special characters  
- Tokenization (for LSTM/BERT)  
- Padding/truncation for sequence models  
- TF-IDF vectorization for ANN  

---

## 7. Model Details

| Model | Input | Architecture | Notes |
|-------|-------|-------------|-------|
| ANN | TF-IDF | Dense layers | Baseline implemented in PyTorch |
| LSTM | Tokenized sequences | Embedding → LSTM → Dense | PyTorch |
| BERT | Tokenized with tokenizer | Pretrained BERT → Classifier head | PyTorch, HuggingFace Transformers |

---

## 8. Evaluation Metrics
- **Accuracy** – overall correct predictions  
- **Precision** – positive predictive value  
- **Recall** – sensitivity  
- **F1-score** – harmonic mean of precision and recall  
- **Training & inference time** – efficiency comparison

---

## 9. Project Milestones
1. **Data Loading & EDA**  
2. **ANN + TF-IDF Implementation** (PyTorch) 
3. **LSTM Implementation** (PyTorch)  
4. **BERT Fine-Tuning** (PyTorch + HuggingFace) 
5. **Model Evaluation & Comparison**   
6. **Report & Visualization**  

---

## 10. Deliverables
- Jupyter Notebook with code and explanations (runnable on Google Colab)  
- EDA visualizations  
- Models trained and evaluated  
- Comparison table of metrics  
- Final project report / presentation

---

## 11. Tools & Libraries
- Python 3.x  
- Pandas, NumPy  
- Matplotlib, Seaborn, Plotly  
- Scikit-learn  
- PyTorch  
- HuggingFace Transformers  
- WordCloud  

---

## 12. Success Criteria
- Models run without errors  
- Visualization and EDA complete  
- Performance comparison clearly documented  
- Achieve reasonable accuracy (>85% baseline for ANN, >90% BERT expected)  

---