# Advanced-NLP-for-Bias-Detection-and-Sentiment-Intelligence

This project focuses on detecting gender bias and classifying sentiment in textual content using advanced NLP techniques. Multiple machine learning models—including Logistic Regression, SVM, and Neural Networks—are implemented and evaluated on a large dataset of textual entries.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Models](#models)
  - [Logistic Regression](#logistic-regression)
  - [SVM](#svm)
  - [Neural Network](#neural-network)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Author](#author)

---

## Project Overview
The goal of this project is to:
- Detect sexism in textual data.
- Classify sentiment as positive/negative or neutral.
- Handle both structured and unstructured text.
- Fine-tune transformer models for bias detection and sentiment intelligence.

This project demonstrates a full NLP pipeline, including preprocessing, vectorization, model training, and evaluation.

---

## Dataset
- **Training Data:** `train_all_tasks.csv` (14,000 entries)  
  Columns:  
  - `rewire_id` – unique ID for each entry  
  - `text` – textual content  
  - `label_sexist` – binary label for sexism (`sexist`, `not sexist`)  
  - `label_category` – detailed category of sexist content  
  - `label_vector` – vectorized label representation  

- **Development Data:** `dev_task_a_entries.csv` – used for model testing and prediction.

---

## Preprocessing
Text preprocessing includes:
- Removing punctuation and digits
- Tokenization using NLTK
- Stopword removal
- Lemmatization

```python
def preprocess_text(text_column):
    # Remove punctuation, lowercase, remove digits
    # Tokenize, remove stopwords, lemmatize
    # Return preprocessed text
