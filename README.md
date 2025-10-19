# Advanced-NLP-for-Bias-Detection-and-Sentiment-Intelligence

This project focuses on detecting gender bias and performing sentiment classification in textual content using advanced NLP techniques. It demonstrates a combination of classical machine learning, neural networks, and transformer-based models to analyze structured and unstructured text.

## Dataset

- **Source:** `train_all_tasks.csv`  
- **Samples:** 14,000 entries  
- **Classes:** Binary classification (`sexist` vs `not sexist`)  
- **Label Distribution:**
  - Not sexist: 10,602
  - Sexist: 3,398
- No missing values present

## Preprocessing

Text data was cleaned and prepared using the following steps:

1. Converted text to lowercase
2. Removed punctuation and numeric characters
3. Tokenized using NLTK
4. Removed stopwords
5. Lemmatized using WordNetLemmatizer
6. Rejoined tokens into cleaned strings

## Models

### Logistic Regression
- Uses TF-IDF vectorization
- Achieved overall accuracy: 82%
- Handles binary classification of sexist vs non-sexist text
- Metrics:
  - Precision (sexist): 0.85
  - Precision (not sexist): 0.81
  - Recall (sexist): 0.31
  - Recall (not sexist): 0.98
  - F1-score (weighted): 0.78

### SVM (Support Vector Machine)
- Uses TF-IDF features
- Achieved overall accuracy: 82%
- High precision on minority class
- Metrics:
  - Precision (sexist): 0.95
  - Precision (not sexist): 0.81
  - Recall (sexist): 0.29
  - Recall (not sexist): 1.00
  - F1-score (weighted): 0.79

### Neural Network (MLP Classifier)
- Uses CountVectorizer for input features
- Hidden layer with 100 neurons
- Accuracy: 77%
- Good performance on majority class
- Metrics:
  - Precision (sexist): 0.56
  - Precision (not sexist): 0.83
  - Recall (sexist): 0.47
  - Recall (not sexist): 0.88
  - F1-score (weighted): 0.77

## Visualization

- Confusion matrices and classification reports were visualized using **Seaborn**.
- Performance metrics for all models were plotted using **Matplotlib**.

## Key Features

- Handles both structured and unstructured text efficiently
- Fine-tunes transformer models for bias detection
- Implements preprocessing pipelines with tokenization, stopword removal, and lemmatization
- Compares multiple ML and neural network models
- Generates reproducible evaluation metrics

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/HumayraMusarrat/Advanced-NLP-for-Bias-Detection-and-Sentiment-Intelligence.git
