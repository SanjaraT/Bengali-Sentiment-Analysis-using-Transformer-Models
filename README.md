<<<<<<< HEAD
# Bengali Sentiment Analysis using Transformer Models

## Overview

This project presents a comparative study of transformer-based architectures for Bengali sentiment classification. Multiple pre-trained transformer models were fine-tuned and evaluated on a Bengali sentiment dataset to analyze their effectiveness in low-resource multilingual NLP tasks.

The project focused not only on model training, but also on:

* data preprocessing
* handling class imbalance
* transformer fine-tuning
* comparative evaluation
* confusion matrix analysis
* qualitative error analysis
* inference API development using FastAPI
* Hugging Face model hosting

The primary objective of this work was to explore how modern multilingual transformer architectures perform on Bengali sentiment understanding.

---

# Dataset

Dataset: Bengali Sentiment Classification Dataset

Source:
[https://www.kaggle.com/datasets/saurabhshahane/bengali-sentiment-classification](https://www.kaggle.com/datasets/saurabhshahane/bengali-sentiment-classification)

The dataset contains Bengali text samples labeled with sentiment polarity.

## Task

Binary Sentiment Classification:

* Positive
* Negative

---

# Models Used

The following transformer architectures were fine-tuned and compared:

| Model       | Description                                                         |
| ----------- | ------------------------------------------------------------------- |
| BERT        | Multilingual BERT baseline model                                    |
| DistilBERT  | Lightweight distilled transformer                                   |
| XLM-RoBERTa | Multilingual transformer with stronger cross-lingual representation |

---

# Project Pipeline

## 1. Data Preprocessing

The dataset was preprocessed before training:

* removed null values
* cleaned noisy text
* tokenized Bengali text using Hugging Face tokenizers
* padded and truncated sequences
* converted labels into numerical format

---

## 2. Handling Class Imbalance

Initial experiments showed severe model bias toward the majority class.

To stabilize training and improve minority-class learning:

* dataset balancing techniques were applied
* class weighting strategies were explored
* training stability improvements were introduced

This significantly improved recall and overall classification performance.

---

## 3. Transformer Fine-Tuning

All models were fine-tuned using PyTorch and Hugging Face Transformers.

Training pipeline included:

* tokenizer-based encoding
* attention masks
* AdamW optimizer
* GPU acceleration (Google Colab)
* mini-batch training
* evaluation after training

---

# Model Performance

## BERT Results

```text
Accuracy: 0.9407281964436918

              precision    recall  f1-score   support

           0       0.84      0.98      0.91       341
           1       0.99      0.93      0.96       840

    accuracy                           0.94      1181
   macro avg       0.92      0.95      0.93      1181
weighted avg       0.95      0.94      0.94      1181
```

---

## DistilBERT Results

```text
Accuracy: 0.939034716342083

              precision    recall  f1-score   support

           0       0.89      0.90      0.89       331
           1       0.96      0.96      0.96       850

    accuracy                           0.94      1181
   macro avg       0.92      0.93      0.92      1181
weighted avg       0.94      0.94      0.94      1181
```

---

## XLM-RoBERTa Results

```text
Accuracy: 0.955969517358171

              precision    recall  f1-score   support

           0       0.91      0.94      0.92       331
           1       0.97      0.96      0.97       850

    accuracy                           0.96      1181
   macro avg       0.94      0.95      0.95      1181
weighted avg       0.96      0.96      0.96      1181
```

---

# Comparative Analysis

Among all evaluated models, XLM-RoBERTa achieved the best overall performance.

Key observations:

* XLM-R demonstrated stronger multilingual contextual understanding
* DistilBERT provided competitive performance with reduced model complexity
* BERT achieved strong baseline results after class balancing
* multilingual transformers significantly improved Bengali sentiment representation

---

# Confusion Matrix Analysis

Confusion matrices were generated for comparative evaluation.

The XLM-R model demonstrated:

* strong true-positive prediction capability
* improved minority-class recognition
* reduced misclassification compared to baseline models

---

# Error Analysis

Professional qualitative error analysis was conducted on misclassified samples.

Errors were categorized into several linguistic groups:

| Error Type        | Description                                              |
| ----------------- | -------------------------------------------------------- |
| Sarcasm           | Positive lexical cues used in negative contexts          |
| Mixed Sentiment   | Sentences containing both positive and negative opinions |
| Ambiguous Wording | Weak or unclear sentiment polarity                       |
| Long Context      | Important contextual information missed by the model     |
| Neutral Tone      | Sentences lacking strong emotional signals               |

Example findings:

* sarcastic Bengali expressions remained difficult for transformer models
* mixed-polarity sentences caused contextual confusion
* long reviews occasionally reduced contextual focus

The analysis demonstrated that even high-performing transformer models still struggle with nuanced Bengali semantic understanding.

---

# FastAPI Inference API

A FastAPI inference pipeline was developed to convert the trained model into an accessible NLP service.

Features:

* real-time sentiment prediction
* transformer inference endpoint
* Hugging Face model loading
* JSON-based API requests

Example Input:

```json
{
  "text": "নাটকটি অসাধারণ ছিল"
}
```

Example Output:

```json
{
  "sentiment": "Positive"
}
```

---

# Technologies Used

| Technology                | Purpose                    |
| ------------------------- | -------------------------- |
| Python                    | Core programming language  |
| PyTorch                   | Deep learning framework    |
| Hugging Face Transformers | Transformer implementation |
| FastAPI                   | Inference API              |
| Google Colab              | GPU training environment   |
| Scikit-learn              | Evaluation metrics         |
| Matplotlib                | Visualization              |
| Pandas                    | Data handling              |

---

# Key Learning Outcomes

Through this project, the following concepts were explored:

* transformer architecture
* multilingual NLP
* transfer learning
* transformer fine-tuning
* handling class imbalance
* evaluation metrics
* qualitative error analysis
* inference pipeline development
* model hosting and deployment workflow

---

# Future Improvements

Potential future extensions include:

* larger Bengali sentiment datasets
* sarcasm-aware sentiment modeling
* Bengali-specific pretraining
* quantized lightweight deployment
* Streamlit-based frontend
* Docker containerization
* cloud deployment optimization


# Conclusion

This project demonstrated the effectiveness of transformer-based multilingual architectures for Bengali sentiment analysis.

The experiments showed that XLM-RoBERTa achieved the strongest performance among the evaluated models, while DistilBERT offered an efficient lightweight alternative.

Beyond model accuracy, the project emphasized practical NLP engineering workflows including:

* preprocessing
* balancing
* fine-tuning
* evaluation
* error analysis
* inference pipeline development

The work highlights both the capabilities and limitations of modern transformer models in understanding nuanced Bengali sentiment.

---

# Author

Sanjara

Undergraduate CSE Student
Interested in Machine Learning, NLP, and Applied AI Systems
=======

>>>>>>> 2350fc4385640b5438c6f446981820c764e3cdf0
