# Chapter 05 Quick Lab Report

## Data Quality

- Total examples: 120
- Missing values: 0
- Unique texts: 70
- Diversity ratio: 0.58

### Class Balance

- shipping_status: 30
- payment_issue: 30
- product_information: 30
- returns_refund: 30

## Model Comparison

| Model | C value | Train accuracy | Test accuracy |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression - stronger regularization | 0.5 | 1.0 | 1.0 |
| TF-IDF + Logistic Regression - weaker regularization | 5.0 | 1.0 | 1.0 |

## Classification Reports

### TF-IDF + Logistic Regression - stronger regularization

```text
                     precision    recall  f1-score   support

      payment_issue       1.00      1.00      1.00         8
product_information       1.00      1.00      1.00         7
     returns_refund       1.00      1.00      1.00         8
    shipping_status       1.00      1.00      1.00         7

           accuracy                           1.00        30
          macro avg       1.00      1.00      1.00        30
       weighted avg       1.00      1.00      1.00        30
```

### TF-IDF + Logistic Regression - weaker regularization

```text
                     precision    recall  f1-score   support

      payment_issue       1.00      1.00      1.00         8
product_information       1.00      1.00      1.00         7
     returns_refund       1.00      1.00      1.00         8
    shipping_status       1.00      1.00      1.00         7

           accuracy                           1.00        30
          macro avg       1.00      1.00      1.00        30
       weighted avg       1.00      1.00      1.00        30
```

## Reflection

This quick lab uses a classical TF-IDF + Logistic Regression baseline. It is lightweight, reproducible and useful for understanding the fundamentals of intent classification. For modern AI Development, this baseline could later be extended with pretrained Transformer models, sentence embeddings or Hugging Face fine-tuning.
