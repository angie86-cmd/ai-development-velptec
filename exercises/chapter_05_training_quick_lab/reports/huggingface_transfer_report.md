# Chapter 05 Hugging Face Transfer Learning Report

## Goal

This report compares a classical TF-IDF baseline with a lightweight Hugging Face transfer-learning approach for customer-support intent classification.

## Model Comparison

| Model | Accuracy |
|---|---:|
| TF-IDF + Logistic Regression baseline | 1.0 |
| Hugging Face sentence embeddings + Logistic Regression | 1.0 |

## Hugging Face Model

- Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Embedding matrix shape: `(90, 384)`
- Average embedding norm: `1.0`

## Baseline Classification Report

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

## Hugging Face Classification Report

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

## Prediction Output

Sample predictions are saved in `reports/huggingface_transfer_predictions.csv`.

## Reflection

The TF-IDF baseline is fast and transparent, but it represents text mainly through word and n-gram frequencies. The Hugging Face approach uses a pretrained Transformer-based sentence embedding model to map customer messages into a semantic vector space. A task-specific classifier is then trained on top of these embeddings.

This is a lightweight form of transfer learning. The pretrained model is reused as a frozen feature extractor, while the downstream classifier is adapted to the synthetic customer-support intents. A full fine-tuning approach would update the Transformer model weights directly, but that is intentionally left out here to keep the lab short and reproducible.
