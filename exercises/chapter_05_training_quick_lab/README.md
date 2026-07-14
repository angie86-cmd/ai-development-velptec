# Chapter 05 Quick Lab - Customer Support Intent Classification

## Goal

This quick lab modernizes the VelpTEC exercise on chatbot model training into a small and reproducible intent-classification pipeline.

The lab focuses on a customer support chatbot for an online shop.

It demonstrates:

- synthetic training data generation
- data-quality evaluation
- classical machine learning baseline
- two hyperparameter settings
- basic overfitting observation
- report generation

## Relation to the VelpTEC Exercise

The original VelpTEC transfer task asks for the following steps:

```text
a) Generate synthetic training data for a chatbot.
b) Evaluate the quality of the generated data.
c) Train a simple chatbot model with the collected data and experiment with at least two hyperparameter settings.
d) Implement transfer learning using a pretrained model and compare the results.
```

This quick lab currently focuses on steps **a)**, **b)** and **c)**.

Step **d)**, transfer learning with a pretrained model, is planned as a later extension using Hugging Face Transformers or pretrained sentence embeddings.

The current implementation is intentionally lightweight, reproducible and suitable as a first baseline before adding a more complex pretrained model.

## Use Case

The chatbot is designed to classify customer messages into support intents.

Implemented intents:

| Intent | Meaning |
|---|---|
| `product_information` | Customer asks for product details |
| `shipping_status` | Customer asks about delivery or package status |
| `returns_refund` | Customer asks about returns or refunds |
| `payment_issue` | Customer reports a payment problem |

The trained model is not a full generative chatbot. Instead, it implements one important component of a chatbot architecture: **Intent Classification**.

In a real chatbot system, this component would identify the user's intention and route the conversation to the correct answer logic, workflow or support process.

## Why this lab is intentionally lightweight

The original exercise mentions model training and transfer learning. For a short practical lab, this implementation first uses a classical machine learning baseline:

```text
Text → TF-IDF → Logistic Regression → Intent
```

This approach is fast, reproducible and useful as a baseline.

A modern extension could later use:

- pretrained sentence embeddings
- Hugging Face Transformers
- DistilBERT fine-tuning
- semantic routing
- evaluation with a larger dataset

## Files

```text
chapter_05_training_quick_lab/
├── README.md
├── quick_intent_lab.py
├── synthetic_customer_support_intents.csv
└── reports/
    └── quick_lab_report.md
```

## How to run

From the repository root:

```powershell
cd C:\dev\ai-development-velptec
.\.venv\Scripts\Activate.ps1
python exercises\chapter_05_training_quick_lab\quick_intent_lab.py
```

Expected output:

```text
Quick lab completed successfully.
Dataset saved to: ...
Report saved to: ...
```

## What the script does

The script performs five steps:

1. Generates synthetic customer-support examples.
2. Saves the generated dataset as CSV.
3. Evaluates data quality.
4. Trains two TF-IDF + Logistic Regression models.
5. Writes a Markdown report with dataset and model results.

## Data Generation

The dataset is generated synthetically using predefined templates for each support intent.

The implemented categories are:

- product information
- shipping status
- returns and refunds
- payment issues

Each category contains multiple templates. The script fills these templates with product names and order IDs to create varied customer-support messages.

Example generated messages:

```text
Ich brauche Informationen zu Laptop.
Wann kommt meine Bestellung A1001 an?
Ich möchte Kopfhörer zurücksenden.
Meine Zahlung wurde abgelehnt.
```

This approach keeps the lab reproducible and fast while still demonstrating the basic idea of training-data generation for chatbot intent classification.

## Data Quality Evaluation

The script evaluates the generated dataset using three simple criteria:

| Criterion | Purpose |
|---|---|
| Completeness | Checks whether text and intent values are available |
| Class balance | Checks whether the intents are represented evenly |
| Diversity | Counts how many unique utterances exist |

The generated report includes:

```text
Total examples
Missing values
Intent distribution
Unique texts
Diversity ratio
```

This is useful because chatbot model quality depends strongly on the quality and balance of the training data.

## Model Training

The quick lab trains a classical machine learning baseline:

```text
TF-IDF Vectorizer + Logistic Regression
```

This corresponds to step **c)** of the VelpTEC task:

```text
Trainiere ein einfaches Chatbot-Modell mit den gesammelten Daten.
Verwende dafür eine gängige Python-Bibliothek für maschinelles Lernen oder natürliche Sprachverarbeitung.
Experimentiere mit mindestens zwei verschiedenen Sets von Hyperparametern.
```

In this lab, the "chatbot model" is implemented as an **Intent Classification model**.

The model learns to classify customer messages into one of the predefined intents:

```text
product_information
shipping_status
returns_refund
payment_issue
```

## Machine Learning Library: scikit-learn

The implementation uses **scikit-learn**, a widely used Python library for classical machine learning.

The following scikit-learn components are used:

| Library Component | Purpose in this lab |
|---|---|
| `TfidfVectorizer` | Converts text messages into numerical features |
| `LogisticRegression` | Trains the intent classification model |
| `train_test_split` | Splits the dataset into training and test data |
| `Pipeline` | Combines text vectorization and classification into one workflow |
| `accuracy_score` | Calculates training and test accuracy |
| `classification_report` | Generates precision, recall and F1-score per intent |

The relevant imports in `quick_intent_lab.py` are:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
```

Although the VelpTEC task mentions TensorFlow, PyTorch or spaCy as examples, the wording uses "z. B.", meaning "for example".

For this quick lab, scikit-learn is a suitable choice because it is lightweight, fast and appropriate for creating a transparent baseline model.

## Model Pipeline

The model pipeline is implemented as follows:

```text
Customer message
→ TF-IDF features
→ Logistic Regression classifier
→ Predicted intent
```

The training step happens in the function:

```python
def train_and_evaluate_model(
    data: list[dict[str, str]],
    model_name: str,
    c_value: float,
) -> dict[str, object]:
```

Inside this function, the dataset is split into training and test data:

```python
x_train, x_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.25,
    random_state=RANDOM_SEED,
    stratify=labels,
)
```

The machine learning pipeline is defined here:

```python
pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        (
            "classifier",
            LogisticRegression(
                C=c_value,
                max_iter=1000,
                random_state=RANDOM_SEED,
            ),
        ),
    ]
)
```

The actual model training happens here:

```python
pipeline.fit(x_train, y_train)
```

After training, the model predicts intents for both the training data and the test data:

```python
train_predictions = pipeline.predict(x_train)
test_predictions = pipeline.predict(x_test)
```

## Hyperparameter Comparison

The script trains two model variants with different values for the Logistic Regression hyperparameter `C`.

```text
C = 0.5
C = 5.0
```

The parameter `C` controls regularization strength:

| C value | Meaning |
|---|---|
| Lower C | Stronger regularization |
| Higher C | Weaker regularization |

The two model variants are created in the main function:

```python
model_1 = train_and_evaluate_model(
    data=data,
    model_name="TF-IDF + Logistic Regression - stronger regularization",
    c_value=0.5,
)

model_2 = train_and_evaluate_model(
    data=data,
    model_name="TF-IDF + Logistic Regression - weaker regularization",
    c_value=5.0,
)
```

The goal is not to find the perfect model, but to observe how different hyperparameter settings can affect training accuracy, test accuracy and possible overfitting.

## Overfitting Observation

To observe possible overfitting, the script compares training accuracy and test accuracy.

The relevant code is:

```python
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)
```

A simple interpretation is:

| Observation | Possible meaning |
|---|---|
| Training accuracy much higher than test accuracy | Possible overfitting |
| Training and test accuracy similar | Better generalization |
| Both training and test accuracy low | Possible underfitting or weak features |

This makes the lab useful for understanding the relationship between hyperparameters, model accuracy and generalization.

## Report

After execution, the script creates a Markdown report in:

```text
reports/quick_lab_report.md
```

The report contains:

- data-quality summary
- class balance
- comparison of both model configurations
- classification reports
- short reflection

This makes the lab easier to review and suitable for GitHub documentation.

## Current Scope and Planned Extension

The current quick lab covers a lightweight implementation of steps **a)**, **b)** and **c)** of the VelpTEC task.

Step **d)** requires transfer learning with a pretrained model. This is not yet implemented in the quick lab.

A planned extension could compare the current scikit-learn baseline with a pretrained model, for example:

```text
Synthetic dataset
→ baseline TF-IDF classifier
→ pretrained sentence embeddings
→ Hugging Face Transformer fine-tuning
→ model comparison
```

This would connect the classical chatbot-training workflow with modern AI Development practices such as transfer learning, contextual embeddings and semantic classification.

## Reflection

This lab is not intended to replace modern LLM- or Transformer-based chatbot development.

Instead, it provides a fast and transparent baseline. Such baselines are useful because they help compare whether a more complex model is actually needed.

The implementation shows that a simple intent-classification model can already be trained with synthetic data and a classical machine learning pipeline.

At the same time, the limitations are clear:

- The dataset is synthetic and small.
- The templates are controlled and predictable.
- The model does not understand language deeply.
- It classifies intents but does not generate full chatbot responses.
- It does not yet use transfer learning or pretrained Transformer models.

For modern AI Development, the next step would be to compare this baseline with a pretrained model, for example using Hugging Face Transformers or sentence embeddings.