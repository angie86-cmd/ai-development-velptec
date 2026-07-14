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
- lightweight transfer learning with a Hugging Face model
- use of pandas and NumPy
- report generation

## Relation to the VelpTEC Exercise

The original VelpTEC transfer task asks for the following steps:

```text
a) Generate synthetic training data for a chatbot.
b) Evaluate the quality of the generated data.
c) Train a simple chatbot model with the collected data and experiment with at least two hyperparameter settings.
d) Implement transfer learning using a pretrained model and compare the results.
```

This quick lab covers all four steps in a lightweight and reproducible way.

| VelpTEC step | Implementation in this lab |
|---|---|
| a) Synthetic training data | `quick_intent_lab.py` generates customer-support examples |
| b) Data-quality evaluation | `quick_intent_lab.py` checks completeness, class balance and diversity |
| c) Simple chatbot model training | `quick_intent_lab.py` trains TF-IDF + Logistic Regression models |
| d) Transfer learning | `huggingface_transfer_lab.py` uses a pretrained Hugging Face SentenceTransformer model |

The current implementation keeps the scope intentionally small. The goal is not to build a full production chatbot, but to demonstrate the main machine learning concepts in a practical way.

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

Then it adds a lightweight transfer-learning extension:

```text
Text → Hugging Face SentenceTransformer embeddings → Logistic Regression → Intent
```

This approach is fast, reproducible and suitable for a portfolio lab.

A more advanced extension could later use:

- full Hugging Face Transformer fine-tuning
- DistilBERT or multilingual BERT for sequence classification
- larger training datasets
- confusion matrices
- experiment tracking
- model persistence
- inference API

## Files

```text
chapter_05_training_quick_lab/
├── README.md
├── quick_intent_lab.py
├── huggingface_transfer_lab.py
├── synthetic_customer_support_intents.csv
└── reports/
    ├── quick_lab_report.md
    ├── huggingface_transfer_report.md
    └── huggingface_transfer_predictions.csv
```

## Setup

From the repository root:

```powershell
cd C:\dev\ai-development-velptec
.\.venv\Scripts\Activate.ps1
```

Install the additional dependency for the Hugging Face transfer-learning lab:

```powershell
pip install sentence-transformers
```

The package `sentence-transformers` loads pretrained embedding models from Hugging Face and makes it easy to encode sentences as vectors.

## How to run the classical quick lab

Run:

```powershell
python exercises\chapter_05_training_quick_lab\quick_intent_lab.py
```

Expected output:

```text
Quick lab completed successfully.
Dataset saved to: ...
Report saved to: ...
```

This script generates the synthetic dataset, trains the TF-IDF baseline models and writes the first report.

## How to run the Hugging Face transfer-learning lab

Run:

```powershell
python exercises\chapter_05_training_quick_lab\huggingface_transfer_lab.py
```

Expected output:

```text
Hugging Face transfer-learning lab completed successfully.
Dataset loaded from: ...
Report saved to: ...
Predictions saved to: ...

Model comparison:
- TF-IDF + Logistic Regression baseline: accuracy=...
- Hugging Face sentence embeddings + Logistic Regression: accuracy=...
```

The first execution may take longer because the pretrained model has to be downloaded.

## What the scripts do

The lab contains two scripts.

### 1. `quick_intent_lab.py`

This script performs the following steps:

1. Generates synthetic customer-support examples.
2. Saves the generated dataset as CSV.
3. Evaluates data quality.
4. Trains two TF-IDF + Logistic Regression models.
5. Writes a Markdown report with dataset and model results.

### 2. `huggingface_transfer_lab.py`

This script performs the following steps:

1. Loads the synthetic dataset with pandas.
2. Trains a classical TF-IDF baseline for comparison.
3. Loads a pretrained Hugging Face SentenceTransformer model.
4. Converts customer messages into semantic sentence embeddings.
5. Uses NumPy to inspect the embedding matrix.
6. Trains a Logistic Regression classifier on top of the embeddings.
7. Compares the baseline with the Hugging Face approach.
8. Saves a Markdown report and prediction CSV.

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
| `accuracy_score` | Calculates model accuracy |
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

The classical model pipeline is implemented as follows:

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

## Transfer Learning with Hugging Face

The second script, `huggingface_transfer_lab.py`, implements step **d)** of the VelpTEC task in a lightweight way.

The task asks for transfer learning with a pretrained model and comparison with the previously trained model.

This lab uses the pretrained Hugging Face model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

The transfer-learning pipeline is:

```text
Customer message
→ pretrained Hugging Face sentence embedding model
→ semantic vector representation
→ Logistic Regression classifier
→ Predicted intent
```

The model converts each customer message into a dense semantic vector. The classifier is then trained on these vectors using the synthetic customer-support dataset.

This is a lightweight transfer-learning approach because the pretrained Transformer model is reused as a frozen feature extractor. The task-specific classifier is trained on top of the pretrained embeddings.

A full fine-tuning approach would update the weights of the Transformer model itself. That would be more computationally expensive and is intentionally left out of this quick lab.

## Hugging Face and Sentence Embeddings

The lab uses the following import:

```python
from sentence_transformers import SentenceTransformer
```

The pretrained model is loaded with:

```python
embedding_model = SentenceTransformer(MODEL_NAME)
```

The embeddings are generated with:

```python
train_embeddings = embedding_model.encode(
    x_train.tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)
```

The resulting embeddings are then used as numerical input for a Logistic Regression classifier.

```python
classifier.fit(train_embeddings, y_train)
```

This demonstrates how a pretrained language model can be reused for a downstream classification task.

## Use of pandas and NumPy

This lab also demonstrates basic use of **pandas** and **NumPy**.

### pandas

pandas is used to load and clean the dataset:

```python
df = pd.read_csv(dataset_path)
df = df.dropna(subset=["text", "intent"])
df["text"] = df["text"].astype(str).str.strip()
df["intent"] = df["intent"].astype(str).str.strip()
```

pandas is also used to create a prediction table:

```python
prediction_df = pd.DataFrame(
    {
        "text": x_test.tolist(),
        "true_intent": y_test.tolist(),
        "predicted_intent": predictions.tolist(),
    }
)
```

The prediction table is saved as CSV:

```python
hf_result["predictions"].to_csv(predictions_path, index=False, encoding="utf-8")
```

### NumPy

NumPy is used to inspect the generated embedding matrix:

```python
embedding_shape = train_embeddings.shape
average_embedding_norm = float(np.mean(np.linalg.norm(train_embeddings, axis=1)))
```

This shows that the text messages have been converted into numerical vector representations.

## Reports

After execution, the lab creates two Markdown reports.

### Classical lab report

```text
reports/quick_lab_report.md
```

This report contains:

- data-quality summary
- class balance
- comparison of both classical model configurations
- classification reports
- short reflection

### Hugging Face transfer-learning report

```text
reports/huggingface_transfer_report.md
```

This report contains:

- baseline accuracy
- Hugging Face transfer-learning accuracy
- embedding matrix information
- classification reports
- reflection on transfer learning

The Hugging Face script also creates:

```text
reports/huggingface_transfer_predictions.csv
```

This file contains sample predictions with:

- customer message
- true intent
- predicted intent

## Current Scope and Limitations

This lab is intentionally small and fast.

The current scope includes:

- synthetic dataset generation
- basic data-quality checks
- classical ML baseline
- two hyperparameter settings
- lightweight Hugging Face transfer learning
- pandas and NumPy usage
- Markdown report generation

The current limitations are:

- The dataset is synthetic and small.
- The templates are controlled and predictable.
- The classifier predicts intents but does not generate full chatbot responses.
- The Hugging Face model is used as a frozen embedding model.
- The Transformer model itself is not fully fine-tuned.
- There is no model persistence or deployment step.

## Reflection

This lab is not intended to replace modern LLM- or production-grade Transformer chatbot development.

Instead, it provides a fast and transparent learning pipeline.

The classical TF-IDF baseline is useful because it is simple, explainable and fast. It helps establish a comparison point before using more complex models.

The Hugging Face extension demonstrates how pretrained Transformer-based sentence embeddings can be reused for a downstream intent-classification task. This connects the classical chatbot-training workflow with modern AI Development practices such as transfer learning, contextual embeddings and semantic classification.

For a future extension, this lab could be expanded with:

```text
Synthetic dataset
→ baseline TF-IDF classifier
→ pretrained sentence embeddings
→ full Hugging Face Transformer fine-tuning
→ model persistence
→ inference API
→ model monitoring
```

This would turn the quick lab into a more complete portfolio project.