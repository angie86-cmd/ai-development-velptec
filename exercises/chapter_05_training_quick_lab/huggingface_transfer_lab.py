# Quick Lab - Chapter 05: Hugging Face Transfer Learning
#
# Goal:
# This script extends the classical TF-IDF baseline with a lightweight
# Hugging Face transfer-learning approach.
#
# It uses:
# - pandas for loading and structuring the dataset
# - numpy for numerical inspection of embeddings
# - sentence-transformers / Hugging Face model for pretrained embeddings
# - scikit-learn for the downstream intent classifier
#
# Important:
# This is a lightweight transfer-learning lab. The pretrained Transformer
# model is used as a frozen semantic feature extractor. A task-specific
# Logistic Regression classifier is trained on top of the pretrained embeddings.
#
# This is intentionally simpler than full Transformer fine-tuning so it can
# be completed quickly and reproducibly on a local machine.

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


RANDOM_SEED = 42
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    # Load the synthetic customer support intent dataset.
    #
    # Expected columns:
    # - text
    # - intent

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Please run quick_intent_lab.py first to generate the CSV dataset."
        )

    df = pd.read_csv(dataset_path)

    required_columns = {"text", "intent"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.dropna(subset=["text", "intent"])
    df["text"] = df["text"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["intent"] != "")]

    return df


def train_tfidf_baseline(df: pd.DataFrame) -> dict[str, object]:
    # Train the classical baseline model:
    #
    # Text -> TF-IDF -> Logistic Regression -> Intent
    #
    # This baseline is used as a comparison point for the Hugging Face
    # transfer-learning approach.

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["intent"],
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=df["intent"],
    )

    baseline_model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            (
                "classifier",
                LogisticRegression(
                    C=5.0,
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    baseline_model.fit(x_train, y_train)

    predictions = baseline_model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)

    return {
        "model_name": "TF-IDF + Logistic Regression baseline",
        "accuracy": round(accuracy, 3),
        "classification_report": report,
    }


def train_huggingface_embedding_classifier(df: pd.DataFrame) -> dict[str, object]:
    # Train a lightweight transfer-learning classifier.
    #
    # Step 1:
    # A pretrained Hugging Face SentenceTransformer model converts each
    # customer message into a semantic embedding.
    #
    # Step 2:
    # A Logistic Regression classifier is trained on these embeddings.
    #
    # The Transformer model itself remains frozen. This keeps the lab fast
    # and reproducible while still demonstrating transfer learning.

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["intent"],
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=df["intent"],
    )

    embedding_model = SentenceTransformer(MODEL_NAME)

    train_embeddings = embedding_model.encode(
        x_train.tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    test_embeddings = embedding_model.encode(
        x_test.tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # NumPy is used here to inspect the generated embedding matrix.
    embedding_shape = train_embeddings.shape
    average_embedding_norm = float(np.mean(np.linalg.norm(train_embeddings, axis=1)))

    classifier = LogisticRegression(
        C=3.0,
        max_iter=1000,
        random_state=RANDOM_SEED,
    )

    classifier.fit(train_embeddings, y_train)

    predictions = classifier.predict(test_embeddings)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)

    prediction_df = pd.DataFrame(
        {
            "text": x_test.tolist(),
            "true_intent": y_test.tolist(),
            "predicted_intent": predictions.tolist(),
        }
    )

    return {
        "model_name": "Hugging Face sentence embeddings + Logistic Regression",
        "hf_model_name": MODEL_NAME,
        "accuracy": round(accuracy, 3),
        "classification_report": report,
        "embedding_shape": embedding_shape,
        "average_embedding_norm": round(average_embedding_norm, 3),
        "predictions": prediction_df,
    }


def save_transfer_report(
    baseline_result: dict[str, object],
    hf_result: dict[str, object],
    report_path: Path,
    predictions_path: Path,
) -> None:
    # Save a Markdown report comparing the baseline model and the
    # Hugging Face transfer-learning approach.

    report_path.parent.mkdir(parents=True, exist_ok=True)

    hf_result["predictions"].to_csv(predictions_path, index=False, encoding="utf-8")

    with report_path.open("w", encoding="utf-8") as file:
        file.write("# Chapter 05 Hugging Face Transfer Learning Report\n\n")

        file.write("## Goal\n\n")
        file.write(
            "This report compares a classical TF-IDF baseline with a lightweight "
            "Hugging Face transfer-learning approach for customer-support "
            "intent classification.\n\n"
        )

        file.write("## Model Comparison\n\n")
        file.write("| Model | Accuracy |\n")
        file.write("|---|---:|\n")
        file.write(
            f"| {baseline_result['model_name']} | "
            f"{baseline_result['accuracy']} |\n"
        )
        file.write(
            f"| {hf_result['model_name']} | "
            f"{hf_result['accuracy']} |\n\n"
        )

        file.write("## Hugging Face Model\n\n")
        file.write(f"- Model: `{hf_result['hf_model_name']}`\n")
        file.write(f"- Embedding matrix shape: `{hf_result['embedding_shape']}`\n")
        file.write(
            f"- Average embedding norm: `{hf_result['average_embedding_norm']}`\n\n"
        )

        file.write("## Baseline Classification Report\n\n")
        file.write("```text\n")
        file.write(str(baseline_result["classification_report"]))
        file.write("```\n\n")

        file.write("## Hugging Face Classification Report\n\n")
        file.write("```text\n")
        file.write(str(hf_result["classification_report"]))
        file.write("```\n\n")

        file.write("## Prediction Output\n\n")
        file.write(
            "Sample predictions are saved in "
            "`reports/huggingface_transfer_predictions.csv`.\n\n"
        )

        file.write("## Reflection\n\n")
        file.write(
            "The TF-IDF baseline is fast and transparent, but it represents text "
            "mainly through word and n-gram frequencies. The Hugging Face approach "
            "uses a pretrained Transformer-based sentence embedding model to map "
            "customer messages into a semantic vector space. A task-specific "
            "classifier is then trained on top of these embeddings.\n\n"
        )
        file.write(
            "This is a lightweight form of transfer learning. The pretrained model "
            "is reused as a frozen feature extractor, while the downstream "
            "classifier is adapted to the synthetic customer-support intents. "
            "A full fine-tuning approach would update the Transformer model "
            "weights directly, but that is intentionally left out here to keep "
            "the lab short and reproducible.\n"
        )


def main() -> None:
    # Main execution flow.

    base_dir = Path(__file__).parent
    dataset_path = base_dir / "synthetic_customer_support_intents.csv"
    report_path = base_dir / "reports" / "huggingface_transfer_report.md"
    predictions_path = base_dir / "reports" / "huggingface_transfer_predictions.csv"

    df = load_dataset(dataset_path)

    baseline_result = train_tfidf_baseline(df)
    hf_result = train_huggingface_embedding_classifier(df)

    save_transfer_report(
        baseline_result=baseline_result,
        hf_result=hf_result,
        report_path=report_path,
        predictions_path=predictions_path,
    )

    print("Hugging Face transfer-learning lab completed successfully.")
    print(f"Dataset loaded from: {dataset_path}")
    print(f"Report saved to: {report_path}")
    print(f"Predictions saved to: {predictions_path}")
    print()
    print("Model comparison:")
    print(
        f"- {baseline_result['model_name']}: "
        f"accuracy={baseline_result['accuracy']}"
    )
    print(
        f"- {hf_result['model_name']}: "
        f"accuracy={hf_result['accuracy']}"
    )


if __name__ == "__main__":
    main()