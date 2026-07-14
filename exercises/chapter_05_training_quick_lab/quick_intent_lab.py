# Quick Lab - Chapter 05: Training a simple intent classifier
#
# Goal:
# This script modernizes the VelpTEC chatbot training exercise into a small,
# reproducible intent-classification pipeline.
#
# It demonstrates:
# - synthetic data generation
# - basic data-quality checks
# - classical ML baseline training with TF-IDF + Logistic Regression
# - comparison of two hyperparameter settings
#
# This is intentionally lightweight and suitable for a 30-minute lab.

from collections import Counter
from pathlib import Path
import csv
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


RANDOM_SEED = 42
random.seed(RANDOM_SEED)


INTENT_TEMPLATES = {
    "product_information": [
        "Ich brauche Informationen zu {product}.",
        "Kannst du mir mehr über {product} sagen?",
        "Welche Eigenschaften hat {product}?",
        "Ist {product} aktuell verfügbar?",
        "Ich möchte Details zu {product} wissen.",
    ],
    "shipping_status": [
        "Wann kommt meine Bestellung {order_id} an?",
        "Gibt es ein Update zur Lieferung {order_id}?",
        "Wo ist mein Paket {order_id}?",
        "Ich warte noch auf meine Bestellung {order_id}.",
        "Wie ist der Versandstatus von {order_id}?",
    ],
    "returns_refund": [
        "Ich möchte {product} zurücksenden.",
        "Wie kann ich {product} retournieren?",
        "Ich brauche Hilfe bei der Rücksendung von {product}.",
        "Kann ich {product} zurückgeben?",
        "Wie bekomme ich eine Erstattung für {product}?",
    ],
    "payment_issue": [
        "Meine Zahlung wurde abgelehnt.",
        "Ich sehe eine doppelte Abbuchung.",
        "Warum wurde meine Kreditkarte nicht akzeptiert?",
        "Ich habe ein Problem mit der Bezahlung.",
        "Die Zahlung für meine Bestellung funktioniert nicht.",
    ],
}


PRODUCTS = [
    "Laptop",
    "Smartphone",
    "Kopfhörer",
    "Kamera",
    "T-Shirt",
    "Buch",
    "Tablet",
    "Monitor",
]

ORDER_IDS = [
    "A1001",
    "B2045",
    "C7788",
    "D9012",
    "E4455",
    "F3009",
]


def generate_synthetic_data(samples_per_intent: int = 30) -> list[dict[str, str]]:
    # Generate synthetic training examples for each intent.
    #
    # Input:
    # - samples_per_intent: number of examples per intent
    #
    # Output:
    # - list of dictionaries with text and intent

    data = []

    for intent, templates in INTENT_TEMPLATES.items():
        for _ in range(samples_per_intent):
            template = random.choice(templates)
            product = random.choice(PRODUCTS)
            order_id = random.choice(ORDER_IDS)

            text = template.format(product=product, order_id=order_id)

            data.append(
                {
                    "text": text,
                    "intent": intent,
                }
            )

    random.shuffle(data)
    return data


def save_dataset(data: list[dict[str, str]], output_path: Path) -> None:
    # Save generated dataset as CSV.

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "intent"])
        writer.writeheader()
        writer.writerows(data)


def evaluate_data_quality(data: list[dict[str, str]]) -> dict[str, object]:
    # Evaluate basic dataset quality.
    #
    # Criteria:
    # - completeness: no missing text or intent
    # - class balance: number of examples per intent
    # - diversity: number of unique user utterances

    total_examples = len(data)
    missing_values = sum(
        1
        for row in data
        if not row["text"].strip() or not row["intent"].strip()
    )

    intent_counts = Counter(row["intent"] for row in data)
    unique_texts = len(set(row["text"] for row in data))

    return {
        "total_examples": total_examples,
        "missing_values": missing_values,
        "intent_counts": dict(intent_counts),
        "unique_texts": unique_texts,
        "diversity_ratio": round(unique_texts / total_examples, 2),
    }


def train_and_evaluate_model(
    data: list[dict[str, str]],
    model_name: str,
    c_value: float,
) -> dict[str, object]:
    # Train and evaluate a classical intent-classification model.
    #
    # Model:
    # - TF-IDF converts text into numeric features.
    # - Logistic Regression predicts the intent.
    #
    # Hyperparameter:
    # - C controls regularization strength in Logistic Regression.
    # - Lower C = stronger regularization.
    # - Higher C = weaker regularization.

    texts = [row["text"] for row in data]
    labels = [row["intent"] for row in data]

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

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

    pipeline.fit(x_train, y_train)

    train_predictions = pipeline.predict(x_train)
    test_predictions = pipeline.predict(x_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    report = classification_report(
        y_test,
        test_predictions,
        zero_division=0,
    )

    return {
        "model_name": model_name,
        "c_value": c_value,
        "train_accuracy": round(train_accuracy, 3),
        "test_accuracy": round(test_accuracy, 3),
        "classification_report": report,
    }


def save_report(
    quality_report: dict[str, object],
    model_reports: list[dict[str, object]],
    output_path: Path,
) -> None:
    # Save a Markdown report with dataset quality and model comparison.

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        file.write("# Chapter 05 Quick Lab Report\n\n")

        file.write("## Data Quality\n\n")
        file.write(f"- Total examples: {quality_report['total_examples']}\n")
        file.write(f"- Missing values: {quality_report['missing_values']}\n")
        file.write(f"- Unique texts: {quality_report['unique_texts']}\n")
        file.write(f"- Diversity ratio: {quality_report['diversity_ratio']}\n\n")

        file.write("### Class Balance\n\n")
        for intent, count in quality_report["intent_counts"].items():
            file.write(f"- {intent}: {count}\n")

        file.write("\n## Model Comparison\n\n")
        file.write("| Model | C value | Train accuracy | Test accuracy |\n")
        file.write("|---|---:|---:|---:|\n")

        for report in model_reports:
            file.write(
                f"| {report['model_name']} | "
                f"{report['c_value']} | "
                f"{report['train_accuracy']} | "
                f"{report['test_accuracy']} |\n"
            )

        file.write("\n## Classification Reports\n\n")

        for report in model_reports:
            file.write(f"### {report['model_name']}\n\n")
            file.write("```text\n")
            file.write(report["classification_report"])
            file.write("```\n\n")

        file.write("## Reflection\n\n")
        file.write(
            "This quick lab uses a classical TF-IDF + Logistic Regression "
            "baseline. It is lightweight, reproducible and useful for "
            "understanding the fundamentals of intent classification. "
            "For modern AI Development, this baseline could later be extended "
            "with pretrained Transformer models, sentence embeddings or "
            "Hugging Face fine-tuning.\n"
        )


def main() -> None:
    # Main execution flow.

    base_dir = Path(__file__).parent
    dataset_path = base_dir / "synthetic_customer_support_intents.csv"
    report_path = base_dir / "reports" / "quick_lab_report.md"

    data = generate_synthetic_data(samples_per_intent=30)
    save_dataset(data, dataset_path)

    quality_report = evaluate_data_quality(data)

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

    save_report(
        quality_report=quality_report,
        model_reports=[model_1, model_2],
        output_path=report_path,
    )

    print("Quick lab completed successfully.")
    print(f"Dataset saved to: {dataset_path}")
    print(f"Report saved to: {report_path}")
    print()
    print("Model comparison:")
    print(
        f"- {model_1['model_name']}: "
        f"train={model_1['train_accuracy']}, "
        f"test={model_1['test_accuracy']}"
    )
    print(
        f"- {model_2['model_name']}: "
        f"train={model_2['train_accuracy']}, "
        f"test={model_2['test_accuracy']}"
    )


if __name__ == "__main__":
    main()