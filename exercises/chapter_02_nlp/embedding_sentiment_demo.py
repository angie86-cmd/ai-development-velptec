from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# This multilingual embedding model allows us to compare sentences
# across German, Spanish, English and French.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# Instead of using a legacy word list approach, we define short prototype
# sentences for each sentiment category.
#
# The classifier will compare new sentences with these prototypes in the
# embedding space.
SENTIMENT_PROTOTYPES = {
    "positive": [
        "Ich liebe diesen Kurs.",
        "Este curso me encanta.",
        "This is an excellent experience.",
        "C'est une très bonne expérience.",
    ],
    "negative": [
        "Das ist enttäuschend.",
        "Esto es frustrante.",
        "This is a terrible experience.",
        "C'est une mauvaise expérience.",
    ],
    "neutral": [
        "Der Termin ist morgen.",
        "La cita es mañana.",
        "This is a factual statement.",
        "C'est une information neutre.",
    ],
}


# Test sentences in multiple languages.
# Some examples are simple, while others include cases that are harder
# for classic word-count sentiment analysis.
TEST_SENTENCES = [
    "Ich liebe diesen Kurs.",
    "Das ist enttäuschend und schlecht organisiert.",
    "Der Termin ist morgen um 10 Uhr.",
    "Das ist nicht schlecht.",
    "Este curso me encanta.",
    "Je suis très déçue par le résultat.",
    "El documento debe entregarse el lunes.",
]


def build_sentiment_centroids(model: SentenceTransformer) -> dict[str, np.ndarray]:
    """Create one average embedding vector for each sentiment category."""

    centroids = {}

    for label, examples in SENTIMENT_PROTOTYPES.items():
        # Encode all prototype examples for a sentiment label.
        embeddings = model.encode(examples)

        # Average the embeddings to create a simple category centroid.
        centroids[label] = np.mean(embeddings, axis=0)

    return centroids


def classify_sentence(
    model: SentenceTransformer,
    sentence: str,
    centroids: dict[str, np.ndarray],
) -> tuple[str, dict[str, float]]:
    """Classify a sentence by comparing it with sentiment centroids."""

    # Convert the input sentence into an embedding vector.
    sentence_embedding = model.encode([sentence])[0].reshape(1, -1)

    scores = {}

    for label, centroid in centroids.items():
        # Compare the sentence embedding with each sentiment centroid.
        centroid_embedding = centroid.reshape(1, -1)
        score = cosine_similarity(sentence_embedding, centroid_embedding)[0][0]
        scores[label] = float(score)

    # Choose the sentiment label with the highest similarity score.
    predicted_label = max(scores, key=scores.get)

    return predicted_label, scores


def main() -> None:
    # Load the multilingual embedding model.
    model = SentenceTransformer(MODEL_NAME)

    # Build one semantic centroid per sentiment category.
    centroids = build_sentiment_centroids(model)

    print("Modern NLP Demo: Embedding-based Sentiment Classification")
    print("=" * 70)

    for sentence in TEST_SENTENCES:
        label, scores = classify_sentence(model, sentence, centroids)

        print(f"\nText: {sentence}")
        print(f"Predicted sentiment: {label}")
        print(
            "Scores: "
            + ", ".join(
                f"{sentiment}={score:.3f}"
                for sentiment, score in scores.items()
            )
        )


if __name__ == "__main__":
    main()