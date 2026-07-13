from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# This model creates multilingual sentence embeddings.
# It allows the chatbot to understand semantically similar user questions
# even when the exact keywords are different.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class SemanticRestaurantBot:
    """A modernized restaurant chatbot using semantic intent routing."""

    def __init__(self) -> None:
        # Load the sentence embedding model.
        self.model = SentenceTransformer(MODEL_NAME)

        # Define intent examples.
        # These examples act as semantic prototypes for each intent.
        self.intent_examples = {
            "opening_hours": [
                "Wann hat das Restaurant geöffnet?",
                "Wie sind die Öffnungszeiten?",
                "Ist das Restaurant heute offen?",
                "What are your opening hours?",
                "¿Cuál es el horario del restaurante?",
            ],
            "menu": [
                "Wo finde ich die Speisekarte?",
                "Was gibt es zu essen?",
                "Welche Gerichte bietet ihr an?",
                "Can I see the menu?",
                "¿Dónde puedo ver el menú?",
            ],
            "reservation": [
                "Ich möchte einen Tisch reservieren.",
                "Kann ich für heute Abend reservieren?",
                "I want to book a table.",
                "Quiero reservar una mesa.",
                "Je voudrais réserver une table.",
            ],
        }

        # Define responses for each routed intent.
        self.responses = {
            "opening_hours": (
                "Unser Restaurant ist täglich von 12:00 bis 23:00 Uhr geöffnet."
            ),
            "menu": (
                "Unsere Speisekarte finden Sie auf unserer Website "
                "unter www.beispielrestaurant.de/speisekarte."
            ),
            "reservation": (
                "Reservierungen sind telefonisch oder online möglich. "
                "Für dieses Demo-System wird noch keine echte Buchung durchgeführt."
            ),
        }

        # Pre-compute intent embeddings once during initialization.
        self.intent_centroids = self._build_intent_centroids()

    def _build_intent_centroids(self) -> dict[str, list[float]]:
        """Create one average embedding vector for each intent."""

        centroids = {}

        for intent, examples in self.intent_examples.items():
            embeddings = self.model.encode(examples)
            centroids[intent] = embeddings.mean(axis=0)

        return centroids

    def classify_intent(self, user_input: str) -> tuple[str, float]:
        """Classify user input by semantic similarity to intent centroids."""

        user_embedding = self.model.encode([user_input])[0].reshape(1, -1)

        scores = {}

        for intent, centroid in self.intent_centroids.items():
            centroid_embedding = centroid.reshape(1, -1)
            score = cosine_similarity(user_embedding, centroid_embedding)[0][0]
            scores[intent] = float(score)

        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        return best_intent, best_score

    def respond_to_user_input(self, user_input: str) -> str:
        """Generate a response based on the routed intent."""

        intent, score = self.classify_intent(user_input)

        # A confidence threshold prevents weak semantic matches from being accepted.
        if score < 0.35:
            return (
                "Entschuldigung, ich bin mir nicht sicher, ob ich die Frage "
                "richtig verstanden habe. Bitte fragen Sie nach Öffnungszeiten, "
                "Speisekarte oder Reservierung."
            )

        return self.responses[intent]


def main() -> None:
    bot = SemanticRestaurantBot()

    test_messages = [
        "Kannst du mir die Öffnungszeiten sagen?",
        "Wo finde ich euer Essen online?",
        "Ich möchte für heute Abend einen Tisch buchen.",
        "Do you have a menu?",
        "Quiero reservar una mesa para dos personas.",
        "Wie ist das Wetter in Berlin?",
    ]

    print("Modern Chatbot Demo: Semantic Restaurant Bot")
    print("=" * 60)

    for message in test_messages:
        response = bot.respond_to_user_input(message)

        print(f"\nUser: {message}")
        print(f"Bot:  {response}")


if __name__ == "__main__":
    main()