from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# This model is used as a lightweight semantic routing layer.
# In a production system, this could be replaced or extended with an LLM,
# tool calling, RAG and guardrails.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class HybridAgenticChatbot:
    """A small agent-ready chatbot with semantic routing and mock tools."""

    def __init__(self) -> None:
        self.model = SentenceTransformer(MODEL_NAME)

        # Intent examples define the semantic meaning of each capability.
        self.intent_examples = {
            "greeting": [
                "Hallo",
                "Guten Tag",
                "Hi",
                "Hello",
                "Hola",
            ],
            "weather": [
                "Wie ist das Wetter in Berlin?",
                "Ist es heute sonnig?",
                "What is the weather like today?",
                "¿Cómo está el clima hoy?",
                "Quel temps fait-il aujourd'hui?",
            ],
            "current_time": [
                "Wie spät ist es?",
                "Kannst du mir die aktuelle Uhrzeit sagen?",
                "What time is it?",
                "¿Qué hora es?",
                "Quelle heure est-il?",
            ],
        }

        self.intent_centroids = self._build_intent_centroids()

    def _build_intent_centroids(self) -> dict[str, list[float]]:
        """Create one average embedding vector for each intent."""

        centroids = {}

        for intent, examples in self.intent_examples.items():
            embeddings = self.model.encode(examples)
            centroids[intent] = embeddings.mean(axis=0)

        return centroids

    def route_intent(self, user_input: str) -> tuple[str, float]:
        """Route a user message to the most likely capability."""

        user_embedding = self.model.encode([user_input])[0].reshape(1, -1)

        scores = {}

        for intent, centroid in self.intent_centroids.items():
            centroid_embedding = centroid.reshape(1, -1)
            score = cosine_similarity(user_embedding, centroid_embedding)[0][0]
            scores[intent] = float(score)

        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        return best_intent, best_score

    def get_weather(self, city: str = "Berlin") -> str:
        """Mock weather tool.

        In a production chatbot, this would call a real weather API.
        """

        mock_weather_data = {
            "berlin": "In Berlin ist es heute leicht bewölkt bei 22°C.",
            "münchen": "In München ist es heute sonnig bei 24°C.",
            "hamburg": "In Hamburg ist es heute regnerisch bei 18°C.",
        }

        normalized_city = city.lower()
        return mock_weather_data.get(
            normalized_city,
            f"Für {city} liegen in diesem Demo-System keine Wetterdaten vor.",
        )

    def get_current_time(self) -> str:
        """Tool-like function that returns the local system time."""

        current_time = datetime.now().strftime("%H:%M")
        return f"Es ist jetzt {current_time} Uhr."

    def extract_city(self, user_input: str) -> str:
        """Extract a city from the user message using a simple controlled list.

        This is intentionally simple. In a real agent, entity extraction could be
        handled by an LLM, an NER model or a validated form/slot mechanism.
        """

        known_cities = ["berlin", "münchen", "hamburg"]

        user_input_lower = user_input.lower()

        for city in known_cities:
            if city in user_input_lower:
                return city.capitalize()

        return "Berlin"

    def generate_response(self, user_input: str) -> str:
        """Generate a response by combining semantic routing and tool usage."""

        intent, score = self.route_intent(user_input)

        # A threshold protects the bot from overconfident weak matches.
        if score < 0.35:
            return (
                "Entschuldigung, ich bin mir nicht sicher, ob ich die Anfrage "
                "richtig verstanden habe. Ich kann aktuell Begrüßungen, Wetterfragen "
                "und Fragen zur Uhrzeit beantworten."
            )

        if intent == "greeting":
            return "Hallo! Wie kann ich Ihnen helfen?"

        if intent == "weather":
            city = self.extract_city(user_input)
            return self.get_weather(city)

        if intent == "current_time":
            return self.get_current_time()

        return "Entschuldigung, darauf kann ich noch nicht antworten."


def main() -> None:
    bot = HybridAgenticChatbot()

    test_messages = [
        "Hallo",
        "Wie ist das Wetter in Berlin?",
        "Brauche ich heute in Hamburg einen Regenschirm?",
        "Wie spät ist es?",
        "What time is it?",
        "¿Qué hora es?",
        "Kannst du mir ein Gedicht schreiben?",
    ]

    print("Modern Chatbot Demo: Hybrid Agentic Chatbot")
    print("=" * 60)

    for message in test_messages:
        response = bot.generate_response(message)

        print(f"\nUser: {message}")
        print(f"Bot:  {response}")


if __name__ == "__main__":
    main()