from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# This model creates multilingual sentence embeddings.
# It can map sentences from different languages into a shared semantic vector space.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main() -> None:
    # Load the embedding model.
    # The first execution may take a while because the model needs to be downloaded.
    model = SentenceTransformer(MODEL_NAME)

    # German query used as the reference sentence.
    query = "Ich möchte meinen Pass verlängern."

    # Candidate sentences in different languages.
    # Some are semantically related to the query, others are unrelated.
    candidates = [
        "Ich will meinen Reisepass erneuern.",
        "Quiero renovar mi pasaporte.",
        "I want to renew my passport.",
        "Je veux renouveler mon passeport.",
        "Der Hund schläft auf dem Sofa.",
        "Heute regnet es in Berlin.",
    ]

    # Encode the query and all candidate sentences as numerical vectors.
    texts = [query] + candidates
    embeddings = model.encode(texts)

    # Separate the query embedding from the candidate embeddings.
    query_embedding = embeddings[0].reshape(1, -1)
    candidate_embeddings = embeddings[1:]

    # Compute cosine similarity between the query and each candidate sentence.
    # Higher values mean stronger semantic similarity.
    similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]

    # Sort the results from most similar to least similar.
    ranked_results = sorted(
        zip(candidates, similarities),
        key=lambda item: item[1],
        reverse=True,
    )

    # Print the ranked semantic similarity results.
    print("Modern NLP Demo: Multilingual Semantic Similarity")
    print("=" * 60)
    print(f"Query: {query}")
    print("-" * 60)

    for candidate, score in ranked_results:
        print(f"{score:.3f} | {candidate}")


if __name__ == "__main__":
    main()