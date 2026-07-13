# Chapter 03 - Modern Chatbot Architecture Lab

This lab modernizes the chatbot exercises from the VelpTEC AI Development module.

The original exercises focus on classic chatbot concepts such as rule-based responses, keyword matching, simple intent recognition and predefined answers.

This implementation keeps those concepts as background knowledge, but updates the architecture toward a modern LLM-ready and agent-ready design.

## Purpose

The goal is to show that classic chatbot concepts can be translated into modern AI Development patterns.

Instead of building only a keyword-based bot, this lab demonstrates:

- semantic routing with sentence embeddings
- multilingual intent recognition
- tool-like functions
- structured response generation
- fallback handling
- a clear path toward LLM agents, RAG and MCP-based tool integration

## Mapping from VelpTEC exercises to modern implementation

| VelpTEC exercise | Classic approach | Modernized implementation |
|---|---|---|
| 03.2.Ü.01 | spaCy tokenization, VADER sentiment, rule-based vs AI chatbot | Architecture notes and semantic understanding with embeddings |
| 03.2.Ü.02 | Rule-based restaurant bot using keywords | Semantic restaurant assistant with intent routing |
| 03.4.A.01 | Hybrid chatbot with greetings and weather intent | Agent-ready chatbot with semantic routing and mock tools |

## Classic chatbot concepts that are still relevant

Classic chatbot concepts are not useless. They are still important for understanding how conversational systems work.

Relevant concepts include:

- intents
- entities
- slots
- dialogue state
- fallback handling
- backend/API integration
- response generation
- testing and evaluation

However, in modern systems these concepts are usually part of a broader architecture.

## Legacy aspects

The following approaches are useful for learning, but should not be the main portfolio focus:

- pure keyword matching
- rigid if/else chatbot logic
- predefined FAQ answers only
- VADER-based sentiment analysis as the main NLP layer
- rule-based NLU as the full intelligence layer

## Modern architecture perspective

A modern chatbot is better understood as an LLM-ready or agent-ready application.

Typical components are:

```text
User / Channel
→ Backend / Session
→ Semantic Router or LLM
→ Retrieval / Knowledge Layer
→ Tools / APIs
→ Guardrails
→ Response Generator
→ Evaluation / Monitoring
```

This lab does not use a paid LLM API yet. Instead, it uses sentence embeddings to simulate a modern semantic routing layer.

## Files

```text
chapter_03_chatbot/
├── README.md
├── semantic_restaurant_bot.py
└── hybrid_agentic_chatbot_demo.py
```

## Demo 1: Semantic Restaurant Bot

Run:

```bash
python exercises/chapter_03_chatbot/semantic_restaurant_bot.py
```

This demo modernizes the classic restaurant chatbot. Instead of relying only on exact keywords such as "Öffnungszeiten" or "Speisekarte", it uses sentence embeddings to route user messages to the closest intent.

## Demo 2: Hybrid Agentic Chatbot

Run:

```bash
python exercises/chapter_03_chatbot/hybrid_agentic_chatbot_demo.py
```

This demo shows a small agent-ready chatbot architecture with:

- semantic intent routing
- greeting responses
- weather mock tool
- current time tool
- fallback handling

## Portfolio relevance

This chapter demonstrates that I understand classic chatbot foundations, but implement them in a modern way.


## Next steps

Later chapters can extend this architecture with:

- RAG-based document retrieval
- real API tools
- evaluation datasets
- guardrails
- multi-channel deployment
- LLM-based orchestration