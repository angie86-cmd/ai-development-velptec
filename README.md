# AI Development - VelpTEC

Hands-on exercises and portfolio extensions for the VelpTEC AI Development module.

## Purpose

This repository documents my practical work during the VelpTEC AI Development module.

The initial focus is the setup of a Python environment for AI and data-processing exercises. Later, the repository will be extended with selected exercises and portfolio-ready mini-projects, especially around chatbot development, NLP, evaluation and multi-channel AI assistants.

## Environment

The project uses a local Python virtual environment with the following core libraries:

- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Installed versions

The following versions were verified in the local environment:

| Library | Version |
|---|---:|
| NumPy | 2.5.1 |
| Pandas | 3.0.3 |
| Matplotlib | 3.11.0 |
| Scikit-learn | 1.9.0 |

## Verification

The installed package versions can be checked with:

```bash
python scripts/check_environment.py
```

## Known issue

Matplotlib is installed, but importing it directly is currently blocked on this Windows environment by an application control policy related to `ft2font`.

The package installation itself was verified through Python package metadata. This issue will be reviewed separately if graphical plotting is needed for later exercises.

## Planned structure

```text
ai-development-velptec/
├── README.md
├── requirements.txt
├── scripts/
│   └── check_environment.py
├── exercises/
├── notebooks/
├── src/
└── tests/
```

## Portfolio direction

Possible portfolio extensions from this module:

1. Context-aware chatbot with intent recognition, entity extraction and conversation state.
2. Multi-channel chatbot using Flask and Telegram.
3. Chatbot evaluation toolkit with metrics, test cases and regression checks.
4. AI assistant architecture with reusable instructions, tools, guardrails and evaluation cases.

## Learning note

This module focuses mainly on classical chatbot and NLP concepts such as intent recognition, entities, slots, context management and basic ML workflows.

For modern AI development, these concepts will be complemented with current practices such as LLM-based applications, RAG, agentic AI, tool calling, evaluation and observability.