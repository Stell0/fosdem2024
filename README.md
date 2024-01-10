# LangChain From 0 To 1: Unveiling the Power of LLM Programming

## Abstract

Unlocking the realm of Artificial Intelligence has never been more accessible than with LangChain and its seamless integration of external APIs or locally hosted OS Language Model Models (LLMs). In this talk, we embark on a journey from zero to Retrieval Augmented Generation (RAG).

Our hands-on exploration will guide you through the basics of LangChain and LLMs, demonstrating how just a few lines of code can transform any application into an intelligent powerhouse. We'll delve into the construction of a simple Python application, serving as a springboard for grasping more intricate functions and concepts within the LangChain framework.

Key Takeaways:

    Querying LLMs via APIs: Witness the simplicity of tapping into the vast capabilities of Language Models through straightforward API calls.

    Textual Data Handling: Learn to load text from a diverse array of documents, enabling your application to process information from various sources.

    Text Tokenization Techniques: Explore the world of text tokenization, understanding different methods to break down textual data into meaningful units.

    Introduction to Embeddings: Gain insights into the fundamental concept of embeddings, unraveling why they are pivotal in enhancing the intelligence of applications.

    Vector Databases: Navigate the landscape of vector databases and understand their role in optimizing data retrieval and manipulation.

    RAG (Retrieval Augmented Generation): Witness the transformative power of RAG as we leverage it to query LLMs over your dataset, showcasing a synergy between retrieval and generation.

    Synthetic Data: An Exemplar Scenario: Conclude the journey with an example of synthetic data generation

Join us in this concise yet comprehensive session, where we demystify LangChain and empower you to harness the full potential of LLM programming. Whether you're a novice or an experienced developer, this talk is your gateway to building intelligent applications with ease.


## Presentation

https://docs.google.com/presentation/d/1qQ8Cx1oP3RyQa2MkQCf39xTYxHjElRJMv_AaL_V1fK0

## Install

(optional) create virtualenv
```
python -m venv fosdemvenv && source fosdemvenv/bin/activate
```

Install requirements
```
pip install -r requirements.txt
```
## API Key

```
export OPENAI_API_KEY=xx-xXxXxXxXxXxXxXxXxXxXxXxXxXxXxXxXxXxXxXxXxXxXxXxX 
```

## run

```
python src/rag/rag.py
```
