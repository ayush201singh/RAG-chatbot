# LangChain RAG Chatbot using Groq API

This repository contains a **Retrieval-Augmented Generation (RAG)** chatbot built using **LangChain** with the **Groq API** as the backend LLM, and wrapped in a **FastAPI** server for web access.

The project demonstrates a robust, production-ready architecture where the chatbot logic is separated into a Python backend and an HTML/JavaScript frontend.

## Overview

This chatbot showcases:

- How to implement the **Retrieval-Augmented Generation (RAG)** pattern for grounded answers.
- The use of **FastAPI** to serve the LangChain RAG logic as a simple /chat API endpoint.
- How to use the low-latency **Groq-hosted LLMs** (Llama 3.1) for generation.
- The essential **Client-Server relationship** between the index.html frontend and the Python backend.

This system is designed to answer questions strictly based on the provided LangChain documentation data source, minimizing "hallucinations."

## Key Concepts Explained

### 1. Retrieval-Augmented Generation (RAG)

**RAG** is the core framework that addresses the problem of LLMs having static or outdated knowledge.

Instead of relying only on its training data, the RAG process:

1. **Retrieves** relevant context (documents) from an external knowledge base.
2. **Augments** the user's question with that retrieved context.
3. **Generates** an answer based _only_ on the augmented information.

This approach grounds the LLM's response in factual, verifiable data, making the chatbot accurate for domain-specific queries.

### 2. Vector Store (Chroma) and Embeddings

In this RAG system, the **Vector Store** (Chroma) acts as the chatbot's specific knowledge base.

Key features:

- **Indexing:** Documents from the LangChain website are converted into numerical representations called **embeddings** (using a HuggingFace model).
- **Storage:** These embeddings are stored in the Vector Store.
- **Retrieval:** When a user asks a question, the Vector Store finds the document chunks whose embeddings are **semantically closest** to the question's embedding. This retrieved data forms the context for the LLM.

### 3. LLM (Large Language Model)

An LLM is the core intelligence of your chatbot. In this project, the LLM is powered by Groq's hosted models (specifically llama-3.1-8b-instant), which are known for:

- Extremely low-latency inference,
- Stable API performance,
- Compatibility with LangChain and FastAPI.

The LLM is the final step, synthesizing the user's question and the retrieved context into a coherent, natural language response.

### 4. LangChain Expression Language (LCEL) & FastAPI

The logic flow is structured using the LangChain Expression Language (LCEL), which uses the pipe operator (|) to connect components (Retriever, Prompt, LLM) into a single, efficient RAG chain in rag_logic.py.

This entire chain is then wrapped by a FastAPI application (rag_logic_UI.py), which:

- Starts a server on http://127.0.0.1:8000.
- Defines the /chat endpoint that the frontend calls.
- Handles the necessary CORS configuration so the HTML file can communicate with the Python server.

## Project Structure

This project uses a client-server architecture with three main files:

- **rag_logic.py** - Contains the RAG core logic using Python, LangChain, and Groq
- **rag_logic_UI.py** - Serves as the backend API server using Python and FastAPI  
- **index.html** - Provides the frontend UI using HTML, CSS (Tailwind), and JavaScript

## Getting Groq API Keys

You'll need a Groq API key to use the LLM.

1. Go to https://console.groq.com/keys
2. Sign in or create a free account.
3. Generate a new API key.
4. Copy the key and store it securely.

You will set this key directly in the rag_logic.py file.

### Configure the API Key in rag_logic.py

1. Open the file `rag_logic.py`.
2. Find the following line:
   ```python
   os.environ["GROQ_API_KEY"] = "ENTER API KEY"
