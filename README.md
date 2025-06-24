[![LangGraph](https://img.shields.io/badge/LangGraph-%23000000.svg?style=flat&logo=python&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![AutoGen](https://img.shields.io/badge/AutoGen-%23181717.svg?style=flat&logo=github&logoColor=white)](https://github.com/microsoft/autogen)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-%234285F4.svg?style=flat&logo=llama&logoColor=white)](https://www.llamaindex.ai/)
[![DSPy](https://img.shields.io/badge/DSPy-%23FF3E00.svg?style=flat&logo=python&logoColor=white)](https://stanfordnlp.github.io/dspy/)
[![GPT-4o](https://img.shields.io/badge/GPT--4o-%230055FF.svg?style=flat&logo=openai&logoColor=white)](https://openai.com/)
[![GPT-4.1](https://img.shields.io/badge/GPT--4.1-%230055FF.svg?style=flat&logo=openai&logoColor=white)](https://openai.com/)
[![GPT-3.5](https://img.shields.io/badge/GPT--3.5--Turbo-%230055FF.svg?style=flat&logo=openai&logoColor=white)](https://openai.com/)
[![LLaMA 3 8B](https://img.shields.io/badge/LLaMA3--8B-%23ffb86c.svg?style=flat&logo=meta&logoColor=white)](https://ai.meta.com/llama/)
[![LLaMA 3 70B](https://img.shields.io/badge/LLaMA3--70B-%23ffb86c.svg?style=flat&logo=meta&logoColor=white)](https://ai.meta.com/llama/)
[![LLaMA 3.3 Versatile](https://img.shields.io/badge/LLaMA--3.3--Versatile-%23ffb86c.svg?style=flat&logo=meta&logoColor=white)](https://ai.meta.com/llama/)
[![FAISS](https://img.shields.io/badge/FAISS-%2300C7B7.svg?style=flat&logo=facebook&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Chroma](https://img.shields.io/badge/Chroma-%23F28D35.svg?style=flat&logo=python&logoColor=white)](https://www.trychroma.com/)
[![Annoy](https://img.shields.io/badge/Annoy-%23339999.svg?style=flat&logo=python&logoColor=white)](https://github.com/spotify/annoy)




# Docker-Agent

## Overview

Doc-Agent is a project for automating Docker workflows using Docker Agents, integrating multiple LLMs, vector stores, and agentic frameworks, with tracing and monitoring of LLM calls. The repository contains two main services:

* **Rag-API**: A FastAPI/UVicorn service handling retrieval-augmented generation (RAG) workflows.
* **Flask-app**: A Flask-based frontend or auxiliary service.

This README provides instructions for setup, local development, Dockerization, configuration of LLMs and vector stores, and tracing/monitoring integration.

## Features

* **Docker Flow Automation**: Dockerfiles and Docker Compose templates to containerize Rag-API and Flask-app services.
* **Multiple LLM Integration**: Support for configuring and switching between different LLM providers.
* **Vector Store Integration**: Connect to FAISS, Chroma, Pinecone, or other vector stores for embeddings storage and retrieval.
* **Agentic Frameworks**: Examples of integrating agent frameworks (e.g., LangChain, LlamaIndex, custom agent patterns) for orchestrating LLM calls and actions.
* **Tracing & Monitoring**: Instrumentation of LLM calls and internal flows using OpenTelemetry (or similar), exporting traces and metrics to observability backends.

## Prerequisites

* **Python** 3.8+ installed.
* **Git** installed.
* **Docker & Docker Compose** installed for containerization (optional for local dev, recommended for production).
* **Local Desktop Engine**: If using a local LLM or embedding engine (e.g., local inference server), ensure it is running before starting services.
* **Environment Variables / Secrets**: API keys or endpoints for LLM providers, vector store credentials, observability endpoints, etc.

## Getting Started (Local Development)

1. **Clone the repository**:

   ```bash
   git clone https://github.com/call-meRavi-SHORT-CODE/Doc-Agent.git
   cd Doc-Agent
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration**:

   * Copy or create a `.env` file in the root (and/or service subfolders) to hold environment variables. Example entries:

     ```ini
     # LLM provider settings
     OPENAI_API_KEY=your_openai_key
     GROQ_API_KEY=your_groq_api_key
     GEMINI_API_KEY=your_gemini_api_key
    
     ```
   * Adjust per your environment and service needs.

5. **Run Rag-API**:
   From the root folder:

   ```bash
   cd Rag-API
   uvicorn app.main:app --reload
   ```

   * The API will start (default on `http://127.0.0.1:8000`).

6. **Run Flask-app**:
   In another terminal (with the same virtual environment activated):

   ```bash
   cd Flask-app
   python app.py
   ```

   * The Flask service will start (default on `http://127.0.0.1:5000`).
  
![Inputs](Flask-app.png)

7. **Open in Browser**:
   Navigate to `http://localhost:5000` (or the configured host/port) to access the frontend or test endpoints.

> **Important**: Ensure your local desktop engine (e.g., a local LLM inference server or embedding engine) is started and accessible before making requests that depend on it.



## Troubleshooting

* **Local desktop engine not started**: Ensure your local LLM or embedding engine (e.g., a local server for embeddings) is running and reachable at the configured endpoint.
* **Port conflicts**: Verify ports 8000 and 5000 are free or update configuration.
* **Environment variables**: Confirm `.env` values are loaded (e.g., using `python-dotenv` or your framework’s loader).



