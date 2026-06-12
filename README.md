# Local AI - Restaurant Review Q&A System

A privacy-focused Q&A system that answers questions about a pizza restaurant using locally-run AI models. This project uses Ollama for local LLM inference and Chroma for vector-based semantic search.

## Overview

This is a **Retrieval-Augmented Generation (RAG)** system that:
- Converts restaurant reviews into embeddings using a local embedding model
- Stores embeddings in a vector database (Chroma)
- Retrieves relevant reviews based on user questions
- Uses a local LLM to generate contextual answers

**No external API calls or cloud services required** — everything runs locally on your machine.

## Project Structure

```
Local AI/
├── main.py                              # Interactive Q&A chatbot
├── vector.py                            # Vector database setup & retrieval
├── realistic_restaurant_reviews.csv     # Restaurant review dataset
├── requirements.txt                     # Python dependencies
├── chroma_langchain_db/                 # Chroma vector database (persisted)
└── README.md                            # This file
```

## Installation

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running on your machine
  - Required models: `llama3.2` (LLM) and `mxbai-embed-large` (embeddings)

### Setup

1. **Create and activate the virtual environment** (already set up):
   ```powershell
   .venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Ensure Ollama models are available**:
   ```bash
   ollama pull llama3.2
   ollama pull mxbai-embed-large
   ```

## Usage

### Start the Q&A System
```powershell
python main.py
```

Then ask questions about the restaurant:
```
Ask your question (q to quit): What do customers say about the pizza?

Ask your question (q to quit): Is the service good?

Ask your question (q to quit): q
```

## How It Works

### Step 1: Vector Database Setup (vector.py)
- Loads restaurant reviews from CSV
- Converts reviews to embeddings using `mxbai-embed-large`
- Stores embeddings in Chroma vector database
- Creates a retriever that fetches the 5 most relevant reviews for any query

### Step 2: Q&A Chain (main.py)
- Accepts user question
- Retrieves top 5 similar reviews from vector database
- Passes reviews + question to `llama3.2` LLM
- Generates contextual answer based on retrieved reviews

## Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` | LLM application framework |
| `langchain-ollama` | Integration with local Ollama models |
| `langchain-chroma` | Vector database interface |
| `pandas` | CSV data processing |
| `chromadb` | Vector database engine |

## Data

The system uses `realistic_restaurant_reviews.csv` containing:
- **Title**: Review headline
- **Review**: Full review text
- **Rating**: Customer rating
- **Date**: Review date

## Features

✅ **Local AI** – No external API calls, complete privacy  
✅ **Fast Retrieval** – Vector similarity search finds relevant reviews instantly  
✅ **Context-Aware** – LLM answers based on actual customer reviews  
✅ **Persistent Storage** – Vector database is cached for fast subsequent runs  

## Customization

### Change the LLM Model
Edit `main.py`:
```python
model = OllamaLLM(model="neural-chat")  # or any model available in Ollama
```

### Change the Embedding Model
Edit `vector.py`:
```python
embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

### Adjust Retrieved Review Count
Edit `vector.py`:
```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 10}  # Retrieve 10 reviews instead of 5
)
```

## Troubleshooting

**Error: "No such file or directory: chroma_langchain_db"**
- This is normal on first run. The database will be created automatically when you run `main.py`.

**Error: "Model not found"**
- Ensure Ollama is running and the required models are pulled:
  ```bash
  ollama pull llama3.2
  ollama pull mxbai-embed-large
  ```

**Slow responses**
- First run creates embeddings and may take time
- Subsequent runs use cached vector database (much faster)
- Model inference speed depends on your hardware

## License

This project is open source and available for personal use.
