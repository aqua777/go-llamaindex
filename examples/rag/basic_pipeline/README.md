# Basic RAG Pipeline Example

This example demonstrates a complete RAG (Retrieval-Augmented Generation) pipeline in Go, corresponding to the Python `low_level/retrieval.ipynb` example.

## Overview

The basic RAG pipeline consists of:
1. **Document Loading** - Load documents using SimpleDirectoryReader
2. **Text Splitting** - Split documents into chunks using SentenceSplitter
3. **Embedding** - Generate embeddings using OpenAI
4. **Vector Storage** - Store embeddings in ChromaDB
5. **Retrieval** - Retrieve relevant chunks based on query
6. **Response Synthesis** - Generate response using LLM

## Prerequisites

- Go 1.21+
- OpenAI API key (set `OPENAI_API_KEY` environment variable)
- Or use a local LLM endpoint (Ollama)

## Usage

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your-api-key

# Run the example
cd examples/rag/basic_pipeline
go run main.go
```

## Sample Data

Place your text files in the `data/` directory. The example will load all `.txt` and `.md` files.

## Components Used

- `rag/reader.SimpleDirectoryReader` - Document loading
- `textsplitter.SentenceSplitter` - Text chunking
- `embedding.OpenAIEmbedding` - Embedding generation
- `rag/store/chromem.ChromemStore` - Vector storage
- `rag.VectorRetriever` - Retrieval
- `rag/synthesizer.SimpleSynthesizer` - Response synthesis
- `rag.RetrieverQueryEngine` - Query orchestration
