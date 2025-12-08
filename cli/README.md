# LlamaIndex CLI (Go)

A command-line interface for LlamaIndex operations, built in Go.

## Installation

```sh
go install github.com/aqua777/go-llamaindex-cli@latest
```

Or build from source:

```sh
cd cmd/llamaindex
go build -o llamaindex .
```

## Usage

```sh
llamaindex rag [flags]
```

### RAG Command

Ask questions to documents using RAG (Retrieval-Augmented Generation).

#### Flags

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--files` | `-f` | Files or directories to ingest | - |
| `--question` | `-q` | Question to ask | - |
| `--chat` | `-c` | Start interactive chat mode | false |
| `--clear` | - | Clear all cached data | false |

#### Global Flags

| Flag | Short | Env Var | Description | Default |
|------|-------|---------|-------------|---------|
| `--cache-dir` | - | `LLAMAINDEX_CACHE_DIR` | Cache directory | `~/.cache/go-llamaindex-cli` |
| `--ollama-url` | - | `OLLAMA_HOST` | Ollama API URL | `http://localhost:11434` |
| `--model` | `-m` | `OLLAMA_MODEL` | Ollama LLM model | `llama3.1` |
| `--embed-model` | `-e` | `OLLAMA_EMBED_MODEL` | Ollama embedding model | `nomic-embed-text` |
| `--collection` | - | `RAG_COLLECTION` | Vector store collection | `default` |
| `--chunk-size` | - | `RAG_CHUNK_SIZE` | Text chunk size | `1024` |
| `--chunk-overlap` | - | `RAG_CHUNK_OVERLAP` | Text chunk overlap | `200` |
| `--top-k` | `-k` | `RAG_TOP_K` | Number of results to retrieve | `5` |
| `--verbose` | `-v` | `LLAMAINDEX_VERBOSE` | Enable verbose output | false |
| `--stream` | `-s` | `LLAMAINDEX_STREAM` | Enable streaming output | false |

### Examples

**Ingest files and ask a question:**

```sh
llamaindex rag -f ./docs -q "What is this about?"
```

**Ingest with glob pattern:**

```sh
llamaindex rag -f "*.txt" -q "Summarize the content"
```

**Interactive chat mode:**

```sh
llamaindex rag -f ./docs -c
```

**Stream responses:**

```sh
llamaindex rag -f ./docs -q "Explain in detail" --stream
```

**Clear cached data:**

```sh
llamaindex rag --clear
```

**Use different Ollama model:**

```sh
llamaindex rag -m mistral -f ./docs -q "What is this?"
```

## Requirements

- [Ollama](https://ollama.ai/) running locally (default: `http://localhost:11434`)
- Embedding model pulled: `ollama pull nomic-embed-text`
- LLM model pulled: `ollama pull llama3.1`

## Data Storage

- **Vector store**: `~/.cache/go-llamaindex-cli/chromem/`
- **Chat history**: `~/.cache/go-llamaindex-cli/chat_history.json`
- **Files history**: `~/.cache/go-llamaindex-cli/files_history.txt`
