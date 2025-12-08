package main

import (
	"context"
	"fmt"
	"os"

	"github.com/aqua777/krait"
)

func main() {
	// Create RAG subcommand with its specific options
	ragCmd := krait.New("rag", "RAG Q&A tool", "Ask questions to documents using RAG (Retrieval-Augmented Generation)").
		// RAG actions
		WithStringSliceP("files", "Files or directories to ingest", "files", "f", "RAG_FILES", nil).
		WithStringP("question", "Question to ask", "question", "q", "RAG_QUESTION", "").
		WithBoolP("chat", "Start interactive chat mode", "chat", "c", "RAG_CHAT", false).
		WithBool("clear", "Clear all cached data", "clear", "RAG_CLEAR", false).
		// RAG pipeline options
		WithStringP(KeyCollection, "Vector store collection name", "collection", "", "RAG_COLLECTION", DefaultCollection).
		WithIntP(KeyChunkSize, "Text chunk size", "chunk-size", "", "RAG_CHUNK_SIZE", DefaultChunkSize).
		WithIntP(KeyChunkOverlap, "Text chunk overlap", "chunk-overlap", "", "RAG_CHUNK_OVERLAP", DefaultChunkOverlap).
		WithIntP(KeyTopK, "Number of results to retrieve", "top-k", "k", "RAG_TOP_K", DefaultTopK).
		WithBoolP(KeyStream, "Enable streaming output", "stream", "s", "RAG_STREAM", false).
		WithRun(runRAG)

	// Create root application with global options
	app := krait.App("llamaindex", "LlamaIndex CLI tool", "A command-line interface for LlamaIndex operations").
		WithConfig("", "config", "", "LLAMAINDEX_CONFIG").
		// Global options (shared across subcommands)
		WithStringP(KeyCacheDir, "Cache directory for persistence", "cache-dir", "", "LLAMAINDEX_CACHE_DIR", DefaultCacheDir()).
		WithStringP(KeyOllamaURL, "Ollama API URL", "ollama-url", "", "OLLAMA_HOST", DefaultOllamaURL).
		WithStringP(KeyOllamaModel, "Ollama model for LLM", "model", "m", "OLLAMA_MODEL", DefaultOllamaModel).
		WithStringP(KeyOllamaEmbedModel, "Ollama model for embeddings", "embed-model", "e", "OLLAMA_EMBED_MODEL", DefaultOllamaEmbedModel).
		WithBoolP(KeyVerbose, "Enable verbose output", "verbose", "v", "LLAMAINDEX_VERBOSE", false).
		WithCommand(ragCmd).
		WithRun(func(args []string) error {
			// Default action: show help
			fmt.Println("LlamaIndex CLI - Use 'llamaindex rag --help' for RAG commands")
			return nil
		})

	if err := app.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func runRAG(args []string) error {
	ctx := context.Background()

	// Check for clear flag first
	if krait.GetBool("clear") {
		rag, err := NewRAGCommand()
		if err != nil {
			return err
		}
		return rag.Clear()
	}

	// Initialize RAG command
	rag, err := NewRAGCommand()
	if err != nil {
		return fmt.Errorf("failed to initialize RAG: %w", err)
	}

	// Handle file ingestion
	files := krait.GetStringSlice("files")
	if len(files) > 0 {
		if err := rag.IngestFiles(ctx, files); err != nil {
			return err
		}
	}

	// Handle question
	question := krait.GetString("question")
	if question != "" {
		return rag.Query(ctx, question)
	}

	// Handle chat mode
	if krait.GetBool("chat") {
		return rag.Chat(ctx)
	}

	// If files were ingested but no question/chat, we're done
	if len(files) > 0 {
		return nil
	}

	// No action specified
	fmt.Println("Usage: llamaindex rag [--files <paths>] [--question <text>] [--chat] [--clear]")
	fmt.Println("\nExamples:")
	fmt.Println("  llamaindex rag -f ./docs -q 'What is this about?'")
	fmt.Println("  llamaindex rag -f '*.txt' -c")
	fmt.Println("  llamaindex rag --clear")
	return nil
}
