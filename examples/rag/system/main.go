package main

import (
	"context"
	"fmt"
	"log"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/rag"
	openai "github.com/sashabaranov/go-openai"
)

func main() {
	ctx := context.Background()

	// 1. Setup Custom Clients (Ollama)
	// We'll use the OpenAI client compatibility for Ollama
	// Ollama usually runs on http://localhost:11434/v1
	ollamaConfig := openai.DefaultConfig("ollama") // API Key doesn't matter for Ollama
	ollamaConfig.BaseURL = "http://host.docker.internal:11434/v1"

	ollamaClient := openai.NewClientWithConfig(ollamaConfig)

	// 2. Create Custom Implementations
	// Using mxbai-embed-large for embeddings
	embedModel := embedding.NewOpenAIEmbeddingWithClient(ollamaClient, "mxbai-embed-large")
	// Using llama3 for LLM
	llmModel := llm.NewOpenAILLMWithClient(ollamaClient, "jan-v1:q6_k")

	// 3. Initialize RAG System without OpenAI Key
	config := rag.RAGConfig{
		ChunkSize:      512,
		ChunkOverlap:   50,
		PersistPath:    "./.chromem-db",
		CollectionName: "demo-docs",
	}

	// Note: We don't pass OpenAIKey here because we are injecting dependencies
	sys, err := rag.NewRAGSystem(config)
	if err != nil {
		log.Fatal(err)
	}

	// 4. Inject Dependencies
	sys.WithEmbedding(embedModel).WithLLM(llmModel)

	// 5. Ingest Data
	// Ingest a directory
	fmt.Println("Ingesting directory...")
	err = sys.IngestDirectory(ctx, "./.data")
	if err != nil {
		log.Fatalf("Failed to ingest directory: %v", err)
	}

	// Ingest a text variable
	fmt.Println("Ingesting text variable...")
	err = sys.IngestText(ctx, "Ollama is a tool for running open-source large language models locally.", "ollama-info")
	if err != nil {
		log.Fatalf("Failed to ingest text: %v", err)
	}

	// 6. Query
	queries := []string{
		"Summarize what happened to Lula Landry?",
	}

	for _, q := range queries {
		fmt.Printf("\nQuery: %s\n", q)
		response, err := sys.Query(ctx, q)
		if err != nil {
			log.Printf("Query failed: %v", err)
			continue
		}
		fmt.Printf("Response: %s\n", response)
	}
}

