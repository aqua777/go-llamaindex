// Package main demonstrates a basic RAG pipeline.
// This example corresponds to Python's low_level/retrieval.ipynb
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/rag"
	"github.com/aqua777/go-llamaindex/rag/reader"
	"github.com/aqua777/go-llamaindex/rag/store/chromem"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/settings"
	"github.com/aqua777/go-llamaindex/textsplitter"
)

func main() {
	ctx := context.Background()

	// 1. Check for API key
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Println("OPENAI_API_KEY not set, using default local endpoint")
	}

	// 2. Create embedding model and LLM (empty strings use defaults/env vars)
	embedModel := embedding.NewOpenAIEmbedding("", "")
	llmInstance := llm.NewOpenAILLM("", "", "")

	// 3. Create vector store (in-memory for this example)
	vectorStore, err := chromem.NewChromemStore("", "basic_pipeline")
	if err != nil {
		log.Fatalf("Failed to create vector store: %v", err)
	}
	fmt.Println("Vector store initialized (in-memory)")

	// 4. Load documents
	inputDir := "./data"
	if _, err := os.Stat(inputDir); os.IsNotExist(err) {
		// Create sample data if directory doesn't exist
		createSampleData(inputDir)
	}

	fmt.Println("Loading documents from", inputDir)
	docReader := reader.NewSimpleDirectoryReader(inputDir)
	docs, err := docReader.LoadData()
	if err != nil {
		log.Fatalf("Failed to load data: %v", err)
	}
	fmt.Printf("Loaded %d documents\n", len(docs))

	// 5. Split documents into chunks
	splitter := textsplitter.NewSentenceSplitter(
		settings.GetChunkSize(),
		settings.GetChunkOverlap(),
		nil, nil,
	)

	var allNodes []schema.Node
	for _, doc := range docs {
		chunks := splitter.SplitText(doc.Text)
		for i, chunk := range chunks {
			node := schema.Node{
				ID:   fmt.Sprintf("%s-chunk-%d", doc.ID, i),
				Text: chunk,
				Type: schema.ObjectTypeText,
				Metadata: map[string]interface{}{
					"source_id": doc.ID,
					"filename":  doc.Metadata["filename"],
					"chunk_idx": i,
				},
			}
			allNodes = append(allNodes, node)
		}
	}
	fmt.Printf("Created %d chunks from %d documents\n", len(allNodes), len(docs))

	// 6. Generate embeddings
	fmt.Println("Generating embeddings...")
	for i := range allNodes {
		emb, err := embedModel.GetTextEmbedding(ctx, allNodes[i].Text)
		if err != nil {
			log.Printf("Warning: Failed to embed chunk %s: %v", allNodes[i].ID, err)
			continue
		}
		allNodes[i].Embedding = emb
	}

	// 7. Add nodes to vector store
	fmt.Println("Adding nodes to vector store...")
	_, err = vectorStore.Add(ctx, allNodes)
	if err != nil {
		log.Fatalf("Failed to add nodes to store: %v", err)
	}

	// 8. Create retriever and query engine
	topK := 3
	retriever := rag.NewVectorRetriever(vectorStore, embedModel, topK)
	synthesizer := rag.NewSimpleSynthesizer(llmInstance)
	queryEngine := rag.NewRetrieverQueryEngine(retriever, synthesizer)

	// 9. Query the pipeline
	queries := []string{
		"What is LlamaIndex?",
		"How does RAG work?",
		"What are the main components?",
	}

	for _, query := range queries {
		fmt.Printf("\n=== Query: %s ===\n", query)
		response, err := queryEngine.Query(ctx, schema.QueryBundle{QueryString: query})
		if err != nil {
			log.Printf("Query failed: %v", err)
			continue
		}

		fmt.Println("\nResponse:")
		fmt.Println(response.Response)

		fmt.Println("\nSource Nodes:")
		for _, n := range response.SourceNodes {
			fmt.Printf("- [Score: %.3f] %s...\n", n.Score, truncate(n.Node.Text, 100))
		}
	}
}

// createSampleData creates sample documents for the example.
func createSampleData(dir string) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		log.Fatalf("Failed to create data directory: %v", err)
	}

	// Sample document about LlamaIndex
	llamaIndexDoc := `# LlamaIndex Overview

LlamaIndex is a data framework for LLM applications to ingest, structure, and access private or domain-specific data.

## Key Features

1. **Data Connectors** - Ingest data from various sources including APIs, PDFs, documents, and databases.
2. **Data Indexes** - Structure your data in intermediate representations optimized for LLMs.
3. **Query Interface** - Accept any input prompt and return a knowledge-augmented response.

## RAG (Retrieval-Augmented Generation)

RAG is a technique that enhances LLM responses by retrieving relevant context from a knowledge base.
The process involves:
- Indexing documents into a vector store
- Retrieving relevant chunks based on query similarity
- Synthesizing a response using the retrieved context

## Main Components

The main components of a RAG pipeline include:
- Document loaders for ingesting data
- Text splitters for chunking documents
- Embedding models for vectorization
- Vector stores for similarity search
- Synthesizers for response generation
`

	ragDoc := `# Understanding RAG

Retrieval-Augmented Generation (RAG) combines the power of large language models with external knowledge retrieval.

## How RAG Works

1. **Indexing Phase**
   - Documents are loaded and split into chunks
   - Each chunk is converted to an embedding vector
   - Vectors are stored in a vector database

2. **Query Phase**
   - User query is converted to an embedding
   - Similar chunks are retrieved from the vector store
   - Retrieved context is passed to the LLM
   - LLM generates a response based on the context

## Benefits of RAG

- Reduces hallucination by grounding responses in actual data
- Enables access to private or up-to-date information
- More cost-effective than fine-tuning for many use cases
- Provides transparency through source attribution
`

	files := map[string]string{
		"llamaindex.md": llamaIndexDoc,
		"rag.md":        ragDoc,
	}

	for filename, content := range files {
		path := dir + "/" + filename
		if err := os.WriteFile(path, []byte(content), 0644); err != nil {
			log.Fatalf("Failed to write %s: %v", filename, err)
		}
	}

	fmt.Println("Created sample data files in", dir)
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
