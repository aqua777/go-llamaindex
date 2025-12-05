// Package main demonstrates the SummaryIndex for document summarization.
// This example corresponds to Python's index_structs/doc_summary/
package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/index"
	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/rag/synthesizer"
	"github.com/aqua777/go-llamaindex/schema"
)

func main() {
	ctx := context.Background()

	// Create LLM and embedding model
	llmInstance := llm.NewOpenAILLM("", "", "")
	embedModel := embedding.NewOpenAIEmbedding("", "")
	fmt.Println("=== Summary Index Demo ===")
	fmt.Println("\nStores documents in a list and synthesizes answers from all nodes.")

	separator := strings.Repeat("=", 60)

	// 1. Create sample documents
	fmt.Println("\n" + separator)
	fmt.Println("=== Creating Sample Documents ===")
	fmt.Println(separator)

	documents := []schema.Document{
		{
			ID:   "intro",
			Text: "Go (also known as Golang) is a statically typed, compiled programming language designed at Google. It was created by Robert Griesemer, Rob Pike, and Ken Thompson. Go is syntactically similar to C, but with memory safety, garbage collection, structural typing, and CSP-style concurrency.",
			Metadata: map[string]interface{}{
				"section": "introduction",
				"topic":   "golang",
			},
		},
		{
			ID:   "features",
			Text: "Go features include fast compilation, garbage collection, and built-in concurrency with goroutines and channels. The language has a simple syntax with only 25 keywords. Go supports interfaces for polymorphism and has a powerful standard library.",
			Metadata: map[string]interface{}{
				"section": "features",
				"topic":   "golang",
			},
		},
		{
			ID:   "concurrency",
			Text: "Concurrency in Go is achieved through goroutines, which are lightweight threads managed by the Go runtime. Channels provide a way for goroutines to communicate and synchronize. The select statement allows waiting on multiple channel operations.",
			Metadata: map[string]interface{}{
				"section": "concurrency",
				"topic":   "golang",
			},
		},
		{
			ID:   "use_cases",
			Text: "Go is commonly used for building web servers, cloud services, and command-line tools. Companies like Google, Uber, and Dropbox use Go in production. Popular Go projects include Docker, Kubernetes, and Terraform.",
			Metadata: map[string]interface{}{
				"section": "use_cases",
				"topic":   "golang",
			},
		},
	}

	fmt.Printf("Created %d documents:\n", len(documents))
	for _, doc := range documents {
		fmt.Printf("  - %s (%s): %s...\n",
			doc.ID,
			doc.Metadata["section"],
			truncate(doc.Text, 40))
	}

	// 2. Build Summary Index
	fmt.Println("\n" + separator)
	fmt.Println("=== Building Summary Index ===")
	fmt.Println(separator)

	summaryIndex, err := index.NewSummaryIndexFromDocuments(
		ctx,
		documents,
		index.WithSummaryIndexEmbedModel(embedModel),
	)
	if err != nil {
		fmt.Printf("Error creating summary index: %v\n", err)
		return
	}

	fmt.Println("Summary index created successfully!")
	fmt.Printf("Index contains %d nodes\n", len(documents))

	// 3. Default retrieval (all nodes)
	fmt.Println("\n" + separator)
	fmt.Println("=== Default Retrieval Mode ===")
	fmt.Println(separator)

	fmt.Println("\nDefault mode returns ALL nodes for comprehensive summarization.")

	retriever := summaryIndex.AsRetriever()
	query := schema.QueryBundle{QueryString: "What is Go programming language?"}

	nodes, err := retriever.Retrieve(ctx, query)
	if err != nil {
		fmt.Printf("Error retrieving: %v\n", err)
	} else {
		fmt.Printf("\nRetrieved %d nodes:\n", len(nodes))
		for i, n := range nodes {
			fmt.Printf("  %d. [Score: %.2f] %s: %s...\n",
				i+1, n.Score,
				n.Node.ID,
				truncate(n.Node.Text, 40))
		}
	}

	// 4. Query with tree summarization
	fmt.Println("\n" + separator)
	fmt.Println("=== Query with Tree Summarization ===")
	fmt.Println(separator)

	fmt.Println("\nTree summarization recursively summarizes nodes into a final answer.")

	queryEngine := summaryIndex.AsQueryEngine(
		index.WithQueryEngineLLM(llmInstance),
		index.WithResponseMode(synthesizer.ResponseModeTreeSummarize),
	)

	queries := []string{
		"Summarize the key features of Go programming language",
		"How does concurrency work in Go?",
		"What are some popular projects built with Go?",
	}

	for _, q := range queries {
		fmt.Printf("\nQuery: %s\n", q)
		response, err := queryEngine.Query(ctx, q)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}
		fmt.Printf("Response: %s\n", truncate(response.Response, 200))
	}

	// 5. Insert new documents
	fmt.Println("\n" + separator)
	fmt.Println("=== Inserting New Documents ===")
	fmt.Println(separator)

	newDoc := schema.Document{
		ID:   "tools",
		Text: "Go has excellent tooling including gofmt for code formatting, go vet for static analysis, and go test for testing. The go mod system handles dependency management. Popular IDEs like VS Code and GoLand provide great Go support.",
		Metadata: map[string]interface{}{
			"section": "tooling",
			"topic":   "golang",
		},
	}

	fmt.Printf("\nInserting new document: %s\n", newDoc.ID)

	newNode := schema.NewTextNode(newDoc.Text)
	newNode.ID = newDoc.ID
	newNode.Metadata = newDoc.Metadata

	err = summaryIndex.InsertNodes(ctx, []schema.Node{*newNode})
	if err != nil {
		fmt.Printf("Error inserting: %v\n", err)
	} else {
		fmt.Println("Document inserted successfully!")
	}

	// Verify insertion
	allNodes, _ := summaryIndex.GetNodes(ctx)
	fmt.Printf("Index now contains %d nodes\n", len(allNodes))

	// 6. Delete documents
	fmt.Println("\n" + separator)
	fmt.Println("=== Deleting Documents ===")
	fmt.Println(separator)

	fmt.Println("\nDeleting document: tools")

	err = summaryIndex.DeleteNodes(ctx, []string{"tools"})
	if err != nil {
		fmt.Printf("Error deleting: %v\n", err)
	} else {
		fmt.Println("Document deleted successfully!")
	}

	allNodes, _ = summaryIndex.GetNodes(ctx)
	fmt.Printf("Index now contains %d nodes\n", len(allNodes))

	// 7. Refresh documents
	fmt.Println("\n" + separator)
	fmt.Println("=== Refreshing Documents ===")
	fmt.Println(separator)

	fmt.Println("\nRefresh updates existing documents and adds new ones.")

	updatedDocs := []schema.Document{
		{
			ID:   "intro",
			Text: "Go (Golang) is a modern programming language created at Google in 2009. It combines the efficiency of compiled languages with the ease of use of dynamic languages. Go 1.0 was released in March 2012.",
			Metadata: map[string]interface{}{
				"section": "introduction",
				"topic":   "golang",
				"updated": true,
			},
		},
		{
			ID:   "new_section",
			Text: "Go 1.18 introduced generics, allowing type parameters in functions and types. This was one of the most requested features and enables more reusable code without sacrificing type safety.",
			Metadata: map[string]interface{}{
				"section": "generics",
				"topic":   "golang",
			},
		},
	}

	refreshed, err := summaryIndex.RefreshDocuments(ctx, updatedDocs)
	if err != nil {
		fmt.Printf("Error refreshing: %v\n", err)
	} else {
		fmt.Println("Refresh results:")
		for i, doc := range updatedDocs {
			status := "unchanged"
			if refreshed[i] {
				status = "updated/added"
			}
			fmt.Printf("  - %s: %s\n", doc.ID, status)
		}
	}

	// 8. Different response modes
	fmt.Println("\n" + separator)
	fmt.Println("=== Different Response Modes ===")
	fmt.Println(separator)

	responseModes := []struct {
		mode synthesizer.ResponseMode
		name string
		desc string
	}{
		{synthesizer.ResponseModeCompact, "Compact", "Combines nodes and generates a single response"},
		{synthesizer.ResponseModeRefine, "Refine", "Iteratively refines answer through each node"},
		{synthesizer.ResponseModeTreeSummarize, "Tree Summarize", "Recursively summarizes nodes"},
	}

	testQuery := "What is Go used for?"

	for _, rm := range responseModes {
		fmt.Printf("\n%s Mode: %s\n", rm.name, rm.desc)

		qe := summaryIndex.AsQueryEngine(
			index.WithQueryEngineLLM(llmInstance),
			index.WithResponseMode(rm.mode),
		)

		response, err := qe.Query(ctx, testQuery)
		if err != nil {
			fmt.Printf("  Error: %v\n", err)
			continue
		}
		fmt.Printf("  Response: %s\n", truncate(response.Response, 150))
	}

	// 9. Summary Index vs Vector Index
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary Index vs Vector Index ===")
	fmt.Println(separator)

	fmt.Println(`
┌─────────────────────────────────────────────────────────────┐
│                    Summary Index                             │
├─────────────────────────────────────────────────────────────┤
│ • Stores all documents in a list                            │
│ • Default: Returns ALL nodes for synthesis                  │
│ • Best for: Comprehensive summarization                     │
│ • Use when: You need to consider all documents              │
│ • Response modes: Tree summarize, Compact, Refine           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Vector Index                              │
├─────────────────────────────────────────────────────────────┤
│ • Stores documents with embeddings                          │
│ • Returns top-k most similar nodes                          │
│ • Best for: Semantic search and retrieval                   │
│ • Use when: You need specific relevant documents            │
│ • Retrieval: Embedding similarity                           │
└─────────────────────────────────────────────────────────────┘
`)

	// 10. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nSummary Index Features:")
	fmt.Println("  - Simple list-based document storage")
	fmt.Println("  - Default retrieval returns all nodes")
	fmt.Println("  - Optional embedding-based retrieval")
	fmt.Println("  - Multiple response synthesis modes")
	fmt.Println("  - Document insert, delete, and refresh")
	fmt.Println()
	fmt.Println("Use Cases:")
	fmt.Println("  - Document summarization")
	fmt.Println("  - Comprehensive Q&A over all documents")
	fmt.Println("  - Report generation")
	fmt.Println("  - When all context is needed for answers")

	fmt.Println("\n=== Summary Index Demo Complete ===")
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
