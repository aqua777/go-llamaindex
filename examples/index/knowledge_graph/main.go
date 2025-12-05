// Package main demonstrates the KnowledgeGraphIndex for building and querying knowledge graphs.
// This example corresponds to Python's index_structs/knowledge_graph/KnowledgeGraphDemo.ipynb
package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/graphstore"
	"github.com/aqua777/go-llamaindex/index"
	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/schema"
)

func main() {
	ctx := context.Background()

	// Create LLM for triplet extraction
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("=== Knowledge Graph Index Demo ===")
	fmt.Println("\nBuilds a knowledge graph by extracting triplets from documents.")

	separator := strings.Repeat("=", 60)

	// 1. Create sample documents
	fmt.Println("\n" + separator)
	fmt.Println("=== Creating Sample Documents ===")
	fmt.Println(separator)

	documents := []schema.Document{
		{
			ID:   "doc1",
			Text: "Albert Einstein was a theoretical physicist born in Germany. He developed the theory of relativity. Einstein won the Nobel Prize in Physics in 1921.",
			Metadata: map[string]interface{}{
				"source": "biography",
				"topic":  "physics",
			},
		},
		{
			ID:   "doc2",
			Text: "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize. Curie discovered polonium and radium.",
			Metadata: map[string]interface{}{
				"source": "biography",
				"topic":  "physics",
			},
		},
		{
			ID:   "doc3",
			Text: "The theory of relativity was published by Einstein in 1905. It revolutionized our understanding of space, time, and gravity. The famous equation E=mc² comes from this theory.",
			Metadata: map[string]interface{}{
				"source": "science",
				"topic":  "physics",
			},
		},
	}

	fmt.Printf("Created %d documents:\n", len(documents))
	for _, doc := range documents {
		fmt.Printf("  - %s: %s...\n", doc.ID, truncate(doc.Text, 50))
	}

	// 2. Build Knowledge Graph Index with LLM extraction
	fmt.Println("\n" + separator)
	fmt.Println("=== Building Knowledge Graph Index ===")
	fmt.Println(separator)

	fmt.Println("\nExtracting triplets using LLM...")

	kgIndex, err := index.NewKnowledgeGraphIndexFromDocuments(
		ctx,
		documents,
		index.WithKGIndexLLM(llmInstance),
		index.WithKGIndexMaxTripletsPerChunk(10),
		index.WithKGIndexGraphStoreQueryDepth(2),
	)
	if err != nil {
		fmt.Printf("Error creating KG index: %v\n", err)
		// Continue with manual triplets demo
	} else {
		fmt.Println("Knowledge graph index created successfully!")

		// Show extracted keywords
		keywords := kgIndex.GetAllKeywords()
		fmt.Printf("\nExtracted %d keywords/entities:\n", len(keywords))
		for i, kw := range keywords {
			if i >= 10 {
				fmt.Printf("  ... and %d more\n", len(keywords)-10)
				break
			}
			fmt.Printf("  - %s\n", kw)
		}
	}

	// 3. Manual triplet insertion demo
	fmt.Println("\n" + separator)
	fmt.Println("=== Manual Triplet Insertion ===")
	fmt.Println(separator)

	// Create a new KG index with manual triplets
	graphStore := graphstore.NewSimpleGraphStore()

	// Insert triplets manually
	triplets := []graphstore.Triplet{
		{Subject: "Einstein", Relation: "was born in", Object: "Germany"},
		{Subject: "Einstein", Relation: "developed", Object: "Theory of Relativity"},
		{Subject: "Einstein", Relation: "won", Object: "Nobel Prize in Physics"},
		{Subject: "Marie Curie", Relation: "discovered", Object: "Polonium"},
		{Subject: "Marie Curie", Relation: "discovered", Object: "Radium"},
		{Subject: "Marie Curie", Relation: "won", Object: "Nobel Prize"},
		{Subject: "Theory of Relativity", Relation: "contains", Object: "E=mc²"},
		{Subject: "Theory of Relativity", Relation: "describes", Object: "Space-time"},
		{Subject: "Nobel Prize", Relation: "awarded for", Object: "Physics"},
	}

	fmt.Println("\nInserting triplets:")
	for _, t := range triplets {
		fmt.Printf("  %s\n", t.String())
		graphStore.UpsertTriplet(ctx, t.Subject, t.Relation, t.Object)
	}

	// 4. Query the graph store
	fmt.Println("\n" + separator)
	fmt.Println("=== Querying Graph Store ===")
	fmt.Println(separator)

	subjects := []string{"Einstein", "Marie Curie", "Theory of Relativity"}

	for _, subj := range subjects {
		fmt.Printf("\nRelations for '%s':\n", subj)
		relations, err := graphStore.Get(ctx, subj)
		if err != nil {
			fmt.Printf("  Error: %v\n", err)
			continue
		}
		if len(relations) == 0 {
			fmt.Println("  (no relations found)")
			continue
		}
		for _, rel := range relations {
			if len(rel) >= 2 {
				fmt.Printf("  - %s -> %s\n", rel[0], rel[1])
			}
		}
	}

	// 5. Graph traversal with depth
	fmt.Println("\n" + separator)
	fmt.Println("=== Graph Traversal (Depth=2) ===")
	fmt.Println(separator)

	fmt.Println("\nStarting from 'Einstein', traversing with depth 2:")

	relMap, err := graphStore.GetRelMap(ctx, []string{"Einstein"}, 2, 100)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		for subj, rels := range relMap {
			fmt.Printf("\n  Subject: %s\n", subj)
			for _, rel := range rels {
				if len(rel) >= 3 {
					fmt.Printf("    -> %s -> %s\n", rel[1], rel[2])
				}
			}
		}
	}

	// 6. Create KG Index with custom extraction function
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom Triplet Extraction ===")
	fmt.Println(separator)

	// Custom extraction function (simple pattern-based)
	customExtractor := func(text string) ([]graphstore.Triplet, error) {
		var triplets []graphstore.Triplet

		// Simple pattern: "X was a Y" -> (X, was a, Y)
		patterns := []struct {
			pattern string
			rel     string
		}{
			{"was a", "was a"},
			{"is a", "is a"},
			{"developed", "developed"},
			{"discovered", "discovered"},
			{"won", "won"},
		}

		words := strings.Fields(text)
		for i := 0; i < len(words)-2; i++ {
			for _, p := range patterns {
				patternWords := strings.Fields(p.pattern)
				if len(patternWords) == 1 && words[i] == patternWords[0] {
					if i > 0 && i < len(words)-1 {
						triplets = append(triplets, graphstore.Triplet{
							Subject:  words[i-1],
							Relation: p.rel,
							Object:   words[i+1],
						})
					}
				}
			}
		}

		return triplets, nil
	}

	fmt.Println("\nUsing custom pattern-based extractor:")

	customDoc := schema.Document{
		ID:   "custom1",
		Text: "Python is a programming language. Guido developed Python. Python won popularity.",
	}

	extractedTriplets, _ := customExtractor(customDoc.Text)
	fmt.Printf("Extracted from: \"%s\"\n", customDoc.Text)
	for _, t := range extractedTriplets {
		fmt.Printf("  %s\n", t.String())
	}

	// 7. Using KG Index as Query Engine
	fmt.Println("\n" + separator)
	fmt.Println("=== KG Index as Query Engine ===")
	fmt.Println(separator)

	if kgIndex != nil {
		fmt.Println("\nQuerying the knowledge graph...")

		queryEngine := kgIndex.AsQueryEngine(
			index.WithQueryEngineLLM(llmInstance),
		)

		queries := []string{
			"What did Einstein develop?",
			"Who discovered radium?",
		}

		for _, query := range queries {
			fmt.Printf("\nQuery: %s\n", query)
			response, err := queryEngine.Query(ctx, query)
			if err != nil {
				fmt.Printf("Error: %v\n", err)
				continue
			}
			fmt.Printf("Response: %s\n", truncate(response.Response, 200))
		}
	}

	// 8. Visualize the knowledge graph structure
	fmt.Println("\n" + separator)
	fmt.Println("=== Knowledge Graph Visualization ===")
	fmt.Println(separator)

	fmt.Println(`
    ┌──────────────┐
    │   Einstein   │
    └──────┬───────┘
           │
    ┌──────┴──────────────────────┐
    │              │              │
    ▼              ▼              ▼
┌────────┐  ┌───────────┐  ┌───────────────┐
│Germany │  │ Theory of │  │ Nobel Prize   │
│(born)  │  │ Relativity│  │ in Physics    │
└────────┘  └─────┬─────┘  └───────────────┘
                  │
           ┌──────┴──────┐
           │             │
           ▼             ▼
      ┌────────┐   ┌───────────┐
      │ E=mc²  │   │Space-time │
      └────────┘   └───────────┘
`)

	// 9. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nKnowledge Graph Index Features:")
	fmt.Println("  - Automatic triplet extraction using LLM")
	fmt.Println("  - Custom triplet extraction functions")
	fmt.Println("  - Graph store for triplet storage")
	fmt.Println("  - Keyword-based retrieval")
	fmt.Println("  - Embedding-based retrieval (optional)")
	fmt.Println("  - Graph traversal with configurable depth")
	fmt.Println()
	fmt.Println("Use Cases:")
	fmt.Println("  - Entity relationship extraction")
	fmt.Println("  - Question answering over structured data")
	fmt.Println("  - Knowledge base construction")
	fmt.Println("  - Fact verification systems")

	fmt.Println("\n=== Knowledge Graph Index Demo Complete ===")
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
