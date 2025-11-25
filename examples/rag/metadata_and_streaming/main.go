package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/rag"
	"github.com/aqua777/go-llamaindex/rag/reader"
	"github.com/aqua777/go-llamaindex/rag/store/chromem"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/textsplitter"
	"github.com/google/uuid"
	openai "github.com/sashabaranov/go-openai"
)

func main() {
	ctx := context.Background()

	// Ensure OPENAI_API_KEY is set
	if os.Getenv("OPENAI_API_KEY") == "" {
		log.Fatal("OPENAI_API_KEY environment variable not set")
	}

	// Setup components
	llmModel := llm.NewOpenAILLM("", openai.GPT3Dot5Turbo, "")
	embeddingModel := embedding.NewOpenAIEmbedding("", string(openai.AdaEmbeddingV2))
	textSplitter := textsplitter.NewSentenceSplitter(512, 50, nil, nil)

	// Prepare data directory with documents from different categories
	dataDir := "./data"
	os.MkdirAll(dataDir, os.ModePerm)
	defer os.RemoveAll(dataDir)

	// Create documents with different metadata
	docs := []struct {
		filename string
		content  string
		category string
		author   string
	}{
		{
			filename: "tech_doc.txt",
			content:  "Go is a statically typed, compiled programming language designed at Google. It is syntactically similar to C, but with memory safety, garbage collection, structural typing, and CSP-style concurrency.",
			category: "technology",
			author:   "alice",
		},
		{
			filename: "science_doc.txt",
			content:  "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy. This chemical energy is stored in carbohydrate molecules, such as sugars.",
			category: "science",
			author:   "bob",
		},
		{
			filename: "tech_doc2.md",
			content:  "# Python Programming\n\nPython is an interpreted, high-level, general-purpose programming language. Its design philosophy emphasizes code readability with its notable use of significant indentation.",
			category: "technology",
			author:   "alice",
		},
		{
			filename: "history_doc.txt",
			content:  "The Renaissance was a period in European history marking the transition from the Middle Ages to modernity. It started in Italy in the 14th century and spread to the rest of Europe.",
			category: "history",
			author:   "charlie",
		},
	}

	for _, doc := range docs {
		filePath := filepath.Join(dataDir, doc.filename)
		content := fmt.Sprintf("Category: %s\nAuthor: %s\n\n%s", doc.category, doc.author, doc.content)
		if err := os.WriteFile(filePath, []byte(content), 0644); err != nil {
			log.Fatalf("Failed to write %s: %v", doc.filename, err)
		}
	}

	// Load documents
	dirReader := reader.NewSimpleDirectoryReader(dataDir)
	documents, err := dirReader.LoadData()
	if err != nil {
		log.Fatalf("Failed to load documents: %v", err)
	}
	fmt.Printf("Loaded %d documents.\n", len(documents))

	// Split documents into nodes and add metadata
	var nodes []schema.Node
	for i, doc := range documents {
		chunks, err := textSplitter.SplitTextMetadataAware(doc.Text, "")
		if err != nil {
			log.Printf("Warning: Failed to split text for document %s: %v", doc.ID, err)
			continue
		}
		for _, chunk := range chunks {
			node := schema.Node{
				ID:       uuid.New().String(),
				Text:     chunk,
				Type:     schema.ObjectTypeText,
				Metadata: doc.Metadata,
			}
			// Add category and author metadata from the document content
			node.Metadata["category"] = docs[i].category
			node.Metadata["author"] = docs[i].author
			nodes = append(nodes, node)
		}
	}
	fmt.Printf("Split into %d nodes.\n", len(nodes))

	// Generate embeddings
	for i := range nodes {
		emb, err := embeddingModel.GetTextEmbedding(ctx, nodes[i].Text)
		if err != nil {
			log.Fatalf("Failed to get embedding for node %s: %v", nodes[i].ID, err)
		}
		nodes[i].Embedding = emb
	}
	fmt.Println("Generated embeddings for all nodes.")

	// Create persistent ChromemStore
	persistPath := "./chromem_db"
	collectionName := "metadata_streaming_collection"
	os.RemoveAll(persistPath)
	defer os.RemoveAll(persistPath)

	vectorStore, err := chromem.NewChromemStore(persistPath, collectionName)
	if err != nil {
		log.Fatalf("Failed to create ChromemStore: %v", err)
	}

	_, err = vectorStore.Add(ctx, nodes)
	if err != nil {
		log.Fatalf("Failed to add nodes to vector store: %v", err)
	}
	fmt.Printf("Added %d nodes to ChromemStore.\n\n", len(nodes))

	// Create RAG components
	retriever := rag.NewVectorRetriever(vectorStore, embeddingModel, 2)
	synthesizer := rag.NewSimpleSynthesizer(llmModel)
	queryEngine := rag.NewRetrieverQueryEngine(retriever, synthesizer)

	// Example 1: Query without filters
	fmt.Println("=== Example 1: Query without filters ===")
	query1 := schema.QueryBundle{QueryString: "What programming languages are mentioned?"}
	response1, err := queryEngine.Query(ctx, query1)
	if err != nil {
		log.Fatalf("Query failed: %v", err)
	}
	fmt.Printf("Query: %s\n", query1.QueryString)
	fmt.Printf("Response: %s\n", response1.Response)
	fmt.Println("Source Nodes:")
	for _, node := range response1.SourceNodes {
		fmt.Printf("  - ID: %s, Category: %v, Author: %v\n", node.Node.ID, node.Node.Metadata["category"], node.Node.Metadata["author"])
	}
	fmt.Println()

	// Example 2: Query with metadata filter (only technology documents)
	fmt.Println("=== Example 2: Query with metadata filter (category=technology) ===")
	query2 := schema.QueryBundle{
		QueryString: "What programming languages are mentioned?",
		Filters: &schema.MetadataFilters{
			Filters: []schema.MetadataFilter{
				{
					Key:      "category",
					Value:    "technology",
					Operator: schema.FilterOperatorEq,
				},
			},
		},
	}
	response2, err := queryEngine.Query(ctx, query2)
	if err != nil {
		log.Fatalf("Query with filter failed: %v", err)
	}
	fmt.Printf("Query: %s (filtered by category=technology)\n", query2.QueryString)
	fmt.Printf("Response: %s\n", response2.Response)
	fmt.Println("Source Nodes:")
	for _, node := range response2.SourceNodes {
		fmt.Printf("  - ID: %s, Category: %v, Author: %v\n", node.Node.ID, node.Node.Metadata["category"], node.Node.Metadata["author"])
	}
	fmt.Println()

	// Example 3: Query with metadata filter (only documents by alice)
	fmt.Println("=== Example 3: Query with metadata filter (author=alice) ===")
	query3 := schema.QueryBundle{
		QueryString: "Tell me about programming",
		Filters: &schema.MetadataFilters{
			Filters: []schema.MetadataFilter{
				{
					Key:      "author",
					Value:    "alice",
					Operator: schema.FilterOperatorEq,
				},
			},
		},
	}
	response3, err := queryEngine.Query(ctx, query3)
	if err != nil {
		log.Fatalf("Query with filter failed: %v", err)
	}
	fmt.Printf("Query: %s (filtered by author=alice)\n", query3.QueryString)
	fmt.Printf("Response: %s\n", response3.Response)
	fmt.Println("Source Nodes:")
	for _, node := range response3.SourceNodes {
		fmt.Printf("  - ID: %s, Category: %v, Author: %v\n", node.Node.ID, node.Node.Metadata["category"], node.Node.Metadata["author"])
	}
	fmt.Println()

	// Example 4: Streaming query
	fmt.Println("=== Example 4: Streaming query ===")
	query4 := schema.QueryBundle{QueryString: "What is photosynthesis?"}
	streamResponse, err := queryEngine.QueryStream(ctx, query4)
	if err != nil {
		log.Fatalf("Streaming query failed: %v", err)
	}
	fmt.Printf("Query: %s\n", query4.QueryString)
	fmt.Print("Streaming Response: ")
	for token := range streamResponse.ResponseStream {
		fmt.Print(token)
	}
	fmt.Println()
	fmt.Println("Source Nodes:")
	for _, node := range streamResponse.SourceNodes {
		fmt.Printf("  - ID: %s, Category: %v, Author: %v\n", node.Node.ID, node.Node.Metadata["category"], node.Node.Metadata["author"])
	}
	fmt.Println()

	// Example 5: Streaming query with filter
	fmt.Println("=== Example 5: Streaming query with filter (category=science) ===")
	query5 := schema.QueryBundle{
		QueryString: "Explain the process",
		Filters: &schema.MetadataFilters{
			Filters: []schema.MetadataFilter{
				{
					Key:      "category",
					Value:    "science",
					Operator: schema.FilterOperatorEq,
				},
			},
		},
	}
	streamResponse5, err := queryEngine.QueryStream(ctx, query5)
	if err != nil {
		log.Fatalf("Streaming query with filter failed: %v", err)
	}
	fmt.Printf("Query: %s (filtered by category=science)\n", query5.QueryString)
	fmt.Print("Streaming Response: ")
	for token := range streamResponse5.ResponseStream {
		fmt.Print(token)
	}
	fmt.Println()
	fmt.Println("Source Nodes:")
	for _, node := range streamResponse5.SourceNodes {
		fmt.Printf("  - ID: %s, Category: %v, Author: %v\n", node.Node.ID, node.Node.Metadata["category"], node.Node.Metadata["author"])
	}
	fmt.Println()

	fmt.Println("Example completed successfully!")
}
