package main

import (
	"context"
	"fmt"
	"log"

	// "os"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/rag"
	"github.com/aqua777/go-llamaindex/rag/reader"
	"github.com/aqua777/go-llamaindex/rag/store/chromem"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/settings"
	"github.com/aqua777/go-llamaindex/textsplitter"
	openai "github.com/sashabaranov/go-openai"
)

func main() {
	// // 1. Initialize Settings (Environment variables should be set for OpenAI API Key)
	// // OPENAI_API_KEY=...
	// if os.Getenv("OPENAI_API_KEY") == "" {
	// 	log.Fatal("OPENAI_API_KEY environment variable is not set")
	// }
	// // Settings are auto-initialized with defaults in init()

	// 2. Create a persistent Vector Store
	persistPath := "./chromem-db"
	vectorStore, err := chromem.NewChromemStore(persistPath, "documents")
	if err != nil {
		log.Fatalf("Failed to create vector store: %v", err)
	}
	fmt.Println("Vector store initialized at", persistPath)

	// 3. Load Data
	inputDir := "./.data"
	fmt.Println("Loading documents from", inputDir)
	reader := reader.NewSimpleDirectoryReader(inputDir, ".not-txt", ".not-md")
	docs, err := reader.LoadData()
	if err != nil {
		log.Fatalf("Failed to load data: %v", err)
	}
	fmt.Printf("Loaded %d documents\n", len(docs))

	// 4. Parse Documents into Nodes (Chunks)
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
				},
			}
			allNodes = append(allNodes, node)
		}
	}
	fmt.Printf("Created %d nodes from %d documents\n", len(allNodes), len(docs))

	// 5. Generate Embeddings
	fmt.Println("Generating embeddings...")
	// Custom embed model with Ollama endpoint
	config := openai.DefaultConfig("not-needed")
	config.BaseURL = "http://host.docker.internal:11434/v1"
	client := openai.NewClientWithConfig(config)
	embedModel := embedding.NewOpenAIEmbeddingWithClient(client, "mxbai-embed-large") // "bge-large") // "all-minilm:22m")
	ctx := context.Background()

	// Batch embedding generation
	for i := range allNodes {
		emb, err := embedModel.GetTextEmbedding(ctx, allNodes[i].Text)
		if err != nil {
			log.Printf("Failed to embed node %s: %v", allNodes[i].ID, err)
			continue
		}
		allNodes[i].Embedding = emb
	}

	// 6. Ingest into Vector Store
	fmt.Println("Adding nodes to vector store...")
	_, err = vectorStore.Add(ctx, allNodes)
	if err != nil {
		log.Fatalf("Failed to add nodes to store: %v", err)
	}

	// 7. Setup Query Engine
	topK := 3
	if len(allNodes) < topK {
		topK = len(allNodes)
	}
	llmInstance := llm.NewOpenAILLMWithClient(client, "jan-v1:q6_k")
	retriever := rag.NewVectorRetriever(vectorStore, embedModel, topK)
	synthesizer := rag.NewSimpleSynthesizer(llmInstance)
	queryEngine := rag.NewRetrieverQueryEngine(retriever, synthesizer)

	// 8. Query
	queryStr := "What happened to Lula Landry?" // "What is LlamaIndex?"
	fmt.Printf("\nQuerying: %s\n", queryStr)
	response, err := queryEngine.Query(ctx, schema.QueryBundle{QueryString: queryStr})
	if err != nil {
		log.Fatalf("Query failed: %v", err)
	}

	fmt.Println("\nResponse:")
	fmt.Println(response.Response)

	fmt.Println("\nSource Nodes:")
	for _, n := range response.SourceNodes {
		fmt.Printf("- [Score: %.3f] (%d) %s\n", n.Score, len(n.Node.Text), n.Node.Text)
		fmt.Println("--------------------------------")
	}
}
