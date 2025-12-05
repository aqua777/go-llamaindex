// Package main demonstrates the document ingestion pipeline.
// This example corresponds to Python's low_level/ingestion.ipynb
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/rag/reader"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/textsplitter"
)

func main() {
	ctx := context.Background()

	// 1. Create embedding model (empty strings use defaults/env vars)
	embedModel := embedding.NewOpenAIEmbedding("", "")
	fmt.Println("Embedding model initialized")

	// 2. Create sample documents
	fmt.Println("\n=== Creating Sample Documents ===")
	inputDir := "./data"
	if _, err := os.Stat(inputDir); os.IsNotExist(err) {
		createSampleData(inputDir)
	}

	// 3. Load documents using SimpleDirectoryReader
	fmt.Println("\n=== Loading Documents ===")
	docReader := reader.NewSimpleDirectoryReader(inputDir)
	docs, err := docReader.LoadData()
	if err != nil {
		log.Fatalf("Failed to load documents: %v", err)
	}
	fmt.Printf("Loaded %d documents\n", len(docs))

	for i, doc := range docs {
		fmt.Printf("  %d. %s (%d chars)\n", i+1, doc.Metadata["filename"], len(doc.Text))
	}

	// 4. Text Splitting with SentenceSplitter
	fmt.Println("\n=== Text Splitting ===")
	splitter := textsplitter.NewSentenceSplitter(
		512, // chunk size
		50,  // chunk overlap
		nil, // default tokenizer
		nil, // default splitter strategy
	)

	var allChunks []string
	for _, doc := range docs {
		chunks := splitter.SplitText(doc.Text)
		fmt.Printf("Document '%s': %d chunks\n", doc.Metadata["filename"], len(chunks))
		allChunks = append(allChunks, chunks...)
	}
	fmt.Printf("Total chunks: %d\n", len(allChunks))

	// 5. Create nodes from chunks with relationships
	fmt.Println("\n=== Creating Nodes with Relationships ===")
	var nodes []schema.Node
	for docIdx, doc := range docs {
		chunks := splitter.SplitText(doc.Text)
		for chunkIdx, chunk := range chunks {
			node := schema.Node{
				ID:   fmt.Sprintf("doc%d-chunk%d", docIdx, chunkIdx),
				Text: chunk,
				Type: schema.ObjectTypeText,
				Metadata: map[string]interface{}{
					"source_doc_id": doc.ID,
					"filename":      doc.Metadata["filename"],
					"chunk_idx":     chunkIdx,
				},
				Relationships: make(schema.NodeRelationships),
			}

			// Set source relationship
			node.Relationships.SetSource(schema.RelatedNodeInfo{
				NodeID:   doc.ID,
				NodeType: schema.ObjectTypeDocument,
			})

			nodes = append(nodes, node)
		}
	}

	// Set previous/next relationships
	for i := range nodes {
		if i > 0 {
			nodes[i].Relationships.SetPrevious(nodes[i-1].AsRelatedNodeInfo())
		}
		if i < len(nodes)-1 {
			nodes[i].Relationships.SetNext(nodes[i+1].AsRelatedNodeInfo())
		}
	}
	fmt.Printf("Created %d nodes with relationships\n", len(nodes))

	// Display sample node relationships
	for i, node := range nodes[:min(3, len(nodes))] {
		fmt.Printf("\nNode %d: %s\n", i+1, truncate(node.Text, 60))
		fmt.Printf("  ID: %s\n", node.ID)
		fmt.Printf("  Metadata: %v\n", node.Metadata)
		if prev := node.Relationships.GetPrevious(); prev != nil {
			fmt.Printf("  Previous: %s\n", prev.NodeID)
		}
		if next := node.Relationships.GetNext(); next != nil {
			fmt.Printf("  Next: %s\n", next.NodeID)
		}
	}

	// 6. Generate Embeddings
	fmt.Println("\n=== Generating Embeddings ===")
	embeddedNodes := make([]schema.Node, len(nodes))
	for i, node := range nodes {
		emb, err := embedModel.GetTextEmbedding(ctx, node.Text)
		if err != nil {
			log.Printf("Warning: Failed to embed node %s: %v", node.ID, err)
			embeddedNodes[i] = node
			continue
		}
		embeddedNodes[i] = node
		embeddedNodes[i].Embedding = emb
		if (i+1)%5 == 0 || i == len(nodes)-1 {
			fmt.Printf("  Embedded %d/%d nodes\n", i+1, len(nodes))
		}
	}

	// 7. Demonstrate metadata-aware splitting
	fmt.Println("\n=== Metadata-Aware Splitting ===")
	metadataSplitter := textsplitter.NewSentenceSplitter(512, 50, nil, nil)

	// Simulate metadata that will be prepended to chunks
	sampleMetadata := "filename: example.txt\nauthor: John Doe\ndate: 2024-01-01"
	sampleText := docs[0].Text

	metadataAwareChunks, err := metadataSplitter.SplitTextMetadataAware(sampleText, sampleMetadata)
	if err != nil {
		log.Printf("Metadata-aware split failed: %v", err)
	} else {
		fmt.Printf("Metadata-aware splitting created %d chunks\n", len(metadataAwareChunks))
		fmt.Printf("(Accounting for %d chars of metadata)\n", len(sampleMetadata))
	}

	// 8. Custom transformation pipeline
	fmt.Println("\n=== Custom Transformation Pipeline ===")
	transformedNodes := runTransformationPipeline(ctx, docs, embedModel)
	fmt.Printf("Pipeline produced %d nodes with embeddings\n", len(transformedNodes))

	// 9. Summary
	fmt.Println("\n=== Ingestion Summary ===")
	fmt.Printf("Documents loaded: %d\n", len(docs))
	fmt.Printf("Nodes created: %d\n", len(nodes))
	fmt.Printf("Nodes with embeddings: %d\n", countNodesWithEmbeddings(embeddedNodes))

	fmt.Println("\n=== Ingestion Pipeline Complete ===")
}

// runTransformationPipeline demonstrates a custom transformation pipeline.
func runTransformationPipeline(ctx context.Context, docs []schema.Node, embedModel *embedding.OpenAIEmbedding) []schema.Node {
	// Step 1: Split documents into smaller chunks
	splitter := textsplitter.NewSentenceSplitter(256, 25, nil, nil)

	var nodes []schema.Node
	for docIdx, doc := range docs {
		chunks := splitter.SplitText(doc.Text)
		for chunkIdx, chunk := range chunks {
			node := schema.Node{
				ID:   fmt.Sprintf("pipeline-doc%d-chunk%d", docIdx, chunkIdx),
				Text: chunk,
				Type: schema.ObjectTypeText,
				Metadata: map[string]interface{}{
					"source_doc_id":    doc.ID,
					"pipeline_version": "1.0",
					"processed":        true,
				},
			}
			nodes = append(nodes, node)
		}
	}

	// Step 2: Generate embeddings
	result := make([]schema.Node, 0, len(nodes))
	for _, node := range nodes {
		emb, err := embedModel.GetTextEmbedding(ctx, node.Text)
		if err != nil {
			continue
		}
		node.Embedding = emb
		result = append(result, node)
	}

	return result
}

// createSampleData creates sample documents for the example.
func createSampleData(dir string) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		log.Fatalf("Failed to create data directory: %v", err)
	}

	docs := map[string]string{
		"ai_overview.txt": `Artificial Intelligence: An Overview

Artificial intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction.

Machine Learning
Machine learning is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves.

Deep Learning
Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.

Natural Language Processing
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.`,

		"rag_explained.txt": `Understanding Retrieval-Augmented Generation

RAG combines the power of large language models with external knowledge retrieval to produce more accurate and contextual responses.

The RAG Process
1. Document Indexing: Documents are processed and stored in a vector database
2. Query Processing: User queries are converted to embeddings
3. Retrieval: Similar documents are retrieved based on embedding similarity
4. Generation: The LLM generates responses using retrieved context

Benefits of RAG
- Reduces hallucination by grounding responses in actual data
- Enables access to private or domain-specific information
- More cost-effective than fine-tuning for many use cases
- Provides transparency through source attribution

Implementation Considerations
When implementing RAG, consider chunk size, overlap, embedding model selection, and retrieval strategy. These factors significantly impact the quality of generated responses.`,

		"vector_stores.txt": `Vector Stores in RAG Systems

Vector stores are specialized databases designed to store and query high-dimensional vectors efficiently. They are essential components in RAG systems.

Types of Vector Stores
1. In-memory stores: Fast but limited by RAM
2. Disk-based stores: Larger capacity, slightly slower
3. Distributed stores: Scalable for large datasets

Popular Vector Stores
- Chroma: Simple, in-memory or persistent
- Pinecone: Managed cloud service
- Weaviate: Open-source, feature-rich
- Milvus: Highly scalable, open-source

Similarity Metrics
Vector stores typically support multiple similarity metrics:
- Cosine similarity: Measures angle between vectors
- Euclidean distance: Measures straight-line distance
- Dot product: Measures projection of one vector onto another`,
	}

	for filename, content := range docs {
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

// countNodesWithEmbeddings counts nodes that have embeddings.
func countNodesWithEmbeddings(nodes []schema.Node) int {
	count := 0
	for _, node := range nodes {
		if len(node.Embedding) > 0 {
			count++
		}
	}
	return count
}

// min returns the minimum of two integers.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
