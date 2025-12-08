// Package main demonstrates document management with the ingestion pipeline.
// This example corresponds to Python's ingestion/document_management_pipeline.ipynb
package main

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/ingestion"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/textsplitter"
)

// SimpleDocStore is a simple in-memory document store for demonstration.
type SimpleDocStore struct {
	mu        sync.RWMutex
	hashes    map[string]string
	documents map[string]schema.Node
}

// NewSimpleDocStore creates a new SimpleDocStore.
func NewSimpleDocStore() *SimpleDocStore {
	return &SimpleDocStore{
		hashes:    make(map[string]string),
		documents: make(map[string]schema.Node),
	}
}

// GetDocumentHash returns the hash for a document ID.
func (s *SimpleDocStore) GetDocumentHash(docID string) (string, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	hash, ok := s.hashes[docID]
	return hash, ok
}

// SetDocumentHash sets the hash for a document ID.
func (s *SimpleDocStore) SetDocumentHash(docID string, hash string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.hashes[docID] = hash
}

// GetAllDocumentHashes returns all document hashes.
func (s *SimpleDocStore) GetAllDocumentHashes() map[string]string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	result := make(map[string]string)
	for k, v := range s.hashes {
		result[k] = v
	}
	return result
}

// AddDocuments adds documents to the store.
func (s *SimpleDocStore) AddDocuments(nodes []schema.Node) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, node := range nodes {
		s.documents[node.ID] = node
	}
	return nil
}

// DeleteDocument deletes a document by ID.
func (s *SimpleDocStore) DeleteDocument(docID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.documents, docID)
	delete(s.hashes, docID)
	return nil
}

// DeleteRefDoc deletes documents by reference doc ID.
func (s *SimpleDocStore) DeleteRefDoc(refDocID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	// Delete all documents that reference this doc
	for id, node := range s.documents {
		if source := node.Relationships.GetSource(); source != nil && source.NodeID == refDocID {
			delete(s.documents, id)
			delete(s.hashes, id)
		}
	}
	delete(s.hashes, refDocID)
	return nil
}

// GetDocument returns a document by ID.
func (s *SimpleDocStore) GetDocument(docID string) (schema.Node, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	doc, ok := s.documents[docID]
	return doc, ok
}

// Count returns the number of documents.
func (s *SimpleDocStore) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.documents)
}

// SimpleVectorStore is a simple in-memory vector store for demonstration.
type SimpleVectorStore struct {
	mu    sync.RWMutex
	nodes map[string]schema.Node
}

// NewSimpleVectorStore creates a new SimpleVectorStore.
func NewSimpleVectorStore() *SimpleVectorStore {
	return &SimpleVectorStore{
		nodes: make(map[string]schema.Node),
	}
}

// Add adds nodes to the vector store.
func (s *SimpleVectorStore) Add(ctx context.Context, nodes []schema.Node) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, node := range nodes {
		s.nodes[node.ID] = node
	}
	return nil
}

// Delete deletes nodes by reference doc ID.
func (s *SimpleVectorStore) Delete(ctx context.Context, refDocID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for id, node := range s.nodes {
		if source := node.Relationships.GetSource(); source != nil && source.NodeID == refDocID {
			delete(s.nodes, id)
		}
	}
	return nil
}

// Count returns the number of nodes.
func (s *SimpleVectorStore) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.nodes)
}

// TextSplitterTransform wraps a text splitter as a TransformComponent.
type TextSplitterTransform struct {
	splitter *textsplitter.SentenceSplitter
}

// NewTextSplitterTransform creates a new TextSplitterTransform.
func NewTextSplitterTransform(chunkSize, chunkOverlap int) *TextSplitterTransform {
	return &TextSplitterTransform{
		splitter: textsplitter.NewSentenceSplitter(chunkSize, chunkOverlap, nil, nil),
	}
}

// Transform splits nodes into smaller chunks.
func (t *TextSplitterTransform) Transform(ctx context.Context, nodes []schema.Node) ([]schema.Node, error) {
	var result []schema.Node
	for _, node := range nodes {
		chunks := t.splitter.SplitText(node.Text)
		for i, chunk := range chunks {
			newNode := schema.Node{
				ID:       fmt.Sprintf("%s-chunk-%d", node.ID, i),
				Text:     chunk,
				Type:     schema.ObjectTypeText,
				Metadata: copyMetadata(node.Metadata),
			}
			newNode.Metadata["chunk_index"] = i
			newNode.Metadata["parent_id"] = node.ID

			// Set source relationship
			newNode.Relationships = make(schema.NodeRelationships)
			newNode.Relationships.SetSource(schema.RelatedNodeInfo{
				NodeID:   node.ID,
				NodeType: schema.ObjectTypeDocument,
			})

			result = append(result, newNode)
		}
	}
	return result, nil
}

// Name returns the transform name.
func (t *TextSplitterTransform) Name() string {
	return "TextSplitterTransform"
}

// EmbeddingTransform adds embeddings to nodes.
type EmbeddingTransform struct {
	embedModel *embedding.OpenAIEmbedding
}

// NewEmbeddingTransform creates a new EmbeddingTransform.
func NewEmbeddingTransform(embedModel *embedding.OpenAIEmbedding) *EmbeddingTransform {
	return &EmbeddingTransform{embedModel: embedModel}
}

// Transform adds embeddings to nodes.
func (t *EmbeddingTransform) Transform(ctx context.Context, nodes []schema.Node) ([]schema.Node, error) {
	result := make([]schema.Node, len(nodes))
	for i, node := range nodes {
		emb, err := t.embedModel.GetTextEmbedding(ctx, node.Text)
		if err != nil {
			// Skip embedding on error, keep node
			result[i] = node
			continue
		}
		node.Embedding = emb
		result[i] = node
	}
	return result, nil
}

// Name returns the transform name.
func (t *EmbeddingTransform) Name() string {
	return "EmbeddingTransform"
}

func main() {
	ctx := context.Background()

	// Create embedding model
	embedModel := embedding.NewOpenAIEmbedding("", "")
	fmt.Println("=== Document Management Pipeline Demo ===")
	fmt.Println("\nDemonstrates document lifecycle management with deduplication.")

	separator := strings.Repeat("=", 60)

	// 1. Create stores
	fmt.Println("\n" + separator)
	fmt.Println("=== Setting Up Stores ===")
	fmt.Println(separator)

	docStore := NewSimpleDocStore()
	vectorStore := NewSimpleVectorStore()

	fmt.Println("Created in-memory document store")
	fmt.Println("Created in-memory vector store")

	// 2. Create initial documents
	fmt.Println("\n" + separator)
	fmt.Println("=== Initial Document Ingestion ===")
	fmt.Println(separator)

	initialDocs := []schema.Document{
		{
			ID:   "doc1",
			Text: "Go is a statically typed, compiled programming language designed at Google. It was created by Robert Griesemer, Rob Pike, and Ken Thompson. Go is syntactically similar to C, but with memory safety and garbage collection.",
			Metadata: map[string]interface{}{
				"source":  "golang_intro.txt",
				"version": 1,
			},
		},
		{
			ID:   "doc2",
			Text: "Python is a high-level, interpreted programming language. It emphasizes code readability with its notable use of significant whitespace. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
			Metadata: map[string]interface{}{
				"source":  "python_intro.txt",
				"version": 1,
			},
		},
		{
			ID:   "doc3",
			Text: "JavaScript is a programming language that conforms to the ECMAScript specification. It is high-level, often just-in-time compiled, and multi-paradigm. JavaScript is one of the core technologies of the World Wide Web.",
			Metadata: map[string]interface{}{
				"source":  "javascript_intro.txt",
				"version": 1,
			},
		},
	}

	fmt.Printf("Initial documents: %d\n", len(initialDocs))
	for _, doc := range initialDocs {
		fmt.Printf("  - %s: %s...\n", doc.ID, truncate(doc.Text, 40))
	}

	// 3. Create pipeline with Upserts strategy
	fmt.Println("\n" + separator)
	fmt.Println("=== Pipeline with Upserts Strategy ===")
	fmt.Println(separator)

	pipeline := ingestion.NewIngestionPipeline(
		ingestion.WithPipelineName("document_management"),
		ingestion.WithDocstore(docStore),
		ingestion.WithVectorStore(vectorStore),
		ingestion.WithDocstoreStrategy(ingestion.DocstoreStrategyUpserts),
		ingestion.WithTransformations([]ingestion.TransformComponent{
			NewTextSplitterTransform(128, 20),
			NewEmbeddingTransform(embedModel),
		}),
	)

	fmt.Println("Pipeline configuration:")
	fmt.Println("  - Strategy: Upserts (update existing, add new)")
	fmt.Println("  - Transforms: TextSplitter -> Embedding")

	// Run initial ingestion
	fmt.Println("\nRunning initial ingestion...")
	nodes, err := pipeline.Run(ctx, initialDocs, nil)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Processed %d nodes\n", len(nodes))
		fmt.Printf("DocStore count: %d\n", docStore.Count())
		fmt.Printf("VectorStore count: %d\n", vectorStore.Count())
	}

	// 4. Re-run with same documents (deduplication)
	fmt.Println("\n" + separator)
	fmt.Println("=== Deduplication Test ===")
	fmt.Println(separator)

	fmt.Println("\nRe-running pipeline with same documents...")
	nodes, err = pipeline.Run(ctx, initialDocs, nil)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Processed %d nodes (should be 0 - all duplicates)\n", len(nodes))
		fmt.Printf("DocStore count: %d (unchanged)\n", docStore.Count())
	}

	// 5. Update a document
	fmt.Println("\n" + separator)
	fmt.Println("=== Document Update ===")
	fmt.Println(separator)

	updatedDocs := []schema.Document{
		{
			ID:   "doc1",
			Text: "Go (also known as Golang) is a statically typed, compiled programming language designed at Google. It features garbage collection, structural typing, and CSP-style concurrency. Go 1.0 was released in March 2012.",
			Metadata: map[string]interface{}{
				"source":  "golang_intro.txt",
				"version": 2,
				"updated": true,
			},
		},
	}

	fmt.Println("Updating doc1 with new content...")
	fmt.Printf("  New text: %s...\n", truncate(updatedDocs[0].Text, 50))

	nodes, err = pipeline.Run(ctx, updatedDocs, nil)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Processed %d nodes (updated document)\n", len(nodes))
	}

	// 6. Add new document
	fmt.Println("\n" + separator)
	fmt.Println("=== Adding New Document ===")
	fmt.Println(separator)

	newDoc := []schema.Document{
		{
			ID:   "doc4",
			Text: "Rust is a multi-paradigm, general-purpose programming language. Rust emphasizes performance, type safety, and concurrency. Rust enforces memory safety without using a garbage collector.",
			Metadata: map[string]interface{}{
				"source":  "rust_intro.txt",
				"version": 1,
			},
		},
	}

	fmt.Println("Adding new document: doc4")
	nodes, err = pipeline.Run(ctx, newDoc, nil)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Processed %d nodes\n", len(nodes))
		fmt.Printf("DocStore count: %d\n", docStore.Count())
	}

	// 7. Demonstrate Upserts and Delete strategy
	fmt.Println("\n" + separator)
	fmt.Println("=== Upserts and Delete Strategy ===")
	fmt.Println(separator)

	// Create new stores for this demo
	docStore2 := NewSimpleDocStore()
	vectorStore2 := NewSimpleVectorStore()

	pipeline2 := ingestion.NewIngestionPipeline(
		ingestion.WithPipelineName("delete_pipeline"),
		ingestion.WithDocstore(docStore2),
		ingestion.WithVectorStore(vectorStore2),
		ingestion.WithDocstoreStrategy(ingestion.DocstoreStrategyUpsertsAndDelete),
		ingestion.WithTransformations([]ingestion.TransformComponent{
			NewTextSplitterTransform(128, 20),
		}),
	)

	fmt.Println("Strategy: UpsertsAndDelete")
	fmt.Println("  - Updates existing documents")
	fmt.Println("  - Adds new documents")
	fmt.Println("  - DELETES documents not in current batch")

	// Initial ingestion with 3 docs
	fmt.Println("\nInitial ingestion with 3 documents...")
	pipeline2.Run(ctx, initialDocs, nil)
	fmt.Printf("DocStore count: %d\n", docStore2.Count())

	// Re-run with only 2 docs (doc3 should be deleted)
	fmt.Println("\nRe-running with only doc1 and doc2...")
	pipeline2.Run(ctx, initialDocs[:2], nil)
	fmt.Printf("DocStore count: %d (doc3 deleted)\n", docStore2.Count())

	// 8. Demonstrate Duplicates Only strategy
	fmt.Println("\n" + separator)
	fmt.Println("=== Duplicates Only Strategy ===")
	fmt.Println(separator)

	docStore3 := NewSimpleDocStore()

	pipeline3 := ingestion.NewIngestionPipeline(
		ingestion.WithPipelineName("duplicates_pipeline"),
		ingestion.WithDocstore(docStore3),
		ingestion.WithDocstoreStrategy(ingestion.DocstoreStrategyDuplicatesOnly),
		ingestion.WithTransformations([]ingestion.TransformComponent{
			NewTextSplitterTransform(128, 20),
		}),
	)

	fmt.Println("Strategy: DuplicatesOnly")
	fmt.Println("  - Only checks for exact duplicates by hash")
	fmt.Println("  - Does not update existing documents")

	fmt.Println("\nFirst ingestion...")
	nodes, _ = pipeline3.Run(ctx, initialDocs, nil)
	fmt.Printf("Processed %d nodes\n", len(nodes))

	fmt.Println("\nSecond ingestion (same docs)...")
	nodes, _ = pipeline3.Run(ctx, initialDocs, nil)
	fmt.Printf("Processed %d nodes (all duplicates skipped)\n", len(nodes))

	// 9. Pipeline with caching
	fmt.Println("\n" + separator)
	fmt.Println("=== Pipeline Caching ===")
	fmt.Println(separator)

	cache := ingestion.NewIngestionCache()

	cachedPipeline := ingestion.NewIngestionPipeline(
		ingestion.WithPipelineName("cached_pipeline"),
		ingestion.WithPipelineCache(cache),
		ingestion.WithTransformations([]ingestion.TransformComponent{
			NewTextSplitterTransform(128, 20),
		}),
	)

	fmt.Println("Pipeline with caching enabled")
	fmt.Println("  - Caches transformation results")
	fmt.Println("  - Skips re-processing identical inputs")

	fmt.Println("\nFirst run (cache miss)...")
	nodes, _ = cachedPipeline.Run(ctx, initialDocs[:1], nil)
	fmt.Printf("Processed %d nodes\n", len(nodes))

	fmt.Println("\nSecond run (cache hit)...")
	nodes, _ = cachedPipeline.Run(ctx, initialDocs[:1], nil)
	fmt.Printf("Processed %d nodes (from cache)\n", len(nodes))

	// 10. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nDocstore Strategies:")
	fmt.Println("  - Upserts: Update existing, add new documents")
	fmt.Println("  - UpsertsAndDelete: Also delete missing documents")
	fmt.Println("  - DuplicatesOnly: Skip exact duplicates by hash")
	fmt.Println()
	fmt.Println("Document Management Features:")
	fmt.Println("  - Hash-based deduplication")
	fmt.Println("  - Automatic document updates")
	fmt.Println("  - Vector store synchronization")
	fmt.Println("  - Transformation caching")
	fmt.Println()
	fmt.Println("Use Cases:")
	fmt.Println("  - Incremental document updates")
	fmt.Println("  - Large-scale document processing")
	fmt.Println("  - Document versioning")
	fmt.Println("  - Efficient re-indexing")

	fmt.Println("\n=== Document Management Pipeline Demo Complete ===")
}

// copyMetadata creates a copy of metadata.
func copyMetadata(m map[string]interface{}) map[string]interface{} {
	if m == nil {
		return make(map[string]interface{})
	}
	result := make(map[string]interface{})
	for k, v := range m {
		result[k] = v
	}
	return result
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
