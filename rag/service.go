package rag

import (
	"context"
	"fmt"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/rag/store"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/textsplitter"
)

// Service provides RAG indexing and query functionality.
// It composes an embedding model, vector store, and text splitter
// to provide a complete indexing pipeline.
type Service struct {
	embedder    embedding.EmbeddingModel
	vectorStore store.VectorStore
	splitter    textsplitter.TextSplitter
	config      ServiceConfig
}

// ServiceConfig holds configuration for the RAG service.
type ServiceConfig struct {
	// ChunkSize is the maximum size of text chunks in tokens.
	ChunkSize int
	// ChunkOverlap is the overlap between consecutive chunks in tokens.
	ChunkOverlap int
	// TopK is the number of results to retrieve in queries.
	TopK int
}

// DefaultServiceConfig returns a ServiceConfig with sensible defaults.
func DefaultServiceConfig() ServiceConfig {
	return ServiceConfig{
		ChunkSize:    1024,
		ChunkOverlap: 200,
		TopK:         5,
	}
}

// ServiceOption is a function that modifies a Service during construction.
type ServiceOption func(*Service) error

// WithServiceEmbedder sets the embedding model.
func WithServiceEmbedder(embedder embedding.EmbeddingModel) ServiceOption {
	return func(s *Service) error {
		if embedder == nil {
			return fmt.Errorf("embedder cannot be nil")
		}
		s.embedder = embedder
		return nil
	}
}

// WithServiceVectorStore sets the vector store.
func WithServiceVectorStore(vs store.VectorStore) ServiceOption {
	return func(s *Service) error {
		if vs == nil {
			return fmt.Errorf("vector store cannot be nil")
		}
		s.vectorStore = vs
		return nil
	}
}

// WithServiceChunkSize sets the chunk size.
func WithServiceChunkSize(size int) ServiceOption {
	return func(s *Service) error {
		if size <= 0 {
			return fmt.Errorf("chunk size must be positive")
		}
		s.config.ChunkSize = size
		return nil
	}
}

// WithServiceChunkOverlap sets the chunk overlap.
func WithServiceChunkOverlap(overlap int) ServiceOption {
	return func(s *Service) error {
		if overlap < 0 {
			return fmt.Errorf("chunk overlap cannot be negative")
		}
		s.config.ChunkOverlap = overlap
		return nil
	}
}

// WithServiceTopK sets the number of results to retrieve.
func WithServiceTopK(k int) ServiceOption {
	return func(s *Service) error {
		if k <= 0 {
			return fmt.Errorf("topK must be positive")
		}
		s.config.TopK = k
		return nil
	}
}

// NewService creates a new RAG service with the given options.
// An embedder must be provided via WithServiceEmbedder.
func NewService(opts ...ServiceOption) (*Service, error) {
	s := &Service{
		config: DefaultServiceConfig(),
	}

	// Apply options
	for _, opt := range opts {
		if err := opt(s); err != nil {
			return nil, err
		}
	}

	// Validate required fields
	if s.embedder == nil {
		return nil, fmt.Errorf("embedder is required (use WithServiceEmbedder)")
	}

	// Use provided vector store or default to in-memory SimpleVectorStore
	if s.vectorStore == nil {
		s.vectorStore = store.NewSimpleVectorStore()
	}

	// Create text splitter
	s.splitter = textsplitter.NewSentenceSplitter(
		s.config.ChunkSize,
		s.config.ChunkOverlap,
		nil, // Use default tokenizer
		nil, // Use default splitter strategy
	)

	return s, nil
}

// Document represents a document to be indexed.
type Document struct {
	ID       string
	Content  string
	Metadata map[string]interface{}
}

// QueryResult represents a result from a query.
type QueryResult struct {
	ID       string
	Content  string
	Score    float64
	Metadata map[string]interface{}
}

// Index indexes a document by splitting it into chunks and storing embeddings.
func (s *Service) Index(ctx context.Context, doc Document) error {
	// Split the document into chunks
	chunks := s.splitter.SplitText(doc.Content)
	if len(chunks) == 0 {
		return nil
	}

	// Create nodes from chunks
	nodes := make([]schema.Node, 0, len(chunks))
	for i, chunk := range chunks {
		// Generate embedding for the chunk
		emb, err := s.embedder.GetTextEmbedding(ctx, chunk)
		if err != nil {
			return fmt.Errorf("failed to embed chunk %d: %w", i, err)
		}

		node := schema.NewTextNode(chunk)
		node.ID = fmt.Sprintf("%s_chunk_%d", doc.ID, i)
		node.Embedding = emb
		node.Metadata = map[string]interface{}{
			"doc_id":      doc.ID,
			"chunk_index": i,
		}
		// Merge document metadata
		for k, v := range doc.Metadata {
			node.Metadata[k] = v
		}

		nodes = append(nodes, *node)
	}

	// Add nodes to vector store
	_, err := s.vectorStore.Add(ctx, nodes)
	if err != nil {
		return fmt.Errorf("failed to add nodes to vector store: %w", err)
	}

	return nil
}

// Query searches for relevant documents based on the query string.
func (s *Service) Query(ctx context.Context, query string) ([]QueryResult, error) {
	// Generate query embedding
	queryEmb, err := s.embedder.GetQueryEmbedding(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to embed query: %w", err)
	}

	// Query vector store
	vsQuery := schema.NewVectorStoreQuery(queryEmb, s.config.TopK)
	results, err := s.vectorStore.Query(ctx, *vsQuery)
	if err != nil {
		return nil, fmt.Errorf("failed to query vector store: %w", err)
	}

	// Convert to QueryResults
	queryResults := make([]QueryResult, 0, len(results))
	for _, r := range results {
		qr := QueryResult{
			ID:       r.Node.ID,
			Content:  r.Node.Text,
			Score:    r.Score,
			Metadata: r.Node.Metadata,
		}
		queryResults = append(queryResults, qr)
	}

	return queryResults, nil
}

// Delete removes a document and its chunks from the vector store.
func (s *Service) Delete(ctx context.Context, docID string) error {
	return s.vectorStore.Delete(ctx, docID)
}

// DeleteByFilter removes all nodes matching the metadata filters.
// Returns an error if the underlying vector store doesn't support bulk operations.
func (s *Service) DeleteByFilter(ctx context.Context, filters *schema.MetadataFilters) (int, error) {
	if bvs, ok := s.vectorStore.(store.BulkVectorStore); ok {
		return bvs.DeleteByFilter(ctx, filters)
	}
	return 0, fmt.Errorf("vector store does not support bulk deletion")
}

// Count returns total nodes matching the optional filters.
// Returns an error if the underlying vector store doesn't support bulk operations.
func (s *Service) Count(ctx context.Context, filters *schema.MetadataFilters) (int, error) {
	if bvs, ok := s.vectorStore.(store.BulkVectorStore); ok {
		return bvs.Count(ctx, filters)
	}
	return 0, fmt.Errorf("vector store does not support count operations")
}

// Embedder returns the underlying embedding model.
func (s *Service) Embedder() embedding.EmbeddingModel {
	return s.embedder
}

// VectorStore returns the underlying vector store.
func (s *Service) VectorStore() store.VectorStore {
	return s.vectorStore
}

// Splitter returns the underlying text splitter.
func (s *Service) Splitter() textsplitter.TextSplitter {
	return s.splitter
}

// Config returns the current configuration.
func (s *Service) Config() ServiceConfig {
	return s.config
}

// PersistPath returns the path where the database persists data.
// Returns empty string for in-memory storage.
func (s *Service) PersistPath() string {
	return s.vectorStore.PersistPath()
}
