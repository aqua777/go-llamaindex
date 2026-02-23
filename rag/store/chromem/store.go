package chromem

import (
	"context"
	"fmt"
	"strings"
	"sync"

	chromem "github.com/philippgille/chromem-go"

	"github.com/aqua777/go-llamaindex/schema"
)

// ChromemStore is a persistent vector store backed by chromem-go.
// It implements the go-llamaindex VectorStore interface.
type ChromemStore struct {
	mu             sync.RWMutex
	db             *chromem.DB
	embedFunc      chromem.EmbeddingFunc
	collectionName string
	collection     *chromem.Collection
	compress       bool
	persistPath    string // Path where the DB persists data
}

// ChromemStoreOption configures a ChromemStore instance.
type ChromemStoreOption func(*ChromemStore)

// WithChromemCompress enables gzip compression for persistence.
func WithChromemCompress(compress bool) ChromemStoreOption {
	return func(s *ChromemStore) {
		s.compress = compress
	}
}

// WithChromemCollection sets the default collection name.
func WithChromemCollection(name string) ChromemStoreOption {
	return func(s *ChromemStore) {
		s.collectionName = name
	}
}

// NewChromemStore creates a new ChromemStore with persistent storage.
// path is the directory for persistence, embedFunc is the embedding function.
func NewChromemStore(path string, embedFunc chromem.EmbeddingFunc, opts ...ChromemStoreOption) (*ChromemStore, error) {
	s := &ChromemStore{
		embedFunc:      embedFunc,
		collectionName: "default",
		compress:       false,
	}

	for _, opt := range opts {
		opt(s)
	}

	db, err := chromem.NewPersistentDB(path, s.compress)
	if err != nil {
		return nil, fmt.Errorf("failed to create persistent chromem DB: %w", err)
	}
	s.db = db
	s.persistPath = path

	// Create default collection
	if err := s.ensureCollection(s.collectionName); err != nil {
		return nil, err
	}

	return s, nil
}

// NewChromemStoreFromDB creates a ChromemStore from an existing chromem.DB instance.
// This allows sharing a DB instance across multiple stores.
func NewChromemStoreFromDB(db *chromem.DB, embedFunc chromem.EmbeddingFunc, opts ...ChromemStoreOption) (*ChromemStore, error) {
	s := &ChromemStore{
		db:             db,
		embedFunc:      embedFunc,
		collectionName: "default",
		compress:       false,
	}

	for _, opt := range opts {
		opt(s)
	}

	// Create default collection
	if err := s.ensureCollection(s.collectionName); err != nil {
		return nil, err
	}

	return s, nil
}

// NewSimpleChromemStore creates a ChromemStore with minimal configuration.
// If persistPath is empty, the store will be in-memory only.
// Convenience wrapper for tests and simple use cases.
func NewSimpleChromemStore(persistPath string, collectionName string) (*ChromemStore, error) {
	var db *chromem.DB
	if persistPath != "" {
		var err error
		db, err = chromem.NewPersistentDB(persistPath, false)
		if err != nil {
			return nil, fmt.Errorf("failed to create persistent chromem db: %w", err)
		}
	} else {
		db = chromem.NewDB()
	}
	return NewChromemStoreFromDB(db, nil, WithChromemCollection(collectionName))
}

// ensureCollection creates or gets the collection.
// This is only called during construction to set up the immutable collection binding.
func (s *ChromemStore) ensureCollection(name string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	collection, err := s.db.GetOrCreateCollection(name, nil, s.embedFunc)
	if err != nil {
		return fmt.Errorf("failed to get or create collection '%s': %w", name, err)
	}
	s.collection = collection
	s.collectionName = name
	return nil
}

// CollectionName returns the current collection name.
// The collection is immutably set at construction time.
func (s *ChromemStore) CollectionName() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.collectionName
}

// Add adds nodes to the store.
func (s *ChromemStore) Add(ctx context.Context, nodes []schema.Node) ([]string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.collection == nil {
		return nil, fmt.Errorf("no collection set")
	}

	ids := make([]string, 0, len(nodes))
	embeddings := make([][]float32, 0, len(nodes))
	metadatas := make([]map[string]string, 0, len(nodes))
	contents := make([]string, 0, len(nodes))

	for _, node := range nodes {
		if node.ID == "" {
			return nil, fmt.Errorf("node ID cannot be empty")
		}

		ids = append(ids, node.ID)
		contents = append(contents, node.Text)

		// Node.Embedding is now float32 - no conversion needed
		if len(node.Embedding) > 0 {
			embeddings = append(embeddings, node.Embedding)
		} else {
			// No embedding provided - chromem will compute it
			embeddings = append(embeddings, nil)
		}

		// Convert metadata to string map (chromem requirement)
		meta := make(map[string]string)
		for k, v := range node.Metadata {
			meta[k] = fmt.Sprintf("%v", v)
		}
		metadatas = append(metadatas, meta)
	}

	// Handle nil embeddings case - if all are nil, pass nil slice
	allNil := true
	for _, emb := range embeddings {
		if emb != nil {
			allNil = false
			break
		}
	}
	if allNil {
		embeddings = nil
	}

	err := s.collection.Add(ctx, ids, embeddings, metadatas, contents)
	if err != nil {
		return nil, fmt.Errorf("failed to add documents to collection: %w", err)
	}

	return ids, nil
}

// Query finds the top-k most similar nodes to the query embedding.
func (s *ChromemStore) Query(ctx context.Context, query schema.VectorStoreQuery) ([]schema.NodeWithScore, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.collection == nil {
		return nil, fmt.Errorf("no collection set")
	}

	topK := query.GetTopK()
	if topK <= 0 {
		topK = 10
	}

	// Build where clause from filters
	where := make(map[string]string)
	if query.Filters != nil {
		for _, filter := range query.Filters.Filters {
			if filter.Operator == schema.FilterOperatorEq {
				where[filter.Key] = fmt.Sprintf("%v", filter.Value)
			}
		}
	}
	if len(where) == 0 {
		where = nil
	}

	// Query.GetEmbedding() now returns float32 - no conversion needed
	queryEmb := query.GetEmbedding()

	// Query using embedding
	results, err := s.collection.QueryEmbedding(ctx, queryEmb, topK, where, nil)
	if err != nil {
		// Handle the case where collection is empty or has fewer documents than topK
		// chromem-go returns this error instead of an empty result set
		if strings.Contains(err.Error(), "nResults must be <= the number of documents") {
			return []schema.NodeWithScore{}, nil
		}
		return nil, fmt.Errorf("failed to query collection: %w", err)
	}

	// Convert results to NodeWithScore
	output := make([]schema.NodeWithScore, 0, len(results))
	for _, r := range results {
		// Convert metadata back to interface{} map
		meta := make(map[string]interface{})
		for k, v := range r.Metadata {
			meta[k] = v
		}

		// Chromem returns float32 embeddings - matches schema.Node now
		node := schema.Node{
			ID:        r.ID,
			Text:      r.Content,
			Type:      schema.ObjectTypeText,
			Metadata:  meta,
			Embedding: r.Embedding,
		}

		output = append(output, schema.NodeWithScore{
			Node:  node,
			Score: float64(r.Similarity),
		})
	}

	return output, nil
}

// Delete removes a node from the store by ID.
// The refDocID should be the chunk ID (e.g., "docID_chunk_0").
// We extract the base doc_id and use metadata filtering to delete.
func (s *ChromemStore) Delete(ctx context.Context, refDocID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.collection == nil {
		return fmt.Errorf("no collection set")
	}

	// Extract the base doc_id from the chunk ID
	// Chunk IDs have format: "docID_chunk_N"
	// We need to extract "docID" to match the doc_id metadata field
	docID := refDocID
	if idx := strings.LastIndex(refDocID, "_chunk_"); idx > 0 {
		docID = refDocID[:idx]
	}

	// Also try extracting chunk_index for more precise deletion
	chunkIndex := ""
	if idx := strings.LastIndex(refDocID, "_chunk_"); idx > 0 {
		chunkIndex = refDocID[idx+7:] // Skip "_chunk_" prefix
	}

	// Use metadata filter to delete by doc_id
	// This matches documents where doc_id metadata equals our extracted docID
	whereFilter := map[string]string{"doc_id": docID}
	if chunkIndex != "" {
		whereFilter["chunk_index"] = chunkIndex
	}

	err := s.collection.Delete(
		ctx,
		whereFilter, // where - metadata filter
		nil,         // whereDocument - not needed
	)

	// If the above fails or doesn't match anything, that's acceptable
	// since the document might not exist
	if err != nil && !strings.Contains(err.Error(), "not found") {
		return fmt.Errorf("failed to delete from collection: %w", err)
	}

	return nil
}

// DB returns the underlying chromem database.
func (s *ChromemStore) DB() *chromem.DB {
	return s.db
}

// PersistPath returns the path where the database persists data.
func (s *ChromemStore) PersistPath() string {
	return s.persistPath
}

// DeleteByFilter removes all nodes matching the metadata filters.
// Returns the number of deleted nodes (-1 if count is not available).
// Filters must not be nil or empty (safety requirement).
// Implements the BulkVectorStore interface from go-llamaindex.
func (s *ChromemStore) DeleteByFilter(ctx context.Context, filters *schema.MetadataFilters) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.collection == nil {
		return 0, fmt.Errorf("no collection set")
	}

	if filters == nil || len(filters.Filters) == 0 {
		return 0, fmt.Errorf("filter cannot be empty for bulk delete")
	}

	// Convert MetadataFilters to map[string]string (chromem requirement)
	whereFilter := make(map[string]string)
	for _, f := range filters.Filters {
		// Only EQ operator is supported by chromem-go
		if f.Operator == schema.FilterOperatorEq {
			whereFilter[f.Key] = fmt.Sprintf("%v", f.Value)
		}
	}

	if len(whereFilter) == 0 {
		return 0, fmt.Errorf("no valid filters after conversion")
	}

	err := s.collection.Delete(ctx, whereFilter, nil)
	if err != nil {
		// Handle empty collection gracefully
		if strings.Contains(err.Error(), "not found") {
			return 0, nil
		}
		return 0, fmt.Errorf("failed to delete by filter: %w", err)
	}

	// chromem-go doesn't return deleted count
	return -1, nil
}

// Count returns total nodes in the collection.
// Note: chromem-go Count() doesn't support filtering, returns total count.
// Pass nil to count all nodes.
// Implements the BulkVectorStore interface from go-llamaindex.
func (s *ChromemStore) Count(ctx context.Context, filters *schema.MetadataFilters) (int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.collection == nil {
		return 0, fmt.Errorf("no collection set")
	}

	// chromem-go Count() returns total documents, filtering not supported
	// If filters are provided, we log a warning since we can't filter
	if filters != nil && len(filters.Filters) > 0 {
		// chromem-go doesn't support filtered count, return total count
		// Callers should be aware this is a limitation
	}

	return s.collection.Count(), nil
}
