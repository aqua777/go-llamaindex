package store

import (
	"context"

	"github.com/aqua777/go-llamaindex/schema"
)

// VectorStore is the interface for storing and querying vectors.
type VectorStore interface {
	// Add adds nodes to the store.
	Add(ctx context.Context, nodes []schema.Node) ([]string, error)
	// Query finds the top-k most similar nodes to the query embedding.
	Query(ctx context.Context, query schema.VectorStoreQuery) ([]schema.NodeWithScore, error)
	// Delete removes a node from the store by ID.
	Delete(ctx context.Context, refDocID string) error
	// PersistPath returns the storage path. Returns "" for in-memory stores.
	PersistPath() string
}

// BulkVectorStore extends VectorStore with bulk operations.
// Implementations that support metadata-based deletion should implement this.
// This is an optional interface - use type assertion to check support.
type BulkVectorStore interface {
	VectorStore

	// DeleteByFilter removes all nodes matching the metadata filters.
	// Returns the number of deleted nodes.
	// Filters must not be nil or empty (safety requirement).
	DeleteByFilter(ctx context.Context, filters *schema.MetadataFilters) (int, error)

	// Count returns total nodes matching the optional filters.
	// Pass nil to count all nodes.
	Count(ctx context.Context, filters *schema.MetadataFilters) (int, error)
}
