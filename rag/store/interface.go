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
}
