package embedding

import "context"

// EmbeddingModel is the interface for generating text embeddings.
type EmbeddingModel interface {
	// GetTextEmbedding generates an embedding for a given text.
	GetTextEmbedding(ctx context.Context, text string) ([]float64, error)
	// GetQueryEmbedding generates an embedding for a given query.
	// This is often the same as GetTextEmbedding, but some models treat them differently.
	GetQueryEmbedding(ctx context.Context, query string) ([]float64, error)
}

