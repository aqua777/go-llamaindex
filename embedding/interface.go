package embedding

import "context"

// EmbeddingModel is the interface for generating text embeddings.
// This is the basic interface that all embedding implementations must satisfy.
type EmbeddingModel interface {
	// GetTextEmbedding generates an embedding for a given text.
	GetTextEmbedding(ctx context.Context, text string) ([]float64, error)
	// GetQueryEmbedding generates an embedding for a given query.
	// This is often the same as GetTextEmbedding, but some models treat them differently.
	GetQueryEmbedding(ctx context.Context, query string) ([]float64, error)
}

// EmbeddingModelWithInfo extends EmbeddingModel with metadata capabilities.
type EmbeddingModelWithInfo interface {
	EmbeddingModel
	// Info returns information about the model's capabilities.
	Info() EmbeddingInfo
}

// EmbeddingModelWithBatch extends EmbeddingModel with batch processing capabilities.
type EmbeddingModelWithBatch interface {
	EmbeddingModel
	// GetTextEmbeddingsBatch generates embeddings for multiple texts.
	// The callback is optional and can be used to track progress.
	GetTextEmbeddingsBatch(ctx context.Context, texts []string, callback ProgressCallback) ([][]float64, error)
}

// MultiModalEmbeddingModel extends EmbeddingModel with image embedding capabilities.
type MultiModalEmbeddingModel interface {
	EmbeddingModel
	// GetImageEmbedding generates an embedding for an image.
	GetImageEmbedding(ctx context.Context, image ImageType) ([]float64, error)
	// SupportsMultiModal returns true if the model supports multi-modal inputs.
	SupportsMultiModal() bool
}

// FullEmbeddingModel combines all embedding capabilities.
type FullEmbeddingModel interface {
	EmbeddingModelWithInfo
	EmbeddingModelWithBatch
	MultiModalEmbeddingModel
}
