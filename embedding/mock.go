package embedding

import "context"

// MockEmbeddingModel is a mock implementation of the EmbeddingModel interface.
// It can be configured to return specific embeddings or errors.
type MockEmbeddingModel struct {
	// Embedding is the embedding to return for single text requests.
	Embedding []float64
	// Embeddings is the embeddings to return for batch requests.
	Embeddings [][]float64
	// Err is the error to return (if any).
	Err error
	// ModelInfo is the embedding info to return.
	ModelInfo *EmbeddingInfo
	// MultiModalSupported indicates if multi-modal is supported.
	MultiModalSupported bool
}

// NewMockEmbeddingModel creates a new MockEmbeddingModel with a fixed embedding.
func NewMockEmbeddingModel(embedding []float64) *MockEmbeddingModel {
	return &MockEmbeddingModel{Embedding: embedding}
}

// NewMockEmbeddingModelWithError creates a new MockEmbeddingModel that returns an error.
func NewMockEmbeddingModelWithError(err error) *MockEmbeddingModel {
	return &MockEmbeddingModel{Err: err}
}

func (m *MockEmbeddingModel) GetTextEmbedding(ctx context.Context, text string) ([]float64, error) {
	return m.Embedding, m.Err
}

func (m *MockEmbeddingModel) GetQueryEmbedding(ctx context.Context, query string) ([]float64, error) {
	return m.Embedding, m.Err
}

// Info returns the mock model info.
func (m *MockEmbeddingModel) Info() EmbeddingInfo {
	if m.ModelInfo != nil {
		return *m.ModelInfo
	}
	return DefaultEmbeddingInfo("mock-embedding-model")
}

// GetTextEmbeddingsBatch returns mock embeddings for batch requests.
func (m *MockEmbeddingModel) GetTextEmbeddingsBatch(ctx context.Context, texts []string, callback ProgressCallback) ([][]float64, error) {
	if m.Err != nil {
		return nil, m.Err
	}

	// If Embeddings is set, use it
	if len(m.Embeddings) > 0 {
		if callback != nil {
			callback(len(texts), len(texts))
		}
		return m.Embeddings, nil
	}

	// Otherwise, return the same embedding for each text
	results := make([][]float64, len(texts))
	for i := range texts {
		results[i] = m.Embedding
		if callback != nil {
			callback(i+1, len(texts))
		}
	}
	return results, nil
}

// SupportsMultiModal returns whether multi-modal is supported.
func (m *MockEmbeddingModel) SupportsMultiModal() bool {
	return m.MultiModalSupported
}

// GetImageEmbedding returns a mock image embedding.
func (m *MockEmbeddingModel) GetImageEmbedding(ctx context.Context, image ImageType) ([]float64, error) {
	if !m.MultiModalSupported {
		return nil, ErrMultiModalNotSupported
	}
	return m.Embedding, m.Err
}

// ErrMultiModalNotSupported is returned when multi-modal embedding is not supported.
var ErrMultiModalNotSupported = &EmbeddingError{Message: "multi-modal embedding not supported"}

// EmbeddingError represents an embedding-specific error.
type EmbeddingError struct {
	Message string
}

func (e *EmbeddingError) Error() string {
	return e.Message
}
