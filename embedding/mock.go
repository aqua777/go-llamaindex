package embedding

import "context"

// MockEmbeddingModel is a mock implementation of the EmbeddingModel interface.
type MockEmbeddingModel struct {
	Embedding []float64
	Err       error
}

func (m *MockEmbeddingModel) GetTextEmbedding(ctx context.Context, text string) ([]float64, error) {
	return m.Embedding, m.Err
}

func (m *MockEmbeddingModel) GetQueryEmbedding(ctx context.Context, query string) ([]float64, error) {
	return m.Embedding, m.Err
}

