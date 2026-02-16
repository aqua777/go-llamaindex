package embedding

import "context"


func float64ToFloat32(embedding []float64) []float32 {
	result := make([]float32, len(embedding))
	for i, v := range embedding {
		result[i] = float32(v)
	}
	return result
}

// GetTextEmbedding generates an embedding for a given text.
func GetTextEmbeddingF32(ctx context.Context, embedder EmbeddingModel, text string) ([]float32, error) {
	embedding, err := embedder.GetTextEmbedding(ctx, text)
	if err != nil {
		return nil, err
	}
	return float64ToFloat32(embedding), nil
}
	
// GetQueryEmbedding generates an embedding for a given query.
// This is often the same as GetTextEmbedding, but some models treat them differently.
func GetQueryEmbeddingF32(ctx context.Context, embedder EmbeddingModel, query string) ([]float32, error) {
	embedding, err := embedder.GetQueryEmbedding(ctx, query)
	if err != nil {
		return nil, err
	}
	return float64ToFloat32(embedding), nil
}
