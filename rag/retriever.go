package rag

import (
	"context"
	"fmt"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/rag/store"
	"github.com/aqua777/go-llamaindex/schema"
)

// VectorRetriever retrieves relevant nodes using a vector store and embedding model.
type VectorRetriever struct {
	vectorStore    store.VectorStore
	embeddingModel embedding.EmbeddingModel
	topK           int
}

// NewVectorRetriever creates a new VectorRetriever.
func NewVectorRetriever(vectorStore store.VectorStore, embeddingModel embedding.EmbeddingModel, topK int) *VectorRetriever {
	return &VectorRetriever{
		vectorStore:    vectorStore,
		embeddingModel: embeddingModel,
		topK:           topK,
	}
}

func (r *VectorRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	queryEmbedding, err := r.embeddingModel.GetQueryEmbedding(ctx, query.QueryString)
	if err != nil {
		return nil, fmt.Errorf("failed to get query embedding: %w", err)
	}

	storeQuery := schema.VectorStoreQuery{
		Embedding: queryEmbedding,
		TopK:      r.topK,
		Filters:   query.Filters,
	}

	nodes, err := r.vectorStore.Query(ctx, storeQuery)
	if err != nil {
		return nil, fmt.Errorf("failed to query vector store: %w", err)
	}

	return nodes, nil
}
