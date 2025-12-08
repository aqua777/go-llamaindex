package retriever

import (
	"context"
	"fmt"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/rag/store"
	"github.com/aqua777/go-llamaindex/schema"
)

// VectorRetriever retrieves relevant nodes using a vector store and embedding model.
type VectorRetriever struct {
	*BaseRetriever
	// VectorStore is the vector store to query.
	VectorStore store.VectorStore
	// EmbeddingModel is the model used to embed queries.
	EmbeddingModel embedding.EmbeddingModel
	// TopK is the number of results to return.
	TopK int
	// Mode is the query mode for the vector store.
	Mode schema.VectorStoreQueryMode
}

// VectorRetrieverOption is a functional option for VectorRetriever.
type VectorRetrieverOption func(*VectorRetriever)

// WithTopK sets the number of results to return.
func WithTopK(topK int) VectorRetrieverOption {
	return func(vr *VectorRetriever) {
		vr.TopK = topK
	}
}

// WithQueryMode sets the query mode.
func WithQueryMode(mode schema.VectorStoreQueryMode) VectorRetrieverOption {
	return func(vr *VectorRetriever) {
		vr.Mode = mode
	}
}

// NewVectorRetriever creates a new VectorRetriever.
func NewVectorRetriever(
	vectorStore store.VectorStore,
	embeddingModel embedding.EmbeddingModel,
	opts ...VectorRetrieverOption,
) *VectorRetriever {
	vr := &VectorRetriever{
		BaseRetriever:  NewBaseRetriever(),
		VectorStore:    vectorStore,
		EmbeddingModel: embeddingModel,
		TopK:           10,
		Mode:           schema.QueryModeDefault,
	}

	for _, opt := range opts {
		opt(vr)
	}

	return vr
}

// Retrieve retrieves nodes from the vector store.
func (vr *VectorRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	// Get query embedding
	queryEmbedding, err := vr.EmbeddingModel.GetQueryEmbedding(ctx, query.QueryString)
	if err != nil {
		return nil, fmt.Errorf("failed to get query embedding: %w", err)
	}

	// Build vector store query
	storeQuery := schema.VectorStoreQuery{
		Embedding: queryEmbedding,
		TopK:      vr.TopK,
		Filters:   query.Filters,
		Mode:      vr.Mode,
	}

	// Query vector store
	nodes, err := vr.VectorStore.Query(ctx, storeQuery)
	if err != nil {
		return nil, fmt.Errorf("failed to query vector store: %w", err)
	}

	// Handle recursive retrieval if needed
	return vr.HandleRecursiveRetrieval(ctx, query, nodes)
}

// Ensure VectorRetriever implements Retriever.
var _ Retriever = (*VectorRetriever)(nil)
