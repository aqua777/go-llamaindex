package rag

import (
	"context"

	"github.com/aqua777/go-llamaindex/schema"
)

// Retriever is an interface for retrieving relevant nodes based on a query.
type Retriever interface {
	Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error)
}

// Synthesizer is an interface for generating a response from query + nodes.
type Synthesizer interface {
	Synthesize(ctx context.Context, query schema.QueryBundle, nodes []schema.NodeWithScore) (schema.EngineResponse, error)
	SynthesizeStream(ctx context.Context, query schema.QueryBundle, nodes []schema.NodeWithScore) (schema.StreamingEngineResponse, error)
}

// QueryEngine is the interface for the end-to-end query flow.
type QueryEngine interface {
	Query(ctx context.Context, query schema.QueryBundle) (schema.EngineResponse, error)
	QueryStream(ctx context.Context, query schema.QueryBundle) (schema.StreamingEngineResponse, error)
}
