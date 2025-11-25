package rag

import (
	"context"
	"fmt"

	"github.com/aqua777/go-llamaindex/schema"
)

// RetrieverQueryEngine ties together a retriever and a synthesizer.
type RetrieverQueryEngine struct {
	retriever   Retriever
	synthesizer Synthesizer
}

// NewRetrieverQueryEngine creates a new RetrieverQueryEngine.
func NewRetrieverQueryEngine(retriever Retriever, synthesizer Synthesizer) *RetrieverQueryEngine {
	return &RetrieverQueryEngine{
		retriever:   retriever,
		synthesizer: synthesizer,
	}
}

// Retrieve returns the relevant nodes for a given query.
func (e *RetrieverQueryEngine) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	return e.retriever.Retrieve(ctx, query)
}

// Synthesize generates a response from the query and a set of nodes.
func (e *RetrieverQueryEngine) Synthesize(ctx context.Context, query schema.QueryBundle, nodes []schema.NodeWithScore) (schema.EngineResponse, error) {
	return e.synthesizer.Synthesize(ctx, query, nodes)
}

// Query performs the end-to-end RAG flow: retrieve -> synthesize.
func (e *RetrieverQueryEngine) Query(ctx context.Context, query schema.QueryBundle) (schema.EngineResponse, error) {
	nodes, err := e.Retrieve(ctx, query)
	if err != nil {
		return schema.EngineResponse{}, fmt.Errorf("retrieve failed: %w", err)
	}

	response, err := e.Synthesize(ctx, query, nodes)
	if err != nil {
		return schema.EngineResponse{}, fmt.Errorf("synthesize failed: %w", err)
	}

	return response, nil
}

// QueryStream performs the end-to-end RAG flow with streaming: retrieve -> synthesize stream.
func (e *RetrieverQueryEngine) QueryStream(ctx context.Context, query schema.QueryBundle) (schema.StreamingEngineResponse, error) {
	nodes, err := e.Retrieve(ctx, query)
	if err != nil {
		return schema.StreamingEngineResponse{}, fmt.Errorf("retrieve failed: %w", err)
	}

	response, err := e.synthesizer.SynthesizeStream(ctx, query, nodes)
	if err != nil {
		return schema.StreamingEngineResponse{}, fmt.Errorf("synthesize stream failed: %w", err)
	}

	return response, nil
}
