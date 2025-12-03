// Package queryengine provides query engine implementations for RAG systems.
package queryengine

import (
	"context"

	"github.com/aqua777/go-llamaindex/prompts"
	"github.com/aqua777/go-llamaindex/rag/retriever"
	"github.com/aqua777/go-llamaindex/rag/synthesizer"
	"github.com/aqua777/go-llamaindex/schema"
)

// QueryEngine is the interface for query engines.
type QueryEngine interface {
	// Query executes a query and returns a response.
	Query(ctx context.Context, query string) (*synthesizer.Response, error)
}

// QueryEngineWithRetrieval extends QueryEngine with separate retrieve/synthesize.
type QueryEngineWithRetrieval interface {
	QueryEngine
	// Retrieve retrieves nodes for a query.
	Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error)
	// Synthesize synthesizes a response from nodes.
	Synthesize(ctx context.Context, query string, nodes []schema.NodeWithScore) (*synthesizer.Response, error)
}

// BaseQueryEngine provides common functionality for query engines.
type BaseQueryEngine struct {
	// Verbose enables verbose logging.
	Verbose bool
	// PromptMixin for prompt management.
	*prompts.BasePromptMixin
}

// NewBaseQueryEngine creates a new BaseQueryEngine.
func NewBaseQueryEngine() *BaseQueryEngine {
	return &BaseQueryEngine{
		Verbose:         false,
		BasePromptMixin: prompts.NewBasePromptMixin(),
	}
}

// BaseQueryEngineOption is a functional option for BaseQueryEngine.
type BaseQueryEngineOption func(*BaseQueryEngine)

// WithQueryEngineVerbose enables verbose logging.
func WithQueryEngineVerbose(verbose bool) BaseQueryEngineOption {
	return func(bqe *BaseQueryEngine) {
		bqe.Verbose = verbose
	}
}

// NewBaseQueryEngineWithOptions creates a new BaseQueryEngine with options.
func NewBaseQueryEngineWithOptions(opts ...BaseQueryEngineOption) *BaseQueryEngine {
	bqe := NewBaseQueryEngine()
	for _, opt := range opts {
		opt(bqe)
	}
	return bqe
}

// RetrieverQueryEngine combines a retriever and synthesizer.
type RetrieverQueryEngine struct {
	*BaseQueryEngine
	// Retriever retrieves relevant nodes.
	Retriever retriever.Retriever
	// Synthesizer generates responses from nodes.
	Synthesizer synthesizer.Synthesizer
}

// RetrieverQueryEngineOption is a functional option.
type RetrieverQueryEngineOption func(*RetrieverQueryEngine)

// WithRetrieverQueryEngineVerbose enables verbose logging.
func WithRetrieverQueryEngineVerbose(verbose bool) RetrieverQueryEngineOption {
	return func(rqe *RetrieverQueryEngine) {
		rqe.Verbose = verbose
	}
}

// NewRetrieverQueryEngine creates a new RetrieverQueryEngine.
func NewRetrieverQueryEngine(
	ret retriever.Retriever,
	synth synthesizer.Synthesizer,
	opts ...RetrieverQueryEngineOption,
) *RetrieverQueryEngine {
	rqe := &RetrieverQueryEngine{
		BaseQueryEngine: NewBaseQueryEngine(),
		Retriever:       ret,
		Synthesizer:     synth,
	}

	for _, opt := range opts {
		opt(rqe)
	}

	return rqe
}

// Query executes a query and returns a response.
func (rqe *RetrieverQueryEngine) Query(ctx context.Context, query string) (*synthesizer.Response, error) {
	queryBundle := schema.QueryBundle{QueryString: query}

	// Retrieve nodes
	nodes, err := rqe.Retrieve(ctx, queryBundle)
	if err != nil {
		return nil, err
	}

	// Synthesize response
	return rqe.Synthesize(ctx, query, nodes)
}

// Retrieve retrieves nodes for a query.
func (rqe *RetrieverQueryEngine) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	return rqe.Retriever.Retrieve(ctx, query)
}

// Synthesize synthesizes a response from nodes.
func (rqe *RetrieverQueryEngine) Synthesize(ctx context.Context, query string, nodes []schema.NodeWithScore) (*synthesizer.Response, error) {
	return rqe.Synthesizer.Synthesize(ctx, query, nodes)
}

// Ensure RetrieverQueryEngine implements interfaces.
var _ QueryEngine = (*RetrieverQueryEngine)(nil)
var _ QueryEngineWithRetrieval = (*RetrieverQueryEngine)(nil)
