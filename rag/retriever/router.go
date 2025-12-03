package retriever

import (
	"context"
	"errors"

	"github.com/aqua777/go-llamaindex/schema"
)

// RetrieverTool wraps a retriever with metadata for routing.
type RetrieverTool struct {
	// Retriever is the underlying retriever.
	Retriever Retriever
	// Name is the name of the retriever.
	Name string
	// Description describes what this retriever is best suited for.
	Description string
}

// NewRetrieverTool creates a new RetrieverTool.
func NewRetrieverTool(retriever Retriever, name, description string) *RetrieverTool {
	return &RetrieverTool{
		Retriever:   retriever,
		Name:        name,
		Description: description,
	}
}

// SelectorResult represents the result of a selector's decision.
type SelectorResult struct {
	// Indices are the selected retriever indices.
	Indices []int
	// Reasons are the reasons for each selection.
	Reasons []string
}

// Selector selects which retriever(s) to use based on the query.
type Selector interface {
	// Select chooses retrievers based on their metadata and the query.
	Select(ctx context.Context, tools []*RetrieverTool, query schema.QueryBundle) (*SelectorResult, error)
}

// SimpleSelector is a basic selector that always selects all retrievers.
type SimpleSelector struct{}

// Select returns all retrievers.
func (s *SimpleSelector) Select(ctx context.Context, tools []*RetrieverTool, query schema.QueryBundle) (*SelectorResult, error) {
	indices := make([]int, len(tools))
	reasons := make([]string, len(tools))
	for i := range tools {
		indices[i] = i
		reasons[i] = "selected by default"
	}
	return &SelectorResult{Indices: indices, Reasons: reasons}, nil
}

// SingleSelector selects only the first retriever.
type SingleSelector struct{}

// Select returns only the first retriever.
func (s *SingleSelector) Select(ctx context.Context, tools []*RetrieverTool, query schema.QueryBundle) (*SelectorResult, error) {
	if len(tools) == 0 {
		return nil, errors.New("no retrievers available")
	}
	return &SelectorResult{
		Indices: []int{0},
		Reasons: []string{"selected first retriever"},
	}, nil
}

// RouterRetriever routes queries to appropriate retrievers based on a selector.
type RouterRetriever struct {
	*BaseRetriever
	// Selector chooses which retriever(s) to use.
	Selector Selector
	// Tools are the available retriever tools.
	Tools []*RetrieverTool
}

// RouterRetrieverOption is a functional option for RouterRetriever.
type RouterRetrieverOption func(*RouterRetriever)

// WithSelector sets the selector for routing.
func WithSelector(selector Selector) RouterRetrieverOption {
	return func(rr *RouterRetriever) {
		rr.Selector = selector
	}
}

// NewRouterRetriever creates a new RouterRetriever.
func NewRouterRetriever(tools []*RetrieverTool, opts ...RouterRetrieverOption) *RouterRetriever {
	rr := &RouterRetriever{
		BaseRetriever: NewBaseRetriever(),
		Selector:      &SimpleSelector{}, // Default to selecting all
		Tools:         tools,
	}

	for _, opt := range opts {
		opt(rr)
	}

	return rr
}

// Retrieve routes the query to selected retrievers and combines results.
func (rr *RouterRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	if len(rr.Tools) == 0 {
		return nil, errors.New("no retrievers configured")
	}

	// Select retrievers
	result, err := rr.Selector.Select(ctx, rr.Tools, query)
	if err != nil {
		return nil, err
	}

	if len(result.Indices) == 0 {
		return nil, errors.New("no retrievers selected")
	}

	// Retrieve from selected retrievers
	allResults := make(map[string]schema.NodeWithScore)

	for _, idx := range result.Indices {
		if idx < 0 || idx >= len(rr.Tools) {
			continue
		}

		tool := rr.Tools[idx]
		nodes, err := tool.Retriever.Retrieve(ctx, query)
		if err != nil {
			return nil, err
		}

		// Deduplicate by node ID
		for _, node := range nodes {
			allResults[node.Node.ID] = node
		}
	}

	// Convert to slice
	var nodes []schema.NodeWithScore
	for _, node := range allResults {
		nodes = append(nodes, node)
	}

	return nodes, nil
}

// Ensure RouterRetriever implements Retriever.
var _ Retriever = (*RouterRetriever)(nil)
