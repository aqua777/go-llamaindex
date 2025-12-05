package queryengine

import (
	"context"
	"errors"
	"strings"

	"github.com/aqua777/go-llamaindex/rag/synthesizer"
	"github.com/aqua777/go-llamaindex/schema"
)

// QueryEngineSelector selects which query engine(s) to use.
type QueryEngineSelector interface {
	// Select chooses query engines based on their metadata and the query.
	Select(ctx context.Context, tools []*QueryEngineTool, query schema.QueryBundle) (*SelectorResult, error)
}

// SelectorResult represents the result of a selector's decision.
type SelectorResult struct {
	// Indices are the selected query engine indices.
	Indices []int
	// Reasons are the reasons for each selection.
	Reasons []string
}

// SingleSelector always selects the first query engine.
type SingleSelector struct{}

// Select returns only the first query engine.
func (s *SingleSelector) Select(ctx context.Context, tools []*QueryEngineTool, query schema.QueryBundle) (*SelectorResult, error) {
	if len(tools) == 0 {
		return nil, errors.New("no query engines available")
	}
	return &SelectorResult{
		Indices: []int{0},
		Reasons: []string{"selected first query engine"},
	}, nil
}

// MultiSelector selects all query engines.
type MultiSelector struct{}

// Select returns all query engines.
func (s *MultiSelector) Select(ctx context.Context, tools []*QueryEngineTool, query schema.QueryBundle) (*SelectorResult, error) {
	if len(tools) == 0 {
		return nil, errors.New("no query engines available")
	}
	indices := make([]int, len(tools))
	reasons := make([]string, len(tools))
	for i := range tools {
		indices[i] = i
		reasons[i] = "selected by default"
	}
	return &SelectorResult{Indices: indices, Reasons: reasons}, nil
}

// RouterQueryEngine routes queries to appropriate query engines.
type RouterQueryEngine struct {
	*BaseQueryEngine
	// Selector chooses which query engine(s) to use.
	Selector QueryEngineSelector
	// Tools are the available query engine tools.
	Tools []*QueryEngineTool
	// Summarizer combines responses from multiple engines.
	Summarizer synthesizer.Synthesizer
}

// RouterQueryEngineOption is a functional option.
type RouterQueryEngineOption func(*RouterQueryEngine)

// WithRouterSelector sets the selector.
func WithRouterSelector(selector QueryEngineSelector) RouterQueryEngineOption {
	return func(rqe *RouterQueryEngine) {
		rqe.Selector = selector
	}
}

// WithRouterSummarizer sets the summarizer for combining responses.
func WithRouterSummarizer(synth synthesizer.Synthesizer) RouterQueryEngineOption {
	return func(rqe *RouterQueryEngine) {
		rqe.Summarizer = synth
	}
}

// NewRouterQueryEngine creates a new RouterQueryEngine.
func NewRouterQueryEngine(tools []*QueryEngineTool, opts ...RouterQueryEngineOption) *RouterQueryEngine {
	rqe := &RouterQueryEngine{
		BaseQueryEngine: NewBaseQueryEngine(),
		Selector:        &SingleSelector{},
		Tools:           tools,
		Summarizer:      nil, // Will use simple concatenation if nil
	}

	for _, opt := range opts {
		opt(rqe)
	}

	return rqe
}

// Query routes the query to selected query engines.
func (rqe *RouterQueryEngine) Query(ctx context.Context, query string) (*synthesizer.Response, error) {
	if len(rqe.Tools) == 0 {
		return nil, errors.New("no query engines configured")
	}

	queryBundle := schema.QueryBundle{QueryString: query}

	// Select query engines
	result, err := rqe.Selector.Select(ctx, rqe.Tools, queryBundle)
	if err != nil {
		return nil, err
	}

	if len(result.Indices) == 0 {
		return nil, errors.New("no query engines selected")
	}

	// Query selected engines
	var responses []*synthesizer.Response
	var allSourceNodes []schema.NodeWithScore

	for _, idx := range result.Indices {
		if idx < 0 || idx >= len(rqe.Tools) {
			continue
		}

		tool := rqe.Tools[idx]
		resp, err := tool.QueryEngine.Query(ctx, query)
		if err != nil {
			return nil, err
		}

		responses = append(responses, resp)
		allSourceNodes = append(allSourceNodes, resp.SourceNodes...)
	}

	// Combine responses
	if len(responses) == 1 {
		return responses[0], nil
	}

	// Multiple responses - combine them
	return rqe.combineResponses(ctx, query, responses, allSourceNodes)
}

// combineResponses combines multiple responses into one.
func (rqe *RouterQueryEngine) combineResponses(
	ctx context.Context,
	query string,
	responses []*synthesizer.Response,
	sourceNodes []schema.NodeWithScore,
) (*synthesizer.Response, error) {
	// If we have a summarizer, use it
	if rqe.Summarizer != nil {
		// Create nodes from responses
		var nodes []schema.NodeWithScore
		for _, resp := range responses {
			node := schema.NewTextNode(resp.Response)
			nodes = append(nodes, schema.NodeWithScore{Node: *node, Score: 1.0})
		}

		response, err := rqe.Summarizer.Synthesize(ctx, query, nodes)
		if err != nil {
			return nil, err
		}
		response.SourceNodes = sourceNodes
		return response, nil
	}

	// Simple concatenation
	var responseTexts []string
	for _, resp := range responses {
		responseTexts = append(responseTexts, resp.Response)
	}

	return &synthesizer.Response{
		Response:    strings.Join(responseTexts, "\n\n"),
		SourceNodes: sourceNodes,
		Metadata:    make(map[string]interface{}),
	}, nil
}

// Ensure RouterQueryEngine implements QueryEngine.
var _ QueryEngine = (*RouterQueryEngine)(nil)
