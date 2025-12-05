package queryengine

import (
	"context"

	"github.com/aqua777/go-llamaindex/rag/synthesizer"
	"github.com/aqua777/go-llamaindex/schema"
)

// QueryTransform transforms a query before execution.
type QueryTransform interface {
	// Transform transforms the query.
	Transform(ctx context.Context, query schema.QueryBundle) (schema.QueryBundle, error)
}

// IdentityTransform returns the query unchanged.
type IdentityTransform struct{}

// Transform returns the query unchanged.
func (t *IdentityTransform) Transform(ctx context.Context, query schema.QueryBundle) (schema.QueryBundle, error) {
	return query, nil
}

// HyDETransform generates a hypothetical document to improve retrieval.
// HyDE = Hypothetical Document Embeddings
type HyDETransform struct {
	// LLM generates the hypothetical document.
	LLM interface {
		Complete(ctx context.Context, prompt string) (string, error)
	}
	// Prompt is the template for generating hypothetical documents.
	Prompt string
}

// Default HyDE prompt.
const defaultHyDEPrompt = `Please write a passage to answer the question.
Try to include as many key details as possible.

Question: {query_str}

Passage:`

// NewHyDETransform creates a new HyDETransform.
func NewHyDETransform(llm interface {
	Complete(ctx context.Context, prompt string) (string, error)
}) *HyDETransform {
	return &HyDETransform{
		LLM:    llm,
		Prompt: defaultHyDEPrompt,
	}
}

// Transform generates a hypothetical document and uses it as the query.
func (t *HyDETransform) Transform(ctx context.Context, query schema.QueryBundle) (schema.QueryBundle, error) {
	// Format prompt
	prompt := query.QueryString
	if t.Prompt != "" {
		prompt = replaceQueryStr(t.Prompt, query.QueryString)
	}

	// Generate hypothetical document
	hydeDoc, err := t.LLM.Complete(ctx, prompt)
	if err != nil {
		return query, err
	}

	// Return new query with hypothetical document as the query string
	// The hypothetical document improves embedding-based retrieval
	return schema.QueryBundle{
		QueryString: hydeDoc,
		Filters:     query.Filters,
	}, nil
}

// replaceQueryStr replaces {query_str} in the template.
func replaceQueryStr(template, query string) string {
	result := template
	for {
		idx := indexOf(result, "{query_str}")
		if idx == -1 {
			break
		}
		result = result[:idx] + query + result[idx+11:]
	}
	return result
}

func indexOf(s, substr string) int {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}

// TransformQueryEngine applies a transform before querying.
type TransformQueryEngine struct {
	*BaseQueryEngine
	// QueryEngine is the underlying query engine.
	QueryEngine QueryEngine
	// Transform transforms queries before execution.
	Transform QueryTransform
}

// TransformQueryEngineOption is a functional option.
type TransformQueryEngineOption func(*TransformQueryEngine)

// WithTransform sets the query transform.
func WithTransform(transform QueryTransform) TransformQueryEngineOption {
	return func(tqe *TransformQueryEngine) {
		tqe.Transform = transform
	}
}

// NewTransformQueryEngine creates a new TransformQueryEngine.
func NewTransformQueryEngine(
	engine QueryEngine,
	transform QueryTransform,
	opts ...TransformQueryEngineOption,
) *TransformQueryEngine {
	tqe := &TransformQueryEngine{
		BaseQueryEngine: NewBaseQueryEngine(),
		QueryEngine:     engine,
		Transform:       transform,
	}

	for _, opt := range opts {
		opt(tqe)
	}

	return tqe
}

// Query transforms the query and executes it.
func (tqe *TransformQueryEngine) Query(ctx context.Context, query string) (*synthesizer.Response, error) {
	queryBundle := schema.QueryBundle{QueryString: query}

	// Transform query
	transformedQuery, err := tqe.Transform.Transform(ctx, queryBundle)
	if err != nil {
		return nil, err
	}

	// Execute query
	return tqe.QueryEngine.Query(ctx, transformedQuery.QueryString)
}

// Retrieve retrieves nodes with transformed query (if underlying engine supports it).
func (tqe *TransformQueryEngine) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	// Transform query
	transformedQuery, err := tqe.Transform.Transform(ctx, query)
	if err != nil {
		return nil, err
	}

	// Check if underlying engine supports retrieval
	if retriever, ok := tqe.QueryEngine.(QueryEngineWithRetrieval); ok {
		return retriever.Retrieve(ctx, transformedQuery)
	}

	return nil, nil
}

// Synthesize synthesizes with transformed query (if underlying engine supports it).
func (tqe *TransformQueryEngine) Synthesize(ctx context.Context, query string, nodes []schema.NodeWithScore) (*synthesizer.Response, error) {
	queryBundle := schema.QueryBundle{QueryString: query}

	// Transform query
	transformedQuery, err := tqe.Transform.Transform(ctx, queryBundle)
	if err != nil {
		return nil, err
	}

	// Check if underlying engine supports synthesis
	if synth, ok := tqe.QueryEngine.(QueryEngineWithRetrieval); ok {
		return synth.Synthesize(ctx, transformedQuery.QueryString, nodes)
	}

	return nil, nil
}

// Ensure TransformQueryEngine implements QueryEngine.
var _ QueryEngine = (*TransformQueryEngine)(nil)
