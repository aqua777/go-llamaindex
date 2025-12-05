package tools

import (
	"context"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/rag/retriever"
	"github.com/aqua777/go-llamaindex/schema"
)

const (
	// DefaultRetrieverToolName is the default name for retriever tools.
	DefaultRetrieverToolName = "retriever_tool"
	// DefaultRetrieverToolDescription is the default description for retriever tools.
	DefaultRetrieverToolDescription = `Useful for running a natural language query against a knowledge base and retrieving a set of relevant documents.`
)

// NodePostprocessor processes retrieved nodes.
type NodePostprocessor interface {
	// PostprocessNodes processes the retrieved nodes.
	PostprocessNodes(ctx context.Context, nodes []schema.NodeWithScore, query schema.QueryBundle) ([]schema.NodeWithScore, error)
}

// RetrieverTool wraps a retriever as a tool.
type RetrieverTool struct {
	*BaseTool
	retriever          retriever.Retriever
	nodePostprocessors []NodePostprocessor
}

// RetrieverToolOption configures a RetrieverTool.
type RetrieverToolOption func(*RetrieverTool)

// WithRetrieverToolName sets the tool name.
func WithRetrieverToolName(name string) RetrieverToolOption {
	return func(rt *RetrieverTool) {
		rt.metadata.Name = name
	}
}

// WithRetrieverToolDescription sets the tool description.
func WithRetrieverToolDescription(description string) RetrieverToolOption {
	return func(rt *RetrieverTool) {
		rt.metadata.Description = description
	}
}

// WithNodePostprocessors sets the node postprocessors.
func WithNodePostprocessors(postprocessors ...NodePostprocessor) RetrieverToolOption {
	return func(rt *RetrieverTool) {
		rt.nodePostprocessors = postprocessors
	}
}

// NewRetrieverTool creates a new RetrieverTool.
func NewRetrieverTool(
	ret retriever.Retriever,
	opts ...RetrieverToolOption,
) *RetrieverTool {
	rt := &RetrieverTool{
		BaseTool: NewBaseTool(&ToolMetadata{
			Name:        DefaultRetrieverToolName,
			Description: DefaultRetrieverToolDescription,
			Parameters:  DefaultParameters(),
		}),
		retriever:          ret,
		nodePostprocessors: []NodePostprocessor{},
	}

	for _, opt := range opts {
		opt(rt)
	}

	return rt
}

// NewRetrieverToolFromDefaults creates a RetrieverTool with explicit configuration.
func NewRetrieverToolFromDefaults(
	ret retriever.Retriever,
	name string,
	description string,
	opts ...RetrieverToolOption,
) *RetrieverTool {
	if name == "" {
		name = DefaultRetrieverToolName
	}
	if description == "" {
		description = DefaultRetrieverToolDescription
	}

	rt := &RetrieverTool{
		BaseTool: NewBaseTool(&ToolMetadata{
			Name:        name,
			Description: description,
			Parameters:  DefaultParameters(),
		}),
		retriever:          ret,
		nodePostprocessors: []NodePostprocessor{},
	}

	for _, opt := range opts {
		opt(rt)
	}

	return rt
}

// Retriever returns the underlying retriever.
func (rt *RetrieverTool) Retriever() retriever.Retriever {
	return rt.retriever
}

// Call executes the retriever with the given input.
func (rt *RetrieverTool) Call(ctx context.Context, input interface{}) (*ToolOutput, error) {
	queryStr, err := rt.getQueryString(input)
	if err != nil {
		return NewErrorToolOutput(rt.metadata.Name, err), err
	}

	queryBundle := schema.QueryBundle{QueryString: queryStr}

	// Retrieve documents
	docs, err := rt.retriever.Retrieve(ctx, queryBundle)
	if err != nil {
		return NewErrorToolOutput(rt.metadata.Name, err), err
	}

	// Apply postprocessors
	docs, err = rt.applyPostprocessors(ctx, docs, queryBundle)
	if err != nil {
		return NewErrorToolOutput(rt.metadata.Name, err), err
	}

	// Build content from documents
	var contentParts []string
	for _, doc := range docs {
		content := doc.Node.GetContent(schema.MetadataModeLLM)
		contentParts = append(contentParts, content)
	}
	content := strings.Join(contentParts, "\n\n")

	rawInput := map[string]interface{}{"input": queryStr}

	return NewToolOutputWithInput(rt.metadata.Name, content, rawInput, docs), nil
}

// getQueryString extracts the query string from the input.
func (rt *RetrieverTool) getQueryString(input interface{}) (string, error) {
	switch v := input.(type) {
	case string:
		return v, nil
	case map[string]interface{}:
		// Try various keys
		for _, key := range []string{"input", "query", "question"} {
			if queryStr, ok := v[key].(string); ok {
				return queryStr, nil
			}
		}
		// Build query from all key-value pairs
		var parts []string
		for k, val := range v {
			parts = append(parts, fmt.Sprintf("%s is %v", k, val))
		}
		if len(parts) > 0 {
			return strings.Join(parts, ", "), nil
		}
		return "", fmt.Errorf("cannot extract query string from input")
	default:
		return fmt.Sprintf("%v", input), nil
	}
}

// applyPostprocessors applies all postprocessors to the nodes.
func (rt *RetrieverTool) applyPostprocessors(
	ctx context.Context,
	nodes []schema.NodeWithScore,
	query schema.QueryBundle,
) ([]schema.NodeWithScore, error) {
	for _, pp := range rt.nodePostprocessors {
		var err error
		nodes, err = pp.PostprocessNodes(ctx, nodes, query)
		if err != nil {
			return nil, err
		}
	}
	return nodes, nil
}

// Ensure RetrieverTool implements Tool.
var _ Tool = (*RetrieverTool)(nil)
