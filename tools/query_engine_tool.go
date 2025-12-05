package tools

import (
	"context"
	"fmt"

	"github.com/aqua777/go-llamaindex/rag/queryengine"
)

const (
	// DefaultQueryEngineToolName is the default name for query engine tools.
	DefaultQueryEngineToolName = "query_engine_tool"
	// DefaultQueryEngineToolDescription is the default description for query engine tools.
	DefaultQueryEngineToolDescription = `Useful for running a natural language query against a knowledge base and get back a natural language response.`
)

// QueryEngineTool wraps a query engine as a tool.
type QueryEngineTool struct {
	*BaseTool
	queryEngine        queryengine.QueryEngine
	resolveInputErrors bool
}

// QueryEngineToolOption configures a QueryEngineTool.
type QueryEngineToolOption func(*QueryEngineTool)

// WithQueryEngineToolName sets the tool name.
func WithQueryEngineToolName(name string) QueryEngineToolOption {
	return func(qet *QueryEngineTool) {
		qet.metadata.Name = name
	}
}

// WithQueryEngineToolDescription sets the tool description.
func WithQueryEngineToolDescription(description string) QueryEngineToolOption {
	return func(qet *QueryEngineTool) {
		qet.metadata.Description = description
	}
}

// WithQueryEngineToolReturnDirect sets whether to return the output directly.
func WithQueryEngineToolReturnDirect(returnDirect bool) QueryEngineToolOption {
	return func(qet *QueryEngineTool) {
		qet.metadata.ReturnDirect = returnDirect
	}
}

// WithResolveInputErrors sets whether to resolve input errors.
func WithResolveInputErrors(resolve bool) QueryEngineToolOption {
	return func(qet *QueryEngineTool) {
		qet.resolveInputErrors = resolve
	}
}

// NewQueryEngineTool creates a new QueryEngineTool.
func NewQueryEngineTool(
	queryEngine queryengine.QueryEngine,
	opts ...QueryEngineToolOption,
) *QueryEngineTool {
	qet := &QueryEngineTool{
		BaseTool: NewBaseTool(&ToolMetadata{
			Name:        DefaultQueryEngineToolName,
			Description: DefaultQueryEngineToolDescription,
			Parameters:  DefaultParameters(),
		}),
		queryEngine:        queryEngine,
		resolveInputErrors: true,
	}

	for _, opt := range opts {
		opt(qet)
	}

	return qet
}

// NewQueryEngineToolFromDefaults creates a QueryEngineTool with explicit configuration.
func NewQueryEngineToolFromDefaults(
	queryEngine queryengine.QueryEngine,
	name string,
	description string,
	opts ...QueryEngineToolOption,
) *QueryEngineTool {
	if name == "" {
		name = DefaultQueryEngineToolName
	}
	if description == "" {
		description = DefaultQueryEngineToolDescription
	}

	qet := &QueryEngineTool{
		BaseTool: NewBaseTool(&ToolMetadata{
			Name:        name,
			Description: description,
			Parameters:  DefaultParameters(),
		}),
		queryEngine:        queryEngine,
		resolveInputErrors: true,
	}

	for _, opt := range opts {
		opt(qet)
	}

	return qet
}

// QueryEngine returns the underlying query engine.
func (qet *QueryEngineTool) QueryEngine() queryengine.QueryEngine {
	return qet.queryEngine
}

// Call executes the query engine with the given input.
func (qet *QueryEngineTool) Call(ctx context.Context, input interface{}) (*ToolOutput, error) {
	queryStr, err := qet.getQueryString(input)
	if err != nil {
		return NewErrorToolOutput(qet.metadata.Name, err), err
	}

	response, err := qet.queryEngine.Query(ctx, queryStr)
	if err != nil {
		return NewErrorToolOutput(qet.metadata.Name, err), err
	}

	content := response.Response
	rawInput := map[string]interface{}{"input": queryStr}

	return NewToolOutputWithInput(qet.metadata.Name, content, rawInput, response), nil
}

// getQueryString extracts the query string from the input.
func (qet *QueryEngineTool) getQueryString(input interface{}) (string, error) {
	switch v := input.(type) {
	case string:
		return v, nil
	case map[string]interface{}:
		if queryStr, ok := v["input"].(string); ok {
			return queryStr, nil
		}
		if queryStr, ok := v["query"].(string); ok {
			return queryStr, nil
		}
		if qet.resolveInputErrors {
			// Convert the entire map to a string
			return fmt.Sprintf("%v", v), nil
		}
		return "", fmt.Errorf("cannot find 'input' or 'query' in input map")
	default:
		if qet.resolveInputErrors {
			return fmt.Sprintf("%v", input), nil
		}
		return "", fmt.Errorf("unsupported input type: %T", input)
	}
}

// Ensure QueryEngineTool implements Tool.
var _ Tool = (*QueryEngineTool)(nil)
