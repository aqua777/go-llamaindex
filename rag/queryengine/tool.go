package queryengine

// QueryEngineTool wraps a query engine with metadata for routing.
type QueryEngineTool struct {
	// QueryEngine is the underlying query engine.
	QueryEngine QueryEngine
	// Name is the name of the query engine.
	Name string
	// Description describes what this query engine is best suited for.
	Description string
}

// NewQueryEngineTool creates a new QueryEngineTool.
func NewQueryEngineTool(engine QueryEngine, name, description string) *QueryEngineTool {
	return &QueryEngineTool{
		QueryEngine: engine,
		Name:        name,
		Description: description,
	}
}

// ToolMetadata returns the tool metadata.
func (qet *QueryEngineTool) ToolMetadata() ToolMetadata {
	return ToolMetadata{
		Name:        qet.Name,
		Description: qet.Description,
	}
}

// ToolMetadata contains metadata about a tool.
type ToolMetadata struct {
	Name        string
	Description string
}
