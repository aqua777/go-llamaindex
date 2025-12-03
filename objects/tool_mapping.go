package objects

import (
	"context"
	"fmt"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/tools"
)

// ToolNodeMapping maps tools to nodes for storage and retrieval.
type ToolNodeMapping struct {
	*BaseObjectNodeMapping
	// tools stores tools by name for quick lookup.
	tools map[string]tools.Tool
}

// NewToolNodeMapping creates a new ToolNodeMapping.
func NewToolNodeMapping() *ToolNodeMapping {
	m := &ToolNodeMapping{
		BaseObjectNodeMapping: NewBaseObjectNodeMapping(),
		tools:                 make(map[string]tools.Tool),
	}

	// Set custom ToNode function for tools
	m.ToNodeFunc = toolToNode
	m.FromNodeFunc = m.toolFromNode

	return m
}

// toolToNode converts a tool to a node.
func toolToNode(obj interface{}) (*schema.Node, error) {
	tool, ok := obj.(tools.Tool)
	if !ok {
		return nil, fmt.Errorf("expected tools.Tool, got %T", obj)
	}

	metadata := tool.Metadata()
	if metadata == nil {
		return nil, fmt.Errorf("tool has no metadata")
	}

	// Create node with tool description as text
	text := metadata.Description
	if text == "" {
		text = metadata.Name
	}

	node := schema.NewTextNode(text)
	node.ID = metadata.Name // Use tool name as ID
	node.Metadata = map[string]interface{}{
		"name":          metadata.Name,
		"description":   metadata.Description,
		"object_type":   "tool",
		"return_direct": metadata.ReturnDirect,
	}

	// Add parameters schema if available
	if metadata.Parameters != nil {
		node.Metadata["parameters"] = metadata.Parameters
	}

	return node, nil
}

// toolFromNode retrieves a tool from a node.
func (m *ToolNodeMapping) toolFromNode(node *schema.Node) (interface{}, error) {
	name, ok := node.Metadata["name"].(string)
	if !ok {
		return nil, fmt.Errorf("node missing tool name")
	}

	tool, exists := m.tools[name]
	if !exists {
		return nil, fmt.Errorf("tool not found: %s", name)
	}

	return tool, nil
}

// AddTool adds a tool to the mapping.
func (m *ToolNodeMapping) AddTool(tool tools.Tool) error {
	metadata := tool.Metadata()
	if metadata == nil {
		return fmt.Errorf("tool has no metadata")
	}

	m.tools[metadata.Name] = tool
	return m.AddObject(tool)
}

// AddTools adds multiple tools to the mapping.
func (m *ToolNodeMapping) AddTools(toolList ...tools.Tool) error {
	for _, tool := range toolList {
		if err := m.AddTool(tool); err != nil {
			return err
		}
	}
	return nil
}

// GetTool retrieves a tool by name.
func (m *ToolNodeMapping) GetTool(name string) (tools.Tool, error) {
	tool, exists := m.tools[name]
	if !exists {
		return nil, fmt.Errorf("tool not found: %s", name)
	}
	return tool, nil
}

// GetTools returns all tools in the mapping.
func (m *ToolNodeMapping) GetTools() []tools.Tool {
	toolList := make([]tools.Tool, 0, len(m.tools))
	for _, tool := range m.tools {
		toolList = append(toolList, tool)
	}
	return toolList
}

// GetToolNames returns all tool names.
func (m *ToolNodeMapping) GetToolNames() []string {
	names := make([]string, 0, len(m.tools))
	for name := range m.tools {
		names = append(names, name)
	}
	return names
}

// ToolRetriever retrieves tools based on query similarity.
type ToolRetriever struct {
	// mapping is the tool node mapping.
	mapping *ToolNodeMapping
	// embedModel is the embedding model for similarity search.
	embedModel embedding.EmbeddingModel
	// nodeEmbeddings stores embeddings for each node.
	nodeEmbeddings map[string][]float64
	// topK is the number of tools to retrieve.
	topK int
}

// ToolRetrieverOption configures a ToolRetriever.
type ToolRetrieverOption func(*ToolRetriever)

// WithToolRetrieverTopK sets the number of tools to retrieve.
func WithToolRetrieverTopK(k int) ToolRetrieverOption {
	return func(r *ToolRetriever) {
		r.topK = k
	}
}

// NewToolRetriever creates a new ToolRetriever.
func NewToolRetriever(mapping *ToolNodeMapping, embedModel embedding.EmbeddingModel, opts ...ToolRetrieverOption) *ToolRetriever {
	r := &ToolRetriever{
		mapping:        mapping,
		embedModel:     embedModel,
		nodeEmbeddings: make(map[string][]float64),
		topK:           3,
	}

	for _, opt := range opts {
		opt(r)
	}

	return r
}

// BuildIndex builds embeddings for all tools.
func (r *ToolRetriever) BuildIndex(ctx context.Context) error {
	nodes := r.mapping.GetNodes()

	for _, node := range nodes {
		text := node.GetContent(schema.MetadataModeNone)
		emb, err := r.embedModel.GetTextEmbedding(ctx, text)
		if err != nil {
			return fmt.Errorf("failed to embed tool %s: %w", node.ID, err)
		}
		r.nodeEmbeddings[node.ID] = emb
	}

	return nil
}

// RetrieveTools retrieves the most relevant tools for a query.
func (r *ToolRetriever) RetrieveTools(ctx context.Context, query string) ([]tools.Tool, error) {
	// Get query embedding
	queryEmb, err := r.embedModel.GetQueryEmbedding(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to embed query: %w", err)
	}

	// Calculate similarities
	type scoredTool struct {
		name  string
		score float64
	}

	var scores []scoredTool
	for nodeID, nodeEmb := range r.nodeEmbeddings {
		score, err := embedding.CosineSimilarity(queryEmb, nodeEmb)
		if err != nil {
			continue // Skip nodes with incompatible embeddings
		}
		scores = append(scores, scoredTool{name: nodeID, score: score})
	}

	// Sort by score descending
	for i := 0; i < len(scores)-1; i++ {
		for j := i + 1; j < len(scores); j++ {
			if scores[j].score > scores[i].score {
				scores[i], scores[j] = scores[j], scores[i]
			}
		}
	}

	// Get top K tools
	k := r.topK
	if k > len(scores) {
		k = len(scores)
	}

	result := make([]tools.Tool, 0, k)
	for i := 0; i < k; i++ {
		tool, err := r.mapping.GetTool(scores[i].name)
		if err != nil {
			continue
		}
		result = append(result, tool)
	}

	return result, nil
}

// RetrieveObjects implements ObjectRetriever.
func (r *ToolRetriever) RetrieveObjects(ctx context.Context, query string) ([]interface{}, error) {
	tools, err := r.RetrieveTools(ctx, query)
	if err != nil {
		return nil, err
	}

	objects := make([]interface{}, len(tools))
	for i, tool := range tools {
		objects[i] = tool
	}

	return objects, nil
}

// Ensure interfaces are implemented.
var _ ObjectNodeMapping = (*ToolNodeMapping)(nil)
var _ ObjectRetriever = (*ToolRetriever)(nil)
