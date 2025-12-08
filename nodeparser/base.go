package nodeparser

import (
	"fmt"

	"github.com/aqua777/go-llamaindex/schema"
	"github.com/google/uuid"
)

// BaseNodeParser provides common functionality for node parsers.
// It handles metadata propagation, relationship establishment, and ID generation.
type BaseNodeParser struct {
	options  NodeParserOptions
	callback NodeParserCallback
}

// NewBaseNodeParser creates a new BaseNodeParser with default options.
func NewBaseNodeParser() *BaseNodeParser {
	return &BaseNodeParser{
		options: DefaultNodeParserOptions(),
	}
}

// NewBaseNodeParserWithOptions creates a new BaseNodeParser with custom options.
func NewBaseNodeParserWithOptions(options NodeParserOptions) *BaseNodeParser {
	return &BaseNodeParser{
		options: options,
	}
}

// WithIncludeMetadata sets whether to include parent metadata in child nodes.
func (p *BaseNodeParser) WithIncludeMetadata(include bool) *BaseNodeParser {
	p.options.IncludeMetadata = include
	return p
}

// WithIncludePrevNextRel sets whether to establish PREVIOUS/NEXT relationships.
func (p *BaseNodeParser) WithIncludePrevNextRel(include bool) *BaseNodeParser {
	p.options.IncludePrevNextRel = include
	return p
}

// WithCallback sets the callback for parsing events.
func (p *BaseNodeParser) WithCallback(callback NodeParserCallback) *BaseNodeParser {
	p.callback = callback
	return p
}

// WithIDFunc sets a custom ID generation function.
func (p *BaseNodeParser) WithIDFunc(idFunc func() string) *BaseNodeParser {
	p.options.IDFunc = idFunc
	return p
}

// Options returns the current options.
func (p *BaseNodeParser) Options() NodeParserOptions {
	return p.options
}

// GenerateID generates a new node ID.
func (p *BaseNodeParser) GenerateID() string {
	if p.options.IDFunc != nil {
		return p.options.IDFunc()
	}
	return uuid.New().String()
}

// PostProcessNodes applies post-processing to nodes:
// - Establishes PREVIOUS/NEXT relationships if enabled
// - Calculates character indices
// - Copies metadata from parent if enabled
func (p *BaseNodeParser) PostProcessNodes(nodes []*schema.Node, parentNode *schema.Node, parentDoc *schema.Document) []*schema.Node {
	if len(nodes) == 0 {
		return nodes
	}

	// Track character position for StartCharIdx/EndCharIdx
	charIdx := 0

	for i, node := range nodes {
		// Set character indices
		startIdx := charIdx
		endIdx := charIdx + len(node.Text)
		node.StartCharIdx = &startIdx
		node.EndCharIdx = &endIdx
		charIdx = endIdx

		// Copy metadata from parent
		if p.options.IncludeMetadata {
			if parentNode != nil {
				p.mergeMetadata(node, parentNode.Metadata)
			}
			if parentDoc != nil {
				p.mergeMetadata(node, parentDoc.Metadata)
			}
		}

		// Set SOURCE relationship
		if parentNode != nil {
			node.Relationships.SetSource(parentNode.AsRelatedNodeInfo())
		} else if parentDoc != nil {
			node.Relationships.SetSource(schema.RelatedNodeInfo{
				NodeID:   parentDoc.ID,
				NodeType: schema.ObjectTypeDocument,
				Metadata: parentDoc.Metadata,
			})
		}

		// Set PREVIOUS/NEXT relationships
		if p.options.IncludePrevNextRel {
			if i > 0 {
				prevNode := nodes[i-1]
				node.Relationships.SetPrevious(prevNode.AsRelatedNodeInfo())
			}
			if i < len(nodes)-1 {
				// We'll set NEXT in a second pass since we don't have the next node yet
			}
		}
	}

	// Second pass for NEXT relationships
	if p.options.IncludePrevNextRel {
		for i := 0; i < len(nodes)-1; i++ {
			nodes[i].Relationships.SetNext(nodes[i+1].AsRelatedNodeInfo())
		}
	}

	return nodes
}

// mergeMetadata merges source metadata into node metadata.
// Existing keys in the node are not overwritten.
func (p *BaseNodeParser) mergeMetadata(node *schema.Node, source map[string]interface{}) {
	if source == nil {
		return
	}
	if node.Metadata == nil {
		node.Metadata = make(map[string]interface{})
	}
	for key, value := range source {
		if _, exists := node.Metadata[key]; !exists {
			node.Metadata[key] = value
		}
	}
}

// BuildNodesFromSplits creates nodes from text splits.
// This is a helper for text splitter-based parsers.
func (p *BaseNodeParser) BuildNodesFromSplits(splits []string, parentNode *schema.Node, parentDoc *schema.Document) []*schema.Node {
	nodes := make([]*schema.Node, len(splits))

	for i, text := range splits {
		node := schema.NewNode()
		node.ID = p.GenerateID()
		node.Text = text
		node.Type = schema.ObjectTypeText
		node.Hash = node.GenerateHash()

		// Add chunk metadata
		node.Metadata["chunk_index"] = i
		node.Metadata["chunk_count"] = len(splits)

		nodes[i] = node
	}

	return p.PostProcessNodes(nodes, parentNode, parentDoc)
}

// emitEvent sends an event to the callback if set.
func (p *BaseNodeParser) emitEvent(event NodeParserEvent) {
	if p.callback != nil {
		p.callback(event)
	}
}

// EmitStart emits a start event.
func (p *BaseNodeParser) EmitStart(docID string) {
	p.emitEvent(NodeParserEvent{
		Type:       EventTypeStart,
		DocumentID: docID,
	})
}

// EmitProgress emits a progress event.
func (p *BaseNodeParser) EmitProgress(docID string, nodeCount int) {
	p.emitEvent(NodeParserEvent{
		Type:       EventTypeProgress,
		DocumentID: docID,
		NodeCount:  nodeCount,
	})
}

// EmitComplete emits a complete event.
func (p *BaseNodeParser) EmitComplete(docID string, nodeCount int) {
	p.emitEvent(NodeParserEvent{
		Type:       EventTypeComplete,
		DocumentID: docID,
		NodeCount:  nodeCount,
	})
}

// EmitError emits an error event.
func (p *BaseNodeParser) EmitError(docID string, err error) {
	p.emitEvent(NodeParserEvent{
		Type:       EventTypeError,
		DocumentID: docID,
		Message:    fmt.Sprintf("error: %v", err),
	})
}
