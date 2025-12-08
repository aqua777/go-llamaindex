package postprocessor

import (
	"context"

	"github.com/aqua777/go-llamaindex/schema"
)

// MetadataReplacementPostprocessor replaces node content with metadata value.
type MetadataReplacementPostprocessor struct {
	*BaseNodePostprocessor
	targetMetadataKey string
}

// MetadataReplacementOption configures a MetadataReplacementPostprocessor.
type MetadataReplacementOption func(*MetadataReplacementPostprocessor)

// WithTargetMetadataKey sets the target metadata key.
func WithTargetMetadataKey(key string) MetadataReplacementOption {
	return func(p *MetadataReplacementPostprocessor) {
		p.targetMetadataKey = key
	}
}

// NewMetadataReplacementPostprocessor creates a new MetadataReplacementPostprocessor.
func NewMetadataReplacementPostprocessor(targetKey string, opts ...MetadataReplacementOption) *MetadataReplacementPostprocessor {
	p := &MetadataReplacementPostprocessor{
		BaseNodePostprocessor: NewBaseNodePostprocessor(
			WithPostprocessorName("MetadataReplacementPostprocessor"),
		),
		targetMetadataKey: targetKey,
	}

	for _, opt := range opts {
		opt(p)
	}

	return p
}

// PostprocessNodes replaces node content with the target metadata value.
func (p *MetadataReplacementPostprocessor) PostprocessNodes(
	ctx context.Context,
	nodes []schema.NodeWithScore,
	queryBundle *schema.QueryBundle,
) ([]schema.NodeWithScore, error) {
	result := make([]schema.NodeWithScore, len(nodes))

	for i, nodeWithScore := range nodes {
		// Get the metadata value
		if metadata := nodeWithScore.Node.GetMetadata(); metadata != nil {
			if value, ok := metadata[p.targetMetadataKey]; ok {
				if strValue, ok := value.(string); ok {
					// Create a copy of the node with replaced content
					newNode := nodeWithScore.Node
					newNode.SetContent(strValue)
					result[i] = schema.NodeWithScore{
						Node:  newNode,
						Score: nodeWithScore.Score,
					}
					continue
				}
			}
		}

		// If no replacement, keep original
		result[i] = nodeWithScore
	}

	return result, nil
}

// TargetMetadataKey returns the target metadata key.
func (p *MetadataReplacementPostprocessor) TargetMetadataKey() string {
	return p.targetMetadataKey
}

// Ensure MetadataReplacementPostprocessor implements NodePostprocessor.
var _ NodePostprocessor = (*MetadataReplacementPostprocessor)(nil)
