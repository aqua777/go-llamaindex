package postprocessor

import (
	"context"
	"sort"

	"github.com/aqua777/go-llamaindex/schema"
)

// LongContextReorder reorders nodes to optimize for long context models.
// Based on the paper: https://arxiv.org/abs/2307.03172
// Models struggle to access significant details found in the center of extended contexts.
// The best performance typically arises when crucial data is positioned at the start
// or conclusion of the input context.
type LongContextReorder struct {
	*BaseNodePostprocessor
}

// NewLongContextReorder creates a new LongContextReorder.
func NewLongContextReorder() *LongContextReorder {
	return &LongContextReorder{
		BaseNodePostprocessor: NewBaseNodePostprocessor(
			WithPostprocessorName("LongContextReorder"),
		),
	}
}

// PostprocessNodes reorders nodes by placing higher-scored nodes at the start and end.
func (p *LongContextReorder) PostprocessNodes(
	ctx context.Context,
	nodes []schema.NodeWithScore,
	queryBundle *schema.QueryBundle,
) ([]schema.NodeWithScore, error) {
	if len(nodes) == 0 {
		return nodes, nil
	}

	// Sort nodes by score (ascending)
	sortedNodes := make([]schema.NodeWithScore, len(nodes))
	copy(sortedNodes, nodes)
	sort.Slice(sortedNodes, func(i, j int) bool {
		return sortedNodes[i].Score < sortedNodes[j].Score
	})

	// Reorder: place nodes alternately at the beginning and end
	// This puts higher-scored nodes at both ends of the context
	reorderedNodes := make([]schema.NodeWithScore, 0, len(sortedNodes))

	for i, node := range sortedNodes {
		if i%2 == 0 {
			// Insert at the beginning
			reorderedNodes = append([]schema.NodeWithScore{node}, reorderedNodes...)
		} else {
			// Append at the end
			reorderedNodes = append(reorderedNodes, node)
		}
	}

	return reorderedNodes, nil
}

// Ensure LongContextReorder implements NodePostprocessor.
var _ NodePostprocessor = (*LongContextReorder)(nil)
