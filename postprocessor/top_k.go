package postprocessor

import (
	"context"
	"sort"

	"github.com/aqua777/go-llamaindex/schema"
)

// TopKPostprocessor limits the number of nodes returned.
type TopKPostprocessor struct {
	*BaseNodePostprocessor
	topK int
}

// TopKPostprocessorOption configures a TopKPostprocessor.
type TopKPostprocessorOption func(*TopKPostprocessor)

// WithTopK sets the maximum number of nodes to return.
func WithTopK(k int) TopKPostprocessorOption {
	return func(p *TopKPostprocessor) {
		p.topK = k
	}
}

// NewTopKPostprocessor creates a new TopKPostprocessor.
func NewTopKPostprocessor(k int, opts ...TopKPostprocessorOption) *TopKPostprocessor {
	p := &TopKPostprocessor{
		BaseNodePostprocessor: NewBaseNodePostprocessor(
			WithPostprocessorName("TopKPostprocessor"),
		),
		topK: k,
	}

	for _, opt := range opts {
		opt(p)
	}

	return p
}

// PostprocessNodes returns only the top K nodes by score.
func (p *TopKPostprocessor) PostprocessNodes(
	ctx context.Context,
	nodes []schema.NodeWithScore,
	queryBundle *schema.QueryBundle,
) ([]schema.NodeWithScore, error) {
	if len(nodes) <= p.topK {
		return nodes, nil
	}

	// Sort by score descending
	sortedNodes := make([]schema.NodeWithScore, len(nodes))
	copy(sortedNodes, nodes)
	sort.Slice(sortedNodes, func(i, j int) bool {
		return sortedNodes[i].Score > sortedNodes[j].Score
	})

	return sortedNodes[:p.topK], nil
}

// TopK returns the current top K value.
func (p *TopKPostprocessor) TopK() int {
	return p.topK
}

// Ensure TopKPostprocessor implements NodePostprocessor.
var _ NodePostprocessor = (*TopKPostprocessor)(nil)
