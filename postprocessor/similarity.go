package postprocessor

import (
	"context"

	"github.com/aqua777/go-llamaindex/schema"
)

// SimilarityPostprocessor filters nodes by similarity score.
type SimilarityPostprocessor struct {
	*BaseNodePostprocessor
	similarityCutoff float64
}

// SimilarityPostprocessorOption configures a SimilarityPostprocessor.
type SimilarityPostprocessorOption func(*SimilarityPostprocessor)

// WithSimilarityCutoff sets the similarity cutoff threshold.
func WithSimilarityCutoff(cutoff float64) SimilarityPostprocessorOption {
	return func(p *SimilarityPostprocessor) {
		p.similarityCutoff = cutoff
	}
}

// NewSimilarityPostprocessor creates a new SimilarityPostprocessor.
func NewSimilarityPostprocessor(opts ...SimilarityPostprocessorOption) *SimilarityPostprocessor {
	p := &SimilarityPostprocessor{
		BaseNodePostprocessor: NewBaseNodePostprocessor(
			WithPostprocessorName("SimilarityPostprocessor"),
		),
		similarityCutoff: 0.0,
	}

	for _, opt := range opts {
		opt(p)
	}

	return p
}

// PostprocessNodes filters nodes below the similarity cutoff.
func (p *SimilarityPostprocessor) PostprocessNodes(
	ctx context.Context,
	nodes []schema.NodeWithScore,
	queryBundle *schema.QueryBundle,
) ([]schema.NodeWithScore, error) {
	result := make([]schema.NodeWithScore, 0, len(nodes))

	for _, node := range nodes {
		// Skip nodes without scores
		if node.Score == 0 {
			continue
		}

		// Keep nodes above the cutoff
		if node.Score >= p.similarityCutoff {
			result = append(result, node)
		}
	}

	return result, nil
}

// SimilarityCutoff returns the current similarity cutoff.
func (p *SimilarityPostprocessor) SimilarityCutoff() float64 {
	return p.similarityCutoff
}

// Ensure SimilarityPostprocessor implements NodePostprocessor.
var _ NodePostprocessor = (*SimilarityPostprocessor)(nil)
