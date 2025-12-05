package retriever

import (
	"context"
	"sort"

	"github.com/aqua777/go-llamaindex/schema"
)

// FusionMode represents the fusion strategy for combining results.
type FusionMode string

const (
	// FusionModeReciprocalRank applies Reciprocal Rank Fusion.
	FusionModeReciprocalRank FusionMode = "reciprocal_rerank"
	// FusionModeRelativeScore applies relative score fusion.
	FusionModeRelativeScore FusionMode = "relative_score"
	// FusionModeDistBasedScore applies distance-based score fusion.
	FusionModeDistBasedScore FusionMode = "dist_based_score"
	// FusionModeSimple applies simple fusion (max score).
	FusionModeSimple FusionMode = "simple"
)

// FusionRetriever combines results from multiple retrievers using fusion strategies.
type FusionRetriever struct {
	*BaseRetriever
	// Retrievers is the list of retrievers to combine.
	Retrievers []Retriever
	// RetrieverWeights are the weights for each retriever (must sum to 1).
	RetrieverWeights []float64
	// Mode is the fusion strategy to use.
	Mode FusionMode
	// SimilarityTopK is the number of results to return.
	SimilarityTopK int
}

// FusionRetrieverOption is a functional option for FusionRetriever.
type FusionRetrieverOption func(*FusionRetriever)

// WithFusionMode sets the fusion mode.
func WithFusionMode(mode FusionMode) FusionRetrieverOption {
	return func(fr *FusionRetriever) {
		fr.Mode = mode
	}
}

// WithSimilarityTopK sets the number of results to return.
func WithSimilarityTopK(topK int) FusionRetrieverOption {
	return func(fr *FusionRetriever) {
		fr.SimilarityTopK = topK
	}
}

// WithRetrieverWeights sets the weights for each retriever.
func WithRetrieverWeights(weights []float64) FusionRetrieverOption {
	return func(fr *FusionRetriever) {
		// Normalize weights to sum to 1
		total := 0.0
		for _, w := range weights {
			total += w
		}
		normalized := make([]float64, len(weights))
		for i, w := range weights {
			normalized[i] = w / total
		}
		fr.RetrieverWeights = normalized
	}
}

// NewFusionRetriever creates a new FusionRetriever.
func NewFusionRetriever(retrievers []Retriever, opts ...FusionRetrieverOption) *FusionRetriever {
	// Default equal weights
	weights := make([]float64, len(retrievers))
	for i := range weights {
		weights[i] = 1.0 / float64(len(retrievers))
	}

	fr := &FusionRetriever{
		BaseRetriever:    NewBaseRetriever(),
		Retrievers:       retrievers,
		RetrieverWeights: weights,
		Mode:             FusionModeSimple,
		SimilarityTopK:   10,
	}

	for _, opt := range opts {
		opt(fr)
	}

	return fr
}

// Retrieve retrieves nodes from all retrievers and fuses the results.
func (fr *FusionRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	// Collect results from all retrievers
	// Key: (query_str, retriever_index), Value: results
	results := make(map[int][]schema.NodeWithScore)

	for i, retriever := range fr.Retrievers {
		nodes, err := retriever.Retrieve(ctx, query)
		if err != nil {
			return nil, err
		}
		results[i] = nodes
	}

	// Apply fusion strategy
	var fusedNodes []schema.NodeWithScore
	switch fr.Mode {
	case FusionModeReciprocalRank:
		fusedNodes = fr.reciprocalRankFusion(results)
	case FusionModeRelativeScore:
		fusedNodes = fr.relativeScoreFusion(results, false)
	case FusionModeDistBasedScore:
		fusedNodes = fr.relativeScoreFusion(results, true)
	case FusionModeSimple:
		fusedNodes = fr.simpleFusion(results)
	default:
		fusedNodes = fr.simpleFusion(results)
	}

	// Limit to top K
	if len(fusedNodes) > fr.SimilarityTopK {
		fusedNodes = fusedNodes[:fr.SimilarityTopK]
	}

	return fusedNodes, nil
}

// reciprocalRankFusion applies Reciprocal Rank Fusion.
// Reference: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
func (fr *FusionRetriever) reciprocalRankFusion(results map[int][]schema.NodeWithScore) []schema.NodeWithScore {
	k := 60.0 // Parameter to control impact of outlier rankings
	fusedScores := make(map[string]float64)
	hashToNode := make(map[string]schema.NodeWithScore)

	for _, nodes := range results {
		// Sort by score descending
		sorted := make([]schema.NodeWithScore, len(nodes))
		copy(sorted, nodes)
		sort.Slice(sorted, func(i, j int) bool {
			return sorted[i].Score > sorted[j].Score
		})

		for rank, node := range sorted {
			hash := node.Node.GenerateHash()
			hashToNode[hash] = node
			fusedScores[hash] += 1.0 / (float64(rank) + k)
		}
	}

	// Convert to slice and sort by fused score
	var fusedNodes []schema.NodeWithScore
	for hash, score := range fusedScores {
		node := hashToNode[hash]
		node.Score = score
		fusedNodes = append(fusedNodes, node)
	}

	sort.Slice(fusedNodes, func(i, j int) bool {
		return fusedNodes[i].Score > fusedNodes[j].Score
	})

	return fusedNodes
}

// relativeScoreFusion applies relative score fusion with optional distance-based scaling.
func (fr *FusionRetriever) relativeScoreFusion(results map[int][]schema.NodeWithScore, distBased bool) []schema.NodeWithScore {
	// Calculate min/max scores for each retriever
	minMaxScores := make(map[int][2]float64)

	for retrieverIdx, nodes := range results {
		if len(nodes) == 0 {
			minMaxScores[retrieverIdx] = [2]float64{0, 0}
			continue
		}

		scores := make([]float64, len(nodes))
		for i, n := range nodes {
			scores[i] = n.Score
		}

		if distBased {
			// Use mean +/- 3 std dev
			mean := 0.0
			for _, s := range scores {
				mean += s
			}
			mean /= float64(len(scores))

			variance := 0.0
			for _, s := range scores {
				variance += (s - mean) * (s - mean)
			}
			stdDev := 0.0
			if len(scores) > 0 {
				stdDev = sqrt(variance / float64(len(scores)))
			}

			minMaxScores[retrieverIdx] = [2]float64{mean - 3*stdDev, mean + 3*stdDev}
		} else {
			minScore, maxScore := scores[0], scores[0]
			for _, s := range scores {
				if s < minScore {
					minScore = s
				}
				if s > maxScore {
					maxScore = s
				}
			}
			minMaxScores[retrieverIdx] = [2]float64{minScore, maxScore}
		}
	}

	// Normalize scores and aggregate
	allNodes := make(map[string]schema.NodeWithScore)

	for retrieverIdx, nodes := range results {
		minScore := minMaxScores[retrieverIdx][0]
		maxScore := minMaxScores[retrieverIdx][1]
		weight := fr.RetrieverWeights[retrieverIdx]

		for _, node := range nodes {
			// Normalize score to [0, 1]
			var normalizedScore float64
			if maxScore == minScore {
				if maxScore > 0 {
					normalizedScore = 1.0
				} else {
					normalizedScore = 0.0
				}
			} else {
				normalizedScore = (node.Score - minScore) / (maxScore - minScore)
			}

			// Apply weight
			weightedScore := normalizedScore * weight

			hash := node.Node.GenerateHash()
			if existing, exists := allNodes[hash]; exists {
				existing.Score += weightedScore
				allNodes[hash] = existing
			} else {
				node.Score = weightedScore
				allNodes[hash] = node
			}
		}
	}

	// Convert to slice and sort
	var fusedNodes []schema.NodeWithScore
	for _, node := range allNodes {
		fusedNodes = append(fusedNodes, node)
	}

	sort.Slice(fusedNodes, func(i, j int) bool {
		return fusedNodes[i].Score > fusedNodes[j].Score
	})

	return fusedNodes
}

// simpleFusion applies simple fusion (max score for duplicates).
func (fr *FusionRetriever) simpleFusion(results map[int][]schema.NodeWithScore) []schema.NodeWithScore {
	allNodes := make(map[string]schema.NodeWithScore)

	for _, nodes := range results {
		for _, node := range nodes {
			hash := node.Node.GenerateHash()
			if existing, exists := allNodes[hash]; exists {
				if node.Score > existing.Score {
					allNodes[hash] = node
				}
			} else {
				allNodes[hash] = node
			}
		}
	}

	// Convert to slice and sort
	var fusedNodes []schema.NodeWithScore
	for _, node := range allNodes {
		fusedNodes = append(fusedNodes, node)
	}

	sort.Slice(fusedNodes, func(i, j int) bool {
		return fusedNodes[i].Score > fusedNodes[j].Score
	})

	return fusedNodes
}

// sqrt is a simple square root implementation.
func sqrt(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x == 0 {
		return 0
	}
	z := x / 2
	for i := 0; i < 100; i++ {
		z = z - (z*z-x)/(2*z)
	}
	return z
}

// Ensure FusionRetriever implements Retriever.
var _ Retriever = (*FusionRetriever)(nil)
