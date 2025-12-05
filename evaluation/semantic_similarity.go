package evaluation

import (
	"context"
	"fmt"
	"math"

	"github.com/aqua777/go-llamaindex/embedding"
)

// SimilarityMode represents the mode for computing similarity.
type SimilarityMode string

const (
	// SimilarityModeCosine uses cosine similarity.
	SimilarityModeCosine SimilarityMode = "cosine"
	// SimilarityModeDotProduct uses dot product similarity.
	SimilarityModeDotProduct SimilarityMode = "dot_product"
	// SimilarityModeEuclidean uses negative euclidean distance.
	SimilarityModeEuclidean SimilarityMode = "euclidean"
)

// SemanticSimilarityEvaluator evaluates the semantic similarity between
// the generated answer and the reference answer using embeddings.
type SemanticSimilarityEvaluator struct {
	*BaseEvaluator
	embedModel          embedding.EmbeddingModel
	similarityMode      SimilarityMode
	similarityThreshold float64
}

// SemanticSimilarityEvaluatorOption configures a SemanticSimilarityEvaluator.
type SemanticSimilarityEvaluatorOption func(*SemanticSimilarityEvaluator)

// WithSemanticSimilarityEmbedModel sets the embedding model.
func WithSemanticSimilarityEmbedModel(model embedding.EmbeddingModel) SemanticSimilarityEvaluatorOption {
	return func(e *SemanticSimilarityEvaluator) {
		e.embedModel = model
	}
}

// WithSemanticSimilarityMode sets the similarity mode.
func WithSemanticSimilarityMode(mode SimilarityMode) SemanticSimilarityEvaluatorOption {
	return func(e *SemanticSimilarityEvaluator) {
		e.similarityMode = mode
	}
}

// WithSemanticSimilarityThreshold sets the similarity threshold for passing.
func WithSemanticSimilarityThreshold(threshold float64) SemanticSimilarityEvaluatorOption {
	return func(e *SemanticSimilarityEvaluator) {
		e.similarityThreshold = threshold
	}
}

// NewSemanticSimilarityEvaluator creates a new SemanticSimilarityEvaluator.
func NewSemanticSimilarityEvaluator(opts ...SemanticSimilarityEvaluatorOption) *SemanticSimilarityEvaluator {
	e := &SemanticSimilarityEvaluator{
		BaseEvaluator:       NewBaseEvaluator(WithEvaluatorName("semantic_similarity")),
		similarityMode:      SimilarityModeCosine,
		similarityThreshold: 0.8,
	}

	for _, opt := range opts {
		opt(e)
	}

	return e
}

// Evaluate evaluates the semantic similarity between response and reference.
func (e *SemanticSimilarityEvaluator) Evaluate(ctx context.Context, input *EvaluateInput) (*EvaluationResult, error) {
	if input.Response == "" {
		return NewEvaluationResult().WithInvalid("response must be provided"), nil
	}
	if input.Reference == "" {
		return NewEvaluationResult().WithInvalid("reference must be provided"), nil
	}
	if e.embedModel == nil {
		return nil, fmt.Errorf("embedding model must be provided for semantic similarity evaluation")
	}

	// Get embeddings for response and reference
	responseEmbedding, err := e.embedModel.GetTextEmbedding(ctx, input.Response)
	if err != nil {
		return nil, fmt.Errorf("failed to get response embedding: %w", err)
	}

	referenceEmbedding, err := e.embedModel.GetTextEmbedding(ctx, input.Reference)
	if err != nil {
		return nil, fmt.Errorf("failed to get reference embedding: %w", err)
	}

	// Calculate similarity
	similarity := computeSimilarity(responseEmbedding, referenceEmbedding, e.similarityMode)
	passing := similarity >= e.similarityThreshold

	return NewEvaluationResult().
		WithQuery(input.Query).
		WithResponse(input.Response).
		WithReference(input.Reference).
		WithPassing(passing).
		WithScore(similarity).
		WithFeedback(fmt.Sprintf("Similarity score: %.4f (threshold: %.4f)", similarity, e.similarityThreshold)), nil
}

// computeSimilarity computes the similarity between two embedding vectors.
func computeSimilarity(vec1, vec2 []float64, mode SimilarityMode) float64 {
	switch mode {
	case SimilarityModeCosine:
		return cosineSimilarity(vec1, vec2)
	case SimilarityModeDotProduct:
		return dotProduct(vec1, vec2)
	case SimilarityModeEuclidean:
		return negativeEuclideanDistance(vec1, vec2)
	default:
		return cosineSimilarity(vec1, vec2)
	}
}

// cosineSimilarity computes the cosine similarity between two vectors.
func cosineSimilarity(vec1, vec2 []float64) float64 {
	if len(vec1) != len(vec2) {
		return 0
	}

	var dot, norm1, norm2 float64
	for i := range vec1 {
		dot += vec1[i] * vec2[i]
		norm1 += vec1[i] * vec1[i]
		norm2 += vec2[i] * vec2[i]
	}

	if norm1 == 0 || norm2 == 0 {
		return 0
	}

	return dot / (math.Sqrt(norm1) * math.Sqrt(norm2))
}

// dotProduct computes the dot product between two vectors.
func dotProduct(vec1, vec2 []float64) float64 {
	if len(vec1) != len(vec2) {
		return 0
	}

	var dot float64
	for i := range vec1 {
		dot += vec1[i] * vec2[i]
	}

	return dot
}

// negativeEuclideanDistance computes the negative euclidean distance.
// Negative because smaller distance = more similar.
func negativeEuclideanDistance(vec1, vec2 []float64) float64 {
	if len(vec1) != len(vec2) {
		return math.Inf(-1)
	}

	var sumSquares float64
	for i := range vec1 {
		diff := vec1[i] - vec2[i]
		sumSquares += diff * diff
	}

	return -math.Sqrt(sumSquares)
}

// CosineSimilarity is a public helper function for computing cosine similarity.
func CosineSimilarity(vec1, vec2 []float64) float64 {
	return cosineSimilarity(vec1, vec2)
}

// DotProduct is a public helper function for computing dot product.
func DotProduct(vec1, vec2 []float64) float64 {
	return dotProduct(vec1, vec2)
}

// EuclideanDistance computes the euclidean distance between two vectors.
func EuclideanDistance(vec1, vec2 []float64) float64 {
	return -negativeEuclideanDistance(vec1, vec2)
}
