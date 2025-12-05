package embedding

import (
	"fmt"
	"math"
)

// SimilarityType represents the type of similarity metric.
type SimilarityType string

const (
	// SimilarityTypeCosine uses cosine similarity (default for most use cases).
	SimilarityTypeCosine SimilarityType = "cosine"
	// SimilarityTypeEuclidean uses Euclidean distance (converted to similarity).
	SimilarityTypeEuclidean SimilarityType = "euclidean"
	// SimilarityTypeDotProduct uses dot product similarity.
	SimilarityTypeDotProduct SimilarityType = "dot_product"
)

// CosineSimilarity calculates the cosine similarity between two vectors.
// Returns a value between -1 and 1, where 1 means identical direction.
// For normalized vectors, this is equivalent to dot product.
func CosineSimilarity(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("vectors must have same length: %d != %d", len(a), len(b))
	}
	if len(a) == 0 {
		return 0, fmt.Errorf("vectors must not be empty")
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0, fmt.Errorf("vectors must not be zero vectors")
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB)), nil
}

// DotProduct calculates the dot product between two vectors.
// For normalized vectors, this equals cosine similarity.
func DotProduct(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("vectors must have same length: %d != %d", len(a), len(b))
	}
	if len(a) == 0 {
		return 0, fmt.Errorf("vectors must not be empty")
	}

	var result float64
	for i := range a {
		result += a[i] * b[i]
	}
	return result, nil
}

// EuclideanDistance calculates the Euclidean distance between two vectors.
// Returns a non-negative value where 0 means identical vectors.
func EuclideanDistance(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("vectors must have same length: %d != %d", len(a), len(b))
	}
	if len(a) == 0 {
		return 0, fmt.Errorf("vectors must not be empty")
	}

	var sum float64
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum), nil
}

// EuclideanSimilarity converts Euclidean distance to a similarity score.
// Returns a value between 0 and 1, where 1 means identical vectors.
func EuclideanSimilarity(a, b []float64) (float64, error) {
	dist, err := EuclideanDistance(a, b)
	if err != nil {
		return 0, err
	}
	// Convert distance to similarity: 1 / (1 + distance)
	return 1.0 / (1.0 + dist), nil
}

// Similarity calculates similarity between two vectors using the specified metric.
func Similarity(a, b []float64, simType SimilarityType) (float64, error) {
	switch simType {
	case SimilarityTypeCosine:
		return CosineSimilarity(a, b)
	case SimilarityTypeDotProduct:
		return DotProduct(a, b)
	case SimilarityTypeEuclidean:
		return EuclideanSimilarity(a, b)
	default:
		return CosineSimilarity(a, b)
	}
}

// Normalize normalizes a vector to unit length (L2 norm = 1).
// Returns a new normalized vector without modifying the original.
func Normalize(v []float64) ([]float64, error) {
	if len(v) == 0 {
		return nil, fmt.Errorf("vector must not be empty")
	}

	var norm float64
	for _, val := range v {
		norm += val * val
	}
	norm = math.Sqrt(norm)

	if norm == 0 {
		return nil, fmt.Errorf("cannot normalize zero vector")
	}

	result := make([]float64, len(v))
	for i, val := range v {
		result[i] = val / norm
	}
	return result, nil
}

// NormalizeInPlace normalizes a vector in place.
func NormalizeInPlace(v []float64) error {
	if len(v) == 0 {
		return fmt.Errorf("vector must not be empty")
	}

	var norm float64
	for _, val := range v {
		norm += val * val
	}
	norm = math.Sqrt(norm)

	if norm == 0 {
		return fmt.Errorf("cannot normalize zero vector")
	}

	for i := range v {
		v[i] /= norm
	}
	return nil
}

// Magnitude calculates the magnitude (L2 norm) of a vector.
func Magnitude(v []float64) float64 {
	var sum float64
	for _, val := range v {
		sum += val * val
	}
	return math.Sqrt(sum)
}

// Add adds two vectors element-wise.
func Add(a, b []float64) ([]float64, error) {
	if len(a) != len(b) {
		return nil, fmt.Errorf("vectors must have same length: %d != %d", len(a), len(b))
	}

	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] + b[i]
	}
	return result, nil
}

// Subtract subtracts vector b from vector a element-wise.
func Subtract(a, b []float64) ([]float64, error) {
	if len(a) != len(b) {
		return nil, fmt.Errorf("vectors must have same length: %d != %d", len(a), len(b))
	}

	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] - b[i]
	}
	return result, nil
}

// Scale multiplies a vector by a scalar.
func Scale(v []float64, scalar float64) []float64 {
	result := make([]float64, len(v))
	for i, val := range v {
		result[i] = val * scalar
	}
	return result
}

// Mean calculates the element-wise mean of multiple vectors.
func Mean(vectors [][]float64) ([]float64, error) {
	if len(vectors) == 0 {
		return nil, fmt.Errorf("must provide at least one vector")
	}

	dim := len(vectors[0])
	for i, v := range vectors {
		if len(v) != dim {
			return nil, fmt.Errorf("all vectors must have same dimension: vector %d has %d, expected %d", i, len(v), dim)
		}
	}

	result := make([]float64, dim)
	for _, v := range vectors {
		for i, val := range v {
			result[i] += val
		}
	}

	n := float64(len(vectors))
	for i := range result {
		result[i] /= n
	}

	return result, nil
}

// TopKSimilar finds the top K most similar vectors to a query vector.
// Returns indices and similarity scores sorted by similarity (descending).
func TopKSimilar(query []float64, vectors [][]float64, k int, simType SimilarityType) ([]int, []float64, error) {
	if k <= 0 {
		return nil, nil, fmt.Errorf("k must be positive")
	}
	if len(vectors) == 0 {
		return nil, nil, nil
	}
	if k > len(vectors) {
		k = len(vectors)
	}

	// Calculate all similarities
	type scoredIndex struct {
		index int
		score float64
	}
	scores := make([]scoredIndex, len(vectors))

	for i, v := range vectors {
		sim, err := Similarity(query, v, simType)
		if err != nil {
			return nil, nil, fmt.Errorf("error computing similarity for vector %d: %w", i, err)
		}
		scores[i] = scoredIndex{index: i, score: sim}
	}

	// Simple selection sort for top K (efficient for small K)
	for i := 0; i < k; i++ {
		maxIdx := i
		for j := i + 1; j < len(scores); j++ {
			if scores[j].score > scores[maxIdx].score {
				maxIdx = j
			}
		}
		scores[i], scores[maxIdx] = scores[maxIdx], scores[i]
	}

	// Extract top K results
	indices := make([]int, k)
	similarities := make([]float64, k)
	for i := 0; i < k; i++ {
		indices[i] = scores[i].index
		similarities[i] = scores[i].score
	}

	return indices, similarities, nil
}
