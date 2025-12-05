package embedding

import (
	"context"
	"fmt"
	"sort"
)

// SparseEmbedding represents a sparse vector embedding.
// Sparse embeddings only store non-zero values with their indices.
type SparseEmbedding struct {
	// Indices are the positions of non-zero values.
	Indices []int `json:"indices"`
	// Values are the non-zero values at the corresponding indices.
	Values []float64 `json:"values"`
	// Dimension is the total dimension of the sparse vector.
	Dimension int `json:"dimension,omitempty"`
}

// NewSparseEmbedding creates a new sparse embedding.
func NewSparseEmbedding(indices []int, values []float64) *SparseEmbedding {
	return &SparseEmbedding{
		Indices: indices,
		Values:  values,
	}
}

// NewSparseEmbeddingWithDimension creates a sparse embedding with explicit dimension.
func NewSparseEmbeddingWithDimension(indices []int, values []float64, dimension int) *SparseEmbedding {
	return &SparseEmbedding{
		Indices:   indices,
		Values:    values,
		Dimension: dimension,
	}
}

// FromDense converts a dense vector to sparse embedding.
func FromDense(dense []float64, threshold float64) *SparseEmbedding {
	var indices []int
	var values []float64

	for i, v := range dense {
		if v != 0 && (threshold == 0 || abs(v) >= threshold) {
			indices = append(indices, i)
			values = append(values, v)
		}
	}

	return &SparseEmbedding{
		Indices:   indices,
		Values:    values,
		Dimension: len(dense),
	}
}

// ToDense converts sparse embedding to dense vector.
func (s *SparseEmbedding) ToDense() []float64 {
	dim := s.Dimension
	if dim == 0 && len(s.Indices) > 0 {
		// Infer dimension from max index
		for _, idx := range s.Indices {
			if idx >= dim {
				dim = idx + 1
			}
		}
	}

	dense := make([]float64, dim)
	for i, idx := range s.Indices {
		if idx < dim {
			dense[idx] = s.Values[i]
		}
	}

	return dense
}

// Len returns the number of non-zero elements.
func (s *SparseEmbedding) Len() int {
	return len(s.Indices)
}

// Get returns the value at a specific index.
func (s *SparseEmbedding) Get(index int) float64 {
	for i, idx := range s.Indices {
		if idx == index {
			return s.Values[i]
		}
	}
	return 0
}

// DotProduct computes the dot product of two sparse embeddings.
func (s *SparseEmbedding) DotProduct(other *SparseEmbedding) float64 {
	// Create a map for faster lookup
	otherMap := make(map[int]float64)
	for i, idx := range other.Indices {
		otherMap[idx] = other.Values[i]
	}

	var result float64
	for i, idx := range s.Indices {
		if val, exists := otherMap[idx]; exists {
			result += s.Values[i] * val
		}
	}

	return result
}

// Magnitude returns the magnitude (L2 norm) of the sparse embedding.
func (s *SparseEmbedding) Magnitude() float64 {
	var sum float64
	for _, v := range s.Values {
		sum += v * v
	}
	return sqrt(sum)
}

// CosineSimilarity computes cosine similarity between two sparse embeddings.
func (s *SparseEmbedding) CosineSimilarity(other *SparseEmbedding) float64 {
	dot := s.DotProduct(other)
	mag1 := s.Magnitude()
	mag2 := other.Magnitude()

	if mag1 == 0 || mag2 == 0 {
		return 0
	}

	return dot / (mag1 * mag2)
}

// Normalize returns a normalized copy of the sparse embedding.
func (s *SparseEmbedding) Normalize() *SparseEmbedding {
	mag := s.Magnitude()
	if mag == 0 {
		return s
	}

	values := make([]float64, len(s.Values))
	for i, v := range s.Values {
		values[i] = v / mag
	}

	return &SparseEmbedding{
		Indices:   s.Indices,
		Values:    values,
		Dimension: s.Dimension,
	}
}

// Add adds two sparse embeddings.
func (s *SparseEmbedding) Add(other *SparseEmbedding) *SparseEmbedding {
	result := make(map[int]float64)

	for i, idx := range s.Indices {
		result[idx] = s.Values[i]
	}

	for i, idx := range other.Indices {
		result[idx] += other.Values[i]
	}

	// Convert back to sparse format
	indices := make([]int, 0, len(result))
	for idx := range result {
		indices = append(indices, idx)
	}
	sort.Ints(indices)

	values := make([]float64, len(indices))
	for i, idx := range indices {
		values[i] = result[idx]
	}

	dim := s.Dimension
	if other.Dimension > dim {
		dim = other.Dimension
	}

	return &SparseEmbedding{
		Indices:   indices,
		Values:    values,
		Dimension: dim,
	}
}

// SparseEmbeddingModel is the interface for sparse embedding models.
type SparseEmbeddingModel interface {
	// GetSparseEmbedding generates a sparse embedding for text.
	GetSparseEmbedding(ctx context.Context, text string) (*SparseEmbedding, error)
	// GetSparseQueryEmbedding generates a sparse embedding for a query.
	GetSparseQueryEmbedding(ctx context.Context, query string) (*SparseEmbedding, error)
}

// SparseEmbeddingModelWithBatch extends SparseEmbeddingModel with batch processing.
type SparseEmbeddingModelWithBatch interface {
	SparseEmbeddingModel
	// GetSparseEmbeddingsBatch generates sparse embeddings for multiple texts.
	GetSparseEmbeddingsBatch(ctx context.Context, texts []string) ([]*SparseEmbedding, error)
}

// HybridEmbeddingModel combines dense and sparse embeddings.
type HybridEmbeddingModel interface {
	EmbeddingModel
	SparseEmbeddingModel
	// GetHybridEmbedding generates both dense and sparse embeddings.
	GetHybridEmbedding(ctx context.Context, text string) (dense []float64, sparse *SparseEmbedding, err error)
}

// HybridEmbedding combines dense and sparse embeddings.
type HybridEmbedding struct {
	Dense  []float64        `json:"dense"`
	Sparse *SparseEmbedding `json:"sparse"`
}

// NewHybridEmbedding creates a new hybrid embedding.
func NewHybridEmbedding(dense []float64, sparse *SparseEmbedding) *HybridEmbedding {
	return &HybridEmbedding{
		Dense:  dense,
		Sparse: sparse,
	}
}

// HybridSimilarity computes hybrid similarity with configurable alpha.
// alpha controls the weight of dense vs sparse: similarity = alpha*dense + (1-alpha)*sparse
func HybridSimilarity(h1, h2 *HybridEmbedding, alpha float64) (float64, error) {
	if alpha < 0 || alpha > 1 {
		return 0, fmt.Errorf("alpha must be between 0 and 1")
	}

	denseSim, err := CosineSimilarity(h1.Dense, h2.Dense)
	if err != nil {
		return 0, err
	}

	sparseSim := h1.Sparse.CosineSimilarity(h2.Sparse)

	return alpha*denseSim + (1-alpha)*sparseSim, nil
}

// Helper functions
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	// Newton's method
	z := x / 2
	for i := 0; i < 100; i++ {
		z = z - (z*z-x)/(2*z)
	}
	return z
}
