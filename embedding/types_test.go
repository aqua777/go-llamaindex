package embedding

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEmbeddingInfo(t *testing.T) {
	// Default info
	info := DefaultEmbeddingInfo("test-model")
	assert.Equal(t, "test-model", info.ModelName)
	assert.Equal(t, 1536, info.Dimensions)
	assert.Equal(t, 8191, info.MaxTokens)

	// OpenAI models
	small := OpenAISmallEmbedding3Info()
	assert.Equal(t, "text-embedding-3-small", small.ModelName)
	assert.Equal(t, 1536, small.Dimensions)

	large := OpenAILargeEmbedding3Info()
	assert.Equal(t, "text-embedding-3-large", large.ModelName)
	assert.Equal(t, 3072, large.Dimensions)

	ada := OpenAIAdaEmbeddingInfo()
	assert.Equal(t, "text-embedding-ada-002", ada.ModelName)

	// Ollama models
	mxbai := MxbaiEmbedLargeInfo()
	assert.Equal(t, "mxbai-embed-large", mxbai.ModelName)
	assert.Equal(t, 1024, mxbai.Dimensions)

	minilm := AllMiniLMInfo()
	assert.Equal(t, "all-minilm", minilm.ModelName)
	assert.Equal(t, 384, minilm.Dimensions)

	nomic := NomicEmbedTextInfo()
	assert.Equal(t, "nomic-embed-text", nomic.ModelName)
	assert.Equal(t, 768, nomic.Dimensions)
}

func TestImageType(t *testing.T) {
	// URL image
	urlImg := NewImageFromURL("https://example.com/image.png")
	assert.Equal(t, "https://example.com/image.png", urlImg.URL)
	assert.False(t, urlImg.IsEmpty())

	// Base64 image
	b64Img := NewImageFromBase64("base64data", "image/png")
	assert.Equal(t, "base64data", b64Img.Base64)
	assert.Equal(t, "image/png", b64Img.MimeType)
	assert.False(t, b64Img.IsEmpty())

	// Path image
	pathImg := NewImageFromPath("/path/to/image.jpg")
	assert.Equal(t, "/path/to/image.jpg", pathImg.Path)
	assert.False(t, pathImg.IsEmpty())

	// Empty image
	emptyImg := ImageType{}
	assert.True(t, emptyImg.IsEmpty())
}

func TestCosineSimilarity(t *testing.T) {
	// Identical vectors
	a := []float64{1, 0, 0}
	b := []float64{1, 0, 0}
	sim, err := CosineSimilarity(a, b)
	require.NoError(t, err)
	assert.InDelta(t, 1.0, sim, 0.0001)

	// Orthogonal vectors
	c := []float64{1, 0, 0}
	d := []float64{0, 1, 0}
	sim, err = CosineSimilarity(c, d)
	require.NoError(t, err)
	assert.InDelta(t, 0.0, sim, 0.0001)

	// Opposite vectors
	e := []float64{1, 0, 0}
	f := []float64{-1, 0, 0}
	sim, err = CosineSimilarity(e, f)
	require.NoError(t, err)
	assert.InDelta(t, -1.0, sim, 0.0001)

	// Different lengths - should error
	_, err = CosineSimilarity([]float64{1, 2}, []float64{1, 2, 3})
	assert.Error(t, err)

	// Empty vectors - should error
	_, err = CosineSimilarity([]float64{}, []float64{})
	assert.Error(t, err)
}

func TestDotProduct(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}
	dot, err := DotProduct(a, b)
	require.NoError(t, err)
	assert.InDelta(t, 32.0, dot, 0.0001) // 1*4 + 2*5 + 3*6 = 32

	// Different lengths - should error
	_, err = DotProduct([]float64{1, 2}, []float64{1, 2, 3})
	assert.Error(t, err)
}

func TestEuclideanDistance(t *testing.T) {
	// Same point
	a := []float64{1, 2, 3}
	b := []float64{1, 2, 3}
	dist, err := EuclideanDistance(a, b)
	require.NoError(t, err)
	assert.InDelta(t, 0.0, dist, 0.0001)

	// Known distance
	c := []float64{0, 0, 0}
	d := []float64{3, 4, 0}
	dist, err = EuclideanDistance(c, d)
	require.NoError(t, err)
	assert.InDelta(t, 5.0, dist, 0.0001) // 3-4-5 triangle

	// Different lengths - should error
	_, err = EuclideanDistance([]float64{1, 2}, []float64{1, 2, 3})
	assert.Error(t, err)
}

func TestEuclideanSimilarity(t *testing.T) {
	// Same point - similarity should be 1
	a := []float64{1, 2, 3}
	b := []float64{1, 2, 3}
	sim, err := EuclideanSimilarity(a, b)
	require.NoError(t, err)
	assert.InDelta(t, 1.0, sim, 0.0001)

	// Far apart - similarity should be low
	c := []float64{0, 0}
	d := []float64{100, 100}
	sim, err = EuclideanSimilarity(c, d)
	require.NoError(t, err)
	assert.Less(t, sim, 0.1)
}

func TestSimilarity(t *testing.T) {
	a := []float64{1, 0, 0}
	b := []float64{1, 0, 0}

	// Cosine
	sim, err := Similarity(a, b, SimilarityTypeCosine)
	require.NoError(t, err)
	assert.InDelta(t, 1.0, sim, 0.0001)

	// Dot product
	sim, err = Similarity(a, b, SimilarityTypeDotProduct)
	require.NoError(t, err)
	assert.InDelta(t, 1.0, sim, 0.0001)

	// Euclidean
	sim, err = Similarity(a, b, SimilarityTypeEuclidean)
	require.NoError(t, err)
	assert.InDelta(t, 1.0, sim, 0.0001)

	// Default (cosine)
	sim, err = Similarity(a, b, "unknown")
	require.NoError(t, err)
	assert.InDelta(t, 1.0, sim, 0.0001)
}

func TestNormalize(t *testing.T) {
	v := []float64{3, 4}
	normalized, err := Normalize(v)
	require.NoError(t, err)

	// Check magnitude is 1
	mag := Magnitude(normalized)
	assert.InDelta(t, 1.0, mag, 0.0001)

	// Check direction is preserved
	assert.InDelta(t, 0.6, normalized[0], 0.0001) // 3/5
	assert.InDelta(t, 0.8, normalized[1], 0.0001) // 4/5

	// Original should be unchanged
	assert.Equal(t, 3.0, v[0])
	assert.Equal(t, 4.0, v[1])

	// Empty vector - should error
	_, err = Normalize([]float64{})
	assert.Error(t, err)

	// Zero vector - should error
	_, err = Normalize([]float64{0, 0, 0})
	assert.Error(t, err)
}

func TestNormalizeInPlace(t *testing.T) {
	v := []float64{3, 4}
	err := NormalizeInPlace(v)
	require.NoError(t, err)

	// Check magnitude is 1
	mag := Magnitude(v)
	assert.InDelta(t, 1.0, mag, 0.0001)
}

func TestMagnitude(t *testing.T) {
	v := []float64{3, 4}
	mag := Magnitude(v)
	assert.InDelta(t, 5.0, mag, 0.0001)

	// Zero vector
	zero := []float64{0, 0, 0}
	assert.Equal(t, 0.0, Magnitude(zero))
}

func TestAdd(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}
	result, err := Add(a, b)
	require.NoError(t, err)
	assert.Equal(t, []float64{5, 7, 9}, result)

	// Different lengths - should error
	_, err = Add([]float64{1, 2}, []float64{1, 2, 3})
	assert.Error(t, err)
}

func TestSubtract(t *testing.T) {
	a := []float64{4, 5, 6}
	b := []float64{1, 2, 3}
	result, err := Subtract(a, b)
	require.NoError(t, err)
	assert.Equal(t, []float64{3, 3, 3}, result)

	// Different lengths - should error
	_, err = Subtract([]float64{1, 2}, []float64{1, 2, 3})
	assert.Error(t, err)
}

func TestScale(t *testing.T) {
	v := []float64{1, 2, 3}
	result := Scale(v, 2.0)
	assert.Equal(t, []float64{2, 4, 6}, result)

	// Original unchanged
	assert.Equal(t, []float64{1, 2, 3}, v)
}

func TestMean(t *testing.T) {
	vectors := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	mean, err := Mean(vectors)
	require.NoError(t, err)
	assert.Equal(t, []float64{4, 5, 6}, mean)

	// Empty - should error
	_, err = Mean([][]float64{})
	assert.Error(t, err)

	// Different dimensions - should error
	_, err = Mean([][]float64{{1, 2}, {1, 2, 3}})
	assert.Error(t, err)
}

func TestTopKSimilar(t *testing.T) {
	query := []float64{1, 0, 0}
	vectors := [][]float64{
		{1, 0, 0},     // Most similar (identical)
		{0.9, 0.1, 0}, // Second most similar
		{0, 1, 0},     // Orthogonal
		{-1, 0, 0},    // Opposite
	}

	indices, scores, err := TopKSimilar(query, vectors, 2, SimilarityTypeCosine)
	require.NoError(t, err)
	assert.Len(t, indices, 2)
	assert.Len(t, scores, 2)

	// First should be the identical vector
	assert.Equal(t, 0, indices[0])
	assert.InDelta(t, 1.0, scores[0], 0.0001)

	// Second should be the similar vector
	assert.Equal(t, 1, indices[1])
	assert.Greater(t, scores[1], 0.9)

	// K larger than vectors
	indices, scores, err = TopKSimilar(query, vectors, 10, SimilarityTypeCosine)
	require.NoError(t, err)
	assert.Len(t, indices, 4)

	// K = 0 should error
	_, _, err = TopKSimilar(query, vectors, 0, SimilarityTypeCosine)
	assert.Error(t, err)

	// Empty vectors
	indices, scores, err = TopKSimilar(query, [][]float64{}, 2, SimilarityTypeCosine)
	require.NoError(t, err)
	assert.Nil(t, indices)
	assert.Nil(t, scores)
}

func TestSimilarityTypes(t *testing.T) {
	assert.Equal(t, SimilarityType("cosine"), SimilarityTypeCosine)
	assert.Equal(t, SimilarityType("euclidean"), SimilarityTypeEuclidean)
	assert.Equal(t, SimilarityType("dot_product"), SimilarityTypeDotProduct)
}

func TestCosineSimilarityNormalizedVectors(t *testing.T) {
	// For normalized vectors, cosine similarity equals dot product
	a := []float64{0.6, 0.8}
	b := []float64{0.8, 0.6}

	// Verify they're normalized
	assert.InDelta(t, 1.0, Magnitude(a), 0.0001)
	assert.InDelta(t, 1.0, Magnitude(b), 0.0001)

	cosine, _ := CosineSimilarity(a, b)
	dot, _ := DotProduct(a, b)
	assert.InDelta(t, cosine, dot, 0.0001)
}

func TestEmbeddingResult(t *testing.T) {
	result := EmbeddingResult{
		Embedding:  []float64{0.1, 0.2, 0.3},
		Text:       "test text",
		TokenCount: 2,
	}
	assert.Len(t, result.Embedding, 3)
	assert.Equal(t, "test text", result.Text)
	assert.Equal(t, 2, result.TokenCount)
}

func TestBatchEmbeddingResult(t *testing.T) {
	batch := BatchEmbeddingResult{
		Embeddings: []EmbeddingResult{
			{Embedding: []float64{0.1, 0.2}, Text: "text1"},
			{Embedding: []float64{0.3, 0.4}, Text: "text2"},
		},
		TotalTokens: 10,
	}
	assert.Len(t, batch.Embeddings, 2)
	assert.Equal(t, 10, batch.TotalTokens)
}

func TestVectorOperationsPreserveOriginal(t *testing.T) {
	original := []float64{1, 2, 3}
	originalCopy := make([]float64, len(original))
	copy(originalCopy, original)

	// Normalize should not modify original
	_, _ = Normalize(original)
	assert.Equal(t, originalCopy, original)

	// Scale should not modify original
	_ = Scale(original, 2.0)
	assert.Equal(t, originalCopy, original)

	// Add should not modify original
	_, _ = Add(original, []float64{1, 1, 1})
	assert.Equal(t, originalCopy, original)

	// Subtract should not modify original
	_, _ = Subtract(original, []float64{1, 1, 1})
	assert.Equal(t, originalCopy, original)
}

func TestSpecialCases(t *testing.T) {
	// Very small values
	small := []float64{1e-10, 1e-10, 1e-10}
	mag := Magnitude(small)
	assert.Greater(t, mag, 0.0)

	// Very large values
	large := []float64{1e10, 1e10, 1e10}
	mag = Magnitude(large)
	assert.False(t, math.IsInf(mag, 0))
	assert.False(t, math.IsNaN(mag))
}
