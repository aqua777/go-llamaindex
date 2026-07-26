package store

import (
	"context"
	"testing"

	"github.com/aqua777/go-llamaindex/schema"
	"github.com/stretchr/testify/suite"
)

// sparseHybridTestNodes returns a deterministic 5-node corpus for sparse/hybrid tests.
func sparseHybridTestNodes() []schema.Node {
	entries := []struct {
		id   string
		text string
		emb  []float32
	}{
		{"n1", "machine learning algorithms optimize models", []float32{1, 0, 0, 0}},
		{"n2", "natural language processing understands text", []float32{0, 1, 0, 0}},
		{"n3", "database indexing improves query performance", []float32{0, 0, 1, 0}},
		{"n4", "neural networks learn from training data", []float32{0, 0, 0, 1}},
		{"n5", "reinforcement learning trains agents with rewards", []float32{0.5, 0.5, 0, 0}},
	}
	nodes := make([]schema.Node, len(entries))
	for i, e := range entries {
		n := schema.NewTextNode(e.text)
		n.ID = e.id
		n.Embedding = e.emb
		nodes[i] = *n
	}
	return nodes
}

type SparseHybridTestSuite struct {
	suite.Suite
	ctx   context.Context
	store *SimpleVectorStore
}

func TestSparseHybridTestSuite(t *testing.T) {
	suite.Run(t, new(SparseHybridTestSuite))
}

func (s *SparseHybridTestSuite) SetupTest() {
	s.ctx = context.Background()
	s.store = NewSimpleVectorStore()
	nodes := sparseHybridTestNodes()
	_, err := s.store.Add(s.ctx, nodes)
	s.NoError(err)
}

// --- Sparse mode tests ---

func (s *SparseHybridTestSuite) TestQuerySparse_ReturnsNodesByBM25Rank() {
	q := schema.VectorStoreQuery{
		QueryStr:       "machine learning",
		SimilarityTopK: 3,
		Mode:           schema.QueryModeSparse,
	}
	results, err := s.store.Query(s.ctx, q)
	s.NoError(err)
	s.NotEmpty(results)
	// Top result must contain the query terms.
	s.Contains(results[0].Node.Text, "machine learning")
}

func (s *SparseHybridTestSuite) TestQuerySparse_EmptyQueryStrReturnsError() {
	q := schema.VectorStoreQuery{
		QueryStr:       "",
		SimilarityTopK: 3,
		Mode:           schema.QueryModeSparse,
	}
	_, err := s.store.Query(s.ctx, q)
	s.Error(err)
}

func (s *SparseHybridTestSuite) TestQuerySparse_AfterAddRefits() {
	extra := schema.NewTextNode("compiler optimization techniques")
	extra.ID = "n6"
	extra.Embedding = []float32{0, 0, 0.5, 0.5}
	_, err := s.store.Add(s.ctx, []schema.Node{*extra})
	s.NoError(err)

	q := schema.VectorStoreQuery{
		QueryStr:       "compiler optimization",
		SimilarityTopK: 3,
		Mode:           schema.QueryModeSparse,
	}
	results, err := s.store.Query(s.ctx, q)
	s.NoError(err)
	s.NotEmpty(results)
	s.Equal("n6", results[0].Node.ID)
}

func (s *SparseHybridTestSuite) TestQuerySparse_AfterDeleteRefits() {
	err := s.store.Delete(s.ctx, "n1")
	s.NoError(err)

	q := schema.VectorStoreQuery{
		QueryStr:       "machine learning",
		SimilarityTopK: 5,
		Mode:           schema.QueryModeSparse,
	}
	results, err := s.store.Query(s.ctx, q)
	s.NoError(err)

	for _, r := range results {
		s.NotEqual("n1", r.Node.ID)
	}
}

// --- Hybrid mode tests ---

func alpha(v float64) *float64 { return &v }

func (s *SparseHybridTestSuite) TestQueryHybrid_ReturnsCombinedResults() {
	q := schema.VectorStoreQuery{
		QueryStr:       "machine learning",
		QueryEmbedding: []float32{1, 0, 0, 0},
		SimilarityTopK: 3,
		Mode:           schema.QueryModeHybrid,
	}
	results, err := s.store.Query(s.ctx, q)
	s.NoError(err)
	s.NotEmpty(results)
}

func (s *SparseHybridTestSuite) TestQueryHybrid_Alpha1_TopResultMatchesDense() {
	denseQ := schema.VectorStoreQuery{
		QueryEmbedding: []float32{1, 0, 0, 0},
		SimilarityTopK: 5,
		Mode:           schema.QueryModeDefault,
	}
	denseResults, err := s.store.Query(s.ctx, denseQ)
	s.NoError(err)

	hybridQ := schema.VectorStoreQuery{
		QueryStr:       "machine learning",
		QueryEmbedding: []float32{1, 0, 0, 0},
		SimilarityTopK: 5,
		Mode:           schema.QueryModeHybrid,
		Alpha:          alpha(1.0),
	}
	hybridResults, err := s.store.Query(s.ctx, hybridQ)
	s.NoError(err)

	s.Equal(len(denseResults), len(hybridResults))
	s.Equal(denseResults[0].Node.ID, hybridResults[0].Node.ID)
}

func (s *SparseHybridTestSuite) TestQueryHybrid_Alpha0_TopResultMatchesSparse() {
	sparseQ := schema.VectorStoreQuery{
		QueryStr:       "machine learning",
		SimilarityTopK: 5,
		Mode:           schema.QueryModeSparse,
	}
	sparseResults, err := s.store.Query(s.ctx, sparseQ)
	s.NoError(err)

	hybridQ := schema.VectorStoreQuery{
		QueryStr:       "machine learning",
		QueryEmbedding: []float32{1, 0, 0, 0},
		SimilarityTopK: 5,
		Mode:           schema.QueryModeHybrid,
		Alpha:          alpha(0.0),
	}
	hybridResults, err := s.store.Query(s.ctx, hybridQ)
	s.NoError(err)

	s.Equal(sparseResults[0].Node.ID, hybridResults[0].Node.ID)
}

func (s *SparseHybridTestSuite) TestQueryHybrid_NilAlphaDefaults() {
	q := schema.VectorStoreQuery{
		QueryStr:       "machine learning",
		QueryEmbedding: []float32{1, 0, 0, 0},
		SimilarityTopK: 3,
		Mode:           schema.QueryModeHybrid,
		Alpha:          nil, // should default to 0.5
	}
	results, err := s.store.Query(s.ctx, q)
	s.NoError(err)
	s.NotEmpty(results)
}

func (s *SparseHybridTestSuite) TestQueryHybrid_InvalidAlphaReturnsError() {
	q := schema.VectorStoreQuery{
		QueryStr:       "machine learning",
		QueryEmbedding: []float32{1, 0, 0, 0},
		SimilarityTopK: 3,
		Mode:           schema.QueryModeHybrid,
		Alpha:          alpha(1.5),
	}
	_, err := s.store.Query(s.ctx, q)
	s.Error(err)
}

func (s *SparseHybridTestSuite) TestQueryHybrid_EmptyEmbeddingReturnsError() {
	q := schema.VectorStoreQuery{
		QueryStr:       "machine learning",
		QueryEmbedding: nil,
		Embedding:      nil,
		SimilarityTopK: 3,
		Mode:           schema.QueryModeHybrid,
	}
	_, err := s.store.Query(s.ctx, q)
	s.Error(err)
}

func (s *SparseHybridTestSuite) TestQueryHybrid_EmptyQueryStrReturnsError() {
	q := schema.VectorStoreQuery{
		QueryStr:       "",
		QueryEmbedding: []float32{1, 0, 0, 0},
		SimilarityTopK: 3,
		Mode:           schema.QueryModeHybrid,
	}
	_, err := s.store.Query(s.ctx, q)
	s.Error(err)
}

func (s *SparseHybridTestSuite) TestQueryUnsupportedModeReturnsError() {
	q := schema.VectorStoreQuery{
		QueryEmbedding: []float32{1, 0, 0, 0},
		SimilarityTopK: 3,
		Mode:           schema.QueryModeMMR,
	}
	_, err := s.store.Query(s.ctx, q)
	s.Error(err)
	s.Contains(err.Error(), "unsupported query mode")
}

func (s *SparseHybridTestSuite) TestExistingDenseTestsStillPass() {
	q := schema.VectorStoreQuery{
		QueryEmbedding: []float32{1, 0, 0, 0},
		SimilarityTopK: 3,
		Mode:           schema.QueryModeDefault,
	}
	results, err := s.store.Query(s.ctx, q)
	s.NoError(err)
	s.NotEmpty(results)
	s.Equal("n1", results[0].Node.ID)
}
