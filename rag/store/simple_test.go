package store

import (
	"context"
	"testing"

	"github.com/aqua777/go-llamaindex/schema"
	"github.com/stretchr/testify/suite"
)

type SimpleVectorStoreTestSuite struct {
	suite.Suite
	ctx   context.Context
	store *SimpleVectorStore
}

func TestSimpleVectorStoreTestSuite(t *testing.T) {
	suite.Run(t, new(SimpleVectorStoreTestSuite))
}

func (s *SimpleVectorStoreTestSuite) SetupTest() {
	s.ctx = context.Background()
	s.store = NewSimpleVectorStore()
}

func (s *SimpleVectorStoreTestSuite) TestDeleteByFilter_RemovesMatchingNodes() {
	// Add nodes with different metadata
	nodes := []schema.Node{
		{ID: "1", Text: "doc1", Metadata: map[string]interface{}{"chat_id": "chat1"}, Embedding: []float32{1, 0}},
		{ID: "2", Text: "doc2", Metadata: map[string]interface{}{"chat_id": "chat1"}, Embedding: []float32{0, 1}},
		{ID: "3", Text: "doc3", Metadata: map[string]interface{}{"chat_id": "chat2"}, Embedding: []float32{1, 1}},
	}
	_, err := s.store.Add(s.ctx, nodes)
	s.NoError(err)

	// Delete nodes matching chat_id=chat1
	filters := schema.NewMetadataFilters(schema.NewMetadataFilter("chat_id", "chat1"))
	count, err := s.store.DeleteByFilter(s.ctx, filters)

	s.NoError(err)
	s.Equal(2, count)

	// Verify remaining nodes
	totalCount, err := s.store.Count(s.ctx, nil)
	s.NoError(err)
	s.Equal(1, totalCount)
}

func (s *SimpleVectorStoreTestSuite) TestDeleteByFilter_EmptyFilterReturnsError() {
	// Empty filters should return error
	count, err := s.store.DeleteByFilter(s.ctx, nil)
	s.Error(err)
	s.Equal(0, count)

	// Empty filters slice should also return error
	count, err = s.store.DeleteByFilter(s.ctx, &schema.MetadataFilters{})
	s.Error(err)
	s.Equal(0, count)
}

func (s *SimpleVectorStoreTestSuite) TestDeleteByFilter_ReturnsCorrectCount() {
	nodes := []schema.Node{
		{ID: "1", Text: "doc1", Metadata: map[string]interface{}{"type": "a"}, Embedding: []float32{1, 0}},
		{ID: "2", Text: "doc2", Metadata: map[string]interface{}{"type": "a"}, Embedding: []float32{0, 1}},
		{ID: "3", Text: "doc3", Metadata: map[string]interface{}{"type": "a"}, Embedding: []float32{1, 1}},
		{ID: "4", Text: "doc4", Metadata: map[string]interface{}{"type": "b"}, Embedding: []float32{0, 0}},
	}
	_, err := s.store.Add(s.ctx, nodes)
	s.NoError(err)

	filters := schema.NewMetadataFilters(schema.NewMetadataFilter("type", "a"))
	count, err := s.store.DeleteByFilter(s.ctx, filters)

	s.NoError(err)
	s.Equal(3, count)
}

func (s *SimpleVectorStoreTestSuite) TestDeleteByFilter_NonMatchingFilterDeletesNothing() {
	nodes := []schema.Node{
		{ID: "1", Text: "doc1", Metadata: map[string]interface{}{"type": "a"}, Embedding: []float32{1, 0}},
		{ID: "2", Text: "doc2", Metadata: map[string]interface{}{"type": "b"}, Embedding: []float32{0, 1}},
	}
	_, err := s.store.Add(s.ctx, nodes)
	s.NoError(err)

	filters := schema.NewMetadataFilters(schema.NewMetadataFilter("type", "nonexistent"))
	count, err := s.store.DeleteByFilter(s.ctx, filters)

	s.NoError(err)
	s.Equal(0, count)

	// Verify all nodes still exist
	totalCount, err := s.store.Count(s.ctx, nil)
	s.NoError(err)
	s.Equal(2, totalCount)
}

func (s *SimpleVectorStoreTestSuite) TestCount_NilFilterReturnsTotalCount() {
	nodes := []schema.Node{
		{ID: "1", Text: "doc1", Metadata: map[string]interface{}{}, Embedding: []float32{1, 0}},
		{ID: "2", Text: "doc2", Metadata: map[string]interface{}{}, Embedding: []float32{0, 1}},
		{ID: "3", Text: "doc3", Metadata: map[string]interface{}{}, Embedding: []float32{1, 1}},
	}
	_, err := s.store.Add(s.ctx, nodes)
	s.NoError(err)

	count, err := s.store.Count(s.ctx, nil)
	s.NoError(err)
	s.Equal(3, count)
}

func (s *SimpleVectorStoreTestSuite) TestCount_WithFilterReturnsFilteredCount() {
	nodes := []schema.Node{
		{ID: "1", Text: "doc1", Metadata: map[string]interface{}{"category": "x"}, Embedding: []float32{1, 0}},
		{ID: "2", Text: "doc2", Metadata: map[string]interface{}{"category": "x"}, Embedding: []float32{0, 1}},
		{ID: "3", Text: "doc3", Metadata: map[string]interface{}{"category": "y"}, Embedding: []float32{1, 1}},
	}
	_, err := s.store.Add(s.ctx, nodes)
	s.NoError(err)

	filters := schema.NewMetadataFilters(schema.NewMetadataFilter("category", "x"))
	count, err := s.store.Count(s.ctx, filters)

	s.NoError(err)
	s.Equal(2, count)
}

func (s *SimpleVectorStoreTestSuite) TestCount_EmptyStoreReturnsZero() {
	count, err := s.store.Count(s.ctx, nil)
	s.NoError(err)
	s.Equal(0, count)
}

func (s *SimpleVectorStoreTestSuite) TestDeleteByFilter_WithNotEqualOperator() {
	nodes := []schema.Node{
		{ID: "1", Text: "doc1", Metadata: map[string]interface{}{"status": "active"}, Embedding: []float32{1, 0}},
		{ID: "2", Text: "doc2", Metadata: map[string]interface{}{"status": "inactive"}, Embedding: []float32{0, 1}},
		{ID: "3", Text: "doc3", Metadata: map[string]interface{}{"status": "active"}, Embedding: []float32{1, 1}},
	}
	_, err := s.store.Add(s.ctx, nodes)
	s.NoError(err)

	// Delete nodes where status != "active"
	filters := schema.NewMetadataFilters(
		schema.NewMetadataFilterWithOp("status", "active", schema.FilterOperatorNe),
	)
	count, err := s.store.DeleteByFilter(s.ctx, filters)

	s.NoError(err)
	s.Equal(1, count)

	// Verify remaining nodes are all active
	remainingCount, err := s.store.Count(s.ctx, nil)
	s.NoError(err)
	s.Equal(2, remainingCount)
}

// TestQuery_UsesGetEmbedding tests the bug fix where Query should use
// GetEmbedding() accessor instead of directly accessing query.Embedding field
func (s *SimpleVectorStoreTestSuite) TestQuery_UsesGetEmbedding() {
	// Add test nodes
	nodes := []schema.Node{
		{ID: "1", Text: "doc1", Embedding: []float32{1.0, 0.0, 0.0}},
		{ID: "2", Text: "doc2", Embedding: []float32{0.0, 1.0, 0.0}},
		{ID: "3", Text: "doc3", Embedding: []float32{0.0, 0.0, 1.0}},
	}
	_, err := s.store.Add(s.ctx, nodes)
	s.NoError(err)

	// Test with QueryEmbedding field set (preferred)
	query := schema.VectorStoreQuery{
		QueryEmbedding: []float32{1.0, 0.0, 0.0},
		Embedding:      nil, // Empty - should not be used
		SimilarityTopK: 2,
	}
	results, err := s.store.Query(s.ctx, query)
	s.NoError(err)
	s.Len(results, 2)
	s.Equal("1", results[0].Node.ID) // Should match doc1

	// Test with only Embedding field set (backward compatibility)
	query2 := schema.VectorStoreQuery{
		QueryEmbedding: nil,
		Embedding:      []float32{0.0, 1.0, 0.0},
		SimilarityTopK: 2,
	}
	results2, err := s.store.Query(s.ctx, query2)
	s.NoError(err)
	s.Len(results2, 2)
	s.Equal("2", results2[0].Node.ID) // Should match doc2

	// Test with empty embedding returns error
	query3 := schema.VectorStoreQuery{
		QueryEmbedding: nil,
		Embedding:      nil,
		SimilarityTopK: 2,
	}
	_, err = s.store.Query(s.ctx, query3)
	s.Error(err)
	s.Contains(err.Error(), "query embedding is empty")
}
