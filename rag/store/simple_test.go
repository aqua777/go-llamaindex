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
		{ID: "1", Text: "doc1", Metadata: map[string]interface{}{"chat_id": "chat1"}, Embedding: []float64{1, 0}},
		{ID: "2", Text: "doc2", Metadata: map[string]interface{}{"chat_id": "chat1"}, Embedding: []float64{0, 1}},
		{ID: "3", Text: "doc3", Metadata: map[string]interface{}{"chat_id": "chat2"}, Embedding: []float64{1, 1}},
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
		{ID: "1", Text: "doc1", Metadata: map[string]interface{}{"type": "a"}, Embedding: []float64{1, 0}},
		{ID: "2", Text: "doc2", Metadata: map[string]interface{}{"type": "a"}, Embedding: []float64{0, 1}},
		{ID: "3", Text: "doc3", Metadata: map[string]interface{}{"type": "a"}, Embedding: []float64{1, 1}},
		{ID: "4", Text: "doc4", Metadata: map[string]interface{}{"type": "b"}, Embedding: []float64{0, 0}},
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
		{ID: "1", Text: "doc1", Metadata: map[string]interface{}{"type": "a"}, Embedding: []float64{1, 0}},
		{ID: "2", Text: "doc2", Metadata: map[string]interface{}{"type": "b"}, Embedding: []float64{0, 1}},
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
		{ID: "1", Text: "doc1", Metadata: map[string]interface{}{}, Embedding: []float64{1, 0}},
		{ID: "2", Text: "doc2", Metadata: map[string]interface{}{}, Embedding: []float64{0, 1}},
		{ID: "3", Text: "doc3", Metadata: map[string]interface{}{}, Embedding: []float64{1, 1}},
	}
	_, err := s.store.Add(s.ctx, nodes)
	s.NoError(err)

	count, err := s.store.Count(s.ctx, nil)
	s.NoError(err)
	s.Equal(3, count)
}

func (s *SimpleVectorStoreTestSuite) TestCount_WithFilterReturnsFilteredCount() {
	nodes := []schema.Node{
		{ID: "1", Text: "doc1", Metadata: map[string]interface{}{"category": "x"}, Embedding: []float64{1, 0}},
		{ID: "2", Text: "doc2", Metadata: map[string]interface{}{"category": "x"}, Embedding: []float64{0, 1}},
		{ID: "3", Text: "doc3", Metadata: map[string]interface{}{"category": "y"}, Embedding: []float64{1, 1}},
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
		{ID: "1", Text: "doc1", Metadata: map[string]interface{}{"status": "active"}, Embedding: []float64{1, 0}},
		{ID: "2", Text: "doc2", Metadata: map[string]interface{}{"status": "inactive"}, Embedding: []float64{0, 1}},
		{ID: "3", Text: "doc3", Metadata: map[string]interface{}{"status": "active"}, Embedding: []float64{1, 1}},
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
