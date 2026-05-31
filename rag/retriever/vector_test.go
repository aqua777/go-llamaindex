package retriever

import (
	"context"
	"testing"

	"github.com/aqua777/go-llamaindex/schema"
	"github.com/stretchr/testify/suite"
)

// captureVectorStore records the last VectorStoreQuery it receives.
type captureVectorStore struct {
	lastQuery schema.VectorStoreQuery
}

func (c *captureVectorStore) Add(ctx context.Context, nodes []schema.Node) ([]string, error) {
	return nil, nil
}

func (c *captureVectorStore) Query(ctx context.Context, query schema.VectorStoreQuery) ([]schema.NodeWithScore, error) {
	c.lastQuery = query
	return nil, nil
}

func (c *captureVectorStore) Delete(ctx context.Context, refDocID string) error {
	return nil
}

func (c *captureVectorStore) PersistPath() string {
	return ""
}

// VectorRetrieverQueryStrTestSuite tests QueryStr forwarding in VectorRetriever.
type VectorRetrieverQueryStrTestSuite struct {
	suite.Suite
	ctx context.Context
}

func TestVectorRetrieverQueryStrTestSuite(t *testing.T) {
	suite.Run(t, new(VectorRetrieverQueryStrTestSuite))
}

func (s *VectorRetrieverQueryStrTestSuite) SetupTest() {
	s.ctx = context.Background()
}

// mockEmbedding returns a fixed embedding regardless of input.
type mockEmbedding struct {
	emb []float32
}

func (m *mockEmbedding) GetTextEmbedding(ctx context.Context, text string) ([]float32, error) {
	return m.emb, nil
}

func (m *mockEmbedding) GetQueryEmbedding(ctx context.Context, query string) ([]float32, error) {
	return m.emb, nil
}

func (s *VectorRetrieverQueryStrTestSuite) TestHybridMode_ForwardsQueryStr() {
	capture := &captureVectorStore{}
	emb := &mockEmbedding{emb: []float32{1, 0, 0}}
	vr := NewVectorRetriever(capture, emb, WithQueryMode(schema.QueryModeHybrid))

	query := schema.QueryBundle{QueryString: "machine learning"}
	_, err := vr.Retrieve(s.ctx, query)
	s.NoError(err)
	s.Equal("machine learning", capture.lastQuery.QueryStr,
		"QueryStr must be forwarded in hybrid mode")
}

func (s *VectorRetrieverQueryStrTestSuite) TestSparseMode_ForwardsQueryStr() {
	capture := &captureVectorStore{}
	emb := &mockEmbedding{emb: []float32{1, 0, 0}}
	vr := NewVectorRetriever(capture, emb, WithQueryMode(schema.QueryModeSparse))

	query := schema.QueryBundle{QueryString: "database indexing"}
	_, err := vr.Retrieve(s.ctx, query)
	s.NoError(err)
	s.Equal("database indexing", capture.lastQuery.QueryStr,
		"QueryStr must be forwarded in sparse mode")
}

func (s *VectorRetrieverQueryStrTestSuite) TestDefaultMode_DoesNotForwardQueryStr() {
	capture := &captureVectorStore{}
	emb := &mockEmbedding{emb: []float32{1, 0, 0}}
	vr := NewVectorRetriever(capture, emb, WithQueryMode(schema.QueryModeDefault))

	query := schema.QueryBundle{QueryString: "anything"}
	_, err := vr.Retrieve(s.ctx, query)
	s.NoError(err)
	s.Equal("", capture.lastQuery.QueryStr,
		"QueryStr must NOT be forwarded in default mode for backward compat")
}
