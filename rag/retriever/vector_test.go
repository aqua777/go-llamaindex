package retriever

import (
	"context"
	"errors"
	"testing"

	"github.com/aqua777/go-llamaindex/schema"
	"github.com/stretchr/testify/suite"
)

// captureVectorStore records the last VectorStoreQuery it receives.
type captureVectorStore struct {
	lastQuery schema.VectorStoreQuery
	err       error
}

func (c *captureVectorStore) Add(ctx context.Context, nodes []schema.Node) ([]string, error) {
	return nil, nil
}

func (c *captureVectorStore) Query(ctx context.Context, query schema.VectorStoreQuery) ([]schema.NodeWithScore, error) {
	c.lastQuery = query
	return nil, c.err
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

// errEmbedding always returns an error from GetQueryEmbedding.
type errEmbedding struct{ embedErr error }

func (e *errEmbedding) GetTextEmbedding(ctx context.Context, text string) ([]float32, error) {
	return nil, nil
}
func (e *errEmbedding) GetQueryEmbedding(ctx context.Context, query string) ([]float32, error) {
	return nil, e.embedErr
}

func (s *VectorRetrieverQueryStrTestSuite) TestRetrieve_EmbeddingFailure_ReturnsError() {
	embErr := errors.New("embedding API unavailable")
	vr := NewVectorRetriever(&captureVectorStore{}, &errEmbedding{embedErr: embErr})

	_, err := vr.Retrieve(s.ctx, schema.QueryBundle{QueryString: "test"})
	s.Error(err)
	s.ErrorContains(err, "embedding API unavailable")
}

func (s *VectorRetrieverQueryStrTestSuite) TestRetrieve_StoreQueryFailure_ReturnsError() {
	storeErr := errors.New("store unavailable")
	capture := &captureVectorStore{err: storeErr}
	emb := &mockEmbedding{emb: []float32{1, 0, 0}}
	vr := NewVectorRetriever(capture, emb)

	_, err := vr.Retrieve(s.ctx, schema.QueryBundle{QueryString: "test"})
	s.Error(err)
	s.ErrorContains(err, "store unavailable")
}
