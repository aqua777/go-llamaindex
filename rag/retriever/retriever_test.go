package retriever

import (
	"context"
	"testing"

	"github.com/aqua777/go-llamaindex/schema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// MockRetriever is a mock retriever for testing.
type MockRetriever struct {
	Nodes []schema.NodeWithScore
	Err   error
}

func (m *MockRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	if m.Err != nil {
		return nil, m.Err
	}
	return m.Nodes, nil
}

func createTestNode(id, text string, score float64) schema.NodeWithScore {
	node := schema.NewTextNode(text)
	node.ID = id
	return schema.NodeWithScore{
		Node:  *node,
		Score: score,
	}
}

func TestBaseRetriever(t *testing.T) {
	br := NewBaseRetriever()
	assert.NotNil(t, br.ObjectMap)
	assert.NotNil(t, br.BasePromptMixin)
}

func TestBaseRetrieverWithOptions(t *testing.T) {
	objectMap := map[string]interface{}{"key": "value"}
	br := NewBaseRetrieverWithOptions(
		WithObjectMap(objectMap),
		WithVerbose(true),
	)

	assert.Equal(t, objectMap, br.ObjectMap)
	assert.True(t, br.Verbose)
}

func TestBaseRetrieverAddGetObject(t *testing.T) {
	br := NewBaseRetriever()

	mockRetriever := &MockRetriever{}
	br.AddObject("test-index", mockRetriever)

	obj := br.GetObject("test-index")
	assert.Equal(t, mockRetriever, obj)
}

func TestFusionRetrieverSimple(t *testing.T) {
	ctx := context.Background()

	// Create mock retrievers with overlapping results
	mock1 := &MockRetriever{
		Nodes: []schema.NodeWithScore{
			createTestNode("node1", "content 1", 0.9),
			createTestNode("node2", "content 2", 0.8),
		},
	}
	mock2 := &MockRetriever{
		Nodes: []schema.NodeWithScore{
			createTestNode("node2", "content 2", 0.85), // Duplicate
			createTestNode("node3", "content 3", 0.7),
		},
	}

	fr := NewFusionRetriever([]Retriever{mock1, mock2}, WithFusionMode(FusionModeSimple))

	query := schema.QueryBundle{QueryString: "test query"}
	results, err := fr.Retrieve(ctx, query)
	require.NoError(t, err)

	// Should have 3 unique nodes
	assert.Len(t, results, 3)

	// Results should be sorted by score
	assert.GreaterOrEqual(t, results[0].Score, results[1].Score)
}

func TestFusionRetrieverReciprocalRank(t *testing.T) {
	ctx := context.Background()

	mock1 := &MockRetriever{
		Nodes: []schema.NodeWithScore{
			createTestNode("node1", "content 1", 0.9),
			createTestNode("node2", "content 2", 0.8),
		},
	}
	mock2 := &MockRetriever{
		Nodes: []schema.NodeWithScore{
			createTestNode("node2", "content 2", 0.95),
			createTestNode("node3", "content 3", 0.7),
		},
	}

	fr := NewFusionRetriever(
		[]Retriever{mock1, mock2},
		WithFusionMode(FusionModeReciprocalRank),
		WithSimilarityTopK(5),
	)

	query := schema.QueryBundle{QueryString: "test query"}
	results, err := fr.Retrieve(ctx, query)
	require.NoError(t, err)

	assert.Len(t, results, 3)
	// node2 should have highest score (appears in both)
}

func TestFusionRetrieverWithWeights(t *testing.T) {
	ctx := context.Background()

	mock1 := &MockRetriever{
		Nodes: []schema.NodeWithScore{
			createTestNode("node1", "content 1", 1.0),
		},
	}
	mock2 := &MockRetriever{
		Nodes: []schema.NodeWithScore{
			createTestNode("node2", "content 2", 1.0),
		},
	}

	fr := NewFusionRetriever(
		[]Retriever{mock1, mock2},
		WithFusionMode(FusionModeRelativeScore),
		WithRetrieverWeights([]float64{0.7, 0.3}),
	)

	query := schema.QueryBundle{QueryString: "test query"}
	results, err := fr.Retrieve(ctx, query)
	require.NoError(t, err)

	assert.Len(t, results, 2)
}

func TestRouterRetriever(t *testing.T) {
	ctx := context.Background()

	mock1 := &MockRetriever{
		Nodes: []schema.NodeWithScore{
			createTestNode("node1", "content 1", 0.9),
		},
	}
	mock2 := &MockRetriever{
		Nodes: []schema.NodeWithScore{
			createTestNode("node2", "content 2", 0.8),
		},
	}

	tools := []*RetrieverTool{
		NewRetrieverTool(mock1, "retriever1", "First retriever"),
		NewRetrieverTool(mock2, "retriever2", "Second retriever"),
	}

	rr := NewRouterRetriever(tools)

	query := schema.QueryBundle{QueryString: "test query"}
	results, err := rr.Retrieve(ctx, query)
	require.NoError(t, err)

	// SimpleSelector selects all, so should have both nodes
	assert.Len(t, results, 2)
}

func TestRouterRetrieverWithSingleSelector(t *testing.T) {
	ctx := context.Background()

	mock1 := &MockRetriever{
		Nodes: []schema.NodeWithScore{
			createTestNode("node1", "content 1", 0.9),
		},
	}
	mock2 := &MockRetriever{
		Nodes: []schema.NodeWithScore{
			createTestNode("node2", "content 2", 0.8),
		},
	}

	tools := []*RetrieverTool{
		NewRetrieverTool(mock1, "retriever1", "First retriever"),
		NewRetrieverTool(mock2, "retriever2", "Second retriever"),
	}

	rr := NewRouterRetriever(tools, WithSelector(&SingleSelector{}))

	query := schema.QueryBundle{QueryString: "test query"}
	results, err := rr.Retrieve(ctx, query)
	require.NoError(t, err)

	// SingleSelector selects only first, so should have 1 node
	assert.Len(t, results, 1)
	assert.Equal(t, "node1", results[0].Node.ID)
}

func TestRetrieverTool(t *testing.T) {
	mock := &MockRetriever{}
	tool := NewRetrieverTool(mock, "test", "Test retriever")

	assert.Equal(t, mock, tool.Retriever)
	assert.Equal(t, "test", tool.Name)
	assert.Equal(t, "Test retriever", tool.Description)
}

func TestFusionModes(t *testing.T) {
	assert.Equal(t, FusionMode("reciprocal_rerank"), FusionModeReciprocalRank)
	assert.Equal(t, FusionMode("relative_score"), FusionModeRelativeScore)
	assert.Equal(t, FusionMode("dist_based_score"), FusionModeDistBasedScore)
	assert.Equal(t, FusionMode("simple"), FusionModeSimple)
}

func TestHandleRecursiveRetrieval(t *testing.T) {
	ctx := context.Background()
	br := NewBaseRetriever()

	// Add a mock retriever to object map
	mockRetriever := &MockRetriever{
		Nodes: []schema.NodeWithScore{
			createTestNode("nested-node", "nested content", 0.95),
		},
	}
	br.AddObject("index-node-id", mockRetriever)

	// Create nodes including one that references the mock retriever
	nodes := []schema.NodeWithScore{
		createTestNode("index-node-id", "index node", 0.9),
		createTestNode("regular-node", "regular content", 0.8),
	}

	query := schema.QueryBundle{QueryString: "test"}
	result, err := br.HandleRecursiveRetrieval(ctx, query, nodes)
	require.NoError(t, err)

	// Should have nested-node (from mock) and regular-node
	assert.Len(t, result, 2)
}

func TestVectorRetrieverOptions(t *testing.T) {
	vr := &VectorRetriever{
		BaseRetriever: NewBaseRetriever(),
		TopK:          10,
		Mode:          schema.QueryModeDefault,
	}

	WithTopK(20)(vr)
	assert.Equal(t, 20, vr.TopK)

	WithQueryMode(schema.QueryModeHybrid)(vr)
	assert.Equal(t, schema.QueryModeHybrid, vr.Mode)
}

func TestSelectorResult(t *testing.T) {
	result := &SelectorResult{
		Indices: []int{0, 2},
		Reasons: []string{"reason1", "reason2"},
	}

	assert.Len(t, result.Indices, 2)
	assert.Len(t, result.Reasons, 2)
}

func TestSimpleSelector(t *testing.T) {
	ctx := context.Background()
	selector := &SimpleSelector{}

	tools := []*RetrieverTool{
		{Name: "tool1"},
		{Name: "tool2"},
		{Name: "tool3"},
	}

	result, err := selector.Select(ctx, tools, schema.QueryBundle{})
	require.NoError(t, err)

	assert.Len(t, result.Indices, 3)
	assert.Equal(t, []int{0, 1, 2}, result.Indices)
}

func TestSingleSelector(t *testing.T) {
	ctx := context.Background()
	selector := &SingleSelector{}

	tools := []*RetrieverTool{
		{Name: "tool1"},
		{Name: "tool2"},
	}

	result, err := selector.Select(ctx, tools, schema.QueryBundle{})
	require.NoError(t, err)

	assert.Len(t, result.Indices, 1)
	assert.Equal(t, 0, result.Indices[0])
}

func TestSingleSelectorEmpty(t *testing.T) {
	ctx := context.Background()
	selector := &SingleSelector{}

	_, err := selector.Select(ctx, []*RetrieverTool{}, schema.QueryBundle{})
	assert.Error(t, err)
}
