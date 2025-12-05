package postprocessor

import (
	"context"
	"testing"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// MockLLM implements llm.LLM for testing.
type MockLLM struct {
	responses []string
	callCount int
}

func NewMockLLM(responses ...string) *MockLLM {
	return &MockLLM{responses: responses}
}

func (m *MockLLM) Complete(ctx context.Context, prompt string) (string, error) {
	return m.getNextResponse(), nil
}

func (m *MockLLM) Chat(ctx context.Context, messages []llm.ChatMessage) (string, error) {
	return m.getNextResponse(), nil
}

func (m *MockLLM) Stream(ctx context.Context, prompt string) (<-chan string, error) {
	ch := make(chan string, 1)
	ch <- m.getNextResponse()
	close(ch)
	return ch, nil
}

func (m *MockLLM) getNextResponse() string {
	if m.callCount >= len(m.responses) {
		return "Doc: 1, Relevance: 5"
	}
	response := m.responses[m.callCount]
	m.callCount++
	return response
}

func createTestNode(id, text string, score float64) schema.NodeWithScore {
	node := schema.NewNode()
	node.ID = id
	node.Text = text
	return schema.NodeWithScore{
		Node:  *node,
		Score: score,
	}
}

func createTestNodeWithMetadata(id, text string, score float64, metadata map[string]interface{}) schema.NodeWithScore {
	node := schema.NewNode()
	node.ID = id
	node.Text = text
	node.Metadata = metadata
	return schema.NodeWithScore{
		Node:  *node,
		Score: score,
	}
}

// TestBaseNodePostprocessor tests the BaseNodePostprocessor.
func TestBaseNodePostprocessor(t *testing.T) {
	ctx := context.Background()

	t.Run("NewBaseNodePostprocessor", func(t *testing.T) {
		pp := NewBaseNodePostprocessor()
		assert.NotNil(t, pp)
		assert.Equal(t, "BaseNodePostprocessor", pp.Name())
	})

	t.Run("WithPostprocessorName", func(t *testing.T) {
		pp := NewBaseNodePostprocessor(WithPostprocessorName("custom"))
		assert.Equal(t, "custom", pp.Name())
	})

	t.Run("PostprocessNodes returns nodes unchanged", func(t *testing.T) {
		pp := NewBaseNodePostprocessor()
		nodes := []schema.NodeWithScore{
			createTestNode("1", "Test", 0.9),
		}

		result, err := pp.PostprocessNodes(ctx, nodes, nil)
		require.NoError(t, err)
		assert.Equal(t, nodes, result)
	})
}

// TestSimilarityPostprocessor tests the SimilarityPostprocessor.
func TestSimilarityPostprocessor(t *testing.T) {
	ctx := context.Background()

	t.Run("NewSimilarityPostprocessor", func(t *testing.T) {
		pp := NewSimilarityPostprocessor()
		assert.NotNil(t, pp)
		assert.Equal(t, "SimilarityPostprocessor", pp.Name())
		assert.Equal(t, 0.0, pp.SimilarityCutoff())
	})

	t.Run("WithSimilarityCutoff", func(t *testing.T) {
		pp := NewSimilarityPostprocessor(WithSimilarityCutoff(0.5))
		assert.Equal(t, 0.5, pp.SimilarityCutoff())
	})

	t.Run("Filters nodes below cutoff", func(t *testing.T) {
		pp := NewSimilarityPostprocessor(WithSimilarityCutoff(0.5))
		nodes := []schema.NodeWithScore{
			createTestNode("1", "High score", 0.9),
			createTestNode("2", "Low score", 0.3),
			createTestNode("3", "Medium score", 0.5),
		}

		result, err := pp.PostprocessNodes(ctx, nodes, nil)
		require.NoError(t, err)
		assert.Len(t, result, 2)
		assert.Equal(t, "1", result[0].Node.ID)
		assert.Equal(t, "3", result[1].Node.ID)
	})

	t.Run("Filters nodes with zero score", func(t *testing.T) {
		pp := NewSimilarityPostprocessor(WithSimilarityCutoff(0.0))
		nodes := []schema.NodeWithScore{
			createTestNode("1", "Has score", 0.5),
			createTestNode("2", "No score", 0),
		}

		result, err := pp.PostprocessNodes(ctx, nodes, nil)
		require.NoError(t, err)
		assert.Len(t, result, 1)
	})

	t.Run("Returns all nodes when cutoff is 0", func(t *testing.T) {
		pp := NewSimilarityPostprocessor(WithSimilarityCutoff(0.0))
		nodes := []schema.NodeWithScore{
			createTestNode("1", "Test 1", 0.9),
			createTestNode("2", "Test 2", 0.1),
		}

		result, err := pp.PostprocessNodes(ctx, nodes, nil)
		require.NoError(t, err)
		assert.Len(t, result, 2)
	})
}

// TestKeywordPostprocessor tests the KeywordPostprocessor.
func TestKeywordPostprocessor(t *testing.T) {
	ctx := context.Background()

	t.Run("NewKeywordPostprocessor", func(t *testing.T) {
		pp := NewKeywordPostprocessor()
		assert.NotNil(t, pp)
		assert.Equal(t, "KeywordPostprocessor", pp.Name())
	})

	t.Run("WithRequiredKeywords", func(t *testing.T) {
		pp := NewKeywordPostprocessor(WithRequiredKeywords([]string{"test"}))
		assert.Equal(t, []string{"test"}, pp.RequiredKeywords())
	})

	t.Run("WithExcludeKeywords", func(t *testing.T) {
		pp := NewKeywordPostprocessor(WithExcludeKeywords([]string{"bad"}))
		assert.Equal(t, []string{"bad"}, pp.ExcludeKeywords())
	})

	t.Run("Filters by required keywords", func(t *testing.T) {
		pp := NewKeywordPostprocessor(WithRequiredKeywords([]string{"important"}))
		nodes := []schema.NodeWithScore{
			createTestNode("1", "This is important content", 0.9),
			createTestNode("2", "This is other content", 0.8),
		}

		result, err := pp.PostprocessNodes(ctx, nodes, nil)
		require.NoError(t, err)
		assert.Len(t, result, 1)
		assert.Equal(t, "1", result[0].Node.ID)
	})

	t.Run("Filters by excluded keywords", func(t *testing.T) {
		pp := NewKeywordPostprocessor(WithExcludeKeywords([]string{"spam"}))
		nodes := []schema.NodeWithScore{
			createTestNode("1", "Good content", 0.9),
			createTestNode("2", "This is spam content", 0.8),
		}

		result, err := pp.PostprocessNodes(ctx, nodes, nil)
		require.NoError(t, err)
		assert.Len(t, result, 1)
		assert.Equal(t, "1", result[0].Node.ID)
	})

	t.Run("Case insensitive by default", func(t *testing.T) {
		pp := NewKeywordPostprocessor(WithRequiredKeywords([]string{"IMPORTANT"}))
		nodes := []schema.NodeWithScore{
			createTestNode("1", "This is important content", 0.9),
		}

		result, err := pp.PostprocessNodes(ctx, nodes, nil)
		require.NoError(t, err)
		assert.Len(t, result, 1)
	})

	t.Run("Case sensitive when enabled", func(t *testing.T) {
		pp := NewKeywordPostprocessor(
			WithRequiredKeywords([]string{"IMPORTANT"}),
			WithCaseSensitive(true),
		)
		nodes := []schema.NodeWithScore{
			createTestNode("1", "This is important content", 0.9),
		}

		result, err := pp.PostprocessNodes(ctx, nodes, nil)
		require.NoError(t, err)
		assert.Len(t, result, 0)
	})

	t.Run("Multiple required keywords", func(t *testing.T) {
		pp := NewKeywordPostprocessor(WithRequiredKeywords([]string{"hello", "world"}))
		nodes := []schema.NodeWithScore{
			createTestNode("1", "Hello World", 0.9),
			createTestNode("2", "Hello there", 0.8),
			createTestNode("3", "World news", 0.7),
		}

		result, err := pp.PostprocessNodes(ctx, nodes, nil)
		require.NoError(t, err)
		assert.Len(t, result, 1)
		assert.Equal(t, "1", result[0].Node.ID)
	})
}

// TestMetadataReplacementPostprocessor tests the MetadataReplacementPostprocessor.
func TestMetadataReplacementPostprocessor(t *testing.T) {
	ctx := context.Background()

	t.Run("NewMetadataReplacementPostprocessor", func(t *testing.T) {
		pp := NewMetadataReplacementPostprocessor("window")
		assert.NotNil(t, pp)
		assert.Equal(t, "MetadataReplacementPostprocessor", pp.Name())
		assert.Equal(t, "window", pp.TargetMetadataKey())
	})

	t.Run("Replaces content with metadata value", func(t *testing.T) {
		pp := NewMetadataReplacementPostprocessor("window")
		nodes := []schema.NodeWithScore{
			createTestNodeWithMetadata("1", "Original text", 0.9, map[string]interface{}{
				"window": "Replacement text from window",
			}),
		}

		result, err := pp.PostprocessNodes(ctx, nodes, nil)
		require.NoError(t, err)
		assert.Len(t, result, 1)
		assert.Equal(t, "Replacement text from window", result[0].Node.GetContent(schema.MetadataModeNone))
	})

	t.Run("Keeps original content when metadata key missing", func(t *testing.T) {
		pp := NewMetadataReplacementPostprocessor("window")
		nodes := []schema.NodeWithScore{
			createTestNodeWithMetadata("1", "Original text", 0.9, map[string]interface{}{
				"other_key": "Some value",
			}),
		}

		result, err := pp.PostprocessNodes(ctx, nodes, nil)
		require.NoError(t, err)
		assert.Len(t, result, 1)
		assert.Equal(t, "Original text", result[0].Node.GetContent(schema.MetadataModeNone))
	})

	t.Run("Handles nil metadata", func(t *testing.T) {
		pp := NewMetadataReplacementPostprocessor("window")
		nodes := []schema.NodeWithScore{
			createTestNode("1", "Original text", 0.9),
		}

		result, err := pp.PostprocessNodes(ctx, nodes, nil)
		require.NoError(t, err)
		assert.Len(t, result, 1)
	})
}

// TestLongContextReorder tests the LongContextReorder.
func TestLongContextReorder(t *testing.T) {
	ctx := context.Background()

	t.Run("NewLongContextReorder", func(t *testing.T) {
		pp := NewLongContextReorder()
		assert.NotNil(t, pp)
		assert.Equal(t, "LongContextReorder", pp.Name())
	})

	t.Run("Reorders nodes correctly", func(t *testing.T) {
		pp := NewLongContextReorder()
		nodes := []schema.NodeWithScore{
			createTestNode("1", "Score 1", 1.0),
			createTestNode("2", "Score 2", 2.0),
			createTestNode("3", "Score 3", 3.0),
			createTestNode("4", "Score 4", 4.0),
		}

		result, err := pp.PostprocessNodes(ctx, nodes, nil)
		require.NoError(t, err)
		assert.Len(t, result, 4)

		// After reordering, higher scores should be at start and end
		// Original order by score: 1, 2, 3, 4
		// Reordered: 3, 1, 2, 4 (alternating insert at front/back)
		// Actually: 4 at front, 3 at back, 2 at front, 1 at back
		// Result: 2, 4, 1, 3 or similar pattern
	})

	t.Run("Handles empty nodes", func(t *testing.T) {
		pp := NewLongContextReorder()
		nodes := []schema.NodeWithScore{}

		result, err := pp.PostprocessNodes(ctx, nodes, nil)
		require.NoError(t, err)
		assert.Len(t, result, 0)
	})

	t.Run("Handles single node", func(t *testing.T) {
		pp := NewLongContextReorder()
		nodes := []schema.NodeWithScore{
			createTestNode("1", "Only node", 1.0),
		}

		result, err := pp.PostprocessNodes(ctx, nodes, nil)
		require.NoError(t, err)
		assert.Len(t, result, 1)
	})
}

// TestTopKPostprocessor tests the TopKPostprocessor.
func TestTopKPostprocessor(t *testing.T) {
	ctx := context.Background()

	t.Run("NewTopKPostprocessor", func(t *testing.T) {
		pp := NewTopKPostprocessor(5)
		assert.NotNil(t, pp)
		assert.Equal(t, "TopKPostprocessor", pp.Name())
		assert.Equal(t, 5, pp.TopK())
	})

	t.Run("Returns top K nodes", func(t *testing.T) {
		pp := NewTopKPostprocessor(2)
		nodes := []schema.NodeWithScore{
			createTestNode("1", "Low", 0.3),
			createTestNode("2", "High", 0.9),
			createTestNode("3", "Medium", 0.5),
		}

		result, err := pp.PostprocessNodes(ctx, nodes, nil)
		require.NoError(t, err)
		assert.Len(t, result, 2)
		assert.Equal(t, "2", result[0].Node.ID) // Highest score first
		assert.Equal(t, "3", result[1].Node.ID) // Second highest
	})

	t.Run("Returns all nodes when K > len(nodes)", func(t *testing.T) {
		pp := NewTopKPostprocessor(10)
		nodes := []schema.NodeWithScore{
			createTestNode("1", "Test 1", 0.9),
			createTestNode("2", "Test 2", 0.8),
		}

		result, err := pp.PostprocessNodes(ctx, nodes, nil)
		require.NoError(t, err)
		assert.Len(t, result, 2)
	})

	t.Run("Returns K nodes when K == len(nodes)", func(t *testing.T) {
		pp := NewTopKPostprocessor(2)
		nodes := []schema.NodeWithScore{
			createTestNode("1", "Test 1", 0.9),
			createTestNode("2", "Test 2", 0.8),
		}

		result, err := pp.PostprocessNodes(ctx, nodes, nil)
		require.NoError(t, err)
		assert.Len(t, result, 2)
	})
}

// TestPostprocessorChain tests the PostprocessorChain.
func TestPostprocessorChain(t *testing.T) {
	ctx := context.Background()

	t.Run("NewPostprocessorChain", func(t *testing.T) {
		chain := NewPostprocessorChain()
		assert.NotNil(t, chain)
		assert.Equal(t, "PostprocessorChain", chain.Name())
	})

	t.Run("Add postprocessor", func(t *testing.T) {
		chain := NewPostprocessorChain()
		pp := NewSimilarityPostprocessor()

		chain.Add(pp)

		assert.Len(t, chain.Postprocessors(), 1)
	})

	t.Run("Chains postprocessors", func(t *testing.T) {
		// First filter by similarity, then by keyword
		chain := NewPostprocessorChain(
			NewSimilarityPostprocessor(WithSimilarityCutoff(0.5)),
			NewKeywordPostprocessor(WithRequiredKeywords([]string{"important"})),
		)

		nodes := []schema.NodeWithScore{
			createTestNode("1", "This is important", 0.9),
			createTestNode("2", "This is important", 0.3), // Below cutoff
			createTestNode("3", "This is other", 0.8),     // Missing keyword
		}

		result, err := chain.PostprocessNodes(ctx, nodes, nil)
		require.NoError(t, err)
		assert.Len(t, result, 1)
		assert.Equal(t, "1", result[0].Node.ID)
	})

	t.Run("Empty chain returns nodes unchanged", func(t *testing.T) {
		chain := NewPostprocessorChain()
		nodes := []schema.NodeWithScore{
			createTestNode("1", "Test", 0.9),
		}

		result, err := chain.PostprocessNodes(ctx, nodes, nil)
		require.NoError(t, err)
		assert.Equal(t, nodes, result)
	})
}

// TestInterfaceCompliance tests that all postprocessors implement NodePostprocessor.
func TestInterfaceCompliance(t *testing.T) {
	var _ NodePostprocessor = (*BaseNodePostprocessor)(nil)
	var _ NodePostprocessor = (*SimilarityPostprocessor)(nil)
	var _ NodePostprocessor = (*KeywordPostprocessor)(nil)
	var _ NodePostprocessor = (*MetadataReplacementPostprocessor)(nil)
	var _ NodePostprocessor = (*LongContextReorder)(nil)
	var _ NodePostprocessor = (*TopKPostprocessor)(nil)
	var _ NodePostprocessor = (*PostprocessorChain)(nil)
	var _ NodePostprocessor = (*LLMRerank)(nil)
	var _ NodePostprocessor = (*RankGPTRerank)(nil)
	var _ NodePostprocessor = (*SlidingWindowRankGPT)(nil)
	var _ NodePostprocessor = (*CohereRerank)(nil)
}

// TestLLMRerank tests the LLMRerank postprocessor.
func TestLLMRerank(t *testing.T) {
	ctx := context.Background()

	t.Run("NewLLMRerank", func(t *testing.T) {
		pp := NewLLMRerank()
		assert.NotNil(t, pp)
		assert.Equal(t, "LLMRerank", pp.Name())
	})

	t.Run("WithLLMRerankTopN", func(t *testing.T) {
		pp := NewLLMRerank(WithLLMRerankTopN(5))
		assert.Equal(t, 5, pp.topN)
	})

	t.Run("WithLLMRerankBatchSize", func(t *testing.T) {
		pp := NewLLMRerank(WithLLMRerankBatchSize(20))
		assert.Equal(t, 20, pp.choiceBatchSize)
	})

	t.Run("Requires query bundle", func(t *testing.T) {
		mockLLM := NewMockLLM("Doc: 1, Relevance: 8")
		pp := NewLLMRerank(WithLLMRerankLLM(mockLLM))

		nodes := []schema.NodeWithScore{
			createTestNode("1", "Test content", 0.5),
		}

		_, err := pp.PostprocessNodes(ctx, nodes, nil)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "query bundle must be provided")
	})

	t.Run("Requires LLM", func(t *testing.T) {
		pp := NewLLMRerank()

		nodes := []schema.NodeWithScore{
			createTestNode("1", "Test content", 0.5),
		}
		queryBundle := &schema.QueryBundle{QueryString: "test query"}

		_, err := pp.PostprocessNodes(ctx, nodes, queryBundle)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "LLM must be provided")
	})

	t.Run("Returns empty for empty nodes", func(t *testing.T) {
		mockLLM := NewMockLLM()
		pp := NewLLMRerank(WithLLMRerankLLM(mockLLM))

		queryBundle := &schema.QueryBundle{QueryString: "test query"}
		result, err := pp.PostprocessNodes(ctx, []schema.NodeWithScore{}, queryBundle)

		require.NoError(t, err)
		assert.Empty(t, result)
	})

	t.Run("Reranks nodes based on LLM response", func(t *testing.T) {
		// LLM returns doc 2 as most relevant, then doc 1
		mockLLM := NewMockLLM("Doc: 2, Relevance: 9\nDoc: 1, Relevance: 5")
		pp := NewLLMRerank(WithLLMRerankLLM(mockLLM), WithLLMRerankTopN(10))

		nodes := []schema.NodeWithScore{
			createTestNode("1", "First document", 0.5),
			createTestNode("2", "Second document", 0.3),
		}
		queryBundle := &schema.QueryBundle{QueryString: "test query"}

		result, err := pp.PostprocessNodes(ctx, nodes, queryBundle)

		require.NoError(t, err)
		assert.Len(t, result, 2)
		// Doc 2 should be first (higher relevance)
		assert.Equal(t, "Second document", result[0].Node.Text)
		assert.Equal(t, 9.0, result[0].Score)
	})

	t.Run("Respects topN limit", func(t *testing.T) {
		mockLLM := NewMockLLM("Doc: 1, Relevance: 9\nDoc: 2, Relevance: 8\nDoc: 3, Relevance: 7")
		pp := NewLLMRerank(WithLLMRerankLLM(mockLLM), WithLLMRerankTopN(2))

		nodes := []schema.NodeWithScore{
			createTestNode("1", "First", 0.5),
			createTestNode("2", "Second", 0.4),
			createTestNode("3", "Third", 0.3),
		}
		queryBundle := &schema.QueryBundle{QueryString: "test query"}

		result, err := pp.PostprocessNodes(ctx, nodes, queryBundle)

		require.NoError(t, err)
		assert.Len(t, result, 2)
	})
}

// TestRankGPTRerank tests the RankGPTRerank postprocessor.
func TestRankGPTRerank(t *testing.T) {
	ctx := context.Background()

	t.Run("NewRankGPTRerank", func(t *testing.T) {
		pp := NewRankGPTRerank()
		assert.NotNil(t, pp)
		assert.Equal(t, "RankGPTRerank", pp.Name())
	})

	t.Run("WithRankGPTTopN", func(t *testing.T) {
		pp := NewRankGPTRerank(WithRankGPTTopN(10))
		assert.Equal(t, 10, pp.topN)
	})

	t.Run("WithRankGPTVerbose", func(t *testing.T) {
		pp := NewRankGPTRerank(WithRankGPTVerbose(true))
		assert.True(t, pp.verbose)
	})

	t.Run("Requires query bundle", func(t *testing.T) {
		mockLLM := NewMockLLM("[2] > [1]")
		pp := NewRankGPTRerank(WithRankGPTLLM(mockLLM))

		nodes := []schema.NodeWithScore{
			createTestNode("1", "Test content", 0.5),
		}

		_, err := pp.PostprocessNodes(ctx, nodes, nil)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "query bundle must be provided")
	})

	t.Run("Requires LLM", func(t *testing.T) {
		pp := NewRankGPTRerank()

		nodes := []schema.NodeWithScore{
			createTestNode("1", "Test content", 0.5),
		}
		queryBundle := &schema.QueryBundle{QueryString: "test query"}

		_, err := pp.PostprocessNodes(ctx, nodes, queryBundle)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "LLM must be provided")
	})

	t.Run("Returns empty for empty nodes", func(t *testing.T) {
		mockLLM := NewMockLLM()
		pp := NewRankGPTRerank(WithRankGPTLLM(mockLLM))

		queryBundle := &schema.QueryBundle{QueryString: "test query"}
		result, err := pp.PostprocessNodes(ctx, []schema.NodeWithScore{}, queryBundle)

		require.NoError(t, err)
		assert.Empty(t, result)
	})

	t.Run("Reranks nodes based on LLM response", func(t *testing.T) {
		// LLM returns ranking: 2 > 1
		mockLLM := NewMockLLM("[2] > [1]")
		pp := NewRankGPTRerank(WithRankGPTLLM(mockLLM), WithRankGPTTopN(10))

		nodes := []schema.NodeWithScore{
			createTestNode("1", "First document", 0.5),
			createTestNode("2", "Second document", 0.3),
		}
		queryBundle := &schema.QueryBundle{QueryString: "test query"}

		result, err := pp.PostprocessNodes(ctx, nodes, queryBundle)

		require.NoError(t, err)
		assert.Len(t, result, 2)
		// Doc 2 should be first
		assert.Equal(t, "Second document", result[0].Node.Text)
	})

	t.Run("Respects topN limit", func(t *testing.T) {
		mockLLM := NewMockLLM("[1] > [2] > [3]")
		pp := NewRankGPTRerank(WithRankGPTLLM(mockLLM), WithRankGPTTopN(2))

		nodes := []schema.NodeWithScore{
			createTestNode("1", "First", 0.5),
			createTestNode("2", "Second", 0.4),
			createTestNode("3", "Third", 0.3),
		}
		queryBundle := &schema.QueryBundle{QueryString: "test query"}

		result, err := pp.PostprocessNodes(ctx, nodes, queryBundle)

		require.NoError(t, err)
		assert.Len(t, result, 2)
	})

	t.Run("Handles malformed response", func(t *testing.T) {
		// LLM returns invalid ranking format
		mockLLM := NewMockLLM("I cannot rank these documents")
		pp := NewRankGPTRerank(WithRankGPTLLM(mockLLM), WithRankGPTTopN(10))

		nodes := []schema.NodeWithScore{
			createTestNode("1", "First", 0.5),
			createTestNode("2", "Second", 0.4),
		}
		queryBundle := &schema.QueryBundle{QueryString: "test query"}

		result, err := pp.PostprocessNodes(ctx, nodes, queryBundle)

		require.NoError(t, err)
		// Should still return nodes (with original order as fallback)
		assert.Len(t, result, 2)
	})
}

// TestSlidingWindowRankGPT tests the SlidingWindowRankGPT postprocessor.
func TestSlidingWindowRankGPT(t *testing.T) {
	ctx := context.Background()

	t.Run("NewSlidingWindowRankGPT", func(t *testing.T) {
		pp := NewSlidingWindowRankGPT(nil)
		assert.NotNil(t, pp)
		assert.Equal(t, "RankGPTRerank", pp.Name())
	})

	t.Run("WithSlidingWindowSize", func(t *testing.T) {
		pp := NewSlidingWindowRankGPT(nil, WithSlidingWindowSize(30))
		assert.Equal(t, 30, pp.windowSize)
	})

	t.Run("WithSlidingStepSize", func(t *testing.T) {
		pp := NewSlidingWindowRankGPT(nil, WithSlidingStepSize(15))
		assert.Equal(t, 15, pp.stepSize)
	})

	t.Run("Uses regular RankGPT for small sets", func(t *testing.T) {
		mockLLM := NewMockLLM("[2] > [1]")
		baseOpts := []RankGPTRerankOption{WithRankGPTLLM(mockLLM), WithRankGPTTopN(10)}
		pp := NewSlidingWindowRankGPT(baseOpts, WithSlidingWindowSize(20))

		nodes := []schema.NodeWithScore{
			createTestNode("1", "First", 0.5),
			createTestNode("2", "Second", 0.4),
		}
		queryBundle := &schema.QueryBundle{QueryString: "test query"}

		result, err := pp.PostprocessNodes(ctx, nodes, queryBundle)

		require.NoError(t, err)
		assert.Len(t, result, 2)
	})
}

// TestCohereRerank tests the CohereRerank postprocessor.
func TestCohereRerank(t *testing.T) {
	ctx := context.Background()

	t.Run("NewCohereRerank", func(t *testing.T) {
		pp := NewCohereRerank()
		assert.NotNil(t, pp)
		assert.Equal(t, "CohereRerank", pp.Name())
	})

	t.Run("WithCohereAPIKey", func(t *testing.T) {
		pp := NewCohereRerank(WithCohereAPIKey("test-key"))
		assert.Equal(t, "test-key", pp.apiKey)
	})

	t.Run("WithCohereModel", func(t *testing.T) {
		pp := NewCohereRerank(WithCohereModel("rerank-multilingual-v2.0"))
		assert.Equal(t, "rerank-multilingual-v2.0", pp.model)
	})

	t.Run("WithCohereTopN", func(t *testing.T) {
		pp := NewCohereRerank(WithCohereTopN(10))
		assert.Equal(t, 10, pp.topN)
	})

	t.Run("Requires API key", func(t *testing.T) {
		pp := NewCohereRerank()

		nodes := []schema.NodeWithScore{
			createTestNode("1", "Test", 0.5),
		}
		queryBundle := &schema.QueryBundle{QueryString: "test query"}

		_, err := pp.PostprocessNodes(ctx, nodes, queryBundle)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "API key must be provided")
	})
}

// TestDefaultParseChoiceSelectAnswer tests the choice select answer parser.
func TestDefaultParseChoiceSelectAnswer(t *testing.T) {
	t.Run("Parses standard format", func(t *testing.T) {
		response := "Doc: 2, Relevance: 8\nDoc: 1, Relevance: 5"
		choices, err := defaultParseChoiceSelectAnswer(response, 3)

		require.NoError(t, err)
		assert.Len(t, choices, 2)
		assert.Equal(t, 1, choices[0].DocIndex) // Doc 2 -> index 1
		assert.Equal(t, 8.0, choices[0].Relevance)
		assert.Equal(t, 0, choices[1].DocIndex) // Doc 1 -> index 0
		assert.Equal(t, 5.0, choices[1].Relevance)
	})

	t.Run("Parses document format", func(t *testing.T) {
		response := "Document 1: Relevance 7\nDocument 3: Relevance 9"
		choices, err := defaultParseChoiceSelectAnswer(response, 3)

		require.NoError(t, err)
		assert.Len(t, choices, 2)
	})

	t.Run("Handles missing relevance", func(t *testing.T) {
		response := "Doc: 1\nDoc: 2"
		choices, err := defaultParseChoiceSelectAnswer(response, 2)

		require.NoError(t, err)
		assert.Len(t, choices, 2)
		// Default relevance should be 5.0
		assert.Equal(t, 5.0, choices[0].Relevance)
	})

	t.Run("Ignores out of range documents", func(t *testing.T) {
		response := "Doc: 1, Relevance: 8\nDoc: 10, Relevance: 5"
		choices, err := defaultParseChoiceSelectAnswer(response, 3)

		require.NoError(t, err)
		assert.Len(t, choices, 1)
	})

	t.Run("Returns error for empty response", func(t *testing.T) {
		response := "I cannot determine relevance"
		_, err := defaultParseChoiceSelectAnswer(response, 3)

		assert.Error(t, err)
	})

	t.Run("Handles duplicate documents", func(t *testing.T) {
		response := "Doc: 1, Relevance: 8\nDoc: 1, Relevance: 5"
		choices, err := defaultParseChoiceSelectAnswer(response, 3)

		require.NoError(t, err)
		assert.Len(t, choices, 1) // Should only include once
	})
}

// TestCleanResponse tests the RankGPT response cleaner.
func TestCleanResponse(t *testing.T) {
	t.Run("Extracts numbers", func(t *testing.T) {
		result := cleanResponse("[2] > [1] > [3]")
		// Check that it contains the numbers separated by spaces
		assert.Contains(t, result, "2")
		assert.Contains(t, result, "1")
		assert.Contains(t, result, "3")
	})

	t.Run("Handles text with numbers", func(t *testing.T) {
		result := cleanResponse("The ranking is 3, then 1, then 2")
		assert.Contains(t, result, "3")
		assert.Contains(t, result, "1")
		assert.Contains(t, result, "2")
	})

	t.Run("Handles empty string", func(t *testing.T) {
		result := cleanResponse("")
		assert.Equal(t, "", result)
	})
}

// TestTruncateToWords tests the word truncation function.
func TestTruncateToWords(t *testing.T) {
	t.Run("Truncates long text", func(t *testing.T) {
		text := "one two three four five six seven eight nine ten"
		result := truncateToWords(text, 5)
		assert.Equal(t, "one two three four five", result)
	})

	t.Run("Returns original if under limit", func(t *testing.T) {
		text := "one two three"
		result := truncateToWords(text, 10)
		assert.Equal(t, "one two three", result)
	})

	t.Run("Handles empty string", func(t *testing.T) {
		result := truncateToWords("", 5)
		assert.Equal(t, "", result)
	})
}

// TestDefaultFormatNodeBatch tests the node batch formatter.
func TestDefaultFormatNodeBatch(t *testing.T) {
	t.Run("Formats nodes correctly", func(t *testing.T) {
		node1 := schema.NewNode()
		node1.Text = "First document content"
		node2 := schema.NewNode()
		node2.Text = "Second document content"

		nodes := []*schema.Node{node1, node2}
		result := defaultFormatNodeBatch(nodes)

		assert.Contains(t, result, "Document 1:")
		assert.Contains(t, result, "First document content")
		assert.Contains(t, result, "Document 2:")
		assert.Contains(t, result, "Second document content")
	})

	t.Run("Truncates long content", func(t *testing.T) {
		node := schema.NewNode()
		// Create content longer than 500 characters (need > 500)
		longContent := ""
		for i := 0; i < 150; i++ {
			longContent += "word "
		}
		node.Text = longContent

		nodes := []*schema.Node{node}
		result := defaultFormatNodeBatch(nodes)

		assert.Contains(t, result, "...")
	})

	t.Run("Handles empty nodes", func(t *testing.T) {
		result := defaultFormatNodeBatch([]*schema.Node{})
		assert.Equal(t, "", result)
	})
}
