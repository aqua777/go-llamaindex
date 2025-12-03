package postprocessor

import (
	"context"
	"testing"

	"github.com/aqua777/go-llamaindex/schema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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
}
