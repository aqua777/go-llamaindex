package extractors

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// MockLLM implements llm.LLM for testing.
type MockLLM struct {
	responses    map[string]string
	defaultResp  string
	callCount    int
	lastPrompt   string
	completeFunc func(prompt string) string
}

func NewMockLLM() *MockLLM {
	return &MockLLM{
		responses:   make(map[string]string),
		defaultResp: "Mock response",
	}
}

func (m *MockLLM) WithResponse(contains, response string) *MockLLM {
	m.responses[contains] = response
	return m
}

func (m *MockLLM) WithDefaultResponse(response string) *MockLLM {
	m.defaultResp = response
	return m
}

func (m *MockLLM) WithCompleteFunc(fn func(string) string) *MockLLM {
	m.completeFunc = fn
	return m
}

func (m *MockLLM) Complete(ctx context.Context, prompt string) (string, error) {
	m.callCount++
	m.lastPrompt = prompt

	if m.completeFunc != nil {
		return m.completeFunc(prompt), nil
	}

	for contains, response := range m.responses {
		if strings.Contains(prompt, contains) {
			return response, nil
		}
	}
	return m.defaultResp, nil
}

func (m *MockLLM) Chat(ctx context.Context, messages []llm.ChatMessage) (string, error) {
	if len(messages) > 0 {
		return m.Complete(ctx, messages[len(messages)-1].Content)
	}
	return m.defaultResp, nil
}

func (m *MockLLM) Stream(ctx context.Context, prompt string) (<-chan string, error) {
	ch := make(chan string, 1)
	resp, _ := m.Complete(ctx, prompt)
	ch <- resp
	close(ch)
	return ch, nil
}

func createTestNode(id, text string) *schema.Node {
	node := schema.NewNode()
	node.ID = id
	node.Text = text
	return node
}

func createTestNodeWithMetadata(id, text string, metadata map[string]interface{}) *schema.Node {
	node := createTestNode(id, text)
	node.Metadata = metadata
	return node
}

// TestBaseExtractor tests the BaseExtractor.
func TestBaseExtractor(t *testing.T) {
	t.Run("NewBaseExtractor", func(t *testing.T) {
		e := NewBaseExtractor()
		assert.NotNil(t, e)
		assert.Equal(t, "BaseExtractor", e.Name())
		assert.True(t, e.IsTextNodeOnly())
		assert.Equal(t, MetadataModeAll, e.MetadataMode())
		assert.True(t, e.InPlace())
		assert.Equal(t, 4, e.NumWorkers())
	})

	t.Run("WithOptions", func(t *testing.T) {
		e := NewBaseExtractor(
			WithExtractorName("CustomExtractor"),
			WithTextNodeOnly(false),
			WithMetadataMode(MetadataModeNone),
			WithInPlace(false),
			WithNumWorkers(8),
		)
		assert.Equal(t, "CustomExtractor", e.Name())
		assert.False(t, e.IsTextNodeOnly())
		assert.Equal(t, MetadataModeNone, e.MetadataMode())
		assert.False(t, e.InPlace())
		assert.Equal(t, 8, e.NumWorkers())
	})

	t.Run("Extract returns empty metadata", func(t *testing.T) {
		e := NewBaseExtractor()
		ctx := context.Background()
		nodes := []*schema.Node{createTestNode("1", "Test")}

		result, err := e.Extract(ctx, nodes)

		require.NoError(t, err)
		assert.Len(t, result, 1)
		assert.Empty(t, result[0])
	})

	t.Run("ProcessNodes in place", func(t *testing.T) {
		e := NewBaseExtractor(WithInPlace(true))
		ctx := context.Background()
		node := createTestNode("1", "Test")
		nodes := []*schema.Node{node}

		result, err := e.ProcessNodes(ctx, nodes)

		require.NoError(t, err)
		assert.Same(t, node, result[0])
	})

	t.Run("ProcessNodes copy", func(t *testing.T) {
		e := NewBaseExtractor(WithInPlace(false))
		ctx := context.Background()
		node := createTestNode("1", "Test")
		nodes := []*schema.Node{node}

		result, err := e.ProcessNodes(ctx, nodes)

		require.NoError(t, err)
		assert.NotSame(t, node, result[0])
		assert.Equal(t, node.ID, result[0].ID)
	})
}

// TestExtractorChain tests the ExtractorChain.
func TestExtractorChain(t *testing.T) {
	t.Run("NewExtractorChain", func(t *testing.T) {
		chain := NewExtractorChain()
		assert.NotNil(t, chain)
		assert.Equal(t, "ExtractorChain", chain.Name())
		assert.Empty(t, chain.Extractors())
	})

	t.Run("Add extractors", func(t *testing.T) {
		chain := NewExtractorChain()
		e1 := NewBaseExtractor(WithExtractorName("E1"))
		e2 := NewBaseExtractor(WithExtractorName("E2"))

		chain.Add(e1)
		chain.Add(e2)

		assert.Len(t, chain.Extractors(), 2)
	})

	t.Run("Extract merges metadata", func(t *testing.T) {
		mockLLM := NewMockLLM().
			WithResponse("keywords", "keyword1, keyword2").
			WithResponse("questions", "Q1?\nQ2?")

		chain := NewExtractorChain(
			NewKeywordsExtractor(WithKeywordsLLM(mockLLM)),
			NewQuestionsAnsweredExtractor(WithQuestionsLLM(mockLLM)),
		)

		ctx := context.Background()
		nodes := []*schema.Node{createTestNode("1", "Test content")}

		result, err := chain.Extract(ctx, nodes)

		require.NoError(t, err)
		assert.Len(t, result, 1)
		assert.Contains(t, result[0], "excerpt_keywords")
		assert.Contains(t, result[0], "questions_this_excerpt_can_answer")
	})
}

// TestTitleExtractor tests the TitleExtractor.
func TestTitleExtractor(t *testing.T) {
	ctx := context.Background()

	t.Run("NewTitleExtractor", func(t *testing.T) {
		e := NewTitleExtractor()
		assert.NotNil(t, e)
		assert.Equal(t, "TitleExtractor", e.Name())
	})

	t.Run("WithOptions", func(t *testing.T) {
		mockLLM := NewMockLLM()
		e := NewTitleExtractor(
			WithTitleLLM(mockLLM),
			WithTitleNodes(3),
			WithTitleNodeTemplate("Custom: {context_str}"),
			WithTitleCombineTemplate("Combine: {context_str}"),
		)
		assert.Equal(t, 3, e.nodes)
		assert.Equal(t, "Custom: {context_str}", e.nodeTemplate)
		assert.Equal(t, "Combine: {context_str}", e.combineTemplate)
	})

	t.Run("Requires LLM", func(t *testing.T) {
		e := NewTitleExtractor()
		nodes := []*schema.Node{createTestNode("1", "Test")}

		_, err := e.Extract(ctx, nodes)

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "LLM must be provided")
	})

	t.Run("Returns empty for empty nodes", func(t *testing.T) {
		mockLLM := NewMockLLM()
		e := NewTitleExtractor(WithTitleLLM(mockLLM))

		result, err := e.Extract(ctx, []*schema.Node{})

		require.NoError(t, err)
		assert.Empty(t, result)
	})

	t.Run("Extracts document title", func(t *testing.T) {
		mockLLM := NewMockLLM().
			WithResponse("Give a title", "Machine Learning Basics").
			WithResponse("comprehensive title", "Introduction to Machine Learning")

		e := NewTitleExtractor(WithTitleLLM(mockLLM))
		nodes := []*schema.Node{
			createTestNode("1", "Machine learning is a subset of AI..."),
			createTestNode("2", "Neural networks are inspired by the brain..."),
		}

		result, err := e.Extract(ctx, nodes)

		require.NoError(t, err)
		assert.Len(t, result, 2)
		assert.Equal(t, "Introduction to Machine Learning", result[0]["document_title"])
		assert.Equal(t, "Introduction to Machine Learning", result[1]["document_title"])
	})

	t.Run("Groups nodes by ref_doc_id", func(t *testing.T) {
		mockLLM := NewMockLLM().
			WithCompleteFunc(func(prompt string) string {
				if strings.Contains(prompt, "comprehensive") {
					if strings.Contains(prompt, "Doc A") {
						return "Title for Doc A"
					}
					return "Title for Doc B"
				}
				if strings.Contains(prompt, "Content A") {
					return "Doc A Title"
				}
				return "Doc B Title"
			})

		e := NewTitleExtractor(WithTitleLLM(mockLLM))
		nodes := []*schema.Node{
			createTestNodeWithMetadata("1", "Content A1", map[string]interface{}{"ref_doc_id": "docA"}),
			createTestNodeWithMetadata("2", "Content A2", map[string]interface{}{"ref_doc_id": "docA"}),
			createTestNodeWithMetadata("3", "Content B1", map[string]interface{}{"ref_doc_id": "docB"}),
		}

		result, err := e.Extract(ctx, nodes)

		require.NoError(t, err)
		assert.Len(t, result, 3)
		// Nodes from same doc should have same title
		assert.Equal(t, result[0]["document_title"], result[1]["document_title"])
		// Nodes from different docs may have different titles
		assert.NotNil(t, result[2]["document_title"])
	})
}

// TestSummaryExtractor tests the SummaryExtractor.
func TestSummaryExtractor(t *testing.T) {
	ctx := context.Background()

	t.Run("NewSummaryExtractor", func(t *testing.T) {
		e := NewSummaryExtractor()
		assert.NotNil(t, e)
		assert.Equal(t, "SummaryExtractor", e.Name())
	})

	t.Run("WithOptions", func(t *testing.T) {
		mockLLM := NewMockLLM()
		e := NewSummaryExtractor(
			WithSummaryLLM(mockLLM),
			WithSummaryTypes(SummaryTypeSelf, SummaryTypePrev, SummaryTypeNext),
			WithSummaryPromptTemplate("Custom: {context_str}"),
		)
		assert.Len(t, e.summaries, 3)
		assert.Equal(t, "Custom: {context_str}", e.promptTemplate)
	})

	t.Run("Requires LLM", func(t *testing.T) {
		e := NewSummaryExtractor()
		nodes := []*schema.Node{createTestNode("1", "Test")}

		_, err := e.Extract(ctx, nodes)

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "LLM must be provided")
	})

	t.Run("Returns empty for empty nodes", func(t *testing.T) {
		mockLLM := NewMockLLM()
		e := NewSummaryExtractor(WithSummaryLLM(mockLLM))

		result, err := e.Extract(ctx, []*schema.Node{})

		require.NoError(t, err)
		assert.Empty(t, result)
	})

	t.Run("Extracts self summary only", func(t *testing.T) {
		mockLLM := NewMockLLM().WithDefaultResponse("Summary of the content")
		e := NewSummaryExtractor(
			WithSummaryLLM(mockLLM),
			WithSummaryTypes(SummaryTypeSelf),
		)
		nodes := []*schema.Node{createTestNode("1", "Test content")}

		result, err := e.Extract(ctx, nodes)

		require.NoError(t, err)
		assert.Len(t, result, 1)
		assert.Equal(t, "Summary of the content", result[0]["section_summary"])
		assert.NotContains(t, result[0], "prev_section_summary")
		assert.NotContains(t, result[0], "next_section_summary")
	})

	t.Run("Extracts adjacent summaries", func(t *testing.T) {
		callNum := 0
		mockLLM := NewMockLLM().WithCompleteFunc(func(prompt string) string {
			callNum++
			return fmt.Sprintf("Summary %d", callNum)
		})

		e := NewSummaryExtractor(
			WithSummaryLLM(mockLLM),
			WithSummaryTypes(SummaryTypeSelf, SummaryTypePrev, SummaryTypeNext),
		)
		nodes := []*schema.Node{
			createTestNode("1", "First section"),
			createTestNode("2", "Second section"),
			createTestNode("3", "Third section"),
		}

		result, err := e.Extract(ctx, nodes)

		require.NoError(t, err)
		assert.Len(t, result, 3)

		// First node: has self and next, no prev
		assert.Contains(t, result[0], "section_summary")
		assert.NotContains(t, result[0], "prev_section_summary")
		assert.Contains(t, result[0], "next_section_summary")

		// Middle node: has all three
		assert.Contains(t, result[1], "section_summary")
		assert.Contains(t, result[1], "prev_section_summary")
		assert.Contains(t, result[1], "next_section_summary")

		// Last node: has self and prev, no next
		assert.Contains(t, result[2], "section_summary")
		assert.Contains(t, result[2], "prev_section_summary")
		assert.NotContains(t, result[2], "next_section_summary")
	})
}

// TestKeywordsExtractor tests the KeywordsExtractor.
func TestKeywordsExtractor(t *testing.T) {
	ctx := context.Background()

	t.Run("NewKeywordsExtractor", func(t *testing.T) {
		e := NewKeywordsExtractor()
		assert.NotNil(t, e)
		assert.Equal(t, "KeywordsExtractor", e.Name())
	})

	t.Run("WithOptions", func(t *testing.T) {
		mockLLM := NewMockLLM()
		e := NewKeywordsExtractor(
			WithKeywordsLLM(mockLLM),
			WithKeywordsCount(10),
			WithKeywordsPromptTemplate("Custom: {context_str} {keywords}"),
		)
		assert.Equal(t, 10, e.keywords)
		assert.Equal(t, "Custom: {context_str} {keywords}", e.promptTemplate)
	})

	t.Run("Requires LLM", func(t *testing.T) {
		e := NewKeywordsExtractor()
		nodes := []*schema.Node{createTestNode("1", "Test")}

		_, err := e.Extract(ctx, nodes)

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "LLM must be provided")
	})

	t.Run("Returns empty for empty nodes", func(t *testing.T) {
		mockLLM := NewMockLLM()
		e := NewKeywordsExtractor(WithKeywordsLLM(mockLLM))

		result, err := e.Extract(ctx, []*schema.Node{})

		require.NoError(t, err)
		assert.Empty(t, result)
	})

	t.Run("Extracts keywords", func(t *testing.T) {
		mockLLM := NewMockLLM().WithDefaultResponse("machine learning, AI, neural networks, deep learning, algorithms")
		e := NewKeywordsExtractor(WithKeywordsLLM(mockLLM))
		nodes := []*schema.Node{createTestNode("1", "Machine learning content...")}

		result, err := e.Extract(ctx, nodes)

		require.NoError(t, err)
		assert.Len(t, result, 1)
		assert.Contains(t, result[0]["excerpt_keywords"], "machine learning")
	})

	t.Run("Extracts from multiple nodes", func(t *testing.T) {
		callNum := 0
		mockLLM := NewMockLLM().WithCompleteFunc(func(prompt string) string {
			callNum++
			return fmt.Sprintf("keyword%d_a, keyword%d_b", callNum, callNum)
		})

		e := NewKeywordsExtractor(WithKeywordsLLM(mockLLM))
		nodes := []*schema.Node{
			createTestNode("1", "First content"),
			createTestNode("2", "Second content"),
		}

		result, err := e.Extract(ctx, nodes)

		require.NoError(t, err)
		assert.Len(t, result, 2)
		assert.NotEqual(t, result[0]["excerpt_keywords"], result[1]["excerpt_keywords"])
	})
}

// TestQuestionsAnsweredExtractor tests the QuestionsAnsweredExtractor.
func TestQuestionsAnsweredExtractor(t *testing.T) {
	ctx := context.Background()

	t.Run("NewQuestionsAnsweredExtractor", func(t *testing.T) {
		e := NewQuestionsAnsweredExtractor()
		assert.NotNil(t, e)
		assert.Equal(t, "QuestionsAnsweredExtractor", e.Name())
		assert.True(t, e.IsEmbeddingOnly())
	})

	t.Run("WithOptions", func(t *testing.T) {
		mockLLM := NewMockLLM()
		e := NewQuestionsAnsweredExtractor(
			WithQuestionsLLM(mockLLM),
			WithQuestionsCount(10),
			WithQuestionsPromptTemplate("Custom: {context_str} {num_questions}"),
			WithEmbeddingOnly(false),
		)
		assert.Equal(t, 10, e.questions)
		assert.Equal(t, "Custom: {context_str} {num_questions}", e.promptTemplate)
		assert.False(t, e.IsEmbeddingOnly())
	})

	t.Run("Requires LLM", func(t *testing.T) {
		e := NewQuestionsAnsweredExtractor()
		nodes := []*schema.Node{createTestNode("1", "Test")}

		_, err := e.Extract(ctx, nodes)

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "LLM must be provided")
	})

	t.Run("Returns empty for empty nodes", func(t *testing.T) {
		mockLLM := NewMockLLM()
		e := NewQuestionsAnsweredExtractor(WithQuestionsLLM(mockLLM))

		result, err := e.Extract(ctx, []*schema.Node{})

		require.NoError(t, err)
		assert.Empty(t, result)
	})

	t.Run("Extracts questions", func(t *testing.T) {
		mockLLM := NewMockLLM().WithDefaultResponse("1. What is machine learning?\n2. How do neural networks work?")
		e := NewQuestionsAnsweredExtractor(WithQuestionsLLM(mockLLM))
		nodes := []*schema.Node{createTestNode("1", "Machine learning content...")}

		result, err := e.Extract(ctx, nodes)

		require.NoError(t, err)
		assert.Len(t, result, 1)
		assert.Contains(t, result[0]["questions_this_excerpt_can_answer"], "What is machine learning?")
	})
}

// TestParseKeywords tests the ParseKeywords function.
func TestParseKeywords(t *testing.T) {
	t.Run("Parses comma-separated keywords", func(t *testing.T) {
		result := ParseKeywords("keyword1, keyword2, keyword3")
		assert.Equal(t, []string{"keyword1", "keyword2", "keyword3"}, result)
	})

	t.Run("Handles extra whitespace", func(t *testing.T) {
		result := ParseKeywords("  keyword1  ,  keyword2  ,  keyword3  ")
		assert.Equal(t, []string{"keyword1", "keyword2", "keyword3"}, result)
	})

	t.Run("Handles empty parts", func(t *testing.T) {
		result := ParseKeywords("keyword1, , keyword2")
		assert.Equal(t, []string{"keyword1", "keyword2"}, result)
	})

	t.Run("Handles empty string", func(t *testing.T) {
		result := ParseKeywords("")
		assert.Empty(t, result)
	})
}

// TestParseQuestions tests the ParseQuestions function.
func TestParseQuestions(t *testing.T) {
	t.Run("Parses numbered questions", func(t *testing.T) {
		result := ParseQuestions("1. What is AI?\n2. How does ML work?\n3. What are neural networks?")
		assert.Len(t, result, 3)
		assert.Equal(t, "What is AI?", result[0])
		assert.Equal(t, "How does ML work?", result[1])
	})

	t.Run("Parses bullet questions", func(t *testing.T) {
		result := ParseQuestions("- What is AI?\n- How does ML work?")
		assert.Len(t, result, 2)
		assert.Equal(t, "What is AI?", result[0])
	})

	t.Run("Handles plain questions", func(t *testing.T) {
		result := ParseQuestions("What is AI?\nHow does ML work?")
		assert.Len(t, result, 2)
	})

	t.Run("Handles empty lines", func(t *testing.T) {
		result := ParseQuestions("What is AI?\n\nHow does ML work?")
		assert.Len(t, result, 2)
	})

	t.Run("Handles empty string", func(t *testing.T) {
		result := ParseQuestions("")
		assert.Empty(t, result)
	})
}

// TestFormatPrompt tests the formatPrompt function.
func TestFormatPrompt(t *testing.T) {
	t.Run("Replaces single variable", func(t *testing.T) {
		result := formatPrompt("Hello {name}!", map[string]string{"name": "World"})
		assert.Equal(t, "Hello World!", result)
	})

	t.Run("Replaces multiple variables", func(t *testing.T) {
		result := formatPrompt("{greeting} {name}!", map[string]string{
			"greeting": "Hello",
			"name":     "World",
		})
		assert.Equal(t, "Hello World!", result)
	})

	t.Run("Handles missing variables", func(t *testing.T) {
		result := formatPrompt("Hello {name}!", map[string]string{})
		assert.Equal(t, "Hello {name}!", result)
	})

	t.Run("Handles empty template", func(t *testing.T) {
		result := formatPrompt("", map[string]string{"name": "World"})
		assert.Equal(t, "", result)
	})
}

// TestInterfaceCompliance tests that all extractors implement MetadataExtractor.
func TestInterfaceCompliance(t *testing.T) {
	var _ MetadataExtractor = (*BaseExtractor)(nil)
	var _ MetadataExtractor = (*ExtractorChain)(nil)
	var _ MetadataExtractor = (*TitleExtractor)(nil)
	var _ MetadataExtractor = (*SummaryExtractor)(nil)
	var _ MetadataExtractor = (*KeywordsExtractor)(nil)
	var _ MetadataExtractor = (*QuestionsAnsweredExtractor)(nil)
}

// TestConcurrentExtraction tests concurrent extraction.
func TestConcurrentExtraction(t *testing.T) {
	ctx := context.Background()

	t.Run("Extracts from many nodes concurrently", func(t *testing.T) {
		callCount := 0
		mockLLM := NewMockLLM().WithCompleteFunc(func(prompt string) string {
			callCount++
			return fmt.Sprintf("keyword_%d", callCount)
		})

		e := NewKeywordsExtractor(
			WithKeywordsLLM(mockLLM),
		)

		// Create many nodes
		nodes := make([]*schema.Node, 20)
		for i := range nodes {
			nodes[i] = createTestNode(fmt.Sprintf("%d", i), fmt.Sprintf("Content %d", i))
		}

		result, err := e.Extract(ctx, nodes)

		require.NoError(t, err)
		assert.Len(t, result, 20)

		// All nodes should have keywords
		for _, m := range result {
			assert.Contains(t, m, "excerpt_keywords")
		}
	})
}

// TestProcessNodesUpdatesMetadata tests that ProcessNodes updates node metadata.
func TestProcessNodesUpdatesMetadata(t *testing.T) {
	ctx := context.Background()

	t.Run("Updates metadata in place", func(t *testing.T) {
		mockLLM := NewMockLLM().WithDefaultResponse("test_keyword")
		e := NewKeywordsExtractor(WithKeywordsLLM(mockLLM))

		node := createTestNode("1", "Test content")
		nodes := []*schema.Node{node}

		result, err := e.ProcessNodes(ctx, nodes)

		require.NoError(t, err)
		assert.Same(t, node, result[0])
		assert.Equal(t, "test_keyword", node.Metadata["excerpt_keywords"])
	})

	t.Run("Preserves existing metadata", func(t *testing.T) {
		mockLLM := NewMockLLM().WithDefaultResponse("new_keyword")
		e := NewKeywordsExtractor(WithKeywordsLLM(mockLLM))

		node := createTestNodeWithMetadata("1", "Test", map[string]interface{}{
			"existing_key": "existing_value",
		})
		nodes := []*schema.Node{node}

		result, err := e.ProcessNodes(ctx, nodes)

		require.NoError(t, err)
		assert.Equal(t, "existing_value", result[0].Metadata["existing_key"])
		assert.Equal(t, "new_keyword", result[0].Metadata["excerpt_keywords"])
	})
}
