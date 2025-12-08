package selector

import (
	"context"
	"testing"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// MockLLM is a mock LLM for testing.
type MockLLM struct {
	Response string
	Err      error
}

func (m *MockLLM) Complete(ctx context.Context, prompt string) (string, error) {
	return m.Response, m.Err
}

func (m *MockLLM) Chat(ctx context.Context, messages []llm.ChatMessage) (string, error) {
	return m.Response, m.Err
}

func (m *MockLLM) Stream(ctx context.Context, prompt string) (<-chan string, error) {
	ch := make(chan string, 1)
	ch <- m.Response
	close(ch)
	return ch, m.Err
}

// TestToolMetadata tests the ToolMetadata struct.
func TestToolMetadata(t *testing.T) {
	t.Run("Create ToolMetadata", func(t *testing.T) {
		meta := ToolMetadata{
			Name:        "search",
			Description: "Search the web",
		}
		assert.Equal(t, "search", meta.Name)
		assert.Equal(t, "Search the web", meta.Description)
	})
}

// TestSingleSelection tests the SingleSelection struct.
func TestSingleSelection(t *testing.T) {
	t.Run("Create SingleSelection", func(t *testing.T) {
		sel := SingleSelection{
			Index:  0,
			Reason: "Most relevant",
		}
		assert.Equal(t, 0, sel.Index)
		assert.Equal(t, "Most relevant", sel.Reason)
	})
}

// TestSelectorResult tests the SelectorResult struct.
func TestSelectorResult(t *testing.T) {
	t.Run("Ind with single selection", func(t *testing.T) {
		result := &SelectorResult{
			Selections: []SingleSelection{
				{Index: 2, Reason: "Best match"},
			},
		}
		ind, err := result.Ind()
		require.NoError(t, err)
		assert.Equal(t, 2, ind)
	})

	t.Run("Ind with multiple selections returns error", func(t *testing.T) {
		result := &SelectorResult{
			Selections: []SingleSelection{
				{Index: 0, Reason: "First"},
				{Index: 1, Reason: "Second"},
			},
		}
		_, err := result.Ind()
		assert.Error(t, err)
	})

	t.Run("Reason with single selection", func(t *testing.T) {
		result := &SelectorResult{
			Selections: []SingleSelection{
				{Index: 0, Reason: "Best match"},
			},
		}
		reason, err := result.Reason()
		require.NoError(t, err)
		assert.Equal(t, "Best match", reason)
	})

	t.Run("Inds returns all indices", func(t *testing.T) {
		result := &SelectorResult{
			Selections: []SingleSelection{
				{Index: 0, Reason: "First"},
				{Index: 2, Reason: "Third"},
			},
		}
		inds := result.Inds()
		assert.Equal(t, []int{0, 2}, inds)
	})

	t.Run("Reasons returns all reasons", func(t *testing.T) {
		result := &SelectorResult{
			Selections: []SingleSelection{
				{Index: 0, Reason: "First"},
				{Index: 1, Reason: "Second"},
			},
		}
		reasons := result.Reasons()
		assert.Equal(t, []string{"First", "Second"}, reasons)
	})
}

// TestBaseSelector tests the BaseSelector.
func TestBaseSelector(t *testing.T) {
	t.Run("NewBaseSelector", func(t *testing.T) {
		sel := NewBaseSelector()
		assert.NotNil(t, sel)
		assert.Equal(t, "BaseSelector", sel.Name())
	})

	t.Run("WithSelectorName", func(t *testing.T) {
		sel := NewBaseSelector(WithSelectorName("custom"))
		assert.Equal(t, "custom", sel.Name())
	})

	t.Run("Select returns empty result", func(t *testing.T) {
		sel := NewBaseSelector()
		ctx := context.Background()
		choices := []ToolMetadata{
			{Name: "tool1", Description: "First tool"},
		}

		result, err := sel.Select(ctx, choices, "test query")
		require.NoError(t, err)
		assert.Empty(t, result.Selections)
	})
}

// TestBuildChoicesText tests the BuildChoicesText function.
func TestBuildChoicesText(t *testing.T) {
	t.Run("Single choice", func(t *testing.T) {
		choices := []ToolMetadata{
			{Name: "search", Description: "Search the web"},
		}
		text := BuildChoicesText(choices)
		assert.Equal(t, "(1) Search the web", text)
	})

	t.Run("Multiple choices", func(t *testing.T) {
		choices := []ToolMetadata{
			{Name: "search", Description: "Search the web"},
			{Name: "calculate", Description: "Perform calculations"},
		}
		text := BuildChoicesText(choices)
		assert.Contains(t, text, "(1) Search the web")
		assert.Contains(t, text, "(2) Perform calculations")
	})
}

// TestSelectionOutputParser tests the SelectionOutputParser.
func TestSelectionOutputParser(t *testing.T) {
	parser := NewSelectionOutputParser()

	t.Run("Parse single selection", func(t *testing.T) {
		output := `[{"choice": 1, "reason": "Most relevant"}]`
		answers, err := parser.Parse(output)
		require.NoError(t, err)
		assert.Len(t, answers, 1)
		assert.Equal(t, 1, answers[0].Choice)
		assert.Equal(t, "Most relevant", answers[0].Reason)
	})

	t.Run("Parse multiple selections", func(t *testing.T) {
		output := `[{"choice": 1, "reason": "First"}, {"choice": 3, "reason": "Third"}]`
		answers, err := parser.Parse(output)
		require.NoError(t, err)
		assert.Len(t, answers, 2)
	})

	t.Run("Parse single object", func(t *testing.T) {
		output := `{"choice": 2, "reason": "Best match"}`
		answers, err := parser.Parse(output)
		require.NoError(t, err)
		assert.Len(t, answers, 1)
		assert.Equal(t, 2, answers[0].Choice)
	})

	t.Run("Parse with surrounding text", func(t *testing.T) {
		output := `Here is my selection: [{"choice": 1, "reason": "Good fit"}] That's my choice.`
		answers, err := parser.Parse(output)
		require.NoError(t, err)
		assert.Len(t, answers, 1)
	})

	t.Run("Parse invalid JSON returns error", func(t *testing.T) {
		output := `This is not JSON`
		_, err := parser.Parse(output)
		assert.Error(t, err)
	})
}

// TestLLMSingleSelector tests the LLMSingleSelector.
func TestLLMSingleSelector(t *testing.T) {
	ctx := context.Background()

	t.Run("NewLLMSingleSelector", func(t *testing.T) {
		mockLLM := &MockLLM{Response: `[{"choice": 1, "reason": "test"}]`}
		sel := NewLLMSingleSelector(mockLLM)
		assert.NotNil(t, sel)
		assert.Equal(t, "LLMSingleSelector", sel.Name())
	})

	t.Run("Select returns single selection", func(t *testing.T) {
		mockLLM := &MockLLM{Response: `[{"choice": 2, "reason": "Best match"}]`}
		sel := NewLLMSingleSelector(mockLLM)

		choices := []ToolMetadata{
			{Name: "tool1", Description: "First tool"},
			{Name: "tool2", Description: "Second tool"},
		}

		result, err := sel.Select(ctx, choices, "test query")
		require.NoError(t, err)
		assert.Len(t, result.Selections, 1)
		assert.Equal(t, 1, result.Selections[0].Index) // 2-1 = 1 (zero-indexed)
		assert.Equal(t, "Best match", result.Selections[0].Reason)
	})
}

// TestLLMMultiSelector tests the LLMMultiSelector.
func TestLLMMultiSelector(t *testing.T) {
	ctx := context.Background()

	t.Run("NewLLMMultiSelector", func(t *testing.T) {
		mockLLM := &MockLLM{Response: `[{"choice": 1, "reason": "test"}]`}
		sel := NewLLMMultiSelector(mockLLM)
		assert.NotNil(t, sel)
		assert.Equal(t, "LLMMultiSelector", sel.Name())
	})

	t.Run("WithMaxOutputs", func(t *testing.T) {
		mockLLM := &MockLLM{Response: `[]`}
		sel := NewLLMMultiSelector(mockLLM, WithMaxOutputs(3))
		assert.Equal(t, 3, sel.MaxOutputs())
	})

	t.Run("Select returns multiple selections", func(t *testing.T) {
		mockLLM := &MockLLM{Response: `[{"choice": 1, "reason": "First"}, {"choice": 3, "reason": "Third"}]`}
		sel := NewLLMMultiSelector(mockLLM)

		choices := []ToolMetadata{
			{Name: "tool1", Description: "First tool"},
			{Name: "tool2", Description: "Second tool"},
			{Name: "tool3", Description: "Third tool"},
		}

		result, err := sel.Select(ctx, choices, "test query")
		require.NoError(t, err)
		assert.Len(t, result.Selections, 2)
		assert.Equal(t, 0, result.Selections[0].Index) // 1-1 = 0
		assert.Equal(t, 2, result.Selections[1].Index) // 3-1 = 2
	})
}

// TestInterfaceCompliance tests that all selectors implement Selector.
func TestInterfaceCompliance(t *testing.T) {
	var _ Selector = (*BaseSelector)(nil)
	var _ Selector = (*LLMSingleSelector)(nil)
	var _ Selector = (*LLMMultiSelector)(nil)
}
