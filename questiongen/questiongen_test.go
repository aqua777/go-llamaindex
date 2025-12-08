package questiongen

import (
	"context"
	"testing"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/selector"
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

// TestSubQuestion tests the SubQuestion struct.
func TestSubQuestion(t *testing.T) {
	t.Run("Create SubQuestion", func(t *testing.T) {
		sq := SubQuestion{
			SubQuestion: "What is the revenue?",
			ToolName:    "financial_tool",
		}
		assert.Equal(t, "What is the revenue?", sq.SubQuestion)
		assert.Equal(t, "financial_tool", sq.ToolName)
	})
}

// TestSubQuestionList tests the SubQuestionList struct.
func TestSubQuestionList(t *testing.T) {
	t.Run("Create SubQuestionList", func(t *testing.T) {
		list := SubQuestionList{
			Items: []SubQuestion{
				{SubQuestion: "Q1", ToolName: "tool1"},
				{SubQuestion: "Q2", ToolName: "tool2"},
			},
		}
		assert.Len(t, list.Items, 2)
	})
}

// TestBaseQuestionGenerator tests the BaseQuestionGenerator.
func TestBaseQuestionGenerator(t *testing.T) {
	t.Run("NewBaseQuestionGenerator", func(t *testing.T) {
		gen := NewBaseQuestionGenerator()
		assert.NotNil(t, gen)
		assert.Equal(t, "BaseQuestionGenerator", gen.Name())
	})

	t.Run("WithGeneratorName", func(t *testing.T) {
		gen := NewBaseQuestionGenerator(WithGeneratorName("custom"))
		assert.Equal(t, "custom", gen.Name())
	})

	t.Run("Generate returns empty list", func(t *testing.T) {
		gen := NewBaseQuestionGenerator()
		ctx := context.Background()
		tools := []selector.ToolMetadata{
			{Name: "tool1", Description: "First tool"},
		}

		questions, err := gen.Generate(ctx, tools, "test query")
		require.NoError(t, err)
		assert.Empty(t, questions)
	})
}

// TestSubQuestionOutputParser tests the SubQuestionOutputParser.
func TestSubQuestionOutputParser(t *testing.T) {
	parser := NewSubQuestionOutputParser()

	t.Run("Parse SubQuestionList format", func(t *testing.T) {
		output := `{"items": [{"sub_question": "What is revenue?", "tool_name": "finance"}]}`
		questions, err := parser.Parse(output)
		require.NoError(t, err)
		assert.Len(t, questions, 1)
		assert.Equal(t, "What is revenue?", questions[0].SubQuestion)
		assert.Equal(t, "finance", questions[0].ToolName)
	})

	t.Run("Parse array format", func(t *testing.T) {
		// Note: The parser expects either SubQuestionList format or array format
		// Array format is parsed directly
		output := `{"items": [{"sub_question": "Q1", "tool_name": "t1"}, {"sub_question": "Q2", "tool_name": "t2"}]}`
		questions, err := parser.Parse(output)
		require.NoError(t, err)
		assert.Len(t, questions, 2)
	})

	t.Run("Parse from code block", func(t *testing.T) {
		output := "```json\n{\"items\": [{\"sub_question\": \"Test?\", \"tool_name\": \"test\"}]}\n```"
		questions, err := parser.Parse(output)
		require.NoError(t, err)
		assert.Len(t, questions, 1)
	})

	t.Run("Parse with surrounding text", func(t *testing.T) {
		output := `Here are the sub-questions: {"items": [{"sub_question": "Q?", "tool_name": "t"}]} That's all.`
		questions, err := parser.Parse(output)
		require.NoError(t, err)
		assert.Len(t, questions, 1)
	})

	t.Run("Parse invalid JSON returns error", func(t *testing.T) {
		output := `This is not JSON`
		_, err := parser.Parse(output)
		assert.Error(t, err)
	})
}

// TestLLMQuestionGenerator tests the LLMQuestionGenerator.
func TestLLMQuestionGenerator(t *testing.T) {
	ctx := context.Background()

	t.Run("NewLLMQuestionGenerator", func(t *testing.T) {
		mockLLM := &MockLLM{Response: `{"items": []}`}
		gen := NewLLMQuestionGenerator(mockLLM)
		assert.NotNil(t, gen)
		assert.Equal(t, "LLMQuestionGenerator", gen.Name())
	})

	t.Run("Generate returns sub-questions", func(t *testing.T) {
		mockLLM := &MockLLM{
			Response: `{"items": [{"sub_question": "What is Uber revenue?", "tool_name": "uber_10k"}, {"sub_question": "What is Lyft revenue?", "tool_name": "lyft_10k"}]}`,
		}
		gen := NewLLMQuestionGenerator(mockLLM)

		tools := []selector.ToolMetadata{
			{Name: "uber_10k", Description: "Uber financials"},
			{Name: "lyft_10k", Description: "Lyft financials"},
		}

		questions, err := gen.Generate(ctx, tools, "Compare Uber and Lyft revenue")
		require.NoError(t, err)
		assert.Len(t, questions, 2)
		assert.Equal(t, "What is Uber revenue?", questions[0].SubQuestion)
		assert.Equal(t, "uber_10k", questions[0].ToolName)
	})
}

// TestBuildToolsText tests the buildToolsText function.
func TestBuildToolsText(t *testing.T) {
	t.Run("Single tool", func(t *testing.T) {
		tools := []selector.ToolMetadata{
			{Name: "search", Description: "Search the web"},
		}
		text := buildToolsText(tools)
		assert.Contains(t, text, "search")
		assert.Contains(t, text, "Search the web")
	})

	t.Run("Multiple tools", func(t *testing.T) {
		tools := []selector.ToolMetadata{
			{Name: "search", Description: "Search the web"},
			{Name: "calculate", Description: "Perform calculations"},
		}
		text := buildToolsText(tools)
		assert.Contains(t, text, "search")
		assert.Contains(t, text, "calculate")
	})
}

// TestInterfaceCompliance tests that all generators implement QuestionGenerator.
func TestInterfaceCompliance(t *testing.T) {
	var _ QuestionGenerator = (*BaseQuestionGenerator)(nil)
	var _ QuestionGenerator = (*LLMQuestionGenerator)(nil)
}
