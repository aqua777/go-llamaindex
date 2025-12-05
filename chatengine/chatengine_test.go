package chatengine

import (
	"context"
	"testing"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/memory"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// MockLLM is a mock LLM for testing.
type MockLLM struct {
	response      string
	chatResponses []string
	callCount     int
}

func NewMockLLM(response string) *MockLLM {
	return &MockLLM{response: response}
}

func (m *MockLLM) Complete(ctx context.Context, prompt string) (string, error) {
	m.callCount++
	return m.response, nil
}

func (m *MockLLM) Chat(ctx context.Context, messages []llm.ChatMessage) (string, error) {
	m.callCount++
	if len(m.chatResponses) > 0 && m.callCount <= len(m.chatResponses) {
		return m.chatResponses[m.callCount-1], nil
	}
	return m.response, nil
}

func (m *MockLLM) Stream(ctx context.Context, prompt string) (<-chan string, error) {
	m.callCount++
	ch := make(chan string, 1)
	go func() {
		defer close(ch)
		for _, word := range []string{"Hello", " ", "World", "!"} {
			ch <- word
		}
	}()
	return ch, nil
}

// MockRetriever is a mock retriever for testing.
type MockRetriever struct {
	nodes []schema.NodeWithScore
}

func NewMockRetriever(nodes []schema.NodeWithScore) *MockRetriever {
	return &MockRetriever{nodes: nodes}
}

func (m *MockRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	return m.nodes, nil
}

// TestChatResponse tests the ChatResponse struct.
func TestChatResponse(t *testing.T) {
	t.Run("NewChatResponse", func(t *testing.T) {
		resp := NewChatResponse("Hello, World!")
		assert.Equal(t, "Hello, World!", resp.Response)
		assert.Empty(t, resp.SourceNodes)
		assert.Empty(t, resp.Sources)
	})

	t.Run("String", func(t *testing.T) {
		resp := NewChatResponse("Test response")
		assert.Equal(t, "Test response", resp.String())
	})
}

// TestStreamingChatResponse tests the StreamingChatResponse struct.
func TestStreamingChatResponse(t *testing.T) {
	t.Run("NewStreamingChatResponse", func(t *testing.T) {
		ch := make(chan string)
		close(ch)
		resp := NewStreamingChatResponse(ch)
		assert.NotNil(t, resp)
		assert.False(t, resp.IsDone())
	})

	t.Run("Consume", func(t *testing.T) {
		ch := make(chan string, 3)
		ch <- "Hello"
		ch <- " "
		ch <- "World"
		close(ch)

		resp := NewStreamingChatResponse(ch)
		result := resp.Consume()

		assert.Equal(t, "Hello World", result)
		assert.True(t, resp.IsDone())
		assert.Equal(t, "Hello World", resp.Response())
	})
}

// TestSimpleChatEngine tests the SimpleChatEngine.
func TestSimpleChatEngine(t *testing.T) {
	ctx := context.Background()

	t.Run("NewSimpleChatEngine", func(t *testing.T) {
		engine := NewSimpleChatEngine()
		assert.NotNil(t, engine)
	})

	t.Run("Chat", func(t *testing.T) {
		mockLLM := NewMockLLM("Hello! How can I help you?")
		engine := NewSimpleChatEngine(
			WithSimpleChatEngineLLM(mockLLM),
		)

		resp, err := engine.Chat(ctx, "Hello")
		require.NoError(t, err)
		assert.Equal(t, "Hello! How can I help you?", resp.Response)
	})

	t.Run("Chat with system prompt", func(t *testing.T) {
		mockLLM := NewMockLLM("I am a helpful assistant.")
		engine := NewSimpleChatEngine(
			WithSimpleChatEngineLLM(mockLLM),
			WithSimpleChatEngineSystemPrompt("You are a helpful assistant."),
		)

		resp, err := engine.Chat(ctx, "Who are you?")
		require.NoError(t, err)
		assert.NotEmpty(t, resp.Response)
	})

	t.Run("ChatWithHistory", func(t *testing.T) {
		mockLLM := NewMockLLM("I remember you said hello!")
		engine := NewSimpleChatEngine(
			WithSimpleChatEngineLLM(mockLLM),
		)

		history := []llm.ChatMessage{
			{Role: llm.MessageRoleUser, Content: "Hello"},
			{Role: llm.MessageRoleAssistant, Content: "Hi there!"},
		}

		resp, err := engine.ChatWithHistory(ctx, "Do you remember what I said?", history)
		require.NoError(t, err)
		assert.NotEmpty(t, resp.Response)
	})

	t.Run("StreamChat", func(t *testing.T) {
		mockLLM := NewMockLLM("streaming response")
		engine := NewSimpleChatEngine(
			WithSimpleChatEngineLLM(mockLLM),
		)

		resp, err := engine.StreamChat(ctx, "Hello")
		require.NoError(t, err)

		result := resp.Consume()
		assert.NotEmpty(t, result)
	})

	t.Run("Reset", func(t *testing.T) {
		mockLLM := NewMockLLM("response")
		engine := NewSimpleChatEngine(
			WithSimpleChatEngineLLM(mockLLM),
		)

		// Add some messages
		_, err := engine.Chat(ctx, "Hello")
		require.NoError(t, err)

		// Reset
		err = engine.Reset(ctx)
		require.NoError(t, err)

		// Check history is empty
		history, err := engine.ChatHistory(ctx)
		require.NoError(t, err)
		assert.Empty(t, history)
	})

	t.Run("ChatHistory", func(t *testing.T) {
		mockLLM := NewMockLLM("response")
		engine := NewSimpleChatEngine(
			WithSimpleChatEngineLLM(mockLLM),
		)

		_, err := engine.Chat(ctx, "Hello")
		require.NoError(t, err)

		history, err := engine.ChatHistory(ctx)
		require.NoError(t, err)
		assert.Len(t, history, 2) // User + Assistant
	})

	t.Run("Chat without LLM", func(t *testing.T) {
		engine := NewSimpleChatEngine()

		_, err := engine.Chat(ctx, "Hello")
		assert.Error(t, err)
	})
}

// TestSimpleChatEngineFromDefaults tests NewSimpleChatEngineFromDefaults.
func TestSimpleChatEngineFromDefaults(t *testing.T) {
	t.Run("With chat history", func(t *testing.T) {
		mockLLM := NewMockLLM("response")
		chatHistory := []llm.ChatMessage{
			{Role: llm.MessageRoleUser, Content: "Previous message"},
		}

		engine, err := NewSimpleChatEngineFromDefaults(mockLLM, chatHistory, "")
		require.NoError(t, err)
		assert.NotNil(t, engine)
	})

	t.Run("With system prompt", func(t *testing.T) {
		mockLLM := NewMockLLM("response")

		engine, err := NewSimpleChatEngineFromDefaults(mockLLM, nil, "You are helpful.")
		require.NoError(t, err)
		assert.NotNil(t, engine)
		assert.Len(t, engine.PrefixMessages(), 1)
	})
}

// TestContextChatEngine tests the ContextChatEngine.
func TestContextChatEngine(t *testing.T) {
	ctx := context.Background()

	t.Run("NewContextChatEngine", func(t *testing.T) {
		engine := NewContextChatEngine()
		assert.NotNil(t, engine)
	})

	t.Run("Chat", func(t *testing.T) {
		mockLLM := NewMockLLM("Based on the context, the answer is 42.")
		mockRetriever := NewMockRetriever([]schema.NodeWithScore{
			{Node: schema.Node{Text: "The answer to everything is 42."}, Score: 0.9},
		})

		engine := NewContextChatEngine(
			WithContextChatEngineLLM(mockLLM),
			WithContextChatEngineRetriever(mockRetriever),
		)

		resp, err := engine.Chat(ctx, "What is the answer?")
		require.NoError(t, err)
		assert.NotEmpty(t, resp.Response)
		assert.Len(t, resp.SourceNodes, 1)
		assert.Len(t, resp.Sources, 1)
	})

	t.Run("Chat with system prompt", func(t *testing.T) {
		mockLLM := NewMockLLM("response")
		mockRetriever := NewMockRetriever([]schema.NodeWithScore{})

		engine := NewContextChatEngine(
			WithContextChatEngineLLM(mockLLM),
			WithContextChatEngineRetriever(mockRetriever),
			WithContextChatEngineSystemPrompt("You are a helpful assistant."),
		)

		resp, err := engine.Chat(ctx, "Hello")
		require.NoError(t, err)
		assert.NotEmpty(t, resp.Response)
	})

	t.Run("StreamChat", func(t *testing.T) {
		mockLLM := NewMockLLM("streaming response")
		mockRetriever := NewMockRetriever([]schema.NodeWithScore{
			{Node: schema.Node{Text: "Context"}, Score: 0.9},
		})

		engine := NewContextChatEngine(
			WithContextChatEngineLLM(mockLLM),
			WithContextChatEngineRetriever(mockRetriever),
		)

		resp, err := engine.StreamChat(ctx, "Hello")
		require.NoError(t, err)
		assert.NotNil(t, resp.SourceNodes)

		result := resp.Consume()
		assert.NotEmpty(t, result)
	})

	t.Run("Reset", func(t *testing.T) {
		mockLLM := NewMockLLM("response")
		mockRetriever := NewMockRetriever([]schema.NodeWithScore{})

		engine := NewContextChatEngine(
			WithContextChatEngineLLM(mockLLM),
			WithContextChatEngineRetriever(mockRetriever),
		)

		_, err := engine.Chat(ctx, "Hello")
		require.NoError(t, err)

		err = engine.Reset(ctx)
		require.NoError(t, err)

		history, err := engine.ChatHistory(ctx)
		require.NoError(t, err)
		assert.Empty(t, history)
	})

	t.Run("Chat without LLM", func(t *testing.T) {
		mockRetriever := NewMockRetriever([]schema.NodeWithScore{})
		engine := NewContextChatEngine(
			WithContextChatEngineRetriever(mockRetriever),
		)

		_, err := engine.Chat(ctx, "Hello")
		assert.Error(t, err)
	})

	t.Run("Chat without retriever", func(t *testing.T) {
		mockLLM := NewMockLLM("response")
		engine := NewContextChatEngine(
			WithContextChatEngineLLM(mockLLM),
		)

		_, err := engine.Chat(ctx, "Hello")
		assert.Error(t, err)
	})
}

// TestContextChatEngineFromDefaults tests NewContextChatEngineFromDefaults.
func TestContextChatEngineFromDefaults(t *testing.T) {
	t.Run("Basic creation", func(t *testing.T) {
		mockLLM := NewMockLLM("response")
		mockRetriever := NewMockRetriever([]schema.NodeWithScore{})

		engine, err := NewContextChatEngineFromDefaults(mockRetriever, mockLLM, nil, "")
		require.NoError(t, err)
		assert.NotNil(t, engine)
	})
}

// TestCondensePlusContextChatEngine tests the CondensePlusContextChatEngine.
func TestCondensePlusContextChatEngine(t *testing.T) {
	ctx := context.Background()

	t.Run("NewCondensePlusContextChatEngine", func(t *testing.T) {
		engine := NewCondensePlusContextChatEngine()
		assert.NotNil(t, engine)
	})

	t.Run("Chat with condensing", func(t *testing.T) {
		mockLLM := &MockLLM{
			chatResponses: []string{
				"What is the meaning of life?", // Condensed question
				"The meaning of life is 42.",   // Final response
			},
			response: "default response",
		}
		mockRetriever := NewMockRetriever([]schema.NodeWithScore{
			{Node: schema.Node{Text: "42 is the answer."}, Score: 0.9},
		})

		engine := NewCondensePlusContextChatEngine(
			WithCondensePlusContextLLM(mockLLM),
			WithCondensePlusContextRetriever(mockRetriever),
		)

		// First message - no condensing needed
		resp, err := engine.Chat(ctx, "What is the meaning of life?")
		require.NoError(t, err)
		assert.NotEmpty(t, resp.Response)
	})

	t.Run("Chat with skip condense", func(t *testing.T) {
		mockLLM := NewMockLLM("Direct response")
		mockRetriever := NewMockRetriever([]schema.NodeWithScore{})

		engine := NewCondensePlusContextChatEngine(
			WithCondensePlusContextLLM(mockLLM),
			WithCondensePlusContextRetriever(mockRetriever),
			WithSkipCondense(true),
		)

		resp, err := engine.Chat(ctx, "Hello")
		require.NoError(t, err)
		assert.NotEmpty(t, resp.Response)
	})

	t.Run("StreamChat", func(t *testing.T) {
		mockLLM := NewMockLLM("streaming response")
		mockRetriever := NewMockRetriever([]schema.NodeWithScore{
			{Node: schema.Node{Text: "Context"}, Score: 0.9},
		})

		engine := NewCondensePlusContextChatEngine(
			WithCondensePlusContextLLM(mockLLM),
			WithCondensePlusContextRetriever(mockRetriever),
			WithSkipCondense(true),
		)

		resp, err := engine.StreamChat(ctx, "Hello")
		require.NoError(t, err)

		result := resp.Consume()
		assert.NotEmpty(t, result)
	})

	t.Run("Reset", func(t *testing.T) {
		mockLLM := NewMockLLM("response")
		mockRetriever := NewMockRetriever([]schema.NodeWithScore{})

		engine := NewCondensePlusContextChatEngine(
			WithCondensePlusContextLLM(mockLLM),
			WithCondensePlusContextRetriever(mockRetriever),
			WithSkipCondense(true),
		)

		_, err := engine.Chat(ctx, "Hello")
		require.NoError(t, err)

		err = engine.Reset(ctx)
		require.NoError(t, err)

		history, err := engine.ChatHistory(ctx)
		require.NoError(t, err)
		assert.Empty(t, history)
	})

	t.Run("Verbose mode", func(t *testing.T) {
		mockLLM := NewMockLLM("response")
		mockRetriever := NewMockRetriever([]schema.NodeWithScore{})

		engine := NewCondensePlusContextChatEngine(
			WithCondensePlusContextLLM(mockLLM),
			WithCondensePlusContextRetriever(mockRetriever),
			WithCondensePlusContextVerbose(true),
			WithSkipCondense(true),
		)

		resp, err := engine.Chat(ctx, "Hello")
		require.NoError(t, err)
		assert.NotEmpty(t, resp.Response)
	})
}

// TestCondensePlusContextChatEngineFromDefaults tests NewCondensePlusContextChatEngineFromDefaults.
func TestCondensePlusContextChatEngineFromDefaults(t *testing.T) {
	t.Run("Basic creation", func(t *testing.T) {
		mockLLM := NewMockLLM("response")
		mockRetriever := NewMockRetriever([]schema.NodeWithScore{})

		engine, err := NewCondensePlusContextChatEngineFromDefaults(mockRetriever, mockLLM, nil, "")
		require.NoError(t, err)
		assert.NotNil(t, engine)
	})

	t.Run("With system prompt", func(t *testing.T) {
		mockLLM := NewMockLLM("response")
		mockRetriever := NewMockRetriever([]schema.NodeWithScore{})

		engine, err := NewCondensePlusContextChatEngineFromDefaults(
			mockRetriever, mockLLM, nil, "You are helpful.",
		)
		require.NoError(t, err)
		assert.Len(t, engine.PrefixMessages(), 1)
	})
}

// TestChatEngineInterface tests that all chat engines implement the ChatEngine interface.
func TestChatEngineInterface(t *testing.T) {
	t.Run("SimpleChatEngine implements ChatEngine", func(t *testing.T) {
		var _ ChatEngine = NewSimpleChatEngine()
	})

	t.Run("ContextChatEngine implements ChatEngine", func(t *testing.T) {
		var _ ChatEngine = NewContextChatEngine()
	})

	t.Run("CondensePlusContextChatEngine implements ChatEngine", func(t *testing.T) {
		var _ ChatEngine = NewCondensePlusContextChatEngine()
	})
}

// TestBaseChatEngine tests the BaseChatEngine.
func TestBaseChatEngine(t *testing.T) {
	t.Run("NewBaseChatEngine", func(t *testing.T) {
		engine := NewBaseChatEngine()
		assert.NotNil(t, engine)
		assert.Empty(t, engine.PrefixMessages())
	})

	t.Run("WithLLM", func(t *testing.T) {
		mockLLM := NewMockLLM("response")
		engine := NewBaseChatEngine(WithLLM(mockLLM))
		assert.Equal(t, mockLLM, engine.LLM())
	})

	t.Run("WithSystemPrompt", func(t *testing.T) {
		engine := NewBaseChatEngine(WithSystemPrompt("You are helpful."))
		assert.Len(t, engine.PrefixMessages(), 1)
		assert.Equal(t, llm.MessageRoleSystem, engine.PrefixMessages()[0].Role)
		assert.Equal(t, "You are helpful.", engine.PrefixMessages()[0].Content)
	})

	t.Run("WithPrefixMessages", func(t *testing.T) {
		messages := []llm.ChatMessage{
			{Role: llm.MessageRoleSystem, Content: "System"},
			{Role: llm.MessageRoleUser, Content: "Example"},
		}
		engine := NewBaseChatEngine(WithPrefixMessages(messages))
		assert.Len(t, engine.PrefixMessages(), 2)
	})
}

// TestCustomMemory tests chat engines with custom memory.
func TestCustomMemory(t *testing.T) {
	ctx := context.Background()

	t.Run("SimpleChatEngine with custom memory", func(t *testing.T) {
		mockLLM := NewMockLLM("response")
		customMemory := memory.NewChatMemoryBuffer(memory.WithTokenLimit(100))

		engine := NewSimpleChatEngine(
			WithSimpleChatEngineLLM(mockLLM),
			WithSimpleChatEngineMemory(customMemory),
		)

		resp, err := engine.Chat(ctx, "Hello")
		require.NoError(t, err)
		assert.NotEmpty(t, resp.Response)
	})
}
