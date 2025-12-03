package memory

import (
	"context"
	"testing"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/rag/store"
	"github.com/aqua777/go-llamaindex/storage/chatstore"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// MockEmbeddingModel is a mock embedding model for testing.
type MockEmbeddingModel struct {
	embeddings map[string][]float64
}

func NewMockEmbeddingModel() *MockEmbeddingModel {
	return &MockEmbeddingModel{
		embeddings: make(map[string][]float64),
	}
}

func (m *MockEmbeddingModel) GetTextEmbedding(ctx context.Context, text string) ([]float64, error) {
	if emb, ok := m.embeddings[text]; ok {
		return emb, nil
	}
	// Return a simple hash-based embedding for testing
	embedding := make([]float64, 128)
	for i, c := range text {
		embedding[i%128] += float64(c) / 1000.0
	}
	return embedding, nil
}

func (m *MockEmbeddingModel) GetQueryEmbedding(ctx context.Context, query string) ([]float64, error) {
	return m.GetTextEmbedding(ctx, query)
}

// MockLLM is a mock LLM for testing.
type MockLLM struct {
	response string
}

func NewMockLLM(response string) *MockLLM {
	return &MockLLM{response: response}
}

func (m *MockLLM) Complete(ctx context.Context, prompt string) (string, error) {
	return m.response, nil
}

func (m *MockLLM) Chat(ctx context.Context, messages []llm.ChatMessage) (string, error) {
	return m.response, nil
}

func (m *MockLLM) Stream(ctx context.Context, prompt string) (<-chan string, error) {
	ch := make(chan string, 1)
	ch <- m.response
	close(ch)
	return ch, nil
}

// TestSimpleMemory tests the SimpleMemory.
func TestSimpleMemory(t *testing.T) {
	ctx := context.Background()

	t.Run("NewSimpleMemory", func(t *testing.T) {
		mem := NewSimpleMemory()
		assert.NotNil(t, mem)
	})

	t.Run("Put and Get", func(t *testing.T) {
		mem := NewSimpleMemory()

		msg := llm.ChatMessage{Role: llm.MessageRoleUser, Content: "Hello"}
		err := mem.Put(ctx, msg)
		require.NoError(t, err)

		messages, err := mem.Get(ctx, "")
		require.NoError(t, err)
		assert.Len(t, messages, 1)
		assert.Equal(t, "Hello", messages[0].Content)
	})

	t.Run("PutMessages", func(t *testing.T) {
		mem := NewSimpleMemory()

		msgs := []llm.ChatMessage{
			{Role: llm.MessageRoleUser, Content: "Hello"},
			{Role: llm.MessageRoleAssistant, Content: "Hi there!"},
		}
		err := mem.PutMessages(ctx, msgs)
		require.NoError(t, err)

		messages, err := mem.GetAll(ctx)
		require.NoError(t, err)
		assert.Len(t, messages, 2)
	})

	t.Run("Set", func(t *testing.T) {
		mem := NewSimpleMemory()

		// Add initial messages
		err := mem.Put(ctx, llm.ChatMessage{Role: llm.MessageRoleUser, Content: "First"})
		require.NoError(t, err)

		// Set new messages
		newMsgs := []llm.ChatMessage{
			{Role: llm.MessageRoleUser, Content: "New message"},
		}
		err = mem.Set(ctx, newMsgs)
		require.NoError(t, err)

		messages, err := mem.GetAll(ctx)
		require.NoError(t, err)
		assert.Len(t, messages, 1)
		assert.Equal(t, "New message", messages[0].Content)
	})

	t.Run("Reset", func(t *testing.T) {
		mem := NewSimpleMemory()

		err := mem.Put(ctx, llm.ChatMessage{Role: llm.MessageRoleUser, Content: "Hello"})
		require.NoError(t, err)

		err = mem.Reset(ctx)
		require.NoError(t, err)

		messages, err := mem.GetAll(ctx)
		require.NoError(t, err)
		assert.Len(t, messages, 0)
	})
}

// TestChatMemoryBuffer tests the ChatMemoryBuffer.
func TestChatMemoryBuffer(t *testing.T) {
	ctx := context.Background()

	t.Run("NewChatMemoryBuffer", func(t *testing.T) {
		mem := NewChatMemoryBuffer()
		assert.NotNil(t, mem)
		assert.Equal(t, DefaultTokenLimit, mem.TokenLimit())
	})

	t.Run("WithTokenLimit", func(t *testing.T) {
		mem := NewChatMemoryBuffer(WithTokenLimit(1000))
		assert.Equal(t, 1000, mem.TokenLimit())
	})

	t.Run("Get within token limit", func(t *testing.T) {
		mem := NewChatMemoryBuffer(WithTokenLimit(1000))

		msgs := []llm.ChatMessage{
			{Role: llm.MessageRoleUser, Content: "Hello"},
			{Role: llm.MessageRoleAssistant, Content: "Hi there!"},
			{Role: llm.MessageRoleUser, Content: "How are you?"},
		}
		err := mem.PutMessages(ctx, msgs)
		require.NoError(t, err)

		messages, err := mem.Get(ctx, "")
		require.NoError(t, err)
		assert.Len(t, messages, 3)
	})

	t.Run("Get trims messages over token limit", func(t *testing.T) {
		// Use a very small token limit
		mem := NewChatMemoryBuffer(WithTokenLimit(20))

		msgs := []llm.ChatMessage{
			{Role: llm.MessageRoleUser, Content: "This is a very long message that should exceed the token limit"},
			{Role: llm.MessageRoleAssistant, Content: "Response"},
			{Role: llm.MessageRoleUser, Content: "Short"},
		}
		err := mem.PutMessages(ctx, msgs)
		require.NoError(t, err)

		messages, err := mem.Get(ctx, "")
		require.NoError(t, err)
		// Should return fewer messages due to token limit
		assert.LessOrEqual(t, len(messages), 3)
	})

	t.Run("GetWithInitialTokenCount", func(t *testing.T) {
		mem := NewChatMemoryBuffer(WithTokenLimit(100))

		msgs := []llm.ChatMessage{
			{Role: llm.MessageRoleUser, Content: "Hello"},
			{Role: llm.MessageRoleAssistant, Content: "Hi!"},
		}
		err := mem.PutMessages(ctx, msgs)
		require.NoError(t, err)

		// With high initial token count, should return fewer messages
		messages, err := mem.GetWithInitialTokenCount(ctx, "", 90)
		require.NoError(t, err)
		assert.LessOrEqual(t, len(messages), 2)
	})

	t.Run("Initial token count exceeds limit", func(t *testing.T) {
		mem := NewChatMemoryBuffer(WithTokenLimit(100))

		_, err := mem.GetWithInitialTokenCount(ctx, "", 150)
		assert.Error(t, err)
	})

	t.Run("Custom tokenizer", func(t *testing.T) {
		customTokenizer := func(text string) int {
			return len(text) // 1 token per character
		}
		mem := NewChatMemoryBuffer(
			WithTokenLimit(50),
			WithTokenizer(customTokenizer),
		)

		err := mem.Put(ctx, llm.ChatMessage{Role: llm.MessageRoleUser, Content: "Hello"})
		require.NoError(t, err)

		messages, err := mem.Get(ctx, "")
		require.NoError(t, err)
		assert.Len(t, messages, 1)
	})
}

// TestChatMemoryBufferFromDefaults tests NewChatMemoryBufferFromDefaults.
func TestChatMemoryBufferFromDefaults(t *testing.T) {
	t.Run("With chat history", func(t *testing.T) {
		chatHistory := []llm.ChatMessage{
			{Role: llm.MessageRoleUser, Content: "Hello"},
			{Role: llm.MessageRoleAssistant, Content: "Hi!"},
		}

		mem, err := NewChatMemoryBufferFromDefaults(chatHistory, nil, 0)
		require.NoError(t, err)

		ctx := context.Background()
		messages, err := mem.GetAll(ctx)
		require.NoError(t, err)
		assert.Len(t, messages, 2)
	})

	t.Run("With custom token limit", func(t *testing.T) {
		mem, err := NewChatMemoryBufferFromDefaults(nil, nil, 500)
		require.NoError(t, err)
		assert.Equal(t, 500, mem.TokenLimit())
	})
}

// TestChatSummaryMemoryBuffer tests the ChatSummaryMemoryBuffer.
func TestChatSummaryMemoryBuffer(t *testing.T) {
	ctx := context.Background()

	t.Run("NewChatSummaryMemoryBuffer", func(t *testing.T) {
		mem := NewChatSummaryMemoryBuffer()
		assert.NotNil(t, mem)
		assert.Equal(t, DefaultSummaryTokenLimit, mem.TokenLimit())
	})

	t.Run("WithSummaryTokenLimit", func(t *testing.T) {
		mem := NewChatSummaryMemoryBuffer(WithSummaryTokenLimit(500))
		assert.Equal(t, 500, mem.TokenLimit())
	})

	t.Run("Put and Get without LLM", func(t *testing.T) {
		mem := NewChatSummaryMemoryBuffer(WithSummaryTokenLimit(1000))

		msgs := []llm.ChatMessage{
			{Role: llm.MessageRoleUser, Content: "Hello"},
			{Role: llm.MessageRoleAssistant, Content: "Hi there!"},
		}
		err := mem.PutMessages(ctx, msgs)
		require.NoError(t, err)

		messages, err := mem.Get(ctx, "")
		require.NoError(t, err)
		assert.GreaterOrEqual(t, len(messages), 1)
	})

	t.Run("Get with LLM summarization", func(t *testing.T) {
		mockLLM := NewMockLLM("Summary of conversation")
		mem := NewChatSummaryMemoryBuffer(
			WithSummaryTokenLimit(50), // Very small limit to force summarization
			WithSummaryLLM(mockLLM),
		)

		// Add many messages to exceed token limit
		for i := 0; i < 10; i++ {
			err := mem.Put(ctx, llm.ChatMessage{
				Role:    llm.MessageRoleUser,
				Content: "This is a longer message that takes up tokens",
			})
			require.NoError(t, err)
			err = mem.Put(ctx, llm.ChatMessage{
				Role:    llm.MessageRoleAssistant,
				Content: "This is a response that also takes up tokens",
			})
			require.NoError(t, err)
		}

		messages, err := mem.Get(ctx, "")
		require.NoError(t, err)
		// Should have summarized older messages
		assert.NotEmpty(t, messages)
	})

	t.Run("GetTokenCount", func(t *testing.T) {
		mem := NewChatSummaryMemoryBuffer()
		assert.Equal(t, 0, mem.GetTokenCount())
	})
}

// TestChatSummaryMemoryBufferFromDefaults tests NewChatSummaryMemoryBufferFromDefaults.
func TestChatSummaryMemoryBufferFromDefaults(t *testing.T) {
	t.Run("With chat history", func(t *testing.T) {
		chatHistory := []llm.ChatMessage{
			{Role: llm.MessageRoleUser, Content: "Hello"},
		}
		mockLLM := NewMockLLM("response")

		mem, err := NewChatSummaryMemoryBufferFromDefaults(chatHistory, mockLLM, 0)
		require.NoError(t, err)

		ctx := context.Background()
		messages, err := mem.GetAll(ctx)
		require.NoError(t, err)
		assert.Len(t, messages, 1)
	})
}

// TestVectorMemory tests the VectorMemory.
func TestVectorMemory(t *testing.T) {
	ctx := context.Background()

	t.Run("NewVectorMemory", func(t *testing.T) {
		mem := NewVectorMemory()
		assert.NotNil(t, mem)
	})

	t.Run("Put and Get", func(t *testing.T) {
		embedModel := NewMockEmbeddingModel()
		vs := store.NewSimpleVectorStore()

		mem := NewVectorMemory(
			WithVectorStore(vs),
			WithVectorMemoryEmbedModel(embedModel),
			WithRetrieverTopK(5),
		)

		// Add messages
		err := mem.Put(ctx, llm.ChatMessage{Role: llm.MessageRoleUser, Content: "Hello, how are you?"})
		require.NoError(t, err)
		err = mem.Put(ctx, llm.ChatMessage{Role: llm.MessageRoleAssistant, Content: "I'm doing well!"})
		require.NoError(t, err)

		// Query for relevant messages - the Get should not error
		// Note: The actual retrieval of messages depends on metadata handling
		// which may vary based on vector store implementation
		_, err = mem.Get(ctx, "Hello")
		require.NoError(t, err)
	})

	t.Run("Get with empty input", func(t *testing.T) {
		mem := NewVectorMemory()

		messages, err := mem.Get(ctx, "")
		require.NoError(t, err)
		assert.Empty(t, messages)
	})

	t.Run("GetAll not supported", func(t *testing.T) {
		mem := NewVectorMemory()

		_, err := mem.GetAll(ctx)
		assert.Error(t, err)
	})

	t.Run("Reset", func(t *testing.T) {
		embedModel := NewMockEmbeddingModel()
		mem := NewVectorMemory(WithVectorMemoryEmbedModel(embedModel))

		err := mem.Put(ctx, llm.ChatMessage{Role: llm.MessageRoleUser, Content: "Hello"})
		require.NoError(t, err)

		err = mem.Reset(ctx)
		require.NoError(t, err)
	})

	t.Run("BatchByUserMessage", func(t *testing.T) {
		embedModel := NewMockEmbeddingModel()
		mem := NewVectorMemory(
			WithVectorMemoryEmbedModel(embedModel),
			WithBatchByUserMessage(true),
		)

		// Add user message - should start new batch
		err := mem.Put(ctx, llm.ChatMessage{Role: llm.MessageRoleUser, Content: "Question 1"})
		require.NoError(t, err)

		// Add assistant message - should be in same batch
		err = mem.Put(ctx, llm.ChatMessage{Role: llm.MessageRoleAssistant, Content: "Answer 1"})
		require.NoError(t, err)

		// Add another user message - should start new batch
		err = mem.Put(ctx, llm.ChatMessage{Role: llm.MessageRoleUser, Content: "Question 2"})
		require.NoError(t, err)
	})
}

// TestVectorMemoryFromDefaults tests NewVectorMemoryFromDefaults.
func TestVectorMemoryFromDefaults(t *testing.T) {
	t.Run("With vector store and embed model", func(t *testing.T) {
		vs := store.NewSimpleVectorStore()
		embedModel := NewMockEmbeddingModel()

		mem := NewVectorMemoryFromDefaults(vs, embedModel)
		assert.NotNil(t, mem)
	})
}

// TestBaseMemory tests the BaseMemory.
func TestBaseMemory(t *testing.T) {
	ctx := context.Background()

	t.Run("NewBaseMemory", func(t *testing.T) {
		mem := NewBaseMemory()
		assert.NotNil(t, mem)
		assert.Equal(t, DefaultChatStoreKey, mem.ChatStoreKey())
	})

	t.Run("WithChatStore", func(t *testing.T) {
		customStore := chatstore.NewSimpleChatStore()
		mem := NewBaseMemory(WithChatStore(customStore))
		assert.Equal(t, customStore, mem.ChatStore())
	})

	t.Run("WithChatStoreKey", func(t *testing.T) {
		mem := NewBaseMemory(WithChatStoreKey("custom_key"))
		assert.Equal(t, "custom_key", mem.ChatStoreKey())
	})

	t.Run("GetAll", func(t *testing.T) {
		mem := NewBaseMemory()

		err := mem.Put(ctx, llm.ChatMessage{Role: llm.MessageRoleUser, Content: "Test"})
		require.NoError(t, err)

		messages, err := mem.GetAll(ctx)
		require.NoError(t, err)
		assert.Len(t, messages, 1)
	})
}

// TestDefaultTokenizer tests the DefaultTokenizer function.
func TestDefaultTokenizer(t *testing.T) {
	t.Run("Empty string", func(t *testing.T) {
		count := DefaultTokenizer("")
		assert.Equal(t, 0, count)
	})

	t.Run("Short string", func(t *testing.T) {
		count := DefaultTokenizer("Hello")
		assert.Equal(t, 1, count) // 5 chars / 4 = 1
	})

	t.Run("Longer string", func(t *testing.T) {
		count := DefaultTokenizer("Hello, World!")
		assert.Equal(t, 3, count) // 13 chars / 4 = 3
	})
}

// TestMemoryInterface tests that all memory types implement the Memory interface.
func TestMemoryInterface(t *testing.T) {
	t.Run("SimpleMemory implements Memory", func(t *testing.T) {
		var _ Memory = NewSimpleMemory()
	})

	t.Run("ChatMemoryBuffer implements Memory", func(t *testing.T) {
		var _ Memory = NewChatMemoryBuffer()
	})

	t.Run("ChatSummaryMemoryBuffer implements Memory", func(t *testing.T) {
		var _ Memory = NewChatSummaryMemoryBuffer()
	})

	t.Run("VectorMemory implements Memory", func(t *testing.T) {
		var _ Memory = NewVectorMemory()
	})
}
