package memory

import (
	"context"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/storage/chatstore"
)

const (
	// DefaultSummaryTokenLimit is the default token limit for summary buffer.
	DefaultSummaryTokenLimit = 2000
	// DefaultSummarizePrompt is the default prompt for summarization.
	DefaultSummarizePrompt = "The following is a conversation between the user and assistant. Write a concise summary about the contents of this conversation."
)

// ChatSummaryMemoryBuffer stores chat history with summarization of older messages.
type ChatSummaryMemoryBuffer struct {
	*BaseMemory
	tokenLimit         int
	tokenizerFn        TokenizerFunc
	llm                llm.LLM
	summarizePrompt    string
	countInitialTokens bool
	tokenCount         int
}

// ChatSummaryMemoryBufferOption configures a ChatSummaryMemoryBuffer.
type ChatSummaryMemoryBufferOption func(*ChatSummaryMemoryBuffer)

// WithSummaryTokenLimit sets the token limit.
func WithSummaryTokenLimit(limit int) ChatSummaryMemoryBufferOption {
	return func(m *ChatSummaryMemoryBuffer) {
		m.tokenLimit = limit
	}
}

// WithSummaryTokenizer sets the tokenizer function.
func WithSummaryTokenizer(fn TokenizerFunc) ChatSummaryMemoryBufferOption {
	return func(m *ChatSummaryMemoryBuffer) {
		m.tokenizerFn = fn
	}
}

// WithSummaryLLM sets the LLM for summarization.
func WithSummaryLLM(l llm.LLM) ChatSummaryMemoryBufferOption {
	return func(m *ChatSummaryMemoryBuffer) {
		m.llm = l
	}
}

// WithSummarizePrompt sets the summarization prompt.
func WithSummarizePrompt(prompt string) ChatSummaryMemoryBufferOption {
	return func(m *ChatSummaryMemoryBuffer) {
		m.summarizePrompt = prompt
	}
}

// WithCountInitialTokens sets whether to count initial tokens.
func WithCountInitialTokens(count bool) ChatSummaryMemoryBufferOption {
	return func(m *ChatSummaryMemoryBuffer) {
		m.countInitialTokens = count
	}
}

// WithSummaryChatStore sets the chat store.
func WithSummaryChatStore(store chatstore.ChatStore) ChatSummaryMemoryBufferOption {
	return func(m *ChatSummaryMemoryBuffer) {
		m.chatStore = store
	}
}

// WithSummaryChatStoreKey sets the chat store key.
func WithSummaryChatStoreKey(key string) ChatSummaryMemoryBufferOption {
	return func(m *ChatSummaryMemoryBuffer) {
		m.chatStoreKey = key
	}
}

// NewChatSummaryMemoryBuffer creates a new ChatSummaryMemoryBuffer.
func NewChatSummaryMemoryBuffer(opts ...ChatSummaryMemoryBufferOption) *ChatSummaryMemoryBuffer {
	m := &ChatSummaryMemoryBuffer{
		BaseMemory:         NewBaseMemory(),
		tokenLimit:         DefaultSummaryTokenLimit,
		tokenizerFn:        DefaultTokenizer,
		summarizePrompt:    DefaultSummarizePrompt,
		countInitialTokens: false,
	}

	for _, opt := range opts {
		opt(m)
	}

	return m
}

// NewChatSummaryMemoryBufferFromDefaults creates a ChatSummaryMemoryBuffer with defaults.
func NewChatSummaryMemoryBufferFromDefaults(
	chatHistory []llm.ChatMessage,
	llmModel llm.LLM,
	tokenLimit int,
	opts ...ChatSummaryMemoryBufferOption,
) (*ChatSummaryMemoryBuffer, error) {
	// Determine token limit
	if tokenLimit <= 0 {
		if llmWithMeta, ok := llmModel.(llm.LLMWithMetadata); ok && llmWithMeta != nil {
			metadata := llmWithMeta.Metadata()
			tokenLimit = int(float64(metadata.ContextWindow) * DefaultTokenLimitRatio)
		} else {
			tokenLimit = DefaultSummaryTokenLimit
		}
	}

	m := NewChatSummaryMemoryBuffer(
		append(opts,
			WithSummaryTokenLimit(tokenLimit),
			WithSummaryLLM(llmModel),
		)...,
	)

	// Set initial chat history
	if len(chatHistory) > 0 {
		ctx := context.Background()
		if err := m.Set(ctx, chatHistory); err != nil {
			return nil, err
		}
	}

	return m, nil
}

// TokenLimit returns the token limit.
func (m *ChatSummaryMemoryBuffer) TokenLimit() int {
	return m.tokenLimit
}

// GetTokenCount returns the current token count.
func (m *ChatSummaryMemoryBuffer) GetTokenCount() int {
	return m.tokenCount
}

// Get retrieves chat history with summarization of older messages.
func (m *ChatSummaryMemoryBuffer) Get(ctx context.Context, input string) ([]llm.ChatMessage, error) {
	return m.GetWithInitialTokenCount(ctx, input, 0)
}

// GetWithInitialTokenCount retrieves chat history accounting for initial tokens.
func (m *ChatSummaryMemoryBuffer) GetWithInitialTokenCount(ctx context.Context, input string, initialTokenCount int) ([]llm.ChatMessage, error) {
	chatHistory, err := m.GetAll(ctx)
	if err != nil {
		return nil, err
	}

	if len(chatHistory) == 0 {
		return chatHistory, nil
	}

	// Reset token count
	if m.countInitialTokens {
		if initialTokenCount > m.tokenLimit {
			return nil, fmt.Errorf("initial token count %d exceeds token limit %d", initialTokenCount, m.tokenLimit)
		}
		m.tokenCount = initialTokenCount
	} else {
		m.tokenCount = 0
	}

	// Split messages into full text and to-be-summarized
	fullTextMessages, toSummarize := m.splitMessagesSummaryOrFullText(chatHistory)

	var updatedHistory []llm.ChatMessage
	if m.llm == nil || len(toSummarize) == 0 {
		// No LLM or nothing to summarize - just use full text messages
		updatedHistory = fullTextMessages
	} else {
		// Summarize older messages
		summaryMsg, err := m.summarizeOldestChatHistory(ctx, toSummarize)
		if err != nil {
			return nil, err
		}
		updatedHistory = append([]llm.ChatMessage{summaryMsg}, fullTextMessages...)
	}

	// Update stored history
	if err := m.Reset(ctx); err != nil {
		return nil, err
	}
	m.tokenCount = 0
	if err := m.Set(ctx, updatedHistory); err != nil {
		return nil, err
	}

	return updatedHistory, nil
}

// splitMessagesSummaryOrFullText splits messages into full text and to-be-summarized.
func (m *ChatSummaryMemoryBuffer) splitMessagesSummaryOrFullText(chatHistory []llm.ChatMessage) ([]llm.ChatMessage, []llm.ChatMessage) {
	var fullTextMessages []llm.ChatMessage
	history := make([]llm.ChatMessage, len(chatHistory))
	copy(history, chatHistory)

	// Traverse from end, adding messages until token limit is reached
	for len(history) > 0 {
		lastMsg := history[len(history)-1]
		msgTokens := m.tokenCountForMessages([]llm.ChatMessage{lastMsg})

		if m.tokenCount+msgTokens > m.tokenLimit {
			break
		}

		m.tokenCount += msgTokens
		fullTextMessages = append([]llm.ChatMessage{lastMsg}, fullTextMessages...)
		history = history[:len(history)-1]
	}

	// Handle assistant/tool messages at the start
	m.handleAssistantAndToolMessages(&fullTextMessages, &history)

	return fullTextMessages, history
}

// handleAssistantAndToolMessages ensures first message isn't assistant/tool.
func (m *ChatSummaryMemoryBuffer) handleAssistantAndToolMessages(fullText *[]llm.ChatMessage, toSummarize *[]llm.ChatMessage) {
	for len(*fullText) > 0 && ((*fullText)[0].Role == llm.MessageRoleAssistant || (*fullText)[0].Role == llm.MessageRoleTool) {
		*toSummarize = append(*toSummarize, (*fullText)[0])
		*fullText = (*fullText)[1:]
	}
}

// summarizeOldestChatHistory uses LLM to summarize older messages.
func (m *ChatSummaryMemoryBuffer) summarizeOldestChatHistory(ctx context.Context, toSummarize []llm.ChatMessage) (llm.ChatMessage, error) {
	// If only a system message, return it as-is
	if len(toSummarize) == 1 && toSummarize[0].Role == llm.MessageRoleSystem {
		return toSummarize[0], nil
	}

	// Build prompt for summarization
	prompt := m.getPromptToSummarize(toSummarize)

	summarizeMessages := []llm.ChatMessage{
		{Role: llm.MessageRoleSystem, Content: m.summarizePrompt},
		{Role: llm.MessageRoleUser, Content: prompt},
	}

	response, err := m.llm.Chat(ctx, summarizeMessages)
	if err != nil {
		return llm.ChatMessage{}, err
	}

	return llm.ChatMessage{
		Role:    llm.MessageRoleSystem,
		Content: response,
	}, nil
}

// getPromptToSummarize builds the prompt for summarization.
func (m *ChatSummaryMemoryBuffer) getPromptToSummarize(messages []llm.ChatMessage) string {
	var sb strings.Builder
	sb.WriteString("Transcript so far:\n")

	for _, msg := range messages {
		sb.WriteString(string(msg.Role))
		sb.WriteString(": ")
		if msg.Content != "" {
			sb.WriteString(msg.Content)
		}
		sb.WriteString("\n\n")
	}

	return sb.String()
}

// tokenCountForMessages counts tokens in a list of messages.
func (m *ChatSummaryMemoryBuffer) tokenCountForMessages(messages []llm.ChatMessage) int {
	if len(messages) == 0 {
		return 0
	}

	var totalContent string
	for _, msg := range messages {
		totalContent += " " + msg.Content
	}

	return m.tokenizerFn(totalContent)
}

// Ensure ChatSummaryMemoryBuffer implements Memory.
var _ Memory = (*ChatSummaryMemoryBuffer)(nil)
