package chatengine

import (
	"context"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/memory"
	"github.com/aqua777/go-llamaindex/rag/retriever"
	"github.com/aqua777/go-llamaindex/schema"
)

const (
	// DefaultContextTemplate is the default template for context.
	DefaultContextTemplate = `Use the context information below to assist the user.
--------------------
%s
--------------------
`
)

// ContextChatEngine is a chat engine that uses a retriever to provide context.
type ContextChatEngine struct {
	*BaseChatEngine
	memory          memory.Memory
	retriever       retriever.Retriever
	contextTemplate string
}

// ContextChatEngineOption configures a ContextChatEngine.
type ContextChatEngineOption func(*ContextChatEngine)

// WithContextChatEngineLLM sets the LLM.
func WithContextChatEngineLLM(l llm.LLM) ContextChatEngineOption {
	return func(e *ContextChatEngine) {
		e.llm = l
	}
}

// WithContextChatEngineMemory sets the memory.
func WithContextChatEngineMemory(m memory.Memory) ContextChatEngineOption {
	return func(e *ContextChatEngine) {
		e.memory = m
	}
}

// WithContextChatEngineRetriever sets the retriever.
func WithContextChatEngineRetriever(r retriever.Retriever) ContextChatEngineOption {
	return func(e *ContextChatEngine) {
		e.retriever = r
	}
}

// WithContextChatEngineSystemPrompt sets the system prompt.
func WithContextChatEngineSystemPrompt(prompt string) ContextChatEngineOption {
	return func(e *ContextChatEngine) {
		e.prefixMessages = []llm.ChatMessage{
			{Role: llm.MessageRoleSystem, Content: prompt},
		}
	}
}

// WithContextChatEnginePrefixMessages sets the prefix messages.
func WithContextChatEnginePrefixMessages(messages []llm.ChatMessage) ContextChatEngineOption {
	return func(e *ContextChatEngine) {
		e.prefixMessages = messages
	}
}

// WithContextTemplate sets the context template.
func WithContextTemplate(template string) ContextChatEngineOption {
	return func(e *ContextChatEngine) {
		e.contextTemplate = template
	}
}

// NewContextChatEngine creates a new ContextChatEngine.
func NewContextChatEngine(opts ...ContextChatEngineOption) *ContextChatEngine {
	e := &ContextChatEngine{
		BaseChatEngine:  NewBaseChatEngine(),
		memory:          memory.NewSimpleMemory(),
		contextTemplate: DefaultContextTemplate,
	}

	for _, opt := range opts {
		opt(e)
	}

	return e
}

// NewContextChatEngineFromDefaults creates a ContextChatEngine with defaults.
func NewContextChatEngineFromDefaults(
	ret retriever.Retriever,
	llmModel llm.LLM,
	chatHistory []llm.ChatMessage,
	systemPrompt string,
	opts ...ContextChatEngineOption,
) (*ContextChatEngine, error) {
	// Create memory with chat history
	mem := memory.NewChatMemoryBuffer()
	if len(chatHistory) > 0 {
		ctx := context.Background()
		if err := mem.Set(ctx, chatHistory); err != nil {
			return nil, err
		}
	}

	// Build options
	allOpts := []ContextChatEngineOption{
		WithContextChatEngineLLM(llmModel),
		WithContextChatEngineMemory(mem),
		WithContextChatEngineRetriever(ret),
	}

	if systemPrompt != "" {
		allOpts = append(allOpts, WithContextChatEngineSystemPrompt(systemPrompt))
	}

	allOpts = append(allOpts, opts...)

	return NewContextChatEngine(allOpts...), nil
}

// Chat sends a message and returns a response.
func (e *ContextChatEngine) Chat(ctx context.Context, message string) (*ChatResponse, error) {
	return e.ChatWithHistory(ctx, message, nil)
}

// ChatWithHistory sends a message with explicit chat history.
func (e *ContextChatEngine) ChatWithHistory(ctx context.Context, message string, chatHistory []llm.ChatMessage) (*ChatResponse, error) {
	if e.llm == nil {
		return nil, fmt.Errorf("LLM not configured")
	}
	if e.retriever == nil {
		return nil, fmt.Errorf("retriever not configured")
	}

	// Set chat history if provided
	if chatHistory != nil {
		if err := e.memory.Set(ctx, chatHistory); err != nil {
			return nil, err
		}
	}

	// Retrieve context nodes
	nodes, err := e.retriever.Retrieve(ctx, schema.QueryBundle{QueryString: message})
	if err != nil {
		return nil, err
	}

	// Build context string
	contextStr := e.buildContextString(nodes)

	// Build messages with context
	allMessages := e.buildMessagesWithContext(ctx, message, contextStr)

	// Call LLM
	response, err := e.llm.Chat(ctx, allMessages)
	if err != nil {
		return nil, err
	}

	// Add messages to memory
	userMessage := llm.ChatMessage{Role: llm.MessageRoleUser, Content: message}
	assistantMessage := llm.ChatMessage{Role: llm.MessageRoleAssistant, Content: response}
	if err := e.memory.Put(ctx, userMessage); err != nil {
		return nil, err
	}
	if err := e.memory.Put(ctx, assistantMessage); err != nil {
		return nil, err
	}

	// Build response
	chatResponse := NewChatResponse(response)
	chatResponse.SourceNodes = nodes
	chatResponse.Sources = []ToolSource{
		{
			ToolName:  "retriever",
			Content:   contextStr,
			RawInput:  map[string]interface{}{"message": message},
			RawOutput: nodes,
		},
	}

	return chatResponse, nil
}

// StreamChat sends a message and returns a streaming response.
func (e *ContextChatEngine) StreamChat(ctx context.Context, message string) (*StreamingChatResponse, error) {
	if e.llm == nil {
		return nil, fmt.Errorf("LLM not configured")
	}
	if e.retriever == nil {
		return nil, fmt.Errorf("retriever not configured")
	}

	// Retrieve context nodes
	nodes, err := e.retriever.Retrieve(ctx, schema.QueryBundle{QueryString: message})
	if err != nil {
		return nil, err
	}

	// Build context string
	contextStr := e.buildContextString(nodes)

	// Build messages with context
	allMessages := e.buildMessagesWithContext(ctx, message, contextStr)

	// Start streaming
	streamChan, err := e.llm.Stream(ctx, formatMessagesForStream(allMessages))
	if err != nil {
		return nil, err
	}

	// Create output channel that also writes to memory
	outputChan := make(chan string)
	go func() {
		defer close(outputChan)
		var fullResponse string
		for token := range streamChan {
			fullResponse += token
			outputChan <- token
		}
		// Add messages to memory
		userMessage := llm.ChatMessage{Role: llm.MessageRoleUser, Content: message}
		assistantMessage := llm.ChatMessage{Role: llm.MessageRoleAssistant, Content: fullResponse}
		_ = e.memory.Put(ctx, userMessage)
		_ = e.memory.Put(ctx, assistantMessage)
	}()

	streamResponse := NewStreamingChatResponse(outputChan)
	streamResponse.SourceNodes = nodes
	streamResponse.Sources = []ToolSource{
		{
			ToolName:  "retriever",
			Content:   contextStr,
			RawInput:  map[string]interface{}{"message": message},
			RawOutput: nodes,
		},
	}

	return streamResponse, nil
}

// Reset clears the conversation state.
func (e *ContextChatEngine) Reset(ctx context.Context) error {
	return e.memory.Reset(ctx)
}

// ChatHistory returns the current chat history.
func (e *ContextChatEngine) ChatHistory(ctx context.Context) ([]llm.ChatMessage, error) {
	return e.memory.GetAll(ctx)
}

// buildContextString builds a context string from nodes.
func (e *ContextChatEngine) buildContextString(nodes []schema.NodeWithScore) string {
	var parts []string
	for _, node := range nodes {
		content := node.Node.GetContent(schema.MetadataModeLLM)
		parts = append(parts, content)
	}
	return strings.Join(parts, "\n\n")
}

// buildMessagesWithContext builds messages with context injected.
func (e *ContextChatEngine) buildMessagesWithContext(ctx context.Context, message string, contextStr string) []llm.ChatMessage {
	var messages []llm.ChatMessage

	// Add prefix messages
	messages = append(messages, e.prefixMessages...)

	// Add context as a system message
	if contextStr != "" {
		contextMessage := llm.ChatMessage{
			Role:    llm.MessageRoleSystem,
			Content: fmt.Sprintf(e.contextTemplate, contextStr),
		}
		messages = append(messages, contextMessage)
	}

	// Get memory messages
	memoryMessages, _ := e.memory.Get(ctx, message)
	messages = append(messages, memoryMessages...)

	// Add current user message
	messages = append(messages, llm.ChatMessage{
		Role:    llm.MessageRoleUser,
		Content: message,
	})

	return messages
}

// Ensure ContextChatEngine implements ChatEngine.
var _ ChatEngine = (*ContextChatEngine)(nil)
