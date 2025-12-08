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
	// DefaultCondensePromptTemplate is the default template for condensing questions.
	DefaultCondensePromptTemplate = `Given the following conversation between a user and an AI assistant and a follow up question from user, rephrase the follow up question to be a standalone question.

Chat History:
%s

Follow Up Input: %s
Standalone question:`

	// DefaultContextPromptTemplate is the default template for context.
	DefaultContextPromptTemplate = `The following is a friendly conversation between a user and an AI assistant.
The assistant is talkative and provides lots of specific details from its context.
If the assistant does not know the answer to a question, it truthfully says it does not know.

Here are the relevant documents for the context:

%s

Instruction: Based on the above documents, provide a detailed answer for the user question below.
Answer "don't know" if not present in the document.`
)

// CondensePlusContextChatEngine condenses conversation history and uses context.
type CondensePlusContextChatEngine struct {
	*BaseChatEngine
	memory                 memory.Memory
	retriever              retriever.Retriever
	condensePromptTemplate string
	contextPromptTemplate  string
	skipCondense           bool
	verbose                bool
}

// CondensePlusContextChatEngineOption configures a CondensePlusContextChatEngine.
type CondensePlusContextChatEngineOption func(*CondensePlusContextChatEngine)

// WithCondensePlusContextLLM sets the LLM.
func WithCondensePlusContextLLM(l llm.LLM) CondensePlusContextChatEngineOption {
	return func(e *CondensePlusContextChatEngine) {
		e.llm = l
	}
}

// WithCondensePlusContextMemory sets the memory.
func WithCondensePlusContextMemory(m memory.Memory) CondensePlusContextChatEngineOption {
	return func(e *CondensePlusContextChatEngine) {
		e.memory = m
	}
}

// WithCondensePlusContextRetriever sets the retriever.
func WithCondensePlusContextRetriever(r retriever.Retriever) CondensePlusContextChatEngineOption {
	return func(e *CondensePlusContextChatEngine) {
		e.retriever = r
	}
}

// WithCondensePlusContextSystemPrompt sets the system prompt.
func WithCondensePlusContextSystemPrompt(prompt string) CondensePlusContextChatEngineOption {
	return func(e *CondensePlusContextChatEngine) {
		e.prefixMessages = []llm.ChatMessage{
			{Role: llm.MessageRoleSystem, Content: prompt},
		}
	}
}

// WithCondensePromptTemplate sets the condense prompt template.
func WithCondensePromptTemplate(template string) CondensePlusContextChatEngineOption {
	return func(e *CondensePlusContextChatEngine) {
		e.condensePromptTemplate = template
	}
}

// WithCondensePlusContextTemplate sets the context prompt template.
func WithCondensePlusContextTemplate(template string) CondensePlusContextChatEngineOption {
	return func(e *CondensePlusContextChatEngine) {
		e.contextPromptTemplate = template
	}
}

// WithSkipCondense sets whether to skip the condense step.
func WithSkipCondense(skip bool) CondensePlusContextChatEngineOption {
	return func(e *CondensePlusContextChatEngine) {
		e.skipCondense = skip
	}
}

// WithCondensePlusContextVerbose sets verbose mode.
func WithCondensePlusContextVerbose(verbose bool) CondensePlusContextChatEngineOption {
	return func(e *CondensePlusContextChatEngine) {
		e.verbose = verbose
	}
}

// NewCondensePlusContextChatEngine creates a new CondensePlusContextChatEngine.
func NewCondensePlusContextChatEngine(opts ...CondensePlusContextChatEngineOption) *CondensePlusContextChatEngine {
	e := &CondensePlusContextChatEngine{
		BaseChatEngine:         NewBaseChatEngine(),
		memory:                 memory.NewSimpleMemory(),
		condensePromptTemplate: DefaultCondensePromptTemplate,
		contextPromptTemplate:  DefaultContextPromptTemplate,
		skipCondense:           false,
		verbose:                false,
	}

	for _, opt := range opts {
		opt(e)
	}

	return e
}

// NewCondensePlusContextChatEngineFromDefaults creates a CondensePlusContextChatEngine with defaults.
func NewCondensePlusContextChatEngineFromDefaults(
	ret retriever.Retriever,
	llmModel llm.LLM,
	chatHistory []llm.ChatMessage,
	systemPrompt string,
	opts ...CondensePlusContextChatEngineOption,
) (*CondensePlusContextChatEngine, error) {
	// Create memory with chat history
	mem := memory.NewChatMemoryBuffer()
	if len(chatHistory) > 0 {
		ctx := context.Background()
		if err := mem.Set(ctx, chatHistory); err != nil {
			return nil, err
		}
	}

	// Build options
	allOpts := []CondensePlusContextChatEngineOption{
		WithCondensePlusContextLLM(llmModel),
		WithCondensePlusContextMemory(mem),
		WithCondensePlusContextRetriever(ret),
	}

	if systemPrompt != "" {
		allOpts = append(allOpts, WithCondensePlusContextSystemPrompt(systemPrompt))
	}

	allOpts = append(allOpts, opts...)

	return NewCondensePlusContextChatEngine(allOpts...), nil
}

// Chat sends a message and returns a response.
func (e *CondensePlusContextChatEngine) Chat(ctx context.Context, message string) (*ChatResponse, error) {
	return e.ChatWithHistory(ctx, message, nil)
}

// ChatWithHistory sends a message with explicit chat history.
func (e *CondensePlusContextChatEngine) ChatWithHistory(ctx context.Context, message string, chatHistory []llm.ChatMessage) (*ChatResponse, error) {
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

	// Get current chat history
	currentHistory, err := e.memory.Get(ctx, message)
	if err != nil {
		return nil, err
	}

	// Condense the question
	condensedQuestion, err := e.condenseQuestion(ctx, currentHistory, message)
	if err != nil {
		return nil, err
	}

	if e.verbose {
		fmt.Printf("Condensed question: %s\n", condensedQuestion)
	}

	// Retrieve context nodes using condensed question
	nodes, err := e.retriever.Retrieve(ctx, schema.QueryBundle{QueryString: condensedQuestion})
	if err != nil {
		return nil, err
	}

	// Build context string
	contextStr := e.buildContextString(nodes)

	// Build messages with context
	allMessages := e.buildMessagesWithContext(currentHistory, message, contextStr)

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
			RawInput:  map[string]interface{}{"message": condensedQuestion},
			RawOutput: nodes,
		},
	}

	return chatResponse, nil
}

// StreamChat sends a message and returns a streaming response.
func (e *CondensePlusContextChatEngine) StreamChat(ctx context.Context, message string) (*StreamingChatResponse, error) {
	if e.llm == nil {
		return nil, fmt.Errorf("LLM not configured")
	}
	if e.retriever == nil {
		return nil, fmt.Errorf("retriever not configured")
	}

	// Get current chat history
	currentHistory, err := e.memory.Get(ctx, message)
	if err != nil {
		return nil, err
	}

	// Condense the question
	condensedQuestion, err := e.condenseQuestion(ctx, currentHistory, message)
	if err != nil {
		return nil, err
	}

	if e.verbose {
		fmt.Printf("Condensed question: %s\n", condensedQuestion)
	}

	// Retrieve context nodes using condensed question
	nodes, err := e.retriever.Retrieve(ctx, schema.QueryBundle{QueryString: condensedQuestion})
	if err != nil {
		return nil, err
	}

	// Build context string
	contextStr := e.buildContextString(nodes)

	// Build messages with context
	allMessages := e.buildMessagesWithContext(currentHistory, message, contextStr)

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
			RawInput:  map[string]interface{}{"message": condensedQuestion},
			RawOutput: nodes,
		},
	}

	return streamResponse, nil
}

// Reset clears the conversation state.
func (e *CondensePlusContextChatEngine) Reset(ctx context.Context) error {
	return e.memory.Reset(ctx)
}

// ChatHistory returns the current chat history.
func (e *CondensePlusContextChatEngine) ChatHistory(ctx context.Context) ([]llm.ChatMessage, error) {
	return e.memory.GetAll(ctx)
}

// condenseQuestion condenses the chat history and latest message into a standalone question.
func (e *CondensePlusContextChatEngine) condenseQuestion(ctx context.Context, chatHistory []llm.ChatMessage, latestMessage string) (string, error) {
	// Skip condensing if configured or no history
	if e.skipCondense || len(chatHistory) == 0 {
		return latestMessage, nil
	}

	// Format chat history
	historyStr := e.formatChatHistory(chatHistory)

	// Build condense prompt
	prompt := fmt.Sprintf(e.condensePromptTemplate, historyStr, latestMessage)

	// Call LLM to condense
	condensed, err := e.llm.Complete(ctx, prompt)
	if err != nil {
		return "", err
	}

	return strings.TrimSpace(condensed), nil
}

// formatChatHistory formats chat history as a string.
func (e *CondensePlusContextChatEngine) formatChatHistory(history []llm.ChatMessage) string {
	var parts []string
	for _, msg := range history {
		parts = append(parts, fmt.Sprintf("%s: %s", msg.Role, msg.Content))
	}
	return strings.Join(parts, "\n")
}

// buildContextString builds a context string from nodes.
func (e *CondensePlusContextChatEngine) buildContextString(nodes []schema.NodeWithScore) string {
	var parts []string
	for _, node := range nodes {
		content := node.Node.GetContent(schema.MetadataModeLLM)
		parts = append(parts, content)
	}
	return strings.Join(parts, "\n\n")
}

// buildMessagesWithContext builds messages with context injected.
func (e *CondensePlusContextChatEngine) buildMessagesWithContext(chatHistory []llm.ChatMessage, message string, contextStr string) []llm.ChatMessage {
	var messages []llm.ChatMessage

	// Add prefix messages (system prompt)
	messages = append(messages, e.prefixMessages...)

	// Add context as a system message
	if contextStr != "" {
		contextMessage := llm.ChatMessage{
			Role:    llm.MessageRoleSystem,
			Content: fmt.Sprintf(e.contextPromptTemplate, contextStr),
		}
		messages = append(messages, contextMessage)
	}

	// Add chat history
	messages = append(messages, chatHistory...)

	// Add current user message
	messages = append(messages, llm.ChatMessage{
		Role:    llm.MessageRoleUser,
		Content: message,
	})

	return messages
}

// Ensure CondensePlusContextChatEngine implements ChatEngine.
var _ ChatEngine = (*CondensePlusContextChatEngine)(nil)
