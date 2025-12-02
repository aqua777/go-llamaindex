package llm

import "context"

// MockLLM is a mock implementation of the LLM interface.
// It can be configured to return specific responses or errors.
type MockLLM struct {
	// Response is the text response to return.
	Response string
	// Err is the error to return (if any).
	Err error
	// CompletionResponse is the full completion response (for tool calling tests).
	CompletionResponse *CompletionResponse
	// ModelMetadata is the metadata to return.
	ModelMetadata *LLMMetadata
	// ToolCallingSupported indicates if tool calling is supported.
	ToolCallingSupported bool
	// StructuredOutputSupported indicates if structured output is supported.
	StructuredOutputSupported bool
}

// NewMockLLM creates a new MockLLM with a simple response.
func NewMockLLM(response string) *MockLLM {
	return &MockLLM{Response: response}
}

// NewMockLLMWithError creates a new MockLLM that returns an error.
func NewMockLLMWithError(err error) *MockLLM {
	return &MockLLM{Err: err}
}

func (m *MockLLM) Complete(ctx context.Context, prompt string) (string, error) {
	return m.Response, m.Err
}

func (m *MockLLM) Chat(ctx context.Context, messages []ChatMessage) (string, error) {
	return m.Response, m.Err
}

func (m *MockLLM) Stream(ctx context.Context, prompt string) (<-chan string, error) {
	ch := make(chan string, 1)
	if m.Err != nil {
		close(ch)
		return ch, m.Err
	}
	ch <- m.Response
	close(ch)
	return ch, nil
}

// Metadata returns the mock model metadata.
func (m *MockLLM) Metadata() LLMMetadata {
	if m.ModelMetadata != nil {
		return *m.ModelMetadata
	}
	return DefaultLLMMetadata("mock-model")
}

// SupportsToolCalling returns whether tool calling is supported.
func (m *MockLLM) SupportsToolCalling() bool {
	return m.ToolCallingSupported
}

// SupportsStructuredOutput returns whether structured output is supported.
func (m *MockLLM) SupportsStructuredOutput() bool {
	return m.StructuredOutputSupported
}

// ChatWithTools returns a mock completion response with tool calls.
func (m *MockLLM) ChatWithTools(ctx context.Context, messages []ChatMessage, tools []*ToolMetadata, opts *ChatCompletionOptions) (CompletionResponse, error) {
	if m.Err != nil {
		return CompletionResponse{}, m.Err
	}
	if m.CompletionResponse != nil {
		return *m.CompletionResponse, nil
	}
	return NewCompletionResponse(m.Response), nil
}

// ChatWithFormat returns a mock response in the specified format.
func (m *MockLLM) ChatWithFormat(ctx context.Context, messages []ChatMessage, format *ResponseFormat) (string, error) {
	return m.Response, m.Err
}

// StreamChat returns a mock streaming response.
func (m *MockLLM) StreamChat(ctx context.Context, messages []ChatMessage) (<-chan StreamToken, error) {
	ch := make(chan StreamToken, 1)
	if m.Err != nil {
		close(ch)
		return ch, m.Err
	}
	ch <- StreamToken{Delta: m.Response, FinishReason: "stop"}
	close(ch)
	return ch, nil
}
