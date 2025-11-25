package llm

import "context"

// MockLLM is a mock implementation of the LLM interface.
type MockLLM struct {
	Response string
	Err      error
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

