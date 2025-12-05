package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
)

const (
	// AnthropicAPIURL is the default Anthropic API endpoint.
	AnthropicAPIURL = "https://api.anthropic.com/v1"
	// AnthropicAPIVersion is the API version header value.
	AnthropicAPIVersion = "2023-06-01"
)

// Anthropic model constants.
const (
	Claude3Opus    = "claude-3-opus-20240229"
	Claude3Sonnet  = "claude-3-sonnet-20240229"
	Claude3Haiku   = "claude-3-haiku-20240307"
	Claude35Sonnet = "claude-3-5-sonnet-20241022"
	Claude35Haiku  = "claude-3-5-haiku-20241022"
)

// AnthropicLLM implements the LLM interface for Anthropic Claude models.
type AnthropicLLM struct {
	apiKey     string
	baseURL    string
	model      string
	maxTokens  int
	httpClient *http.Client
	logger     *slog.Logger
}

// AnthropicOption configures an AnthropicLLM.
type AnthropicOption func(*AnthropicLLM)

// WithAnthropicAPIKey sets the API key.
func WithAnthropicAPIKey(apiKey string) AnthropicOption {
	return func(a *AnthropicLLM) {
		a.apiKey = apiKey
	}
}

// WithAnthropicBaseURL sets the base URL.
func WithAnthropicBaseURL(baseURL string) AnthropicOption {
	return func(a *AnthropicLLM) {
		a.baseURL = baseURL
	}
}

// WithAnthropicModel sets the model.
func WithAnthropicModel(model string) AnthropicOption {
	return func(a *AnthropicLLM) {
		a.model = model
	}
}

// WithAnthropicMaxTokens sets the max tokens.
func WithAnthropicMaxTokens(maxTokens int) AnthropicOption {
	return func(a *AnthropicLLM) {
		a.maxTokens = maxTokens
	}
}

// WithAnthropicHTTPClient sets a custom HTTP client.
func WithAnthropicHTTPClient(client *http.Client) AnthropicOption {
	return func(a *AnthropicLLM) {
		a.httpClient = client
	}
}

// NewAnthropicLLM creates a new Anthropic LLM client.
func NewAnthropicLLM(opts ...AnthropicOption) *AnthropicLLM {
	a := &AnthropicLLM{
		apiKey:     os.Getenv("ANTHROPIC_API_KEY"),
		baseURL:    AnthropicAPIURL,
		model:      Claude35Sonnet,
		maxTokens:  4096,
		httpClient: http.DefaultClient,
		logger:     slog.New(slog.NewJSONHandler(os.Stdout, nil)),
	}

	for _, opt := range opts {
		opt(a)
	}

	return a
}

// anthropicMessage represents a message in the Anthropic API format.
type anthropicMessage struct {
	Role    string             `json:"role"`
	Content []anthropicContent `json:"content"`
}

// anthropicContent represents content in the Anthropic API format.
type anthropicContent struct {
	Type      string                 `json:"type"`
	Text      string                 `json:"text,omitempty"`
	ID        string                 `json:"id,omitempty"`
	Name      string                 `json:"name,omitempty"`
	Input     map[string]interface{} `json:"input,omitempty"`
	ToolUseID string                 `json:"tool_use_id,omitempty"`
	Content   string                 `json:"content,omitempty"`
}

// anthropicRequest represents a request to the Anthropic API.
type anthropicRequest struct {
	Model       string             `json:"model"`
	Messages    []anthropicMessage `json:"messages"`
	MaxTokens   int                `json:"max_tokens"`
	System      string             `json:"system,omitempty"`
	Temperature *float32           `json:"temperature,omitempty"`
	TopP        *float32           `json:"top_p,omitempty"`
	Stream      bool               `json:"stream,omitempty"`
	Tools       []anthropicTool    `json:"tools,omitempty"`
}

// anthropicTool represents a tool in the Anthropic API format.
type anthropicTool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	InputSchema map[string]interface{} `json:"input_schema"`
}

// anthropicResponse represents a response from the Anthropic API.
type anthropicResponse struct {
	ID           string             `json:"id"`
	Type         string             `json:"type"`
	Role         string             `json:"role"`
	Content      []anthropicContent `json:"content"`
	Model        string             `json:"model"`
	StopReason   string             `json:"stop_reason"`
	StopSequence string             `json:"stop_sequence,omitempty"`
	Usage        struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

// anthropicStreamEvent represents a streaming event from the Anthropic API.
type anthropicStreamEvent struct {
	Type         string `json:"type"`
	Index        int    `json:"index,omitempty"`
	ContentBlock *struct {
		Type string `json:"type"`
		Text string `json:"text,omitempty"`
	} `json:"content_block,omitempty"`
	Delta *struct {
		Type string `json:"type"`
		Text string `json:"text,omitempty"`
	} `json:"delta,omitempty"`
	Message *anthropicResponse `json:"message,omitempty"`
}

// anthropicError represents an error from the Anthropic API.
type anthropicError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// Complete generates a completion for a given prompt.
func (a *AnthropicLLM) Complete(ctx context.Context, prompt string) (string, error) {
	messages := []ChatMessage{NewUserMessage(prompt)}
	return a.Chat(ctx, messages)
}

// Chat generates a response for a list of chat messages.
func (a *AnthropicLLM) Chat(ctx context.Context, messages []ChatMessage) (string, error) {
	a.logger.Info("Chat called", "model", a.model, "message_count", len(messages))

	anthropicMessages, systemPrompt := a.convertMessages(messages)

	reqBody := anthropicRequest{
		Model:     a.model,
		Messages:  anthropicMessages,
		MaxTokens: a.maxTokens,
		System:    systemPrompt,
	}

	resp, err := a.doRequest(ctx, "/messages", reqBody)
	if err != nil {
		return "", err
	}

	// Extract text from response
	var text string
	for _, content := range resp.Content {
		if content.Type == "text" {
			text += content.Text
		}
	}

	return text, nil
}

// Stream generates a streaming completion for a given prompt.
func (a *AnthropicLLM) Stream(ctx context.Context, prompt string) (<-chan string, error) {
	a.logger.Info("Stream called", "model", a.model, "prompt_len", len(prompt))

	messages := []anthropicMessage{
		{
			Role:    "user",
			Content: []anthropicContent{{Type: "text", Text: prompt}},
		},
	}

	reqBody := anthropicRequest{
		Model:     a.model,
		Messages:  messages,
		MaxTokens: a.maxTokens,
		Stream:    true,
	}

	return a.doStreamRequest(ctx, "/messages", reqBody)
}

// Metadata returns information about the model's capabilities.
func (a *AnthropicLLM) Metadata() LLMMetadata {
	return getAnthropicModelMetadata(a.model)
}

// SupportsToolCalling returns true if the model supports tool calling.
func (a *AnthropicLLM) SupportsToolCalling() bool {
	return true // All Claude 3+ models support tool calling
}

// SupportsStructuredOutput returns true if the model supports structured output.
func (a *AnthropicLLM) SupportsStructuredOutput() bool {
	return true
}

// ChatWithTools generates a response that may include tool calls.
func (a *AnthropicLLM) ChatWithTools(ctx context.Context, messages []ChatMessage, tools []*ToolMetadata, opts *ChatCompletionOptions) (CompletionResponse, error) {
	a.logger.Info("ChatWithTools called", "model", a.model, "message_count", len(messages), "tool_count", len(tools))

	anthropicMessages, systemPrompt := a.convertMessages(messages)
	anthropicTools := a.convertTools(tools)

	reqBody := anthropicRequest{
		Model:     a.model,
		Messages:  anthropicMessages,
		MaxTokens: a.maxTokens,
		System:    systemPrompt,
		Tools:     anthropicTools,
	}

	if opts != nil {
		if opts.Temperature != nil {
			reqBody.Temperature = opts.Temperature
		}
		if opts.TopP != nil {
			reqBody.TopP = opts.TopP
		}
		if opts.MaxTokens != nil {
			reqBody.MaxTokens = *opts.MaxTokens
		}
	}

	resp, err := a.doRequest(ctx, "/messages", reqBody)
	if err != nil {
		return CompletionResponse{}, err
	}

	return a.convertResponse(resp), nil
}

// ChatWithFormat generates a response in the specified format.
func (a *AnthropicLLM) ChatWithFormat(ctx context.Context, messages []ChatMessage, format *ResponseFormat) (string, error) {
	// Anthropic doesn't have a native JSON mode, but we can instruct via system prompt
	if format != nil && (format.Type == "json_object" || format.Type == "json_schema") {
		// Prepend JSON instruction to system message
		jsonInstruction := "You must respond with valid JSON only. Do not include any text outside the JSON object."

		// Find or create system message
		hasSystem := false
		for i, msg := range messages {
			if msg.Role == MessageRoleSystem {
				messages[i].Content = jsonInstruction + "\n\n" + msg.Content
				hasSystem = true
				break
			}
		}
		if !hasSystem {
			messages = append([]ChatMessage{NewSystemMessage(jsonInstruction)}, messages...)
		}
	}

	return a.Chat(ctx, messages)
}

// StreamChat generates a streaming response for chat messages.
func (a *AnthropicLLM) StreamChat(ctx context.Context, messages []ChatMessage) (<-chan StreamToken, error) {
	a.logger.Info("StreamChat called", "model", a.model, "message_count", len(messages))

	anthropicMessages, systemPrompt := a.convertMessages(messages)

	reqBody := anthropicRequest{
		Model:     a.model,
		Messages:  anthropicMessages,
		MaxTokens: a.maxTokens,
		System:    systemPrompt,
		Stream:    true,
	}

	stringChan, err := a.doStreamRequest(ctx, "/messages", reqBody)
	if err != nil {
		return nil, err
	}

	// Convert string channel to StreamToken channel
	tokenChan := make(chan StreamToken)
	go func() {
		defer close(tokenChan)
		for delta := range stringChan {
			select {
			case tokenChan <- StreamToken{Delta: delta}:
			case <-ctx.Done():
				return
			}
		}
	}()

	return tokenChan, nil
}

// convertMessages converts ChatMessage slice to Anthropic format.
func (a *AnthropicLLM) convertMessages(messages []ChatMessage) ([]anthropicMessage, string) {
	var anthropicMessages []anthropicMessage
	var systemPrompt string

	for _, msg := range messages {
		if msg.Role == MessageRoleSystem {
			systemPrompt = msg.GetTextContent()
			continue
		}

		role := "user"
		if msg.Role == MessageRoleAssistant {
			role = "assistant"
		}

		var content []anthropicContent

		// Handle simple text content
		if msg.Content != "" {
			content = append(content, anthropicContent{
				Type: "text",
				Text: msg.Content,
			})
		}

		// Handle content blocks
		for _, block := range msg.Blocks {
			switch block.Type {
			case ContentBlockTypeText:
				content = append(content, anthropicContent{
					Type: "text",
					Text: block.Text,
				})
			case ContentBlockTypeToolCall:
				if block.ToolCall != nil {
					var input map[string]interface{}
					json.Unmarshal([]byte(block.ToolCall.Arguments), &input)
					content = append(content, anthropicContent{
						Type:  "tool_use",
						ID:    block.ToolCall.ID,
						Name:  block.ToolCall.Name,
						Input: input,
					})
				}
			case ContentBlockTypeToolResult:
				if block.ToolResult != nil {
					content = append(content, anthropicContent{
						Type:      "tool_result",
						ToolUseID: block.ToolResult.ToolCallID,
						Content:   block.ToolResult.Content,
					})
				}
			}
		}

		if len(content) > 0 {
			anthropicMessages = append(anthropicMessages, anthropicMessage{
				Role:    role,
				Content: content,
			})
		}
	}

	return anthropicMessages, systemPrompt
}

// convertTools converts ToolMetadata slice to Anthropic format.
func (a *AnthropicLLM) convertTools(tools []*ToolMetadata) []anthropicTool {
	anthropicTools := make([]anthropicTool, len(tools))
	for i, tool := range tools {
		inputSchema := make(map[string]interface{})
		if tool.Parameters != nil {
			inputSchema = tool.Parameters
		} else {
			inputSchema["type"] = "object"
			inputSchema["properties"] = map[string]interface{}{}
		}

		anthropicTools[i] = anthropicTool{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: inputSchema,
		}
	}
	return anthropicTools
}

// convertResponse converts Anthropic response to CompletionResponse.
func (a *AnthropicLLM) convertResponse(resp *anthropicResponse) CompletionResponse {
	var text string
	msg := ChatMessage{
		Role: MessageRoleAssistant,
	}

	for _, content := range resp.Content {
		switch content.Type {
		case "text":
			text += content.Text
		case "tool_use":
			args, _ := json.Marshal(content.Input)
			msg.Blocks = append(msg.Blocks, NewToolCallBlock(&ToolCall{
				ID:        content.ID,
				Name:      content.Name,
				Arguments: string(args),
			}))
		}
	}

	msg.Content = text
	return CompletionResponse{
		Text:    text,
		Message: &msg,
	}
}

// doRequest performs an HTTP request to the Anthropic API.
func (a *AnthropicLLM) doRequest(ctx context.Context, path string, body interface{}) (*anthropicResponse, error) {
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", a.baseURL+path, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", a.apiKey)
	req.Header.Set("anthropic-version", AnthropicAPIVersion)

	resp, err := a.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var apiErr struct {
			Error anthropicError `json:"error"`
		}
		json.Unmarshal(respBody, &apiErr)
		return nil, fmt.Errorf("anthropic API error (%d): %s", resp.StatusCode, apiErr.Error.Message)
	}

	var result anthropicResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &result, nil
}

// doStreamRequest performs a streaming HTTP request to the Anthropic API.
func (a *AnthropicLLM) doStreamRequest(ctx context.Context, path string, body interface{}) (<-chan string, error) {
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", a.baseURL+path, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", a.apiKey)
	req.Header.Set("anthropic-version", AnthropicAPIVersion)
	req.Header.Set("Accept", "text/event-stream")

	resp, err := a.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		var apiErr struct {
			Error anthropicError `json:"error"`
		}
		json.Unmarshal(respBody, &apiErr)
		return nil, fmt.Errorf("anthropic API error (%d): %s", resp.StatusCode, apiErr.Error.Message)
	}

	tokenChan := make(chan string)

	go func() {
		defer close(tokenChan)
		defer resp.Body.Close()

		decoder := json.NewDecoder(resp.Body)
		for {
			select {
			case <-ctx.Done():
				return
			default:
			}

			// Read SSE events
			var line string
			if _, err := fmt.Fscanln(resp.Body, &line); err != nil {
				if err == io.EOF {
					return
				}
				continue
			}

			// Parse event data
			if len(line) > 6 && line[:6] == "data: " {
				data := line[6:]
				if data == "[DONE]" {
					return
				}

				var event anthropicStreamEvent
				if err := json.Unmarshal([]byte(data), &event); err != nil {
					continue
				}

				if event.Type == "content_block_delta" && event.Delta != nil && event.Delta.Text != "" {
					select {
					case tokenChan <- event.Delta.Text:
					case <-ctx.Done():
						return
					}
				}
			}
		}
		_ = decoder // Silence unused variable warning
	}()

	return tokenChan, nil
}

// getAnthropicModelMetadata returns metadata for Anthropic models.
func getAnthropicModelMetadata(model string) LLMMetadata {
	switch model {
	case Claude3Opus:
		return LLMMetadata{
			ModelName:         model,
			ContextWindow:     200000,
			NumOutputTokens:   4096,
			IsChat:            true,
			IsFunctionCalling: true,
			IsMultiModal:      true,
			SystemRole:        "system",
		}
	case Claude3Sonnet, Claude35Sonnet:
		return LLMMetadata{
			ModelName:         model,
			ContextWindow:     200000,
			NumOutputTokens:   8192,
			IsChat:            true,
			IsFunctionCalling: true,
			IsMultiModal:      true,
			SystemRole:        "system",
		}
	case Claude3Haiku, Claude35Haiku:
		return LLMMetadata{
			ModelName:         model,
			ContextWindow:     200000,
			NumOutputTokens:   4096,
			IsChat:            true,
			IsFunctionCalling: true,
			IsMultiModal:      true,
			SystemRole:        "system",
		}
	default:
		return DefaultLLMMetadata(model)
	}
}

// Ensure AnthropicLLM implements the interfaces.
var _ LLM = (*AnthropicLLM)(nil)
var _ LLMWithMetadata = (*AnthropicLLM)(nil)
var _ LLMWithToolCalling = (*AnthropicLLM)(nil)
var _ LLMWithStructuredOutput = (*AnthropicLLM)(nil)
var _ FullLLM = (*AnthropicLLM)(nil)
