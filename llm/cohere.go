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
	// CohereAPIURL is the default Cohere API endpoint.
	CohereAPIURL = "https://api.cohere.ai/v1"
)

// Cohere model constants.
const (
	CohereCommand      = "command"
	CohereCommandLight = "command-light"
	CohereCommandR     = "command-r"
	CohereCommandRPlus = "command-r-plus"
	CohereCommandR7B   = "command-r7b-12-2024"
)

// CohereLLM implements the LLM interface for Cohere models.
type CohereLLM struct {
	apiKey      string
	baseURL     string
	model       string
	maxTokens   int
	temperature *float32
	httpClient  *http.Client
	logger      *slog.Logger
}

// CohereOption configures a CohereLLM.
type CohereOption func(*CohereLLM)

// WithCohereAPIKey sets the API key.
func WithCohereAPIKey(apiKey string) CohereOption {
	return func(c *CohereLLM) {
		c.apiKey = apiKey
	}
}

// WithCohereBaseURL sets the base URL.
func WithCohereBaseURL(baseURL string) CohereOption {
	return func(c *CohereLLM) {
		c.baseURL = baseURL
	}
}

// WithCohereModel sets the model.
func WithCohereModel(model string) CohereOption {
	return func(c *CohereLLM) {
		c.model = model
	}
}

// WithCohereMaxTokens sets the max tokens.
func WithCohereMaxTokens(maxTokens int) CohereOption {
	return func(c *CohereLLM) {
		c.maxTokens = maxTokens
	}
}

// WithCohereTemperature sets the temperature.
func WithCohereTemperature(temp float32) CohereOption {
	return func(c *CohereLLM) {
		c.temperature = &temp
	}
}

// WithCohereHTTPClient sets a custom HTTP client.
func WithCohereHTTPClient(client *http.Client) CohereOption {
	return func(c *CohereLLM) {
		c.httpClient = client
	}
}

// NewCohereLLM creates a new Cohere LLM client.
func NewCohereLLM(opts ...CohereOption) *CohereLLM {
	c := &CohereLLM{
		apiKey:     os.Getenv("COHERE_API_KEY"),
		baseURL:    CohereAPIURL,
		model:      CohereCommandRPlus,
		maxTokens:  4096,
		httpClient: http.DefaultClient,
		logger:     slog.New(slog.NewJSONHandler(os.Stdout, nil)),
	}

	for _, opt := range opts {
		opt(c)
	}

	return c
}

// cohereGenerateRequest represents a request to the Cohere generate API.
type cohereGenerateRequest struct {
	Model       string   `json:"model"`
	Prompt      string   `json:"prompt"`
	MaxTokens   int      `json:"max_tokens,omitempty"`
	Temperature *float32 `json:"temperature,omitempty"`
	K           int      `json:"k,omitempty"`
	P           float32  `json:"p,omitempty"`
	StopSeqs    []string `json:"stop_sequences,omitempty"`
}

// cohereChatRequest represents a request to the Cohere chat API.
type cohereChatRequest struct {
	Model          string                `json:"model"`
	Message        string                `json:"message"`
	ChatHistory    []cohereChatMessage   `json:"chat_history,omitempty"`
	Preamble       string                `json:"preamble,omitempty"`
	MaxTokens      int                   `json:"max_tokens,omitempty"`
	Temperature    *float32              `json:"temperature,omitempty"`
	Tools          []cohereTool          `json:"tools,omitempty"`
	ToolResults    []cohereToolResult    `json:"tool_results,omitempty"`
	ResponseFormat *cohereResponseFormat `json:"response_format,omitempty"`
}

// cohereChatMessage represents a message in the Cohere chat API format.
type cohereChatMessage struct {
	Role    string `json:"role"` // USER, CHATBOT, SYSTEM, TOOL
	Message string `json:"message"`
}

// cohereTool represents a tool in the Cohere API format.
type cohereTool struct {
	Name                 string                 `json:"name"`
	Description          string                 `json:"description"`
	ParameterDefinitions map[string]cohereParam `json:"parameter_definitions,omitempty"`
}

// cohereParam represents a tool parameter in the Cohere API format.
type cohereParam struct {
	Description string `json:"description"`
	Type        string `json:"type"`
	Required    bool   `json:"required"`
}

// cohereToolResult represents a tool result in the Cohere API format.
type cohereToolResult struct {
	Call    cohereToolCall `json:"call"`
	Outputs []interface{}  `json:"outputs"`
}

// cohereToolCall represents a tool call in the Cohere API format.
type cohereToolCall struct {
	Name       string                 `json:"name"`
	Parameters map[string]interface{} `json:"parameters"`
}

// cohereResponseFormat represents the response format in the Cohere API.
type cohereResponseFormat struct {
	Type string `json:"type"` // "text" or "json_object"
}

// cohereGenerateResponse represents a response from the Cohere generate API.
type cohereGenerateResponse struct {
	ID          string `json:"id"`
	Generations []struct {
		ID   string `json:"id"`
		Text string `json:"text"`
	} `json:"generations"`
	Prompt string `json:"prompt"`
	Meta   struct {
		APIVersion struct {
			Version string `json:"version"`
		} `json:"api_version"`
	} `json:"meta"`
}

// cohereChatResponse represents a response from the Cohere chat API.
type cohereChatResponse struct {
	ResponseID   string              `json:"response_id"`
	Text         string              `json:"text"`
	GenerationID string              `json:"generation_id"`
	ChatHistory  []cohereChatMessage `json:"chat_history,omitempty"`
	ToolCalls    []cohereToolCall    `json:"tool_calls,omitempty"`
	FinishReason string              `json:"finish_reason"`
	Meta         struct {
		APIVersion struct {
			Version string `json:"version"`
		} `json:"api_version"`
		Tokens struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"tokens"`
	} `json:"meta"`
}

// Complete generates a completion for a given prompt.
func (c *CohereLLM) Complete(ctx context.Context, prompt string) (string, error) {
	c.logger.Info("Complete called", "model", c.model, "prompt_len", len(prompt))

	reqBody := cohereGenerateRequest{
		Model:       c.model,
		Prompt:      prompt,
		MaxTokens:   c.maxTokens,
		Temperature: c.temperature,
	}

	resp, err := c.doGenerateRequest(ctx, reqBody)
	if err != nil {
		return "", err
	}

	if len(resp.Generations) == 0 {
		return "", fmt.Errorf("cohere returned no generations")
	}

	return resp.Generations[0].Text, nil
}

// Chat generates a response for a list of chat messages.
func (c *CohereLLM) Chat(ctx context.Context, messages []ChatMessage) (string, error) {
	c.logger.Info("Chat called", "model", c.model, "message_count", len(messages))

	chatHistory, currentMessage, preamble := c.convertMessages(messages)

	reqBody := cohereChatRequest{
		Model:       c.model,
		Message:     currentMessage,
		ChatHistory: chatHistory,
		Preamble:    preamble,
		MaxTokens:   c.maxTokens,
		Temperature: c.temperature,
	}

	resp, err := c.doChatRequest(ctx, reqBody)
	if err != nil {
		return "", err
	}

	return resp.Text, nil
}

// Stream generates a streaming completion for a given prompt.
func (c *CohereLLM) Stream(ctx context.Context, prompt string) (<-chan string, error) {
	c.logger.Info("Stream called", "model", c.model, "prompt_len", len(prompt))

	// Cohere streaming uses the same endpoint with stream parameter
	// For simplicity, we'll use non-streaming and return the full response
	// A full implementation would use SSE streaming
	tokenChan := make(chan string, 1)

	go func() {
		defer close(tokenChan)
		resp, err := c.Complete(ctx, prompt)
		if err != nil {
			c.logger.Error("Stream error", "error", err)
			return
		}
		select {
		case tokenChan <- resp:
		case <-ctx.Done():
		}
	}()

	return tokenChan, nil
}

// Metadata returns information about the model's capabilities.
func (c *CohereLLM) Metadata() LLMMetadata {
	return getCohereModelMetadata(c.model)
}

// SupportsToolCalling returns true if the model supports tool calling.
func (c *CohereLLM) SupportsToolCalling() bool {
	// Command R models support tool calling
	switch c.model {
	case CohereCommandR, CohereCommandRPlus, CohereCommandR7B:
		return true
	default:
		return false
	}
}

// SupportsStructuredOutput returns true if the model supports structured output.
func (c *CohereLLM) SupportsStructuredOutput() bool {
	return true
}

// ChatWithTools generates a response that may include tool calls.
func (c *CohereLLM) ChatWithTools(ctx context.Context, messages []ChatMessage, tools []*ToolMetadata, opts *ChatCompletionOptions) (CompletionResponse, error) {
	c.logger.Info("ChatWithTools called", "model", c.model, "message_count", len(messages), "tool_count", len(tools))

	chatHistory, currentMessage, preamble := c.convertMessages(messages)
	cohereTools := c.convertTools(tools)

	reqBody := cohereChatRequest{
		Model:       c.model,
		Message:     currentMessage,
		ChatHistory: chatHistory,
		Preamble:    preamble,
		MaxTokens:   c.maxTokens,
		Tools:       cohereTools,
	}

	if opts != nil {
		if opts.Temperature != nil {
			reqBody.Temperature = opts.Temperature
		}
		if opts.MaxTokens != nil {
			reqBody.MaxTokens = *opts.MaxTokens
		}
	}

	resp, err := c.doChatRequest(ctx, reqBody)
	if err != nil {
		return CompletionResponse{}, err
	}

	return c.convertChatResponse(resp), nil
}

// ChatWithFormat generates a response in the specified format.
func (c *CohereLLM) ChatWithFormat(ctx context.Context, messages []ChatMessage, format *ResponseFormat) (string, error) {
	c.logger.Info("ChatWithFormat called", "model", c.model, "message_count", len(messages))

	chatHistory, currentMessage, preamble := c.convertMessages(messages)

	reqBody := cohereChatRequest{
		Model:       c.model,
		Message:     currentMessage,
		ChatHistory: chatHistory,
		Preamble:    preamble,
		MaxTokens:   c.maxTokens,
		Temperature: c.temperature,
	}

	if format != nil && format.Type == "json_object" {
		reqBody.ResponseFormat = &cohereResponseFormat{Type: "json_object"}
	}

	resp, err := c.doChatRequest(ctx, reqBody)
	if err != nil {
		return "", err
	}

	return resp.Text, nil
}

// StreamChat generates a streaming response for chat messages.
func (c *CohereLLM) StreamChat(ctx context.Context, messages []ChatMessage) (<-chan StreamToken, error) {
	c.logger.Info("StreamChat called", "model", c.model, "message_count", len(messages))

	// For simplicity, use non-streaming
	tokenChan := make(chan StreamToken, 1)

	go func() {
		defer close(tokenChan)
		resp, err := c.Chat(ctx, messages)
		if err != nil {
			c.logger.Error("StreamChat error", "error", err)
			return
		}
		select {
		case tokenChan <- StreamToken{Delta: resp}:
		case <-ctx.Done():
		}
	}()

	return tokenChan, nil
}

// convertMessages converts ChatMessage slice to Cohere format.
func (c *CohereLLM) convertMessages(messages []ChatMessage) ([]cohereChatMessage, string, string) {
	var chatHistory []cohereChatMessage
	var currentMessage string
	var preamble string

	for i, msg := range messages {
		content := msg.GetTextContent()

		switch msg.Role {
		case MessageRoleSystem:
			preamble = content
		case MessageRoleUser:
			if i == len(messages)-1 {
				// Last user message is the current message
				currentMessage = content
			} else {
				chatHistory = append(chatHistory, cohereChatMessage{
					Role:    "USER",
					Message: content,
				})
			}
		case MessageRoleAssistant:
			chatHistory = append(chatHistory, cohereChatMessage{
				Role:    "CHATBOT",
				Message: content,
			})
		case MessageRoleTool:
			chatHistory = append(chatHistory, cohereChatMessage{
				Role:    "TOOL",
				Message: content,
			})
		}
	}

	// If no current message, use the last user message from history
	if currentMessage == "" && len(messages) > 0 {
		lastMsg := messages[len(messages)-1]
		currentMessage = lastMsg.GetTextContent()
	}

	return chatHistory, currentMessage, preamble
}

// convertTools converts ToolMetadata slice to Cohere format.
func (c *CohereLLM) convertTools(tools []*ToolMetadata) []cohereTool {
	cohereTools := make([]cohereTool, len(tools))
	for i, tool := range tools {
		cohereTools[i] = cohereTool{
			Name:        tool.Name,
			Description: tool.Description,
		}

		// Convert parameters
		if tool.Parameters != nil {
			if props, ok := tool.Parameters["properties"].(map[string]interface{}); ok {
				cohereTools[i].ParameterDefinitions = make(map[string]cohereParam)
				required := make(map[string]bool)
				if reqList, ok := tool.Parameters["required"].([]interface{}); ok {
					for _, r := range reqList {
						if s, ok := r.(string); ok {
							required[s] = true
						}
					}
				}

				for name, prop := range props {
					if propMap, ok := prop.(map[string]interface{}); ok {
						param := cohereParam{
							Required: required[name],
						}
						if desc, ok := propMap["description"].(string); ok {
							param.Description = desc
						}
						if typ, ok := propMap["type"].(string); ok {
							param.Type = typ
						}
						cohereTools[i].ParameterDefinitions[name] = param
					}
				}
			}
		}
	}
	return cohereTools
}

// convertChatResponse converts Cohere chat response to CompletionResponse.
func (c *CohereLLM) convertChatResponse(resp *cohereChatResponse) CompletionResponse {
	msg := ChatMessage{
		Role:    MessageRoleAssistant,
		Content: resp.Text,
	}

	for _, tc := range resp.ToolCalls {
		args, _ := json.Marshal(tc.Parameters)
		msg.Blocks = append(msg.Blocks, NewToolCallBlock(&ToolCall{
			Name:      tc.Name,
			Arguments: string(args),
		}))
	}

	return CompletionResponse{
		Text:    resp.Text,
		Message: &msg,
	}
}

// doGenerateRequest performs a generate request to the Cohere API.
func (c *CohereLLM) doGenerateRequest(ctx context.Context, body cohereGenerateRequest) (*cohereGenerateResponse, error) {
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/generate", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("cohere API error (%d): %s", resp.StatusCode, string(respBody))
	}

	var result cohereGenerateResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &result, nil
}

// doChatRequest performs a chat request to the Cohere API.
func (c *CohereLLM) doChatRequest(ctx context.Context, body cohereChatRequest) (*cohereChatResponse, error) {
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/chat", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("cohere API error (%d): %s", resp.StatusCode, string(respBody))
	}

	var result cohereChatResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &result, nil
}

// getCohereModelMetadata returns metadata for Cohere models.
func getCohereModelMetadata(model string) LLMMetadata {
	switch model {
	case CohereCommandRPlus:
		return LLMMetadata{
			ModelName:         model,
			ContextWindow:     128000,
			NumOutputTokens:   4096,
			IsChat:            true,
			IsFunctionCalling: true,
			IsMultiModal:      false,
			SystemRole:        "system",
		}
	case CohereCommandR, CohereCommandR7B:
		return LLMMetadata{
			ModelName:         model,
			ContextWindow:     128000,
			NumOutputTokens:   4096,
			IsChat:            true,
			IsFunctionCalling: true,
			IsMultiModal:      false,
			SystemRole:        "system",
		}
	case CohereCommand, CohereCommandLight:
		return LLMMetadata{
			ModelName:         model,
			ContextWindow:     4096,
			NumOutputTokens:   4096,
			IsChat:            true,
			IsFunctionCalling: false,
			IsMultiModal:      false,
			SystemRole:        "system",
		}
	default:
		return DefaultLLMMetadata(model)
	}
}

// Ensure CohereLLM implements the interfaces.
var _ LLM = (*CohereLLM)(nil)
var _ LLMWithMetadata = (*CohereLLM)(nil)
var _ LLMWithToolCalling = (*CohereLLM)(nil)
var _ LLMWithStructuredOutput = (*CohereLLM)(nil)
var _ FullLLM = (*CohereLLM)(nil)
