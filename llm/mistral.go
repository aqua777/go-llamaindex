package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
)

const (
	// MistralAPIURL is the default Mistral AI API endpoint.
	MistralAPIURL = "https://api.mistral.ai/v1"
	// DefaultMistralModel is the default model to use.
	DefaultMistralModel = "mistral-large-latest"
	// DefaultMistralMaxTokens is the default max tokens.
	DefaultMistralMaxTokens = 512
)

// Mistral model constants.
const (
	MistralTiny           = "mistral-tiny"
	MistralSmall          = "mistral-small"
	MistralMedium         = "mistral-medium"
	MistralSmallLatest    = "mistral-small-latest"
	MistralLargeLatest    = "mistral-large-latest"
	CodestralLatest       = "codestral-latest"
	PixtralLargeLatest    = "pixtral-large-latest"
	MistralSabaLatest     = "mistral-saba-latest"
	Ministral3BLatest     = "ministral-3b-latest"
	Ministral8BLatest     = "ministral-8b-latest"
	MistralEmbed          = "mistral-embed"
	OpenMistralNemo       = "open-mistral-nemo"
	MagistralMediumLatest = "magistral-medium-latest"
	MagistralSmallLatest  = "magistral-small-latest"
)

// mistralModelContextWindows maps model names to their context window sizes.
var mistralModelContextWindows = map[string]int{
	MistralTiny:           32000,
	MistralSmall:          32000,
	MistralMedium:         32000,
	MistralSmallLatest:    32000,
	MistralLargeLatest:    131000,
	CodestralLatest:       256000,
	PixtralLargeLatest:    131000,
	MistralSabaLatest:     32000,
	Ministral3BLatest:     131000,
	Ministral8BLatest:     131000,
	MistralEmbed:          8000,
	OpenMistralNemo:       128000,
	MagistralMediumLatest: 128000,
	MagistralSmallLatest:  128000,
}

// mistralToolCallingModels lists models that support tool calling.
var mistralToolCallingModels = map[string]bool{
	MistralSmallLatest:    true,
	MistralLargeLatest:    true,
	CodestralLatest:       true,
	PixtralLargeLatest:    true,
	Ministral8BLatest:     true,
	Ministral3BLatest:     true,
	OpenMistralNemo:       true,
	MagistralMediumLatest: true,
	MagistralSmallLatest:  true,
}

// MistralLLM implements the LLM interface for Mistral AI models.
type MistralLLM struct {
	apiKey      string
	baseURL     string
	model       string
	maxTokens   int
	temperature float32
	topP        float32
	safeMode    bool
	randomSeed  *int
	httpClient  *http.Client
	logger      *slog.Logger
}

// MistralOption configures a MistralLLM.
type MistralOption func(*MistralLLM)

// WithMistralAPIKey sets the API key.
func WithMistralAPIKey(apiKey string) MistralOption {
	return func(m *MistralLLM) {
		m.apiKey = apiKey
	}
}

// WithMistralBaseURL sets the base URL.
func WithMistralBaseURL(baseURL string) MistralOption {
	return func(m *MistralLLM) {
		m.baseURL = baseURL
	}
}

// WithMistralModel sets the model.
func WithMistralModel(model string) MistralOption {
	return func(m *MistralLLM) {
		m.model = model
	}
}

// WithMistralMaxTokens sets the max tokens.
func WithMistralMaxTokens(maxTokens int) MistralOption {
	return func(m *MistralLLM) {
		m.maxTokens = maxTokens
	}
}

// WithMistralTemperature sets the temperature.
func WithMistralTemperature(temperature float32) MistralOption {
	return func(m *MistralLLM) {
		m.temperature = temperature
	}
}

// WithMistralTopP sets the top_p value.
func WithMistralTopP(topP float32) MistralOption {
	return func(m *MistralLLM) {
		m.topP = topP
	}
}

// WithMistralSafeMode enables safe mode.
func WithMistralSafeMode(safeMode bool) MistralOption {
	return func(m *MistralLLM) {
		m.safeMode = safeMode
	}
}

// WithMistralRandomSeed sets the random seed for reproducibility.
func WithMistralRandomSeed(seed int) MistralOption {
	return func(m *MistralLLM) {
		m.randomSeed = &seed
	}
}

// WithMistralHTTPClient sets a custom HTTP client.
func WithMistralHTTPClient(client *http.Client) MistralOption {
	return func(m *MistralLLM) {
		m.httpClient = client
	}
}

// NewMistralLLM creates a new Mistral AI LLM client.
func NewMistralLLM(opts ...MistralOption) *MistralLLM {
	apiKey := os.Getenv("MISTRAL_API_KEY")
	baseURL := os.Getenv("MISTRAL_ENDPOINT")
	if baseURL == "" {
		baseURL = MistralAPIURL
	}

	m := &MistralLLM{
		apiKey:      apiKey,
		baseURL:     baseURL,
		model:       DefaultMistralModel,
		maxTokens:   DefaultMistralMaxTokens,
		temperature: 0.1,
		topP:        1.0,
		safeMode:    false,
		httpClient:  http.DefaultClient,
		logger:      slog.New(slog.NewJSONHandler(os.Stdout, nil)),
	}

	for _, opt := range opts {
		opt(m)
	}

	return m
}

// mistralMessage represents a message in the Mistral API format.
type mistralMessage struct {
	Role       string        `json:"role"`
	Content    string        `json:"content"`
	ToolCalls  []mistralTool `json:"tool_calls,omitempty"`
	ToolCallID string        `json:"tool_call_id,omitempty"`
	Name       string        `json:"name,omitempty"`
}

// mistralTool represents a tool call in the Mistral API format.
type mistralTool struct {
	ID       string              `json:"id,omitempty"`
	Type     string              `json:"type"`
	Function mistralFunctionCall `json:"function"`
}

// mistralFunctionCall represents a function call in the Mistral API format.
type mistralFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments,omitempty"`
}

// mistralToolDefinition represents a tool definition for the Mistral API.
type mistralToolDefinition struct {
	Type     string             `json:"type"`
	Function mistralFunctionDef `json:"function"`
}

// mistralFunctionDef represents a function definition.
type mistralFunctionDef struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

// mistralRequest represents a request to the Mistral API.
type mistralRequest struct {
	Model       string                  `json:"model"`
	Messages    []mistralMessage        `json:"messages"`
	MaxTokens   int                     `json:"max_tokens,omitempty"`
	Temperature float32                 `json:"temperature,omitempty"`
	TopP        float32                 `json:"top_p,omitempty"`
	Stream      bool                    `json:"stream,omitempty"`
	SafePrompt  bool                    `json:"safe_prompt,omitempty"`
	RandomSeed  *int                    `json:"random_seed,omitempty"`
	Tools       []mistralToolDefinition `json:"tools,omitempty"`
	ToolChoice  interface{}             `json:"tool_choice,omitempty"`
}

// mistralResponse represents a response from the Mistral API.
type mistralResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int            `json:"index"`
		Message      mistralMessage `json:"message"`
		FinishReason string         `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// mistralStreamResponse represents a streaming response chunk from the Mistral API.
type mistralStreamResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Role      string        `json:"role,omitempty"`
			Content   string        `json:"content,omitempty"`
			ToolCalls []mistralTool `json:"tool_calls,omitempty"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason,omitempty"`
	} `json:"choices"`
}

// mistralError represents an error from the Mistral API.
type mistralError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}

// Complete generates a completion for a given prompt.
func (m *MistralLLM) Complete(ctx context.Context, prompt string) (string, error) {
	messages := []ChatMessage{NewUserMessage(prompt)}
	return m.Chat(ctx, messages)
}

// Chat generates a response for a list of chat messages.
func (m *MistralLLM) Chat(ctx context.Context, messages []ChatMessage) (string, error) {
	m.logger.Info("Chat called", "model", m.model, "message_count", len(messages))

	mistralMessages := m.convertMessages(messages)

	reqBody := mistralRequest{
		Model:       m.model,
		Messages:    mistralMessages,
		MaxTokens:   m.maxTokens,
		Temperature: m.temperature,
		TopP:        m.topP,
		SafePrompt:  m.safeMode,
		RandomSeed:  m.randomSeed,
	}

	resp, err := m.doRequest(ctx, "/chat/completions", reqBody)
	if err != nil {
		return "", err
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("mistral returned no choices")
	}

	return resp.Choices[0].Message.Content, nil
}

// Stream generates a streaming completion for a given prompt.
func (m *MistralLLM) Stream(ctx context.Context, prompt string) (<-chan string, error) {
	m.logger.Info("Stream called", "model", m.model, "prompt_len", len(prompt))

	messages := []mistralMessage{
		{
			Role:    "user",
			Content: prompt,
		},
	}

	reqBody := mistralRequest{
		Model:       m.model,
		Messages:    messages,
		MaxTokens:   m.maxTokens,
		Temperature: m.temperature,
		TopP:        m.topP,
		SafePrompt:  m.safeMode,
		RandomSeed:  m.randomSeed,
		Stream:      true,
	}

	return m.doStreamRequest(ctx, "/chat/completions", reqBody)
}

// Metadata returns information about the model's capabilities.
func (m *MistralLLM) Metadata() LLMMetadata {
	return getMistralModelMetadata(m.model)
}

// SupportsToolCalling returns true if the model supports tool calling.
func (m *MistralLLM) SupportsToolCalling() bool {
	return mistralToolCallingModels[m.model]
}

// SupportsStructuredOutput returns true if the model supports structured output.
func (m *MistralLLM) SupportsStructuredOutput() bool {
	// Mistral supports JSON mode via response_format
	return true
}

// ChatWithTools generates a response that may include tool calls.
func (m *MistralLLM) ChatWithTools(ctx context.Context, messages []ChatMessage, tools []*ToolMetadata, opts *ChatCompletionOptions) (CompletionResponse, error) {
	m.logger.Info("ChatWithTools called", "model", m.model, "message_count", len(messages), "tool_count", len(tools))

	mistralMessages := m.convertMessages(messages)
	mistralTools := m.convertTools(tools)

	reqBody := mistralRequest{
		Model:       m.model,
		Messages:    mistralMessages,
		MaxTokens:   m.maxTokens,
		Temperature: m.temperature,
		TopP:        m.topP,
		SafePrompt:  m.safeMode,
		RandomSeed:  m.randomSeed,
		Tools:       mistralTools,
		ToolChoice:  "auto",
	}

	if opts != nil {
		if opts.Temperature != nil {
			reqBody.Temperature = *opts.Temperature
		}
		if opts.TopP != nil {
			reqBody.TopP = *opts.TopP
		}
		if opts.MaxTokens != nil {
			reqBody.MaxTokens = *opts.MaxTokens
		}
		if opts.ToolChoice != nil {
			switch tc := opts.ToolChoice.(type) {
			case ToolChoice:
				reqBody.ToolChoice = string(tc)
			case string:
				reqBody.ToolChoice = tc
			default:
				reqBody.ToolChoice = tc
			}
		}
	}

	resp, err := m.doRequest(ctx, "/chat/completions", reqBody)
	if err != nil {
		return CompletionResponse{}, err
	}

	return m.convertResponse(resp), nil
}

// ChatWithFormat generates a response in the specified format.
func (m *MistralLLM) ChatWithFormat(ctx context.Context, messages []ChatMessage, format *ResponseFormat) (string, error) {
	m.logger.Info("ChatWithFormat called", "model", m.model, "message_count", len(messages))

	// Mistral supports JSON mode via instructions in the prompt
	if format != nil && (format.Type == "json_object" || format.Type == "json_schema") {
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

	return m.Chat(ctx, messages)
}

// StreamChat generates a streaming response for chat messages.
func (m *MistralLLM) StreamChat(ctx context.Context, messages []ChatMessage) (<-chan StreamToken, error) {
	m.logger.Info("StreamChat called", "model", m.model, "message_count", len(messages))

	mistralMessages := m.convertMessages(messages)

	reqBody := mistralRequest{
		Model:       m.model,
		Messages:    mistralMessages,
		MaxTokens:   m.maxTokens,
		Temperature: m.temperature,
		TopP:        m.topP,
		SafePrompt:  m.safeMode,
		RandomSeed:  m.randomSeed,
		Stream:      true,
	}

	stringChan, err := m.doStreamRequest(ctx, "/chat/completions", reqBody)
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

// convertMessages converts ChatMessage slice to Mistral format.
func (m *MistralLLM) convertMessages(messages []ChatMessage) []mistralMessage {
	var mistralMessages []mistralMessage

	for _, msg := range messages {
		role := string(msg.Role)
		if msg.Role == MessageRoleTool {
			role = "tool"
		}

		mistralMsg := mistralMessage{
			Role:       role,
			Content:    msg.GetTextContent(),
			ToolCallID: msg.ToolCallID,
			Name:       msg.Name,
		}

		// Handle tool calls in assistant messages
		if msg.Role == MessageRoleAssistant && msg.HasToolCalls() {
			for _, tc := range msg.GetToolCalls() {
				mistralMsg.ToolCalls = append(mistralMsg.ToolCalls, mistralTool{
					ID:   tc.ID,
					Type: "function",
					Function: mistralFunctionCall{
						Name:      tc.Name,
						Arguments: tc.Arguments,
					},
				})
			}
		}

		mistralMessages = append(mistralMessages, mistralMsg)
	}

	return mistralMessages
}

// convertTools converts ToolMetadata slice to Mistral format.
func (m *MistralLLM) convertTools(tools []*ToolMetadata) []mistralToolDefinition {
	mistralTools := make([]mistralToolDefinition, len(tools))
	for i, tool := range tools {
		params := make(map[string]interface{})
		if tool.Parameters != nil {
			params = tool.Parameters
		} else {
			params["type"] = "object"
			params["properties"] = map[string]interface{}{}
		}

		mistralTools[i] = mistralToolDefinition{
			Type: "function",
			Function: mistralFunctionDef{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  params,
			},
		}
	}
	return mistralTools
}

// convertResponse converts Mistral response to CompletionResponse.
func (m *MistralLLM) convertResponse(resp *mistralResponse) CompletionResponse {
	if len(resp.Choices) == 0 {
		return CompletionResponse{}
	}

	choice := resp.Choices[0]
	msg := ChatMessage{
		Role:    MessageRoleAssistant,
		Content: choice.Message.Content,
	}

	// Convert tool calls if present
	if len(choice.Message.ToolCalls) > 0 {
		for _, tc := range choice.Message.ToolCalls {
			msg.Blocks = append(msg.Blocks, NewToolCallBlock(&ToolCall{
				ID:        tc.ID,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			}))
		}
	}

	return CompletionResponse{
		Text:    choice.Message.Content,
		Message: &msg,
	}
}

// doRequest performs an HTTP request to the Mistral API.
func (m *MistralLLM) doRequest(ctx context.Context, path string, body interface{}) (*mistralResponse, error) {
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", m.baseURL+path, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.apiKey)
	req.Header.Set("Accept", "application/json")

	resp, err := m.httpClient.Do(req)
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
			Error mistralError `json:"error"`
		}
		json.Unmarshal(respBody, &apiErr)
		if apiErr.Error.Message != "" {
			return nil, fmt.Errorf("mistral API error (%d): %s", resp.StatusCode, apiErr.Error.Message)
		}
		return nil, fmt.Errorf("mistral API error (%d): %s", resp.StatusCode, string(respBody))
	}

	var result mistralResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &result, nil
}

// doStreamRequest performs a streaming HTTP request to the Mistral API.
func (m *MistralLLM) doStreamRequest(ctx context.Context, path string, body interface{}) (<-chan string, error) {
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", m.baseURL+path, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.apiKey)
	req.Header.Set("Accept", "text/event-stream")

	resp, err := m.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		var apiErr struct {
			Error mistralError `json:"error"`
		}
		json.Unmarshal(respBody, &apiErr)
		if apiErr.Error.Message != "" {
			return nil, fmt.Errorf("mistral API error (%d): %s", resp.StatusCode, apiErr.Error.Message)
		}
		return nil, fmt.Errorf("mistral API error (%d): %s", resp.StatusCode, string(respBody))
	}

	tokenChan := make(chan string)

	go func() {
		defer close(tokenChan)
		defer resp.Body.Close()

		reader := bufio.NewReader(resp.Body)
		for {
			select {
			case <-ctx.Done():
				return
			default:
			}

			line, err := reader.ReadString('\n')
			if err != nil {
				if err == io.EOF {
					return
				}
				m.logger.Error("Stream read error", "error", err)
				return
			}

			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}

			// Parse SSE data
			if strings.HasPrefix(line, "data: ") {
				data := strings.TrimPrefix(line, "data: ")
				if data == "[DONE]" {
					return
				}

				var streamResp mistralStreamResponse
				if err := json.Unmarshal([]byte(data), &streamResp); err != nil {
					m.logger.Error("Failed to unmarshal stream response", "error", err)
					continue
				}

				if len(streamResp.Choices) > 0 {
					delta := streamResp.Choices[0].Delta.Content
					if delta != "" {
						select {
						case tokenChan <- delta:
						case <-ctx.Done():
							return
						}
					}
				}
			}
		}
	}()

	return tokenChan, nil
}

// getMistralModelMetadata returns metadata for Mistral models.
func getMistralModelMetadata(model string) LLMMetadata {
	contextWindow := 32000 // default
	if cw, ok := mistralModelContextWindows[model]; ok {
		contextWindow = cw
	}

	return LLMMetadata{
		ModelName:         model,
		ContextWindow:     contextWindow,
		NumOutputTokens:   4096,
		IsChat:            true,
		IsFunctionCalling: mistralToolCallingModels[model],
		IsMultiModal:      model == PixtralLargeLatest, // Pixtral supports images
		SystemRole:        "system",
	}
}

// IsMistralFunctionCallingModel returns true if the model supports function calling.
func IsMistralFunctionCallingModel(model string) bool {
	return mistralToolCallingModels[model]
}

// MistralModelContextSize returns the context window size for a model.
func MistralModelContextSize(model string) int {
	if cw, ok := mistralModelContextWindows[model]; ok {
		return cw
	}
	return 32000 // default
}

// Ensure MistralLLM implements the interfaces.
var _ LLM = (*MistralLLM)(nil)
var _ LLMWithMetadata = (*MistralLLM)(nil)
var _ LLMWithToolCalling = (*MistralLLM)(nil)
var _ LLMWithStructuredOutput = (*MistralLLM)(nil)
var _ FullLLM = (*MistralLLM)(nil)
