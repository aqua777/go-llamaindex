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
)

const (
	// OllamaDefaultURL is the default Ollama API endpoint.
	OllamaDefaultURL = "http://localhost:11434"
)

// Common Ollama model names.
const (
	OllamaLlama2    = "llama2"
	OllamaLlama3    = "llama3"
	OllamaLlama31   = "llama3.1"
	OllamaMistral   = "mistral"
	OllamaCodeLlama = "codellama"
	OllamaGemma     = "gemma"
	OllamaGemma2    = "gemma2"
	OllamaQwen      = "qwen"
	OllamaQwen2     = "qwen2"
	OllamaPhi3      = "phi3"
	OllamaDeepseek  = "deepseek-coder"
)

// OllamaLLM implements the LLM interface for Ollama local models.
type OllamaLLM struct {
	baseURL    string
	model      string
	httpClient *http.Client
	logger     *slog.Logger
	// Generation options
	temperature *float32
	topP        *float32
	topK        *int
	numPredict  *int
	numCtx      *int
	seed        *int
	stop        []string
}

// OllamaOption configures an OllamaLLM.
type OllamaOption func(*OllamaLLM)

// WithOllamaBaseURL sets the base URL.
func WithOllamaBaseURL(baseURL string) OllamaOption {
	return func(o *OllamaLLM) {
		o.baseURL = baseURL
	}
}

// WithOllamaModel sets the model.
func WithOllamaModel(model string) OllamaOption {
	return func(o *OllamaLLM) {
		o.model = model
	}
}

// WithOllamaHTTPClient sets a custom HTTP client.
func WithOllamaHTTPClient(client *http.Client) OllamaOption {
	return func(o *OllamaLLM) {
		o.httpClient = client
	}
}

// WithOllamaTemperature sets the temperature.
func WithOllamaTemperature(temp float32) OllamaOption {
	return func(o *OllamaLLM) {
		o.temperature = &temp
	}
}

// WithOllamaTopP sets the top_p value.
func WithOllamaTopP(topP float32) OllamaOption {
	return func(o *OllamaLLM) {
		o.topP = &topP
	}
}

// WithOllamaTopK sets the top_k value.
func WithOllamaTopK(topK int) OllamaOption {
	return func(o *OllamaLLM) {
		o.topK = &topK
	}
}

// WithOllamaNumPredict sets the max tokens to generate.
func WithOllamaNumPredict(numPredict int) OllamaOption {
	return func(o *OllamaLLM) {
		o.numPredict = &numPredict
	}
}

// WithOllamaNumCtx sets the context window size.
func WithOllamaNumCtx(numCtx int) OllamaOption {
	return func(o *OllamaLLM) {
		o.numCtx = &numCtx
	}
}

// WithOllamaSeed sets the random seed.
func WithOllamaSeed(seed int) OllamaOption {
	return func(o *OllamaLLM) {
		o.seed = &seed
	}
}

// WithOllamaStop sets the stop sequences.
func WithOllamaStop(stop []string) OllamaOption {
	return func(o *OllamaLLM) {
		o.stop = stop
	}
}

// NewOllamaLLM creates a new Ollama LLM client.
func NewOllamaLLM(opts ...OllamaOption) *OllamaLLM {
	baseURL := os.Getenv("OLLAMA_HOST")
	if baseURL == "" {
		baseURL = OllamaDefaultURL
	}

	o := &OllamaLLM{
		baseURL:    baseURL,
		model:      OllamaLlama31,
		httpClient: http.DefaultClient,
		logger:     slog.New(slog.NewJSONHandler(os.Stdout, nil)),
	}

	for _, opt := range opts {
		opt(o)
	}

	return o
}

// ollamaGenerateRequest represents a request to the Ollama generate API.
type ollamaGenerateRequest struct {
	Model    string                 `json:"model"`
	Prompt   string                 `json:"prompt"`
	System   string                 `json:"system,omitempty"`
	Stream   bool                   `json:"stream"`
	Options  map[string]interface{} `json:"options,omitempty"`
	Template string                 `json:"template,omitempty"`
	Context  []int                  `json:"context,omitempty"`
}

// ollamaChatRequest represents a request to the Ollama chat API.
type ollamaChatRequest struct {
	Model    string                 `json:"model"`
	Messages []ollamaMessage        `json:"messages"`
	Stream   bool                   `json:"stream"`
	Options  map[string]interface{} `json:"options,omitempty"`
	Tools    []ollamaTool           `json:"tools,omitempty"`
}

// ollamaMessage represents a message in the Ollama API format.
type ollamaMessage struct {
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	Images    []string         `json:"images,omitempty"`
	ToolCalls []ollamaToolCall `json:"tool_calls,omitempty"`
}

// ollamaTool represents a tool in the Ollama API format.
type ollamaTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string                 `json:"name"`
		Description string                 `json:"description"`
		Parameters  map[string]interface{} `json:"parameters"`
	} `json:"function"`
}

// ollamaToolCall represents a tool call in the Ollama API format.
type ollamaToolCall struct {
	Function struct {
		Name      string                 `json:"name"`
		Arguments map[string]interface{} `json:"arguments"`
	} `json:"function"`
}

// ollamaGenerateResponse represents a response from the Ollama generate API.
type ollamaGenerateResponse struct {
	Model              string `json:"model"`
	CreatedAt          string `json:"created_at"`
	Response           string `json:"response"`
	Done               bool   `json:"done"`
	Context            []int  `json:"context,omitempty"`
	TotalDuration      int64  `json:"total_duration,omitempty"`
	LoadDuration       int64  `json:"load_duration,omitempty"`
	PromptEvalCount    int    `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64  `json:"prompt_eval_duration,omitempty"`
	EvalCount          int    `json:"eval_count,omitempty"`
	EvalDuration       int64  `json:"eval_duration,omitempty"`
}

// ollamaChatResponse represents a response from the Ollama chat API.
type ollamaChatResponse struct {
	Model              string        `json:"model"`
	CreatedAt          string        `json:"created_at"`
	Message            ollamaMessage `json:"message"`
	Done               bool          `json:"done"`
	TotalDuration      int64         `json:"total_duration,omitempty"`
	LoadDuration       int64         `json:"load_duration,omitempty"`
	PromptEvalCount    int           `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64         `json:"prompt_eval_duration,omitempty"`
	EvalCount          int           `json:"eval_count,omitempty"`
	EvalDuration       int64         `json:"eval_duration,omitempty"`
}

// Complete generates a completion for a given prompt.
func (o *OllamaLLM) Complete(ctx context.Context, prompt string) (string, error) {
	o.logger.Info("Complete called", "model", o.model, "prompt_len", len(prompt))

	reqBody := ollamaGenerateRequest{
		Model:   o.model,
		Prompt:  prompt,
		Stream:  false,
		Options: o.buildOptions(),
	}

	resp, err := o.doGenerateRequest(ctx, reqBody)
	if err != nil {
		return "", err
	}

	return resp.Response, nil
}

// Chat generates a response for a list of chat messages.
func (o *OllamaLLM) Chat(ctx context.Context, messages []ChatMessage) (string, error) {
	o.logger.Info("Chat called", "model", o.model, "message_count", len(messages))

	ollamaMessages := o.convertMessages(messages)

	reqBody := ollamaChatRequest{
		Model:    o.model,
		Messages: ollamaMessages,
		Stream:   false,
		Options:  o.buildOptions(),
	}

	resp, err := o.doChatRequest(ctx, reqBody)
	if err != nil {
		return "", err
	}

	return resp.Message.Content, nil
}

// Stream generates a streaming completion for a given prompt.
func (o *OllamaLLM) Stream(ctx context.Context, prompt string) (<-chan string, error) {
	o.logger.Info("Stream called", "model", o.model, "prompt_len", len(prompt))

	reqBody := ollamaGenerateRequest{
		Model:   o.model,
		Prompt:  prompt,
		Stream:  true,
		Options: o.buildOptions(),
	}

	return o.doStreamGenerateRequest(ctx, reqBody)
}

// Metadata returns information about the model's capabilities.
func (o *OllamaLLM) Metadata() LLMMetadata {
	return getOllamaModelMetadata(o.model)
}

// SupportsToolCalling returns true if the model supports tool calling.
func (o *OllamaLLM) SupportsToolCalling() bool {
	// Some Ollama models support tool calling
	switch o.model {
	case OllamaLlama31, OllamaMistral, OllamaQwen2:
		return true
	default:
		return false
	}
}

// SupportsStructuredOutput returns true if the model supports structured output.
func (o *OllamaLLM) SupportsStructuredOutput() bool {
	return true // Most models can output JSON when instructed
}

// ChatWithTools generates a response that may include tool calls.
func (o *OllamaLLM) ChatWithTools(ctx context.Context, messages []ChatMessage, tools []*ToolMetadata, opts *ChatCompletionOptions) (CompletionResponse, error) {
	o.logger.Info("ChatWithTools called", "model", o.model, "message_count", len(messages), "tool_count", len(tools))

	ollamaMessages := o.convertMessages(messages)
	ollamaTools := o.convertTools(tools)

	options := o.buildOptions()
	if opts != nil {
		if opts.Temperature != nil {
			options["temperature"] = *opts.Temperature
		}
		if opts.TopP != nil {
			options["top_p"] = *opts.TopP
		}
		if opts.MaxTokens != nil {
			options["num_predict"] = *opts.MaxTokens
		}
	}

	reqBody := ollamaChatRequest{
		Model:    o.model,
		Messages: ollamaMessages,
		Stream:   false,
		Options:  options,
		Tools:    ollamaTools,
	}

	resp, err := o.doChatRequest(ctx, reqBody)
	if err != nil {
		return CompletionResponse{}, err
	}

	return o.convertChatResponse(resp), nil
}

// ChatWithFormat generates a response in the specified format.
func (o *OllamaLLM) ChatWithFormat(ctx context.Context, messages []ChatMessage, format *ResponseFormat) (string, error) {
	// Add JSON instruction to system message
	if format != nil && (format.Type == "json_object" || format.Type == "json_schema") {
		jsonInstruction := "You must respond with valid JSON only. Do not include any text outside the JSON object."

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

	return o.Chat(ctx, messages)
}

// StreamChat generates a streaming response for chat messages.
func (o *OllamaLLM) StreamChat(ctx context.Context, messages []ChatMessage) (<-chan StreamToken, error) {
	o.logger.Info("StreamChat called", "model", o.model, "message_count", len(messages))

	ollamaMessages := o.convertMessages(messages)

	reqBody := ollamaChatRequest{
		Model:    o.model,
		Messages: ollamaMessages,
		Stream:   true,
		Options:  o.buildOptions(),
	}

	stringChan, err := o.doStreamChatRequest(ctx, reqBody)
	if err != nil {
		return nil, err
	}

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

// buildOptions builds the options map for Ollama requests.
func (o *OllamaLLM) buildOptions() map[string]interface{} {
	options := make(map[string]interface{})

	if o.temperature != nil {
		options["temperature"] = *o.temperature
	}
	if o.topP != nil {
		options["top_p"] = *o.topP
	}
	if o.topK != nil {
		options["top_k"] = *o.topK
	}
	if o.numPredict != nil {
		options["num_predict"] = *o.numPredict
	}
	if o.numCtx != nil {
		options["num_ctx"] = *o.numCtx
	}
	if o.seed != nil {
		options["seed"] = *o.seed
	}
	if len(o.stop) > 0 {
		options["stop"] = o.stop
	}

	return options
}

// convertMessages converts ChatMessage slice to Ollama format.
func (o *OllamaLLM) convertMessages(messages []ChatMessage) []ollamaMessage {
	ollamaMessages := make([]ollamaMessage, 0, len(messages))

	for _, msg := range messages {
		role := string(msg.Role)
		content := msg.GetTextContent()

		ollamaMsg := ollamaMessage{
			Role:    role,
			Content: content,
		}

		// Handle tool calls
		if msg.HasToolCalls() {
			for _, tc := range msg.GetToolCalls() {
				var args map[string]interface{}
				json.Unmarshal([]byte(tc.Arguments), &args)
				ollamaMsg.ToolCalls = append(ollamaMsg.ToolCalls, ollamaToolCall{
					Function: struct {
						Name      string                 `json:"name"`
						Arguments map[string]interface{} `json:"arguments"`
					}{
						Name:      tc.Name,
						Arguments: args,
					},
				})
			}
		}

		ollamaMessages = append(ollamaMessages, ollamaMsg)
	}

	return ollamaMessages
}

// convertTools converts ToolMetadata slice to Ollama format.
func (o *OllamaLLM) convertTools(tools []*ToolMetadata) []ollamaTool {
	ollamaTools := make([]ollamaTool, len(tools))
	for i, tool := range tools {
		ollamaTools[i] = ollamaTool{
			Type: "function",
		}
		ollamaTools[i].Function.Name = tool.Name
		ollamaTools[i].Function.Description = tool.Description
		if tool.Parameters != nil {
			ollamaTools[i].Function.Parameters = tool.Parameters
		} else {
			ollamaTools[i].Function.Parameters = map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
			}
		}
	}
	return ollamaTools
}

// convertChatResponse converts Ollama chat response to CompletionResponse.
func (o *OllamaLLM) convertChatResponse(resp *ollamaChatResponse) CompletionResponse {
	msg := ChatMessage{
		Role:    MessageRoleAssistant,
		Content: resp.Message.Content,
	}

	for _, tc := range resp.Message.ToolCalls {
		args, _ := json.Marshal(tc.Function.Arguments)
		msg.Blocks = append(msg.Blocks, NewToolCallBlock(&ToolCall{
			Name:      tc.Function.Name,
			Arguments: string(args),
		}))
	}

	return CompletionResponse{
		Text:    resp.Message.Content,
		Message: &msg,
	}
}

// doGenerateRequest performs a generate request to the Ollama API.
func (o *OllamaLLM) doGenerateRequest(ctx context.Context, body ollamaGenerateRequest) (*ollamaGenerateResponse, error) {
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", o.baseURL+"/api/generate", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := o.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama API error (%d): %s", resp.StatusCode, string(respBody))
	}

	var result ollamaGenerateResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &result, nil
}

// doChatRequest performs a chat request to the Ollama API.
func (o *OllamaLLM) doChatRequest(ctx context.Context, body ollamaChatRequest) (*ollamaChatResponse, error) {
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", o.baseURL+"/api/chat", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := o.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama API error (%d): %s", resp.StatusCode, string(respBody))
	}

	var result ollamaChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &result, nil
}

// doStreamGenerateRequest performs a streaming generate request.
func (o *OllamaLLM) doStreamGenerateRequest(ctx context.Context, body ollamaGenerateRequest) (<-chan string, error) {
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", o.baseURL+"/api/generate", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := o.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("ollama API error (%d): %s", resp.StatusCode, string(respBody))
	}

	tokenChan := make(chan string)

	go func() {
		defer close(tokenChan)
		defer resp.Body.Close()

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			select {
			case <-ctx.Done():
				return
			default:
			}

			var streamResp ollamaGenerateResponse
			if err := json.Unmarshal(scanner.Bytes(), &streamResp); err != nil {
				continue
			}

			if streamResp.Response != "" {
				select {
				case tokenChan <- streamResp.Response:
				case <-ctx.Done():
					return
				}
			}

			if streamResp.Done {
				return
			}
		}
	}()

	return tokenChan, nil
}

// doStreamChatRequest performs a streaming chat request.
func (o *OllamaLLM) doStreamChatRequest(ctx context.Context, body ollamaChatRequest) (<-chan string, error) {
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", o.baseURL+"/api/chat", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := o.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("ollama API error (%d): %s", resp.StatusCode, string(respBody))
	}

	tokenChan := make(chan string)

	go func() {
		defer close(tokenChan)
		defer resp.Body.Close()

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			select {
			case <-ctx.Done():
				return
			default:
			}

			var streamResp ollamaChatResponse
			if err := json.Unmarshal(scanner.Bytes(), &streamResp); err != nil {
				continue
			}

			if streamResp.Message.Content != "" {
				select {
				case tokenChan <- streamResp.Message.Content:
				case <-ctx.Done():
					return
				}
			}

			if streamResp.Done {
				return
			}
		}
	}()

	return tokenChan, nil
}

// getOllamaModelMetadata returns metadata for Ollama models.
func getOllamaModelMetadata(model string) LLMMetadata {
	// Default context window for most Ollama models
	contextWindow := 4096
	numOutputTokens := 2048

	switch model {
	case OllamaLlama31:
		contextWindow = 128000
		numOutputTokens = 4096
	case OllamaLlama3:
		contextWindow = 8192
		numOutputTokens = 4096
	case OllamaMistral:
		contextWindow = 32768
		numOutputTokens = 4096
	case OllamaCodeLlama:
		contextWindow = 16384
		numOutputTokens = 4096
	case OllamaGemma2:
		contextWindow = 8192
		numOutputTokens = 4096
	case OllamaQwen2:
		contextWindow = 32768
		numOutputTokens = 4096
	}

	return LLMMetadata{
		ModelName:         model,
		ContextWindow:     contextWindow,
		NumOutputTokens:   numOutputTokens,
		IsChat:            true,
		IsFunctionCalling: model == OllamaLlama31 || model == OllamaMistral || model == OllamaQwen2,
		IsMultiModal:      false,
		SystemRole:        "system",
	}
}

// Ensure OllamaLLM implements the interfaces.
var _ LLM = (*OllamaLLM)(nil)
var _ LLMWithMetadata = (*OllamaLLM)(nil)
var _ LLMWithToolCalling = (*OllamaLLM)(nil)
var _ LLMWithStructuredOutput = (*OllamaLLM)(nil)
var _ FullLLM = (*OllamaLLM)(nil)
