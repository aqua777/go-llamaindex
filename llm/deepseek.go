package llm

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"

	openai "github.com/sashabaranov/go-openai"
)

const (
	// DeepSeekAPIURL is the default DeepSeek API endpoint (OpenAI-compatible).
	DeepSeekAPIURL = "https://api.deepseek.com/v1"
	// DefaultDeepSeekModel is the default model to use.
	DefaultDeepSeekModel = "deepseek-chat"
)

// DeepSeek model constants.
const (
	// Chat models
	DeepSeekChat     = "deepseek-chat"
	DeepSeekReasoner = "deepseek-reasoner"

	// Coder models
	DeepSeekCoder = "deepseek-coder"
)

// deepseekModelContextWindows maps model names to their context window sizes.
var deepseekModelContextWindows = map[string]int{
	DeepSeekChat:     64000,
	DeepSeekReasoner: 64000,
	DeepSeekCoder:    128000,
}

// deepseekToolCallingModels lists models that support tool/function calling.
var deepseekToolCallingModels = map[string]bool{
	DeepSeekChat:  true,
	DeepSeekCoder: true,
	// DeepSeekReasoner does not support function calling
}

// DeepSeekLLM implements the LLM interface for DeepSeek's API.
// DeepSeek provides high-performance AI models with an OpenAI-compatible API.
type DeepSeekLLM struct {
	client *openai.Client
	model  string
	logger *slog.Logger
}

// DeepSeekOption configures a DeepSeekLLM.
type DeepSeekOption func(*DeepSeekLLM)

// WithDeepSeekAPIKey sets the API key.
func WithDeepSeekAPIKey(apiKey string) DeepSeekOption {
	return func(d *DeepSeekLLM) {
		config := openai.DefaultConfig(apiKey)
		config.BaseURL = DeepSeekAPIURL
		d.client = openai.NewClientWithConfig(config)
	}
}

// WithDeepSeekModel sets the model.
func WithDeepSeekModel(model string) DeepSeekOption {
	return func(d *DeepSeekLLM) {
		d.model = model
	}
}

// WithDeepSeekBaseURL sets a custom base URL.
func WithDeepSeekBaseURL(baseURL string) DeepSeekOption {
	return func(d *DeepSeekLLM) {
		apiKey := os.Getenv("DEEPSEEK_API_KEY")
		config := openai.DefaultConfig(apiKey)
		config.BaseURL = baseURL
		d.client = openai.NewClientWithConfig(config)
	}
}

// WithDeepSeekClient sets a custom OpenAI client (for testing).
func WithDeepSeekClient(client *openai.Client) DeepSeekOption {
	return func(d *DeepSeekLLM) {
		d.client = client
	}
}

// NewDeepSeekLLM creates a new DeepSeek LLM client.
func NewDeepSeekLLM(opts ...DeepSeekOption) *DeepSeekLLM {
	apiKey := os.Getenv("DEEPSEEK_API_KEY")

	config := openai.DefaultConfig(apiKey)
	config.BaseURL = DeepSeekAPIURL

	d := &DeepSeekLLM{
		client: openai.NewClientWithConfig(config),
		model:  DefaultDeepSeekModel,
		logger: slog.New(slog.NewJSONHandler(os.Stdout, nil)),
	}

	for _, opt := range opts {
		opt(d)
	}

	return d
}

// Complete generates a completion for a given prompt.
func (d *DeepSeekLLM) Complete(ctx context.Context, prompt string) (string, error) {
	d.logger.Info("Complete called", "model", d.model, "prompt_len", len(prompt))

	resp, err := d.client.CreateChatCompletion(
		ctx,
		openai.ChatCompletionRequest{
			Model: d.model,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: prompt,
				},
			},
		},
	)

	if err != nil {
		d.logger.Error("Complete failed", "error", err)
		return "", fmt.Errorf("deepseek completion failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("deepseek returned no choices")
	}

	return resp.Choices[0].Message.Content, nil
}

// Chat generates a response for a list of chat messages.
func (d *DeepSeekLLM) Chat(ctx context.Context, messages []ChatMessage) (string, error) {
	d.logger.Info("Chat called", "model", d.model, "message_count", len(messages))

	openaiMessages := convertToOpenAIMessages(messages)

	resp, err := d.client.CreateChatCompletion(
		ctx,
		openai.ChatCompletionRequest{
			Model:    d.model,
			Messages: openaiMessages,
		},
	)

	if err != nil {
		d.logger.Error("Chat failed", "error", err)
		return "", fmt.Errorf("deepseek chat failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("deepseek returned no choices")
	}

	return resp.Choices[0].Message.Content, nil
}

// Stream generates a streaming completion for a given prompt.
func (d *DeepSeekLLM) Stream(ctx context.Context, prompt string) (<-chan string, error) {
	d.logger.Info("Stream called", "model", d.model, "prompt_len", len(prompt))

	stream, err := d.client.CreateChatCompletionStream(
		ctx,
		openai.ChatCompletionRequest{
			Model: d.model,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: prompt,
				},
			},
			Stream: true,
		},
	)

	if err != nil {
		d.logger.Error("Stream failed", "error", err)
		return nil, fmt.Errorf("deepseek stream failed: %w", err)
	}

	tokenChan := make(chan string)

	go func() {
		defer close(tokenChan)
		defer stream.Close()

		for {
			response, err := stream.Recv()
			if err == io.EOF {
				return
			}
			if err != nil {
				d.logger.Error("Stream receive error", "error", err)
				return
			}

			if len(response.Choices) > 0 {
				delta := response.Choices[0].Delta.Content
				if delta != "" {
					select {
					case tokenChan <- delta:
					case <-ctx.Done():
						return
					}
				}
			}
		}
	}()

	return tokenChan, nil
}

// Metadata returns information about the model's capabilities.
func (d *DeepSeekLLM) Metadata() LLMMetadata {
	return getDeepSeekModelMetadata(d.model)
}

// SupportsToolCalling returns true if the model supports tool calling.
func (d *DeepSeekLLM) SupportsToolCalling() bool {
	return deepseekToolCallingModels[d.model]
}

// SupportsStructuredOutput returns true if the model supports structured output.
func (d *DeepSeekLLM) SupportsStructuredOutput() bool {
	// DeepSeek supports JSON mode via response_format
	return true
}

// ChatWithTools generates a response that may include tool calls.
func (d *DeepSeekLLM) ChatWithTools(ctx context.Context, messages []ChatMessage, tools []*ToolMetadata, opts *ChatCompletionOptions) (CompletionResponse, error) {
	d.logger.Info("ChatWithTools called", "model", d.model, "message_count", len(messages), "tool_count", len(tools))

	openaiMessages := convertToOpenAIMessages(messages)
	openaiTools := convertToOpenAITools(tools)

	req := openai.ChatCompletionRequest{
		Model:    d.model,
		Messages: openaiMessages,
		Tools:    openaiTools,
	}

	// Apply options
	if opts != nil {
		if opts.Temperature != nil {
			req.Temperature = *opts.Temperature
		}
		if opts.MaxTokens != nil {
			req.MaxTokens = *opts.MaxTokens
		}
		if opts.TopP != nil {
			req.TopP = *opts.TopP
		}
		if opts.Stop != nil {
			req.Stop = opts.Stop
		}
		if opts.ToolChoice != nil {
			switch tc := opts.ToolChoice.(type) {
			case ToolChoice:
				req.ToolChoice = string(tc)
			case string:
				req.ToolChoice = tc
			case map[string]interface{}:
				req.ToolChoice = tc
			}
		}
	}

	resp, err := d.client.CreateChatCompletion(ctx, req)
	if err != nil {
		d.logger.Error("ChatWithTools failed", "error", err)
		return CompletionResponse{}, fmt.Errorf("deepseek chat with tools failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return CompletionResponse{}, fmt.Errorf("deepseek returned no choices")
	}

	choice := resp.Choices[0]
	response := CompletionResponse{
		Text: choice.Message.Content,
	}

	// Convert tool calls if present
	if len(choice.Message.ToolCalls) > 0 {
		msg := ChatMessage{
			Role:    MessageRoleAssistant,
			Content: choice.Message.Content,
		}
		for _, tc := range choice.Message.ToolCalls {
			msg.Blocks = append(msg.Blocks, NewToolCallBlock(&ToolCall{
				ID:        tc.ID,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			}))
		}
		response.Message = &msg
	}

	return response, nil
}

// ChatWithFormat generates a response in the specified format.
func (d *DeepSeekLLM) ChatWithFormat(ctx context.Context, messages []ChatMessage, format *ResponseFormat) (string, error) {
	d.logger.Info("ChatWithFormat called", "model", d.model, "message_count", len(messages), "format", format.Type)

	openaiMessages := convertToOpenAIMessages(messages)

	req := openai.ChatCompletionRequest{
		Model:    d.model,
		Messages: openaiMessages,
	}

	if format != nil {
		switch format.Type {
		case "json_object":
			req.ResponseFormat = &openai.ChatCompletionResponseFormat{
				Type: openai.ChatCompletionResponseFormatTypeJSONObject,
			}
		case "json_schema":
			// Fall back to json_object mode for DeepSeek
			req.ResponseFormat = &openai.ChatCompletionResponseFormat{
				Type: openai.ChatCompletionResponseFormatTypeJSONObject,
			}
		}
	}

	resp, err := d.client.CreateChatCompletion(ctx, req)
	if err != nil {
		d.logger.Error("ChatWithFormat failed", "error", err)
		return "", fmt.Errorf("deepseek chat with format failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("deepseek returned no choices")
	}

	return resp.Choices[0].Message.Content, nil
}

// StreamChat generates a streaming response for chat messages.
func (d *DeepSeekLLM) StreamChat(ctx context.Context, messages []ChatMessage) (<-chan StreamToken, error) {
	d.logger.Info("StreamChat called", "model", d.model, "message_count", len(messages))

	openaiMessages := convertToOpenAIMessages(messages)

	stream, err := d.client.CreateChatCompletionStream(
		ctx,
		openai.ChatCompletionRequest{
			Model:    d.model,
			Messages: openaiMessages,
			Stream:   true,
		},
	)

	if err != nil {
		d.logger.Error("StreamChat failed", "error", err)
		return nil, fmt.Errorf("deepseek stream chat failed: %w", err)
	}

	tokenChan := make(chan StreamToken)

	go func() {
		defer close(tokenChan)
		defer stream.Close()

		for {
			response, err := stream.Recv()
			if err == io.EOF {
				return
			}
			if err != nil {
				d.logger.Error("StreamChat receive error", "error", err)
				return
			}

			if len(response.Choices) > 0 {
				choice := response.Choices[0]
				token := StreamToken{
					Delta:        choice.Delta.Content,
					FinishReason: string(choice.FinishReason),
				}

				// Handle tool calls in streaming
				if len(choice.Delta.ToolCalls) > 0 {
					for _, tc := range choice.Delta.ToolCalls {
						token.ToolCalls = append(token.ToolCalls, &ToolCall{
							ID:        tc.ID,
							Name:      tc.Function.Name,
							Arguments: tc.Function.Arguments,
						})
					}
				}

				select {
				case tokenChan <- token:
				case <-ctx.Done():
					return
				}
			}
		}
	}()

	return tokenChan, nil
}

// getDeepSeekModelMetadata returns metadata for DeepSeek models.
func getDeepSeekModelMetadata(model string) LLMMetadata {
	contextWindow := 64000 // default
	if cw, ok := deepseekModelContextWindows[model]; ok {
		contextWindow = cw
	}

	return LLMMetadata{
		ModelName:         model,
		ContextWindow:     contextWindow,
		NumOutputTokens:   8192,
		IsChat:            true,
		IsFunctionCalling: deepseekToolCallingModels[model],
		IsMultiModal:      false, // DeepSeek doesn't support multi-modal yet
		SystemRole:        "system",
	}
}

// IsDeepSeekFunctionCallingModel returns true if the model supports function calling.
func IsDeepSeekFunctionCallingModel(model string) bool {
	return deepseekToolCallingModels[model]
}

// DeepSeekModelContextSize returns the context window size for a model.
func DeepSeekModelContextSize(model string) int {
	if cw, ok := deepseekModelContextWindows[model]; ok {
		return cw
	}
	return 64000 // default
}

// Ensure DeepSeekLLM implements the interfaces.
var _ LLM = (*DeepSeekLLM)(nil)
var _ LLMWithMetadata = (*DeepSeekLLM)(nil)
var _ LLMWithToolCalling = (*DeepSeekLLM)(nil)
var _ LLMWithStructuredOutput = (*DeepSeekLLM)(nil)
var _ FullLLM = (*DeepSeekLLM)(nil)
