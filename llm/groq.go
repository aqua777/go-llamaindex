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
	// GroqAPIURL is the default Groq API endpoint (OpenAI-compatible).
	GroqAPIURL = "https://api.groq.com/openai/v1"
	// DefaultGroqModel is the default model to use.
	DefaultGroqModel = "llama-3.3-70b-versatile"
)

// Groq model constants.
const (
	// Llama models
	GroqLlama31_8B     = "llama-3.1-8b-instant"
	GroqLlama33_70B    = "llama-3.3-70b-versatile"
	GroqLlama31_70B    = "llama-3.1-70b-versatile"
	GroqLlama31_405B   = "llama-3.1-405b-reasoning"
	GroqLlamaGuard3_8B = "llama-guard-3-8b"
	GroqLlama4Scout17B = "meta-llama/llama-4-scout-17b-16e-instruct"
	GroqLlama4Maverick = "meta-llama/llama-4-maverick-17b-128e-instruct"

	// Mixtral models
	GroqMixtral8x7B = "mixtral-8x7b-32768"

	// Gemma models
	GroqGemma2_9B = "gemma2-9b-it"
	GroqGemma7B   = "gemma-7b-it"

	// DeepSeek models
	GroqDeepSeekR1_70B = "deepseek-r1-distill-llama-70b"

	// Qwen models
	GroqQwenQWQ32B = "qwen-qwq-32b"

	// Whisper (audio transcription)
	GroqWhisperLargeV3       = "whisper-large-v3"
	GroqWhisperLargeV3Turbo  = "whisper-large-v3-turbo"
	GroqDistilWhisperLargeV3 = "distil-whisper-large-v3-en"
)

// groqModelContextWindows maps model names to their context window sizes.
var groqModelContextWindows = map[string]int{
	GroqLlama31_8B:           128000,
	GroqLlama33_70B:          128000,
	GroqLlama31_70B:          128000,
	GroqLlama31_405B:         128000,
	GroqLlamaGuard3_8B:       8192,
	GroqLlama4Scout17B:       131072,
	GroqLlama4Maverick:       131072,
	GroqMixtral8x7B:          32768,
	GroqGemma2_9B:            8192,
	GroqGemma7B:              8192,
	GroqDeepSeekR1_70B:       128000,
	GroqQwenQWQ32B:           32768,
	GroqWhisperLargeV3:       0, // Audio model
	GroqWhisperLargeV3Turbo:  0, // Audio model
	GroqDistilWhisperLargeV3: 0, // Audio model
}

// groqToolCallingModels lists models that support tool/function calling.
var groqToolCallingModels = map[string]bool{
	GroqGemma2_9B:      true,
	GroqLlama31_8B:     true,
	GroqLlama33_70B:    true,
	GroqLlama31_70B:    true,
	GroqDeepSeekR1_70B: true,
	GroqQwenQWQ32B:     true,
	GroqLlama4Scout17B: true,
	GroqLlama4Maverick: true,
}

// GroqLLM implements the LLM interface for Groq's API.
// Groq provides ultra-fast inference using their LPU (Language Processing Unit).
type GroqLLM struct {
	client *openai.Client
	model  string
	logger *slog.Logger
}

// GroqOption configures a GroqLLM.
type GroqOption func(*GroqLLM)

// WithGroqAPIKey sets the API key.
func WithGroqAPIKey(apiKey string) GroqOption {
	return func(g *GroqLLM) {
		config := openai.DefaultConfig(apiKey)
		config.BaseURL = GroqAPIURL
		g.client = openai.NewClientWithConfig(config)
	}
}

// WithGroqModel sets the model.
func WithGroqModel(model string) GroqOption {
	return func(g *GroqLLM) {
		g.model = model
	}
}

// WithGroqBaseURL sets a custom base URL.
func WithGroqBaseURL(baseURL string) GroqOption {
	return func(g *GroqLLM) {
		apiKey := os.Getenv("GROQ_API_KEY")
		config := openai.DefaultConfig(apiKey)
		config.BaseURL = baseURL
		g.client = openai.NewClientWithConfig(config)
	}
}

// WithGroqClient sets a custom OpenAI client (for testing).
func WithGroqClient(client *openai.Client) GroqOption {
	return func(g *GroqLLM) {
		g.client = client
	}
}

// NewGroqLLM creates a new Groq LLM client.
func NewGroqLLM(opts ...GroqOption) *GroqLLM {
	apiKey := os.Getenv("GROQ_API_KEY")

	config := openai.DefaultConfig(apiKey)
	config.BaseURL = GroqAPIURL

	g := &GroqLLM{
		client: openai.NewClientWithConfig(config),
		model:  DefaultGroqModel,
		logger: slog.New(slog.NewJSONHandler(os.Stdout, nil)),
	}

	for _, opt := range opts {
		opt(g)
	}

	return g
}

// Complete generates a completion for a given prompt.
func (g *GroqLLM) Complete(ctx context.Context, prompt string) (string, error) {
	g.logger.Info("Complete called", "model", g.model, "prompt_len", len(prompt))

	resp, err := g.client.CreateChatCompletion(
		ctx,
		openai.ChatCompletionRequest{
			Model: g.model,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: prompt,
				},
			},
		},
	)

	if err != nil {
		g.logger.Error("Complete failed", "error", err)
		return "", fmt.Errorf("groq completion failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("groq returned no choices")
	}

	return resp.Choices[0].Message.Content, nil
}

// Chat generates a response for a list of chat messages.
func (g *GroqLLM) Chat(ctx context.Context, messages []ChatMessage) (string, error) {
	g.logger.Info("Chat called", "model", g.model, "message_count", len(messages))

	openaiMessages := convertToOpenAIMessages(messages)

	resp, err := g.client.CreateChatCompletion(
		ctx,
		openai.ChatCompletionRequest{
			Model:    g.model,
			Messages: openaiMessages,
		},
	)

	if err != nil {
		g.logger.Error("Chat failed", "error", err)
		return "", fmt.Errorf("groq chat failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("groq returned no choices")
	}

	return resp.Choices[0].Message.Content, nil
}

// Stream generates a streaming completion for a given prompt.
func (g *GroqLLM) Stream(ctx context.Context, prompt string) (<-chan string, error) {
	g.logger.Info("Stream called", "model", g.model, "prompt_len", len(prompt))

	stream, err := g.client.CreateChatCompletionStream(
		ctx,
		openai.ChatCompletionRequest{
			Model: g.model,
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
		g.logger.Error("Stream failed", "error", err)
		return nil, fmt.Errorf("groq stream failed: %w", err)
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
				g.logger.Error("Stream receive error", "error", err)
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
func (g *GroqLLM) Metadata() LLMMetadata {
	return getGroqModelMetadata(g.model)
}

// SupportsToolCalling returns true if the model supports tool calling.
func (g *GroqLLM) SupportsToolCalling() bool {
	return groqToolCallingModels[g.model]
}

// SupportsStructuredOutput returns true if the model supports structured output.
func (g *GroqLLM) SupportsStructuredOutput() bool {
	// Groq supports JSON mode via response_format
	return true
}

// ChatWithTools generates a response that may include tool calls.
func (g *GroqLLM) ChatWithTools(ctx context.Context, messages []ChatMessage, tools []*ToolMetadata, opts *ChatCompletionOptions) (CompletionResponse, error) {
	g.logger.Info("ChatWithTools called", "model", g.model, "message_count", len(messages), "tool_count", len(tools))

	openaiMessages := convertToOpenAIMessages(messages)
	openaiTools := convertToOpenAITools(tools)

	req := openai.ChatCompletionRequest{
		Model:    g.model,
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

	resp, err := g.client.CreateChatCompletion(ctx, req)
	if err != nil {
		g.logger.Error("ChatWithTools failed", "error", err)
		return CompletionResponse{}, fmt.Errorf("groq chat with tools failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return CompletionResponse{}, fmt.Errorf("groq returned no choices")
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
func (g *GroqLLM) ChatWithFormat(ctx context.Context, messages []ChatMessage, format *ResponseFormat) (string, error) {
	g.logger.Info("ChatWithFormat called", "model", g.model, "message_count", len(messages), "format", format.Type)

	openaiMessages := convertToOpenAIMessages(messages)

	req := openai.ChatCompletionRequest{
		Model:    g.model,
		Messages: openaiMessages,
	}

	if format != nil {
		switch format.Type {
		case "json_object":
			req.ResponseFormat = &openai.ChatCompletionResponseFormat{
				Type: openai.ChatCompletionResponseFormatTypeJSONObject,
			}
		case "json_schema":
			// Fall back to json_object mode for Groq
			req.ResponseFormat = &openai.ChatCompletionResponseFormat{
				Type: openai.ChatCompletionResponseFormatTypeJSONObject,
			}
		}
	}

	resp, err := g.client.CreateChatCompletion(ctx, req)
	if err != nil {
		g.logger.Error("ChatWithFormat failed", "error", err)
		return "", fmt.Errorf("groq chat with format failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("groq returned no choices")
	}

	return resp.Choices[0].Message.Content, nil
}

// StreamChat generates a streaming response for chat messages.
func (g *GroqLLM) StreamChat(ctx context.Context, messages []ChatMessage) (<-chan StreamToken, error) {
	g.logger.Info("StreamChat called", "model", g.model, "message_count", len(messages))

	openaiMessages := convertToOpenAIMessages(messages)

	stream, err := g.client.CreateChatCompletionStream(
		ctx,
		openai.ChatCompletionRequest{
			Model:    g.model,
			Messages: openaiMessages,
			Stream:   true,
		},
	)

	if err != nil {
		g.logger.Error("StreamChat failed", "error", err)
		return nil, fmt.Errorf("groq stream chat failed: %w", err)
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
				g.logger.Error("StreamChat receive error", "error", err)
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

// getGroqModelMetadata returns metadata for Groq models.
func getGroqModelMetadata(model string) LLMMetadata {
	contextWindow := 8192 // default
	if cw, ok := groqModelContextWindows[model]; ok && cw > 0 {
		contextWindow = cw
	}

	return LLMMetadata{
		ModelName:         model,
		ContextWindow:     contextWindow,
		NumOutputTokens:   8192,
		IsChat:            true,
		IsFunctionCalling: groqToolCallingModels[model],
		IsMultiModal:      false, // Groq doesn't support multi-modal yet
		SystemRole:        "system",
	}
}

// IsGroqFunctionCallingModel returns true if the model supports function calling.
func IsGroqFunctionCallingModel(model string) bool {
	return groqToolCallingModels[model]
}

// GroqModelContextSize returns the context window size for a model.
func GroqModelContextSize(model string) int {
	if cw, ok := groqModelContextWindows[model]; ok {
		return cw
	}
	return 8192 // default
}

// Ensure GroqLLM implements the interfaces.
var _ LLM = (*GroqLLM)(nil)
var _ LLMWithMetadata = (*GroqLLM)(nil)
var _ LLMWithToolCalling = (*GroqLLM)(nil)
var _ LLMWithStructuredOutput = (*GroqLLM)(nil)
var _ FullLLM = (*GroqLLM)(nil)
