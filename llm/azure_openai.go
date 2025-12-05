package llm

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"

	openai "github.com/sashabaranov/go-openai"
)

// AzureOpenAILLM implements the LLM interface for Azure OpenAI models.
// It uses the same underlying client as OpenAI but with Azure-specific configuration.
type AzureOpenAILLM struct {
	client     *openai.Client
	model      string // This is the deployment name in Azure
	logger     *slog.Logger
	apiVersion string
}

// AzureOpenAIOption configures an AzureOpenAILLM.
type AzureOpenAIOption func(*AzureOpenAILLM)

// WithAzureDeployment sets the deployment name (model).
func WithAzureDeployment(deployment string) AzureOpenAIOption {
	return func(a *AzureOpenAILLM) {
		a.model = deployment
	}
}

// WithAzureAPIVersion sets the API version.
func WithAzureAPIVersion(version string) AzureOpenAIOption {
	return func(a *AzureOpenAILLM) {
		a.apiVersion = version
	}
}

// NewAzureOpenAILLM creates a new Azure OpenAI LLM client.
// It requires the Azure endpoint and API key, which can be provided via
// environment variables AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY.
func NewAzureOpenAILLM(opts ...AzureOpenAIOption) *AzureOpenAILLM {
	endpoint := os.Getenv("AZURE_OPENAI_ENDPOINT")
	apiKey := os.Getenv("AZURE_OPENAI_API_KEY")
	deployment := os.Getenv("AZURE_OPENAI_DEPLOYMENT")

	a := &AzureOpenAILLM{
		model:      deployment,
		apiVersion: "2024-02-15-preview",
		logger:     slog.New(slog.NewJSONHandler(os.Stdout, nil)),
	}

	for _, opt := range opts {
		opt(a)
	}

	// Create Azure OpenAI config
	config := openai.DefaultAzureConfig(apiKey, endpoint)
	config.APIVersion = a.apiVersion

	a.client = openai.NewClientWithConfig(config)

	return a
}

// NewAzureOpenAILLMWithConfig creates a new Azure OpenAI LLM client with explicit configuration.
func NewAzureOpenAILLMWithConfig(endpoint, apiKey, deployment, apiVersion string) *AzureOpenAILLM {
	if apiVersion == "" {
		apiVersion = "2024-02-15-preview"
	}

	config := openai.DefaultAzureConfig(apiKey, endpoint)
	config.APIVersion = apiVersion

	return &AzureOpenAILLM{
		client:     openai.NewClientWithConfig(config),
		model:      deployment,
		apiVersion: apiVersion,
		logger:     slog.New(slog.NewJSONHandler(os.Stdout, nil)),
	}
}

// Complete generates a completion for a given prompt.
func (a *AzureOpenAILLM) Complete(ctx context.Context, prompt string) (string, error) {
	a.logger.Info("Complete called", "deployment", a.model, "prompt_len", len(prompt))

	resp, err := a.client.CreateChatCompletion(
		ctx,
		openai.ChatCompletionRequest{
			Model: a.model,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: prompt,
				},
			},
		},
	)

	if err != nil {
		a.logger.Error("Complete failed", "error", err)
		return "", fmt.Errorf("azure openai completion failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("azure openai returned no choices")
	}

	return resp.Choices[0].Message.Content, nil
}

// Chat generates a response for a list of chat messages.
func (a *AzureOpenAILLM) Chat(ctx context.Context, messages []ChatMessage) (string, error) {
	a.logger.Info("Chat called", "deployment", a.model, "message_count", len(messages))

	openaiMessages := convertToOpenAIMessages(messages)

	resp, err := a.client.CreateChatCompletion(
		ctx,
		openai.ChatCompletionRequest{
			Model:    a.model,
			Messages: openaiMessages,
		},
	)

	if err != nil {
		a.logger.Error("Chat failed", "error", err)
		return "", fmt.Errorf("azure openai chat failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("azure openai returned no choices")
	}

	return resp.Choices[0].Message.Content, nil
}

// Stream generates a streaming completion for a given prompt.
func (a *AzureOpenAILLM) Stream(ctx context.Context, prompt string) (<-chan string, error) {
	a.logger.Info("Stream called", "deployment", a.model, "prompt_len", len(prompt))

	stream, err := a.client.CreateChatCompletionStream(
		ctx,
		openai.ChatCompletionRequest{
			Model: a.model,
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
		a.logger.Error("Stream failed", "error", err)
		return nil, fmt.Errorf("azure openai stream failed: %w", err)
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
				a.logger.Error("Stream receive error", "error", err)
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
func (a *AzureOpenAILLM) Metadata() LLMMetadata {
	// Azure deployments can use various models, return generic metadata
	return LLMMetadata{
		ModelName:         a.model,
		ContextWindow:     128000, // Assume modern model
		NumOutputTokens:   4096,
		IsChat:            true,
		IsFunctionCalling: true,
		IsMultiModal:      true,
		SystemRole:        "system",
	}
}

// SupportsToolCalling returns true if the model supports tool calling.
func (a *AzureOpenAILLM) SupportsToolCalling() bool {
	return true // Most Azure OpenAI deployments support tool calling
}

// SupportsStructuredOutput returns true if the model supports structured output.
func (a *AzureOpenAILLM) SupportsStructuredOutput() bool {
	return true
}

// ChatWithTools generates a response that may include tool calls.
func (a *AzureOpenAILLM) ChatWithTools(ctx context.Context, messages []ChatMessage, tools []*ToolMetadata, opts *ChatCompletionOptions) (CompletionResponse, error) {
	a.logger.Info("ChatWithTools called", "deployment", a.model, "message_count", len(messages), "tool_count", len(tools))

	openaiMessages := convertToOpenAIMessages(messages)
	openaiTools := convertToOpenAITools(tools)

	req := openai.ChatCompletionRequest{
		Model:    a.model,
		Messages: openaiMessages,
		Tools:    openaiTools,
	}

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

	resp, err := a.client.CreateChatCompletion(ctx, req)
	if err != nil {
		a.logger.Error("ChatWithTools failed", "error", err)
		return CompletionResponse{}, fmt.Errorf("azure openai chat with tools failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return CompletionResponse{}, fmt.Errorf("azure openai returned no choices")
	}

	choice := resp.Choices[0]
	response := CompletionResponse{
		Text: choice.Message.Content,
	}

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
func (a *AzureOpenAILLM) ChatWithFormat(ctx context.Context, messages []ChatMessage, format *ResponseFormat) (string, error) {
	a.logger.Info("ChatWithFormat called", "deployment", a.model, "message_count", len(messages))

	openaiMessages := convertToOpenAIMessages(messages)

	req := openai.ChatCompletionRequest{
		Model:    a.model,
		Messages: openaiMessages,
	}

	if format != nil {
		switch format.Type {
		case "json_object":
			req.ResponseFormat = &openai.ChatCompletionResponseFormat{
				Type: openai.ChatCompletionResponseFormatTypeJSONObject,
			}
		case "json_schema":
			req.ResponseFormat = &openai.ChatCompletionResponseFormat{
				Type: openai.ChatCompletionResponseFormatTypeJSONObject,
			}
		}
	}

	resp, err := a.client.CreateChatCompletion(ctx, req)
	if err != nil {
		a.logger.Error("ChatWithFormat failed", "error", err)
		return "", fmt.Errorf("azure openai chat with format failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("azure openai returned no choices")
	}

	return resp.Choices[0].Message.Content, nil
}

// StreamChat generates a streaming response for chat messages.
func (a *AzureOpenAILLM) StreamChat(ctx context.Context, messages []ChatMessage) (<-chan StreamToken, error) {
	a.logger.Info("StreamChat called", "deployment", a.model, "message_count", len(messages))

	openaiMessages := convertToOpenAIMessages(messages)

	stream, err := a.client.CreateChatCompletionStream(
		ctx,
		openai.ChatCompletionRequest{
			Model:    a.model,
			Messages: openaiMessages,
			Stream:   true,
		},
	)

	if err != nil {
		a.logger.Error("StreamChat failed", "error", err)
		return nil, fmt.Errorf("azure openai stream chat failed: %w", err)
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
				a.logger.Error("StreamChat receive error", "error", err)
				return
			}

			if len(response.Choices) > 0 {
				choice := response.Choices[0]
				token := StreamToken{
					Delta:        choice.Delta.Content,
					FinishReason: string(choice.FinishReason),
				}

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

// Ensure AzureOpenAILLM implements the interfaces.
var _ LLM = (*AzureOpenAILLM)(nil)
var _ LLMWithMetadata = (*AzureOpenAILLM)(nil)
var _ LLMWithToolCalling = (*AzureOpenAILLM)(nil)
var _ LLMWithStructuredOutput = (*AzureOpenAILLM)(nil)
var _ FullLLM = (*AzureOpenAILLM)(nil)
