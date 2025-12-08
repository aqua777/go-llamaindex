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
	OpenAI_API_URL_v1 = "https://api.openai.com/v1"
)

type OpenAILLM struct {
	client *openai.Client
	model  string
	logger *slog.Logger
}

func NewOpenAILLM(baseUrl, model, apiKey string) *OpenAILLM {
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}

	if baseUrl == "" {
		baseUrl = os.Getenv("OPENAI_URL")
		if baseUrl == "" {
			baseUrl = OpenAI_API_URL_v1
		}
	}

	// Default to gpt-3.5-turbo if not specified
	if model == "" {
		model = openai.GPT3Dot5Turbo
	}

	var client *openai.Client
	if baseUrl != "" {
		config := openai.DefaultConfig(apiKey)
		config.BaseURL = baseUrl
		client = openai.NewClientWithConfig(config)
	} else {
		client = openai.NewClient(apiKey)
	}

	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))

	return &OpenAILLM{
		client: client,
		model:  model,
		logger: logger,
	}
}

func NewOpenAILLMWithClient(client *openai.Client, model string) *OpenAILLM {
	// Default to gpt-3.5-turbo if not specified
	if model == "" {
		model = openai.GPT3Dot5Turbo
	}

	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))

	return &OpenAILLM{
		client: client,
		model:  model,
		logger: logger,
	}
}

func (o *OpenAILLM) Complete(ctx context.Context, prompt string) (string, error) {
	o.logger.Info("Complete called", "model", o.model, "prompt_len", len(prompt))

	resp, err := o.client.CreateChatCompletion(
		ctx,
		openai.ChatCompletionRequest{
			Model: o.model,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: prompt,
				},
			},
		},
	)

	if err != nil {
		o.logger.Error("Complete failed", "error", err)
		return "", fmt.Errorf("openai completion failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("openai returned no choices")
	}

	return resp.Choices[0].Message.Content, nil
}

func (o *OpenAILLM) Chat(ctx context.Context, messages []ChatMessage) (string, error) {
	o.logger.Info("Chat called", "model", o.model, "message_count", len(messages))

	openaiMessages := convertToOpenAIMessages(messages)

	resp, err := o.client.CreateChatCompletion(
		ctx,
		openai.ChatCompletionRequest{
			Model:    o.model,
			Messages: openaiMessages,
		},
	)

	if err != nil {
		o.logger.Error("Chat failed", "error", err)
		return "", fmt.Errorf("openai chat failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("openai returned no choices")
	}

	return resp.Choices[0].Message.Content, nil
}

func (o *OpenAILLM) Stream(ctx context.Context, prompt string) (<-chan string, error) {
	o.logger.Info("Stream called", "model", o.model, "prompt_len", len(prompt))

	stream, err := o.client.CreateChatCompletionStream(
		ctx,
		openai.ChatCompletionRequest{
			Model: o.model,
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
		o.logger.Error("Stream failed", "error", err)
		return nil, fmt.Errorf("openai stream failed: %w", err)
	}

	// Create a channel to send tokens
	tokenChan := make(chan string)

	// Start a goroutine to read from the stream and send to the channel
	go func() {
		defer close(tokenChan)
		defer stream.Close()

		for {
			response, err := stream.Recv()
			if err == io.EOF {
				return
			}
			if err != nil {
				o.logger.Error("Stream receive error", "error", err)
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
func (o *OpenAILLM) Metadata() LLMMetadata {
	return getModelMetadata(o.model)
}

// SupportsToolCalling returns true if the model supports tool calling.
func (o *OpenAILLM) SupportsToolCalling() bool {
	meta := o.Metadata()
	return meta.IsFunctionCalling
}

// SupportsStructuredOutput returns true if the model supports structured output.
func (o *OpenAILLM) SupportsStructuredOutput() bool {
	// Most modern OpenAI models support JSON mode
	return true
}

// ChatWithTools generates a response that may include tool calls.
func (o *OpenAILLM) ChatWithTools(ctx context.Context, messages []ChatMessage, tools []*ToolMetadata, opts *ChatCompletionOptions) (CompletionResponse, error) {
	o.logger.Info("ChatWithTools called", "model", o.model, "message_count", len(messages), "tool_count", len(tools))

	openaiMessages := convertToOpenAIMessages(messages)
	openaiTools := convertToOpenAITools(tools)

	req := openai.ChatCompletionRequest{
		Model:    o.model,
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

	resp, err := o.client.CreateChatCompletion(ctx, req)
	if err != nil {
		o.logger.Error("ChatWithTools failed", "error", err)
		return CompletionResponse{}, fmt.Errorf("openai chat with tools failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return CompletionResponse{}, fmt.Errorf("openai returned no choices")
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
func (o *OpenAILLM) ChatWithFormat(ctx context.Context, messages []ChatMessage, format *ResponseFormat) (string, error) {
	o.logger.Info("ChatWithFormat called", "model", o.model, "message_count", len(messages), "format", format.Type)

	openaiMessages := convertToOpenAIMessages(messages)

	req := openai.ChatCompletionRequest{
		Model:    o.model,
		Messages: openaiMessages,
	}

	if format != nil {
		switch format.Type {
		case "json_object":
			req.ResponseFormat = &openai.ChatCompletionResponseFormat{
				Type: openai.ChatCompletionResponseFormatTypeJSONObject,
			}
		case "json_schema":
			// Note: JSON schema support requires the schema to implement json.Marshaler
			// For now, we fall back to json_object mode
			req.ResponseFormat = &openai.ChatCompletionResponseFormat{
				Type: openai.ChatCompletionResponseFormatTypeJSONObject,
			}
		}
	}

	resp, err := o.client.CreateChatCompletion(ctx, req)
	if err != nil {
		o.logger.Error("ChatWithFormat failed", "error", err)
		return "", fmt.Errorf("openai chat with format failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("openai returned no choices")
	}

	return resp.Choices[0].Message.Content, nil
}

// StreamChat generates a streaming response for chat messages.
func (o *OpenAILLM) StreamChat(ctx context.Context, messages []ChatMessage) (<-chan StreamToken, error) {
	o.logger.Info("StreamChat called", "model", o.model, "message_count", len(messages))

	openaiMessages := convertToOpenAIMessages(messages)

	stream, err := o.client.CreateChatCompletionStream(
		ctx,
		openai.ChatCompletionRequest{
			Model:    o.model,
			Messages: openaiMessages,
			Stream:   true,
		},
	)

	if err != nil {
		o.logger.Error("StreamChat failed", "error", err)
		return nil, fmt.Errorf("openai stream chat failed: %w", err)
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
				o.logger.Error("StreamChat receive error", "error", err)
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

// Helper functions

// convertToOpenAIMessages converts ChatMessage slice to OpenAI format.
func convertToOpenAIMessages(messages []ChatMessage) []openai.ChatCompletionMessage {
	openaiMessages := make([]openai.ChatCompletionMessage, len(messages))
	for i, msg := range messages {
		openaiMsg := openai.ChatCompletionMessage{
			Role:    string(msg.Role),
			Content: msg.GetTextContent(),
		}
		if msg.Name != "" {
			openaiMsg.Name = msg.Name
		}
		if msg.ToolCallID != "" {
			openaiMsg.ToolCallID = msg.ToolCallID
		}
		// Handle tool calls in the message
		if msg.HasToolCalls() {
			for _, tc := range msg.GetToolCalls() {
				openaiMsg.ToolCalls = append(openaiMsg.ToolCalls, openai.ToolCall{
					ID:   tc.ID,
					Type: openai.ToolTypeFunction,
					Function: openai.FunctionCall{
						Name:      tc.Name,
						Arguments: tc.Arguments,
					},
				})
			}
		}
		openaiMessages[i] = openaiMsg
	}
	return openaiMessages
}

// convertToOpenAITools converts ToolMetadata slice to OpenAI format.
func convertToOpenAITools(tools []*ToolMetadata) []openai.Tool {
	openaiTools := make([]openai.Tool, len(tools))
	for i, tool := range tools {
		openaiTools[i] = openai.Tool{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.Parameters,
			},
		}
	}
	return openaiTools
}

// getModelMetadata returns metadata for known models.
func getModelMetadata(model string) LLMMetadata {
	switch model {
	case "gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106":
		return GPT35TurboMetadata()
	case "gpt-4", "gpt-4-0613":
		return GPT4Metadata()
	case "gpt-4-turbo", "gpt-4-turbo-preview", "gpt-4-1106-preview":
		return GPT4TurboMetadata()
	case "gpt-4o", "gpt-4o-2024-05-13", "gpt-4o-mini":
		return GPT4oMetadata()
	default:
		return DefaultLLMMetadata(model)
	}
}
