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
	OpenAIAPIURLv1 = "https://api.openai.com/v1"
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
			baseUrl = OpenAIAPIURLv1
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

	openaiMessages := make([]openai.ChatCompletionMessage, len(messages))
	for i, msg := range messages {
		openaiMessages[i] = openai.ChatCompletionMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

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
