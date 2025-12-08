// Package bedrock provides an AWS Bedrock LLM implementation.
// This is a separate sub-module to avoid pulling AWS SDK dependencies
// unless this provider is explicitly used.
package bedrock

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

const (
	// DefaultModel is the default model to use.
	DefaultModel = "anthropic.claude-3-5-sonnet-20241022-v2:0"
	// DefaultMaxTokens is the default max tokens.
	DefaultMaxTokens = 1024
)

// Model constants - Anthropic Claude models.
const (
	ClaudeInstantV1  = "anthropic.claude-instant-v1"
	ClaudeV2         = "anthropic.claude-v2"
	ClaudeV21        = "anthropic.claude-v2:1"
	Claude3Sonnet    = "anthropic.claude-3-sonnet-20240229-v1:0"
	Claude3Haiku     = "anthropic.claude-3-haiku-20240307-v1:0"
	Claude3Opus      = "anthropic.claude-3-opus-20240229-v1:0"
	Claude35Sonnet   = "anthropic.claude-3-5-sonnet-20240620-v1:0"
	Claude35SonnetV2 = "anthropic.claude-3-5-sonnet-20241022-v2:0"
	Claude35Haiku    = "anthropic.claude-3-5-haiku-20241022-v1:0"
	Claude37Sonnet   = "anthropic.claude-3-7-sonnet-20250219-v1:0"
	Claude4Sonnet    = "anthropic.claude-sonnet-4-20250514-v1:0"
	Claude4Opus      = "anthropic.claude-opus-4-20250514-v1:0"
	Claude41Opus     = "anthropic.claude-opus-4-1-20250805-v1:0"
	Claude45Sonnet   = "anthropic.claude-sonnet-4-5-20250929-v1:0"
)

// Model constants - Amazon models.
const (
	TitanTextExpressV1 = "amazon.titan-text-express-v1"
	TitanTextLiteV1    = "amazon.titan-text-lite-v1"
	NovaPremierV1      = "amazon.nova-premier-v1:0"
	NovaProV1          = "amazon.nova-pro-v1:0"
	NovaLiteV1         = "amazon.nova-lite-v1:0"
	NovaMicroV1        = "amazon.nova-micro-v1:0"
)

// Model constants - Meta Llama models.
const (
	Llama2_13BChat       = "meta.llama2-13b-chat-v1"
	Llama2_70BChat       = "meta.llama2-70b-chat-v1"
	Llama3_8BInstruct    = "meta.llama3-8b-instruct-v1:0"
	Llama3_70BInstruct   = "meta.llama3-70b-instruct-v1:0"
	Llama31_8BInstruct   = "meta.llama3-1-8b-instruct-v1:0"
	Llama31_70BInstruct  = "meta.llama3-1-70b-instruct-v1:0"
	Llama31_405BInstruct = "meta.llama3-1-405b-instruct-v1:0"
	Llama32_1BInstruct   = "meta.llama3-2-1b-instruct-v1:0"
	Llama32_3BInstruct   = "meta.llama3-2-3b-instruct-v1:0"
	Llama32_11BInstruct  = "meta.llama3-2-11b-instruct-v1:0"
	Llama32_90BInstruct  = "meta.llama3-2-90b-instruct-v1:0"
	Llama33_70BInstruct  = "meta.llama3-3-70b-instruct-v1:0"
)

// Model constants - Mistral models.
const (
	Mistral7BInstruct   = "mistral.mistral-7b-instruct-v0:2"
	Mixtral8x7BInstruct = "mistral.mixtral-8x7b-instruct-v0:1"
	MistralLarge2402    = "mistral.mistral-large-2402-v1:0"
)

// Model constants - Cohere models.
const (
	CohereCommandTextV14 = "cohere.command-text-v14"
	CohereCommandRV1     = "cohere.command-r-v1:0"
	CohereCommandRPlusV1 = "cohere.command-r-plus-v1:0"
)

// modelContextWindows maps model names to their context window sizes.
var modelContextWindows = map[string]int{
	// Amazon models
	NovaPremierV1:      1000000,
	NovaProV1:          300000,
	NovaLiteV1:         300000,
	NovaMicroV1:        128000,
	TitanTextExpressV1: 8192,
	TitanTextLiteV1:    4096,

	// Anthropic Claude models
	ClaudeInstantV1:  100000,
	ClaudeV2:         100000,
	ClaudeV21:        200000,
	Claude3Sonnet:    200000,
	Claude3Haiku:     200000,
	Claude3Opus:      200000,
	Claude35Sonnet:   200000,
	Claude35SonnetV2: 200000,
	Claude35Haiku:    200000,
	Claude37Sonnet:   200000,
	Claude4Sonnet:    200000,
	Claude4Opus:      200000,
	Claude41Opus:     200000,
	Claude45Sonnet:   200000,

	// Meta Llama models
	Llama2_13BChat:       2048,
	Llama2_70BChat:       4096,
	Llama3_8BInstruct:    8192,
	Llama3_70BInstruct:   8192,
	Llama31_8BInstruct:   128000,
	Llama31_70BInstruct:  128000,
	Llama31_405BInstruct: 128000,
	Llama32_1BInstruct:   131000,
	Llama32_3BInstruct:   131000,
	Llama32_11BInstruct:  128000,
	Llama32_90BInstruct:  128000,
	Llama33_70BInstruct:  128000,

	// Mistral models
	Mistral7BInstruct:   32000,
	Mixtral8x7BInstruct: 32000,
	MistralLarge2402:    32000,

	// Cohere models
	CohereCommandTextV14: 4096,
	CohereCommandRV1:     128000,
	CohereCommandRPlusV1: 128000,
}

// toolCallingModels lists models that support tool/function calling.
var toolCallingModels = map[string]bool{
	// Amazon Nova models
	NovaPremierV1: true,
	NovaProV1:     true,
	NovaLiteV1:    true,
	NovaMicroV1:   true,

	// Anthropic Claude 3+ models
	Claude3Sonnet:    true,
	Claude3Haiku:     true,
	Claude3Opus:      true,
	Claude35Sonnet:   true,
	Claude35SonnetV2: true,
	Claude35Haiku:    true,
	Claude37Sonnet:   true,
	Claude4Sonnet:    true,
	Claude4Opus:      true,
	Claude41Opus:     true,
	Claude45Sonnet:   true,

	// Meta Llama 3.1+ models
	Llama31_8BInstruct:   true,
	Llama31_70BInstruct:  true,
	Llama31_405BInstruct: true,
	Llama32_11BInstruct:  true,
	Llama32_90BInstruct:  true,
	Llama33_70BInstruct:  true,

	// Cohere Command R models
	CohereCommandRV1:     true,
	CohereCommandRPlusV1: true,

	// Mistral Large
	MistralLarge2402: true,
}

// multiModalModels lists models that support multi-modal inputs.
var multiModalModels = map[string]bool{
	Claude3Sonnet:    true,
	Claude3Haiku:     true,
	Claude3Opus:      true,
	Claude35Sonnet:   true,
	Claude35SonnetV2: true,
	Claude35Haiku:    true,
	Claude37Sonnet:   true,
	Claude4Sonnet:    true,
	Claude4Opus:      true,
	Claude41Opus:     true,
	Claude45Sonnet:   true,
	NovaProV1:        true,
	NovaLiteV1:       true,
}

// LLM implements the llm.LLM interface for AWS Bedrock using the Converse API.
type LLM struct {
	client      *bedrockruntime.Client
	model       string
	maxTokens   int
	temperature float32
	topP        float32
	region      string
	logger      *slog.Logger
}

// Option configures an LLM.
type Option func(*LLM)

// WithModel sets the model.
func WithModel(model string) Option {
	return func(b *LLM) {
		b.model = model
	}
}

// WithMaxTokens sets the max tokens.
func WithMaxTokens(maxTokens int) Option {
	return func(b *LLM) {
		b.maxTokens = maxTokens
	}
}

// WithTemperature sets the temperature.
func WithTemperature(temperature float32) Option {
	return func(b *LLM) {
		b.temperature = temperature
	}
}

// WithTopP sets the top_p value.
func WithTopP(topP float32) Option {
	return func(b *LLM) {
		b.topP = topP
	}
}

// WithRegion sets the AWS region.
func WithRegion(region string) Option {
	return func(b *LLM) {
		b.region = region
	}
}

// WithCredentials sets explicit AWS credentials.
func WithCredentials(accessKeyID, secretAccessKey, sessionToken string) Option {
	return func(b *LLM) {
		cfg, err := config.LoadDefaultConfig(context.Background(),
			config.WithRegion(b.region),
			config.WithCredentialsProvider(credentials.NewStaticCredentialsProvider(
				accessKeyID,
				secretAccessKey,
				sessionToken,
			)),
		)
		if err == nil {
			b.client = bedrockruntime.NewFromConfig(cfg)
		}
	}
}

// WithClient sets a custom Bedrock client (for testing).
func WithClient(client *bedrockruntime.Client) Option {
	return func(b *LLM) {
		b.client = client
	}
}

// New creates a new AWS Bedrock LLM client.
func New(opts ...Option) *LLM {
	region := os.Getenv("AWS_REGION")
	if region == "" {
		region = os.Getenv("AWS_DEFAULT_REGION")
	}
	if region == "" {
		region = "us-east-1"
	}

	b := &LLM{
		model:       DefaultModel,
		maxTokens:   DefaultMaxTokens,
		temperature: 0.1,
		topP:        1.0,
		region:      region,
		logger:      slog.New(slog.NewJSONHandler(os.Stdout, nil)),
	}

	// Apply options first to get region
	for _, opt := range opts {
		opt(b)
	}

	// Initialize client if not already set
	if b.client == nil {
		cfg, err := config.LoadDefaultConfig(context.Background(),
			config.WithRegion(b.region),
		)
		if err == nil {
			b.client = bedrockruntime.NewFromConfig(cfg)
		}
	}

	return b
}

// Complete generates a completion for a given prompt.
func (b *LLM) Complete(ctx context.Context, prompt string) (string, error) {
	b.logger.Info("Complete called", "model", b.model, "prompt_len", len(prompt))

	messages := []llm.ChatMessage{llm.NewUserMessage(prompt)}
	return b.Chat(ctx, messages)
}

// Chat generates a response for a list of chat messages.
func (b *LLM) Chat(ctx context.Context, messages []llm.ChatMessage) (string, error) {
	b.logger.Info("Chat called", "model", b.model, "message_count", len(messages))

	converseMessages, systemPrompts := b.convertMessages(messages)

	input := &bedrockruntime.ConverseInput{
		ModelId:  aws.String(b.model),
		Messages: converseMessages,
		InferenceConfig: &types.InferenceConfiguration{
			MaxTokens:   aws.Int32(int32(b.maxTokens)),
			Temperature: aws.Float32(b.temperature),
			TopP:        aws.Float32(b.topP),
		},
	}

	if len(systemPrompts) > 0 {
		input.System = systemPrompts
	}

	resp, err := b.client.Converse(ctx, input)
	if err != nil {
		b.logger.Error("Chat failed", "error", err)
		return "", fmt.Errorf("bedrock converse failed: %w", err)
	}

	return b.extractTextFromResponse(resp), nil
}

// Stream generates a streaming completion for a given prompt.
func (b *LLM) Stream(ctx context.Context, prompt string) (<-chan string, error) {
	b.logger.Info("Stream called", "model", b.model, "prompt_len", len(prompt))

	messages := []llm.ChatMessage{llm.NewUserMessage(prompt)}
	converseMessages, systemPrompts := b.convertMessages(messages)

	input := &bedrockruntime.ConverseStreamInput{
		ModelId:  aws.String(b.model),
		Messages: converseMessages,
		InferenceConfig: &types.InferenceConfiguration{
			MaxTokens:   aws.Int32(int32(b.maxTokens)),
			Temperature: aws.Float32(b.temperature),
			TopP:        aws.Float32(b.topP),
		},
	}

	if len(systemPrompts) > 0 {
		input.System = systemPrompts
	}

	resp, err := b.client.ConverseStream(ctx, input)
	if err != nil {
		b.logger.Error("Stream failed", "error", err)
		return nil, fmt.Errorf("bedrock stream failed: %w", err)
	}

	tokenChan := make(chan string)

	go func() {
		defer close(tokenChan)

		stream := resp.GetStream()
		for event := range stream.Events() {
			switch v := event.(type) {
			case *types.ConverseStreamOutputMemberContentBlockDelta:
				if textDelta, ok := v.Value.Delta.(*types.ContentBlockDeltaMemberText); ok {
					select {
					case tokenChan <- textDelta.Value:
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
func (b *LLM) Metadata() llm.LLMMetadata {
	return GetModelMetadata(b.model)
}

// SupportsToolCalling returns true if the model supports tool calling.
func (b *LLM) SupportsToolCalling() bool {
	return IsFunctionCallingModel(b.model)
}

// SupportsStructuredOutput returns true if the model supports structured output.
func (b *LLM) SupportsStructuredOutput() bool {
	// Bedrock Converse API supports tool use which can be used for structured output
	return b.SupportsToolCalling()
}

// ChatWithTools generates a response that may include tool calls.
func (b *LLM) ChatWithTools(ctx context.Context, messages []llm.ChatMessage, tools []*llm.ToolMetadata, opts *llm.ChatCompletionOptions) (llm.CompletionResponse, error) {
	b.logger.Info("ChatWithTools called", "model", b.model, "message_count", len(messages), "tool_count", len(tools))

	converseMessages, systemPrompts := b.convertMessages(messages)
	converseTools := b.convertTools(tools)

	input := &bedrockruntime.ConverseInput{
		ModelId:  aws.String(b.model),
		Messages: converseMessages,
		InferenceConfig: &types.InferenceConfiguration{
			MaxTokens:   aws.Int32(int32(b.maxTokens)),
			Temperature: aws.Float32(b.temperature),
			TopP:        aws.Float32(b.topP),
		},
	}

	if len(systemPrompts) > 0 {
		input.System = systemPrompts
	}

	if len(converseTools) > 0 {
		input.ToolConfig = &types.ToolConfiguration{
			Tools: converseTools,
		}
	}

	// Apply options
	if opts != nil {
		if opts.Temperature != nil {
			input.InferenceConfig.Temperature = aws.Float32(*opts.Temperature)
		}
		if opts.MaxTokens != nil {
			input.InferenceConfig.MaxTokens = aws.Int32(int32(*opts.MaxTokens))
		}
		if opts.TopP != nil {
			input.InferenceConfig.TopP = aws.Float32(*opts.TopP)
		}
	}

	resp, err := b.client.Converse(ctx, input)
	if err != nil {
		b.logger.Error("ChatWithTools failed", "error", err)
		return llm.CompletionResponse{}, fmt.Errorf("bedrock converse with tools failed: %w", err)
	}

	return b.convertResponse(resp), nil
}

// ChatWithFormat generates a response in the specified format.
func (b *LLM) ChatWithFormat(ctx context.Context, messages []llm.ChatMessage, format *llm.ResponseFormat) (string, error) {
	b.logger.Info("ChatWithFormat called", "model", b.model, "message_count", len(messages), "format", format.Type)

	// For JSON format, we can add instructions to the system prompt
	if format != nil && (format.Type == "json_object" || format.Type == "json_schema") {
		// Prepend JSON instruction to messages
		jsonInstruction := "You must respond with valid JSON only. Do not include any text outside the JSON object."
		if format.JSONSchema != nil {
			schemaBytes, _ := json.Marshal(format.JSONSchema)
			jsonInstruction = fmt.Sprintf("You must respond with valid JSON that conforms to this schema: %s", string(schemaBytes))
		}

		// Add system message with JSON instruction
		newMessages := make([]llm.ChatMessage, 0, len(messages)+1)
		newMessages = append(newMessages, llm.NewSystemMessage(jsonInstruction))
		newMessages = append(newMessages, messages...)
		messages = newMessages
	}

	return b.Chat(ctx, messages)
}

// StreamChat generates a streaming response for chat messages.
func (b *LLM) StreamChat(ctx context.Context, messages []llm.ChatMessage) (<-chan llm.StreamToken, error) {
	b.logger.Info("StreamChat called", "model", b.model, "message_count", len(messages))

	converseMessages, systemPrompts := b.convertMessages(messages)

	input := &bedrockruntime.ConverseStreamInput{
		ModelId:  aws.String(b.model),
		Messages: converseMessages,
		InferenceConfig: &types.InferenceConfiguration{
			MaxTokens:   aws.Int32(int32(b.maxTokens)),
			Temperature: aws.Float32(b.temperature),
			TopP:        aws.Float32(b.topP),
		},
	}

	if len(systemPrompts) > 0 {
		input.System = systemPrompts
	}

	resp, err := b.client.ConverseStream(ctx, input)
	if err != nil {
		b.logger.Error("StreamChat failed", "error", err)
		return nil, fmt.Errorf("bedrock stream chat failed: %w", err)
	}

	tokenChan := make(chan llm.StreamToken)

	go func() {
		defer close(tokenChan)

		var currentToolCall *llm.ToolCall
		stream := resp.GetStream()

		for event := range stream.Events() {
			switch v := event.(type) {
			case *types.ConverseStreamOutputMemberContentBlockDelta:
				token := llm.StreamToken{}

				if textDelta, ok := v.Value.Delta.(*types.ContentBlockDeltaMemberText); ok {
					token.Delta = textDelta.Value
				}

				if toolDelta, ok := v.Value.Delta.(*types.ContentBlockDeltaMemberToolUse); ok {
					if currentToolCall != nil && toolDelta.Value.Input != nil {
						currentToolCall.Arguments += aws.ToString(toolDelta.Value.Input)
					}
				}

				select {
				case tokenChan <- token:
				case <-ctx.Done():
					return
				}

			case *types.ConverseStreamOutputMemberContentBlockStart:
				if toolStart, ok := v.Value.Start.(*types.ContentBlockStartMemberToolUse); ok {
					currentToolCall = &llm.ToolCall{
						ID:   aws.ToString(toolStart.Value.ToolUseId),
						Name: aws.ToString(toolStart.Value.Name),
					}
				}

			case *types.ConverseStreamOutputMemberContentBlockStop:
				if currentToolCall != nil {
					token := llm.StreamToken{
						ToolCalls: []*llm.ToolCall{currentToolCall},
					}
					select {
					case tokenChan <- token:
					case <-ctx.Done():
						return
					}
					currentToolCall = nil
				}

			case *types.ConverseStreamOutputMemberMessageStop:
				token := llm.StreamToken{
					FinishReason: string(v.Value.StopReason),
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

// convertMessages converts ChatMessages to Bedrock Converse format.
func (b *LLM) convertMessages(messages []llm.ChatMessage) ([]types.Message, []types.SystemContentBlock) {
	var converseMessages []types.Message
	var systemPrompts []types.SystemContentBlock

	for _, msg := range messages {
		content := msg.GetTextContent()

		switch msg.Role {
		case llm.MessageRoleSystem:
			systemPrompts = append(systemPrompts, &types.SystemContentBlockMemberText{
				Value: content,
			})

		case llm.MessageRoleUser:
			converseMessages = append(converseMessages, types.Message{
				Role: types.ConversationRoleUser,
				Content: []types.ContentBlock{
					&types.ContentBlockMemberText{Value: content},
				},
			})

		case llm.MessageRoleAssistant:
			contentBlocks := []types.ContentBlock{}

			// Check for tool calls in blocks
			if msg.HasToolCalls() {
				for _, tc := range msg.GetToolCalls() {
					// Parse arguments JSON to interface{} for document
					var args interface{}
					if tc.Arguments != "" {
						json.Unmarshal([]byte(tc.Arguments), &args)
					}
					if args == nil {
						args = map[string]interface{}{}
					}

					contentBlocks = append(contentBlocks, &types.ContentBlockMemberToolUse{
						Value: types.ToolUseBlock{
							ToolUseId: aws.String(tc.ID),
							Name:      aws.String(tc.Name),
							Input:     document.NewLazyDocument(args),
						},
					})
				}
			}

			if content != "" {
				contentBlocks = append(contentBlocks, &types.ContentBlockMemberText{Value: content})
			}

			if len(contentBlocks) > 0 {
				converseMessages = append(converseMessages, types.Message{
					Role:    types.ConversationRoleAssistant,
					Content: contentBlocks,
				})
			}

		case llm.MessageRoleTool:
			// Tool results go as user messages with tool result content
			converseMessages = append(converseMessages, types.Message{
				Role: types.ConversationRoleUser,
				Content: []types.ContentBlock{
					&types.ContentBlockMemberToolResult{
						Value: types.ToolResultBlock{
							ToolUseId: aws.String(msg.ToolCallID),
							Content: []types.ToolResultContentBlock{
								&types.ToolResultContentBlockMemberText{Value: content},
							},
						},
					},
				},
			})
		}
	}

	return converseMessages, systemPrompts
}

// convertTools converts ToolMetadata to Bedrock tool specifications.
func (b *LLM) convertTools(tools []*llm.ToolMetadata) []types.Tool {
	var converseTools []types.Tool

	for _, tool := range tools {
		converseTools = append(converseTools, &types.ToolMemberToolSpec{
			Value: types.ToolSpecification{
				Name:        aws.String(tool.Name),
				Description: aws.String(tool.Description),
				InputSchema: &types.ToolInputSchemaMemberJson{
					Value: document.NewLazyDocument(tool.Parameters),
				},
			},
		})
	}

	return converseTools
}

// convertResponse converts Bedrock response to CompletionResponse.
func (b *LLM) convertResponse(resp *bedrockruntime.ConverseOutput) llm.CompletionResponse {
	response := llm.CompletionResponse{
		Text: b.extractTextFromResponse(resp),
	}

	// Extract tool calls
	var toolCalls []*llm.ToolCall
	if resp.Output != nil {
		if msgOutput, ok := resp.Output.(*types.ConverseOutputMemberMessage); ok {
			for _, block := range msgOutput.Value.Content {
				if toolUse, ok := block.(*types.ContentBlockMemberToolUse); ok {
					// Convert document.Interface to JSON string
					var argsStr string
					if toolUse.Value.Input != nil {
						var args interface{}
						if err := toolUse.Value.Input.UnmarshalSmithyDocument(&args); err == nil {
							if argsBytes, err := json.Marshal(args); err == nil {
								argsStr = string(argsBytes)
							}
						}
					}
					toolCalls = append(toolCalls, &llm.ToolCall{
						ID:        aws.ToString(toolUse.Value.ToolUseId),
						Name:      aws.ToString(toolUse.Value.Name),
						Arguments: argsStr,
					})
				}
			}
		}
	}

	if len(toolCalls) > 0 {
		msg := llm.ChatMessage{
			Role:    llm.MessageRoleAssistant,
			Content: response.Text,
		}
		for _, tc := range toolCalls {
			msg.Blocks = append(msg.Blocks, llm.NewToolCallBlock(tc))
		}
		response.Message = &msg
	}

	return response
}

// extractTextFromResponse extracts text content from Bedrock response.
func (b *LLM) extractTextFromResponse(resp *bedrockruntime.ConverseOutput) string {
	if resp.Output == nil {
		return ""
	}

	if msgOutput, ok := resp.Output.(*types.ConverseOutputMemberMessage); ok {
		var textParts []string
		for _, block := range msgOutput.Value.Content {
			if textBlock, ok := block.(*types.ContentBlockMemberText); ok {
				textParts = append(textParts, textBlock.Value)
			}
		}
		return strings.Join(textParts, "")
	}

	return ""
}

// GetModelMetadata returns metadata for Bedrock models.
func GetModelMetadata(model string) llm.LLMMetadata {
	// Handle region-prefixed models (us., eu., apac.)
	baseModel := model
	prefixes := []string{"us.", "eu.", "apac.", "jp.", "global."}
	for _, prefix := range prefixes {
		if strings.HasPrefix(model, prefix) {
			baseModel = model[len(prefix):]
			break
		}
	}

	contextWindow := 128000 // default
	if cw, ok := modelContextWindows[baseModel]; ok {
		contextWindow = cw
	}

	return llm.LLMMetadata{
		ModelName:         model,
		ContextWindow:     contextWindow,
		NumOutputTokens:   4096,
		IsChat:            true,
		IsFunctionCalling: IsFunctionCallingModel(model),
		IsMultiModal:      multiModalModels[baseModel],
		SystemRole:        "system",
	}
}

// IsFunctionCallingModel returns true if the model supports function calling.
func IsFunctionCallingModel(model string) bool {
	// Handle region-prefixed models
	baseModel := model
	prefixes := []string{"us.", "eu.", "apac.", "jp.", "global."}
	for _, prefix := range prefixes {
		if strings.HasPrefix(model, prefix) {
			baseModel = model[len(prefix):]
			break
		}
	}
	return toolCallingModels[baseModel]
}

// ModelContextSize returns the context window size for a model.
func ModelContextSize(model string) int {
	// Handle region-prefixed models
	baseModel := model
	prefixes := []string{"us.", "eu.", "apac.", "jp.", "global."}
	for _, prefix := range prefixes {
		if strings.HasPrefix(model, prefix) {
			baseModel = model[len(prefix):]
			break
		}
	}

	if cw, ok := modelContextWindows[baseModel]; ok {
		return cw
	}
	return 128000 // default
}

// Ensure LLM implements the interfaces.
var _ llm.LLM = (*LLM)(nil)
var _ llm.LLMWithMetadata = (*LLM)(nil)
var _ llm.LLMWithToolCalling = (*LLM)(nil)
var _ llm.LLMWithStructuredOutput = (*LLM)(nil)
var _ llm.FullLLM = (*LLM)(nil)
