package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

const (
	// DefaultBedrockModel is the default model to use.
	DefaultBedrockModel = "anthropic.claude-3-5-sonnet-20241022-v2:0"
	// DefaultBedrockMaxTokens is the default max tokens.
	DefaultBedrockMaxTokens = 1024
)

// Bedrock model constants - Anthropic Claude models.
const (
	BedrockClaudeInstantV1  = "anthropic.claude-instant-v1"
	BedrockClaudeV2         = "anthropic.claude-v2"
	BedrockClaudeV21        = "anthropic.claude-v2:1"
	BedrockClaude3Sonnet    = "anthropic.claude-3-sonnet-20240229-v1:0"
	BedrockClaude3Haiku     = "anthropic.claude-3-haiku-20240307-v1:0"
	BedrockClaude3Opus      = "anthropic.claude-3-opus-20240229-v1:0"
	BedrockClaude35Sonnet   = "anthropic.claude-3-5-sonnet-20240620-v1:0"
	BedrockClaude35SonnetV2 = "anthropic.claude-3-5-sonnet-20241022-v2:0"
	BedrockClaude35Haiku    = "anthropic.claude-3-5-haiku-20241022-v1:0"
	BedrockClaude37Sonnet   = "anthropic.claude-3-7-sonnet-20250219-v1:0"
	BedrockClaude4Sonnet    = "anthropic.claude-sonnet-4-20250514-v1:0"
	BedrockClaude4Opus      = "anthropic.claude-opus-4-20250514-v1:0"
	BedrockClaude41Opus     = "anthropic.claude-opus-4-1-20250805-v1:0"
	BedrockClaude45Sonnet   = "anthropic.claude-sonnet-4-5-20250929-v1:0"
)

// Bedrock model constants - Amazon models.
const (
	BedrockTitanTextExpressV1 = "amazon.titan-text-express-v1"
	BedrockTitanTextLiteV1    = "amazon.titan-text-lite-v1"
	BedrockNovaPremierV1      = "amazon.nova-premier-v1:0"
	BedrockNovaProV1          = "amazon.nova-pro-v1:0"
	BedrockNovaLiteV1         = "amazon.nova-lite-v1:0"
	BedrockNovaMicroV1        = "amazon.nova-micro-v1:0"
)

// Bedrock model constants - Meta Llama models.
const (
	BedrockLlama2_13BChat       = "meta.llama2-13b-chat-v1"
	BedrockLlama2_70BChat       = "meta.llama2-70b-chat-v1"
	BedrockLlama3_8BInstruct    = "meta.llama3-8b-instruct-v1:0"
	BedrockLlama3_70BInstruct   = "meta.llama3-70b-instruct-v1:0"
	BedrockLlama31_8BInstruct   = "meta.llama3-1-8b-instruct-v1:0"
	BedrockLlama31_70BInstruct  = "meta.llama3-1-70b-instruct-v1:0"
	BedrockLlama31_405BInstruct = "meta.llama3-1-405b-instruct-v1:0"
	BedrockLlama32_1BInstruct   = "meta.llama3-2-1b-instruct-v1:0"
	BedrockLlama32_3BInstruct   = "meta.llama3-2-3b-instruct-v1:0"
	BedrockLlama32_11BInstruct  = "meta.llama3-2-11b-instruct-v1:0"
	BedrockLlama32_90BInstruct  = "meta.llama3-2-90b-instruct-v1:0"
	BedrockLlama33_70BInstruct  = "meta.llama3-3-70b-instruct-v1:0"
)

// Bedrock model constants - Mistral models.
const (
	BedrockMistral7BInstruct   = "mistral.mistral-7b-instruct-v0:2"
	BedrockMixtral8x7BInstruct = "mistral.mixtral-8x7b-instruct-v0:1"
	BedrockMistralLarge2402    = "mistral.mistral-large-2402-v1:0"
)

// Bedrock model constants - Cohere models.
const (
	BedrockCohereCommandTextV14 = "cohere.command-text-v14"
	BedrockCohereCommandRV1     = "cohere.command-r-v1:0"
	BedrockCohereCommandRPlusV1 = "cohere.command-r-plus-v1:0"
)

// bedrockModelContextWindows maps model names to their context window sizes.
var bedrockModelContextWindows = map[string]int{
	// Amazon models
	BedrockNovaPremierV1:      1000000,
	BedrockNovaProV1:          300000,
	BedrockNovaLiteV1:         300000,
	BedrockNovaMicroV1:        128000,
	BedrockTitanTextExpressV1: 8192,
	BedrockTitanTextLiteV1:    4096,

	// Anthropic Claude models
	BedrockClaudeInstantV1:  100000,
	BedrockClaudeV2:         100000,
	BedrockClaudeV21:        200000,
	BedrockClaude3Sonnet:    200000,
	BedrockClaude3Haiku:     200000,
	BedrockClaude3Opus:      200000,
	BedrockClaude35Sonnet:   200000,
	BedrockClaude35SonnetV2: 200000,
	BedrockClaude35Haiku:    200000,
	BedrockClaude37Sonnet:   200000,
	BedrockClaude4Sonnet:    200000,
	BedrockClaude4Opus:      200000,
	BedrockClaude41Opus:     200000,
	BedrockClaude45Sonnet:   200000,

	// Meta Llama models
	BedrockLlama2_13BChat:       2048,
	BedrockLlama2_70BChat:       4096,
	BedrockLlama3_8BInstruct:    8192,
	BedrockLlama3_70BInstruct:   8192,
	BedrockLlama31_8BInstruct:   128000,
	BedrockLlama31_70BInstruct:  128000,
	BedrockLlama31_405BInstruct: 128000,
	BedrockLlama32_1BInstruct:   131000,
	BedrockLlama32_3BInstruct:   131000,
	BedrockLlama32_11BInstruct:  128000,
	BedrockLlama32_90BInstruct:  128000,
	BedrockLlama33_70BInstruct:  128000,

	// Mistral models
	BedrockMistral7BInstruct:   32000,
	BedrockMixtral8x7BInstruct: 32000,
	BedrockMistralLarge2402:    32000,

	// Cohere models
	BedrockCohereCommandTextV14: 4096,
	BedrockCohereCommandRV1:     128000,
	BedrockCohereCommandRPlusV1: 128000,
}

// bedrockToolCallingModels lists models that support tool/function calling.
var bedrockToolCallingModels = map[string]bool{
	// Amazon Nova models
	BedrockNovaPremierV1: true,
	BedrockNovaProV1:     true,
	BedrockNovaLiteV1:    true,
	BedrockNovaMicroV1:   true,

	// Anthropic Claude 3+ models
	BedrockClaude3Sonnet:    true,
	BedrockClaude3Haiku:     true,
	BedrockClaude3Opus:      true,
	BedrockClaude35Sonnet:   true,
	BedrockClaude35SonnetV2: true,
	BedrockClaude35Haiku:    true,
	BedrockClaude37Sonnet:   true,
	BedrockClaude4Sonnet:    true,
	BedrockClaude4Opus:      true,
	BedrockClaude41Opus:     true,
	BedrockClaude45Sonnet:   true,

	// Meta Llama 3.1+ models
	BedrockLlama31_8BInstruct:   true,
	BedrockLlama31_70BInstruct:  true,
	BedrockLlama31_405BInstruct: true,
	BedrockLlama32_11BInstruct:  true,
	BedrockLlama32_90BInstruct:  true,
	BedrockLlama33_70BInstruct:  true,

	// Cohere Command R models
	BedrockCohereCommandRV1:     true,
	BedrockCohereCommandRPlusV1: true,

	// Mistral Large
	BedrockMistralLarge2402: true,
}

// bedrockMultiModalModels lists models that support multi-modal inputs.
var bedrockMultiModalModels = map[string]bool{
	BedrockClaude3Sonnet:    true,
	BedrockClaude3Haiku:     true,
	BedrockClaude3Opus:      true,
	BedrockClaude35Sonnet:   true,
	BedrockClaude35SonnetV2: true,
	BedrockClaude35Haiku:    true,
	BedrockClaude37Sonnet:   true,
	BedrockClaude4Sonnet:    true,
	BedrockClaude4Opus:      true,
	BedrockClaude41Opus:     true,
	BedrockClaude45Sonnet:   true,
	BedrockNovaProV1:        true,
	BedrockNovaLiteV1:       true,
}

// BedrockLLM implements the LLM interface for AWS Bedrock using the Converse API.
type BedrockLLM struct {
	client      *bedrockruntime.Client
	model       string
	maxTokens   int
	temperature float32
	topP        float32
	region      string
	logger      *slog.Logger
}

// BedrockOption configures a BedrockLLM.
type BedrockOption func(*BedrockLLM)

// WithBedrockModel sets the model.
func WithBedrockModel(model string) BedrockOption {
	return func(b *BedrockLLM) {
		b.model = model
	}
}

// WithBedrockMaxTokens sets the max tokens.
func WithBedrockMaxTokens(maxTokens int) BedrockOption {
	return func(b *BedrockLLM) {
		b.maxTokens = maxTokens
	}
}

// WithBedrockTemperature sets the temperature.
func WithBedrockTemperature(temperature float32) BedrockOption {
	return func(b *BedrockLLM) {
		b.temperature = temperature
	}
}

// WithBedrockTopP sets the top_p value.
func WithBedrockTopP(topP float32) BedrockOption {
	return func(b *BedrockLLM) {
		b.topP = topP
	}
}

// WithBedrockRegion sets the AWS region.
func WithBedrockRegion(region string) BedrockOption {
	return func(b *BedrockLLM) {
		b.region = region
	}
}

// WithBedrockCredentials sets explicit AWS credentials.
func WithBedrockCredentials(accessKeyID, secretAccessKey, sessionToken string) BedrockOption {
	return func(b *BedrockLLM) {
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

// WithBedrockClient sets a custom Bedrock client (for testing).
func WithBedrockClient(client *bedrockruntime.Client) BedrockOption {
	return func(b *BedrockLLM) {
		b.client = client
	}
}

// NewBedrockLLM creates a new AWS Bedrock LLM client.
func NewBedrockLLM(opts ...BedrockOption) *BedrockLLM {
	region := os.Getenv("AWS_REGION")
	if region == "" {
		region = os.Getenv("AWS_DEFAULT_REGION")
	}
	if region == "" {
		region = "us-east-1"
	}

	b := &BedrockLLM{
		model:       DefaultBedrockModel,
		maxTokens:   DefaultBedrockMaxTokens,
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
func (b *BedrockLLM) Complete(ctx context.Context, prompt string) (string, error) {
	b.logger.Info("Complete called", "model", b.model, "prompt_len", len(prompt))

	messages := []ChatMessage{NewUserMessage(prompt)}
	return b.Chat(ctx, messages)
}

// Chat generates a response for a list of chat messages.
func (b *BedrockLLM) Chat(ctx context.Context, messages []ChatMessage) (string, error) {
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
func (b *BedrockLLM) Stream(ctx context.Context, prompt string) (<-chan string, error) {
	b.logger.Info("Stream called", "model", b.model, "prompt_len", len(prompt))

	messages := []ChatMessage{NewUserMessage(prompt)}
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
func (b *BedrockLLM) Metadata() LLMMetadata {
	return getBedrockModelMetadata(b.model)
}

// SupportsToolCalling returns true if the model supports tool calling.
func (b *BedrockLLM) SupportsToolCalling() bool {
	return IsBedrockFunctionCallingModel(b.model)
}

// SupportsStructuredOutput returns true if the model supports structured output.
func (b *BedrockLLM) SupportsStructuredOutput() bool {
	// Bedrock Converse API supports tool use which can be used for structured output
	return b.SupportsToolCalling()
}

// ChatWithTools generates a response that may include tool calls.
func (b *BedrockLLM) ChatWithTools(ctx context.Context, messages []ChatMessage, tools []*ToolMetadata, opts *ChatCompletionOptions) (CompletionResponse, error) {
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
		return CompletionResponse{}, fmt.Errorf("bedrock converse with tools failed: %w", err)
	}

	return b.convertResponse(resp), nil
}

// ChatWithFormat generates a response in the specified format.
func (b *BedrockLLM) ChatWithFormat(ctx context.Context, messages []ChatMessage, format *ResponseFormat) (string, error) {
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
		newMessages := make([]ChatMessage, 0, len(messages)+1)
		newMessages = append(newMessages, NewSystemMessage(jsonInstruction))
		newMessages = append(newMessages, messages...)
		messages = newMessages
	}

	return b.Chat(ctx, messages)
}

// StreamChat generates a streaming response for chat messages.
func (b *BedrockLLM) StreamChat(ctx context.Context, messages []ChatMessage) (<-chan StreamToken, error) {
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

	tokenChan := make(chan StreamToken)

	go func() {
		defer close(tokenChan)

		var currentToolCall *ToolCall
		stream := resp.GetStream()

		for event := range stream.Events() {
			switch v := event.(type) {
			case *types.ConverseStreamOutputMemberContentBlockDelta:
				token := StreamToken{}

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
					currentToolCall = &ToolCall{
						ID:   aws.ToString(toolStart.Value.ToolUseId),
						Name: aws.ToString(toolStart.Value.Name),
					}
				}

			case *types.ConverseStreamOutputMemberContentBlockStop:
				if currentToolCall != nil {
					token := StreamToken{
						ToolCalls: []*ToolCall{currentToolCall},
					}
					select {
					case tokenChan <- token:
					case <-ctx.Done():
						return
					}
					currentToolCall = nil
				}

			case *types.ConverseStreamOutputMemberMessageStop:
				token := StreamToken{
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
func (b *BedrockLLM) convertMessages(messages []ChatMessage) ([]types.Message, []types.SystemContentBlock) {
	var converseMessages []types.Message
	var systemPrompts []types.SystemContentBlock

	for _, msg := range messages {
		content := msg.GetTextContent()

		switch msg.Role {
		case MessageRoleSystem:
			systemPrompts = append(systemPrompts, &types.SystemContentBlockMemberText{
				Value: content,
			})

		case MessageRoleUser:
			converseMessages = append(converseMessages, types.Message{
				Role: types.ConversationRoleUser,
				Content: []types.ContentBlock{
					&types.ContentBlockMemberText{Value: content},
				},
			})

		case MessageRoleAssistant:
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

		case MessageRoleTool:
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
func (b *BedrockLLM) convertTools(tools []*ToolMetadata) []types.Tool {
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
func (b *BedrockLLM) convertResponse(resp *bedrockruntime.ConverseOutput) CompletionResponse {
	response := CompletionResponse{
		Text: b.extractTextFromResponse(resp),
	}

	// Extract tool calls
	var toolCalls []*ToolCall
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
					toolCalls = append(toolCalls, &ToolCall{
						ID:        aws.ToString(toolUse.Value.ToolUseId),
						Name:      aws.ToString(toolUse.Value.Name),
						Arguments: argsStr,
					})
				}
			}
		}
	}

	if len(toolCalls) > 0 {
		msg := ChatMessage{
			Role:    MessageRoleAssistant,
			Content: response.Text,
		}
		for _, tc := range toolCalls {
			msg.Blocks = append(msg.Blocks, NewToolCallBlock(tc))
		}
		response.Message = &msg
	}

	return response
}

// extractTextFromResponse extracts text content from Bedrock response.
func (b *BedrockLLM) extractTextFromResponse(resp *bedrockruntime.ConverseOutput) string {
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

// getBedrockModelMetadata returns metadata for Bedrock models.
func getBedrockModelMetadata(model string) LLMMetadata {
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
	if cw, ok := bedrockModelContextWindows[baseModel]; ok {
		contextWindow = cw
	}

	return LLMMetadata{
		ModelName:         model,
		ContextWindow:     contextWindow,
		NumOutputTokens:   4096,
		IsChat:            true,
		IsFunctionCalling: IsBedrockFunctionCallingModel(model),
		IsMultiModal:      bedrockMultiModalModels[baseModel],
		SystemRole:        "system",
	}
}

// IsBedrockFunctionCallingModel returns true if the model supports function calling.
func IsBedrockFunctionCallingModel(model string) bool {
	// Handle region-prefixed models
	baseModel := model
	prefixes := []string{"us.", "eu.", "apac.", "jp.", "global."}
	for _, prefix := range prefixes {
		if strings.HasPrefix(model, prefix) {
			baseModel = model[len(prefix):]
			break
		}
	}
	return bedrockToolCallingModels[baseModel]
}

// BedrockModelContextSize returns the context window size for a model.
func BedrockModelContextSize(model string) int {
	// Handle region-prefixed models
	baseModel := model
	prefixes := []string{"us.", "eu.", "apac.", "jp.", "global."}
	for _, prefix := range prefixes {
		if strings.HasPrefix(model, prefix) {
			baseModel = model[len(prefix):]
			break
		}
	}

	if cw, ok := bedrockModelContextWindows[baseModel]; ok {
		return cw
	}
	return 128000 // default
}

// Ensure BedrockLLM implements the interfaces.
var _ LLM = (*BedrockLLM)(nil)
var _ LLMWithMetadata = (*BedrockLLM)(nil)
var _ LLMWithToolCalling = (*BedrockLLM)(nil)
var _ LLMWithStructuredOutput = (*BedrockLLM)(nil)
var _ FullLLM = (*BedrockLLM)(nil)
