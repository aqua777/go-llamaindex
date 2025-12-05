package llm

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMessageRoles(t *testing.T) {
	assert.Equal(t, MessageRole("system"), MessageRoleSystem)
	assert.Equal(t, MessageRole("user"), MessageRoleUser)
	assert.Equal(t, MessageRole("assistant"), MessageRoleAssistant)
	assert.Equal(t, MessageRole("tool"), MessageRoleTool)
}

func TestNewChatMessage(t *testing.T) {
	msg := NewChatMessage(MessageRoleUser, "Hello")
	assert.Equal(t, MessageRoleUser, msg.Role)
	assert.Equal(t, "Hello", msg.Content)
}

func TestNewSystemMessage(t *testing.T) {
	msg := NewSystemMessage("You are a helpful assistant")
	assert.Equal(t, MessageRoleSystem, msg.Role)
	assert.Equal(t, "You are a helpful assistant", msg.Content)
}

func TestNewUserMessage(t *testing.T) {
	msg := NewUserMessage("What is 2+2?")
	assert.Equal(t, MessageRoleUser, msg.Role)
	assert.Equal(t, "What is 2+2?", msg.Content)
}

func TestNewAssistantMessage(t *testing.T) {
	msg := NewAssistantMessage("The answer is 4")
	assert.Equal(t, MessageRoleAssistant, msg.Role)
	assert.Equal(t, "The answer is 4", msg.Content)
}

func TestNewToolMessage(t *testing.T) {
	msg := NewToolMessage("call_123", "Result: success")
	assert.Equal(t, MessageRoleTool, msg.Role)
	assert.Equal(t, "Result: success", msg.Content)
	assert.Equal(t, "call_123", msg.ToolCallID)
}

func TestChatMessageGetTextContent(t *testing.T) {
	// Simple content
	msg1 := NewUserMessage("Hello")
	assert.Equal(t, "Hello", msg1.GetTextContent())

	// Multi-modal with text blocks
	msg2 := NewMultiModalMessage(MessageRoleUser,
		NewTextBlock("First part"),
		NewTextBlock(" Second part"),
	)
	assert.Equal(t, "First part Second part", msg2.GetTextContent())

	// Empty message
	msg3 := ChatMessage{Role: MessageRoleUser}
	assert.Equal(t, "", msg3.GetTextContent())
}

func TestContentBlocks(t *testing.T) {
	// Text block
	textBlock := NewTextBlock("Hello world")
	assert.Equal(t, ContentBlockTypeText, textBlock.Type)
	assert.Equal(t, "Hello world", textBlock.Text)

	// Image URL block
	imgURLBlock := NewImageURLBlock("https://example.com/image.png", "image/png")
	assert.Equal(t, ContentBlockTypeImage, imgURLBlock.Type)
	assert.Equal(t, "https://example.com/image.png", imgURLBlock.ImageURL)
	assert.Equal(t, "image/png", imgURLBlock.ImageMimeType)

	// Image base64 block
	imgB64Block := NewImageBase64Block("base64data", "image/jpeg")
	assert.Equal(t, ContentBlockTypeImage, imgB64Block.Type)
	assert.Equal(t, "base64data", imgB64Block.ImageBase64)
	assert.Equal(t, "image/jpeg", imgB64Block.ImageMimeType)
}

func TestToolCallBlock(t *testing.T) {
	tc := NewToolCall("call_123", "get_weather", `{"location": "NYC"}`)
	block := NewToolCallBlock(tc)

	assert.Equal(t, ContentBlockTypeToolCall, block.Type)
	assert.NotNil(t, block.ToolCall)
	assert.Equal(t, "call_123", block.ToolCall.ID)
	assert.Equal(t, "get_weather", block.ToolCall.Name)
}

func TestToolResultBlock(t *testing.T) {
	tr := NewToolResult("call_123", "get_weather", "Sunny, 72°F")
	block := NewToolResultBlock(tr)

	assert.Equal(t, ContentBlockTypeToolResult, block.Type)
	assert.NotNil(t, block.ToolResult)
	assert.Equal(t, "call_123", block.ToolResult.ToolCallID)
	assert.Equal(t, "Sunny, 72°F", block.ToolResult.Content)
}

func TestChatMessageHasToolCalls(t *testing.T) {
	// Message without tool calls
	msg1 := NewAssistantMessage("Hello")
	assert.False(t, msg1.HasToolCalls())

	// Message with tool calls
	tc := NewToolCall("call_123", "get_weather", `{}`)
	msg2 := NewMultiModalMessage(MessageRoleAssistant, NewToolCallBlock(tc))
	assert.True(t, msg2.HasToolCalls())
}

func TestChatMessageGetToolCalls(t *testing.T) {
	tc1 := NewToolCall("call_1", "func1", `{}`)
	tc2 := NewToolCall("call_2", "func2", `{}`)

	msg := NewMultiModalMessage(MessageRoleAssistant,
		NewTextBlock("I'll call some functions"),
		NewToolCallBlock(tc1),
		NewToolCallBlock(tc2),
	)

	calls := msg.GetToolCalls()
	assert.Len(t, calls, 2)
	assert.Equal(t, "call_1", calls[0].ID)
	assert.Equal(t, "call_2", calls[1].ID)
}

func TestLLMMetadata(t *testing.T) {
	// Default metadata
	meta := DefaultLLMMetadata("test-model")
	assert.Equal(t, "test-model", meta.ModelName)
	assert.Equal(t, 4096, meta.ContextWindow)
	assert.True(t, meta.IsChat)
	assert.False(t, meta.IsFunctionCalling)

	// GPT-4o metadata
	gpt4o := GPT4oMetadata()
	assert.Equal(t, "gpt-4o", gpt4o.ModelName)
	assert.Equal(t, 128000, gpt4o.ContextWindow)
	assert.True(t, gpt4o.IsFunctionCalling)
	assert.True(t, gpt4o.IsMultiModal)
}

func TestToolCall(t *testing.T) {
	tc := NewToolCall("call_123", "get_weather", `{"location": "NYC", "unit": "celsius"}`)

	assert.Equal(t, "call_123", tc.ID)
	assert.Equal(t, "get_weather", tc.Name)

	// Parse arguments
	args, err := tc.ParseArguments()
	require.NoError(t, err)
	assert.Equal(t, "NYC", args["location"])
	assert.Equal(t, "celsius", args["unit"])

	// Parse into struct
	var params struct {
		Location string `json:"location"`
		Unit     string `json:"unit"`
	}
	err = tc.ParseArgumentsInto(&params)
	require.NoError(t, err)
	assert.Equal(t, "NYC", params.Location)
	assert.Equal(t, "celsius", params.Unit)
}

func TestToolResult(t *testing.T) {
	// Success result
	tr := NewToolResult("call_123", "get_weather", "Sunny, 72°F")
	assert.Equal(t, "call_123", tr.ToolCallID)
	assert.Equal(t, "get_weather", tr.ToolName)
	assert.Equal(t, "Sunny, 72°F", tr.Content)
	assert.False(t, tr.IsError)

	// Error result
	trErr := NewToolResultError("call_456", "get_weather", "Location not found")
	assert.True(t, trErr.IsError)
	assert.Equal(t, "Location not found", trErr.Content)
}

func TestToolMetadata(t *testing.T) {
	tm := NewToolMetadata("get_weather", "Get the current weather for a location")

	assert.Equal(t, "get_weather", tm.Name)
	assert.Equal(t, "Get the current weather for a location", tm.Description)
	assert.NotNil(t, tm.Parameters)

	// With parameters
	params := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"location": map[string]interface{}{
				"type":        "string",
				"description": "The city name",
			},
		},
		"required": []string{"location"},
	}
	tm.WithParameters(params)
	assert.Equal(t, params, tm.Parameters)

	// ToOpenAITool
	openaiTool := tm.ToOpenAITool()
	assert.Equal(t, "function", openaiTool["type"])
	fn := openaiTool["function"].(map[string]interface{})
	assert.Equal(t, "get_weather", fn["name"])
}

func TestCompletionResponse(t *testing.T) {
	// Simple text response
	resp := NewCompletionResponse("Hello world")
	assert.Equal(t, "Hello world", resp.Text)
	assert.Nil(t, resp.Message)

	// Chat completion response
	msg := NewAssistantMessage("Hello from assistant")
	chatResp := NewChatCompletionResponse(msg)
	assert.Equal(t, "Hello from assistant", chatResp.Text)
	assert.NotNil(t, chatResp.Message)
}

func TestResponseFormat(t *testing.T) {
	// JSON format
	jsonFormat := NewJSONResponseFormat()
	assert.Equal(t, "json_object", jsonFormat.Type)

	// JSON schema format
	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"name": map[string]interface{}{"type": "string"},
		},
	}
	schemaFormat := NewJSONSchemaResponseFormat(schema)
	assert.Equal(t, "json_schema", schemaFormat.Type)
	assert.Equal(t, schema, schemaFormat.JSONSchema)
}

func TestToolChoice(t *testing.T) {
	assert.Equal(t, ToolChoice("auto"), ToolChoiceAuto)
	assert.Equal(t, ToolChoice("none"), ToolChoiceNone)
	assert.Equal(t, ToolChoice("required"), ToolChoiceRequired)

	// Specific tool choice
	specific := SpecificToolChoice("get_weather")
	assert.Equal(t, "function", specific["type"])
	fn := specific["function"].(map[string]interface{})
	assert.Equal(t, "get_weather", fn["name"])
}
