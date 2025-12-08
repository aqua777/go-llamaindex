package agent

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/aqua777/go-llamaindex/tools"
)

// CallTool executes a tool with the given input.
func CallTool(ctx context.Context, tool tools.Tool, input map[string]interface{}) (*tools.ToolOutput, error) {
	if tool == nil {
		return nil, fmt.Errorf("tool is nil")
	}

	output, err := tool.Call(ctx, input)
	if err != nil {
		return tools.NewErrorToolOutput(tool.Metadata().Name, err), err
	}

	return output, nil
}

// CallToolByName finds and executes a tool by name.
func CallToolByName(ctx context.Context, agentTools []tools.Tool, toolName string, input map[string]interface{}) (*tools.ToolOutput, error) {
	for _, tool := range agentTools {
		if tool.Metadata().Name == toolName {
			return CallTool(ctx, tool, input)
		}
	}
	return nil, fmt.Errorf("tool not found: %s", toolName)
}

// GetToolsByName returns a map of tools by name.
func GetToolsByName(agentTools []tools.Tool) map[string]tools.Tool {
	result := make(map[string]tools.Tool, len(agentTools))
	for _, tool := range agentTools {
		result[tool.Metadata().Name] = tool
	}
	return result
}

// ValidateToolSelection validates that a tool selection is valid.
func ValidateToolSelection(selection *ToolSelection, agentTools []tools.Tool) error {
	if selection == nil {
		return fmt.Errorf("tool selection is nil")
	}

	if selection.ToolName == "" {
		return fmt.Errorf("tool name is empty")
	}

	// Check if tool exists
	toolsByName := GetToolsByName(agentTools)
	if _, ok := toolsByName[selection.ToolName]; !ok {
		return fmt.Errorf("tool not found: %s", selection.ToolName)
	}

	return nil
}

// FormatToolOutput formats a tool output for display.
func FormatToolOutput(output *tools.ToolOutput) string {
	if output == nil {
		return ""
	}

	if output.IsError {
		return fmt.Sprintf("Error: %s", output.Content)
	}

	return output.Content
}

// ParseToolArguments parses tool arguments from a JSON string.
func ParseToolArguments(argsJSON string) (map[string]interface{}, error) {
	if argsJSON == "" || argsJSON == "{}" {
		return map[string]interface{}{}, nil
	}

	var args map[string]interface{}
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return nil, fmt.Errorf("failed to parse tool arguments: %w", err)
	}

	return args, nil
}

// ToolArgsToJSON converts tool arguments to a JSON string.
func ToolArgsToJSON(args map[string]interface{}) (string, error) {
	if len(args) == 0 {
		return "{}", nil
	}

	data, err := json.Marshal(args)
	if err != nil {
		return "", fmt.Errorf("failed to marshal tool arguments: %w", err)
	}

	return string(data), nil
}

// ExtractResponseFromReasoning extracts the final response from reasoning steps.
func ExtractResponseFromReasoning(steps []BaseReasoningStep) string {
	for i := len(steps) - 1; i >= 0; i-- {
		if step, ok := steps[i].(*ResponseReasoningStep); ok {
			return step.Response
		}
	}
	return ""
}

// FormatReasoningSteps formats reasoning steps for display.
func FormatReasoningSteps(steps []BaseReasoningStep) string {
	var result string
	for _, step := range steps {
		result += step.GetContent() + "\n"
	}
	return result
}

// GenerateToolID generates a unique tool ID.
func GenerateToolID() string {
	// Simple implementation using a counter or UUID would be better in production
	// For now, we use a simple approach
	return fmt.Sprintf("call_%d", generateID())
}

// Simple ID generator (not thread-safe, for demonstration)
var idCounter int

func generateID() int {
	idCounter++
	return idCounter
}

// ResetIDCounter resets the ID counter (useful for testing).
func ResetIDCounter() {
	idCounter = 0
}

// BuildToolsText builds a text description of available tools.
func BuildToolsText(agentTools []tools.Tool) string {
	var result string
	for i, tool := range agentTools {
		meta := tool.Metadata()
		result += fmt.Sprintf("%d. %s: %s\n", i+1, meta.Name, meta.Description)
	}
	return result
}

// BuildToolChoicesText builds a numbered list of tool choices.
func BuildToolChoicesText(agentTools []tools.Tool) string {
	var result string
	for i, tool := range agentTools {
		meta := tool.Metadata()
		result += fmt.Sprintf("(%d) %s: %s\n", i+1, meta.Name, meta.Description)
	}
	return result
}

// IsValidToolName checks if a tool name is valid.
func IsValidToolName(name string, agentTools []tools.Tool) bool {
	for _, tool := range agentTools {
		if tool.Metadata().Name == name {
			return true
		}
	}
	return false
}

// CleanResponse removes common prefixes from agent responses.
func CleanResponse(response string) string {
	// Remove "Answer:" prefix if present
	if idx := findSubstring(response, "Answer:"); idx != -1 {
		return trimSpace(response[idx+7:])
	}
	return trimSpace(response)
}

// findSubstring finds the index of a substring (case-insensitive).
func findSubstring(s, substr string) int {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}

// trimSpace trims whitespace from a string.
func trimSpace(s string) string {
	start := 0
	end := len(s)

	for start < end && (s[start] == ' ' || s[start] == '\t' || s[start] == '\n' || s[start] == '\r') {
		start++
	}

	for end > start && (s[end-1] == ' ' || s[end-1] == '\t' || s[end-1] == '\n' || s[end-1] == '\r') {
		end--
	}

	return s[start:end]
}
