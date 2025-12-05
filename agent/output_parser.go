package agent

import (
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
)

// ReasoningStepType represents the type of reasoning step.
type ReasoningStepType string

const (
	// ReasoningStepTypeAction indicates an action step (tool call).
	ReasoningStepTypeAction ReasoningStepType = "action"
	// ReasoningStepTypeObservation indicates an observation step (tool result).
	ReasoningStepTypeObservation ReasoningStepType = "observation"
	// ReasoningStepTypeResponse indicates a final response step.
	ReasoningStepTypeResponse ReasoningStepType = "response"
)

// BaseReasoningStep is the interface for all reasoning steps.
type BaseReasoningStep interface {
	// GetContent returns the formatted content of the step.
	GetContent() string
	// IsDone returns true if this is the final step.
	IsDone() bool
	// StepType returns the type of reasoning step.
	StepType() ReasoningStepType
}

// ActionReasoningStep represents a thought-action step.
type ActionReasoningStep struct {
	// Thought is the agent's reasoning.
	Thought string `json:"thought"`
	// Action is the tool name to call.
	Action string `json:"action"`
	// ActionInput is the input to the tool.
	ActionInput map[string]interface{} `json:"action_input"`
}

// GetContent returns the formatted content.
func (s *ActionReasoningStep) GetContent() string {
	inputJSON, _ := json.Marshal(s.ActionInput)
	return fmt.Sprintf("Thought: %s\nAction: %s\nAction Input: %s", s.Thought, s.Action, string(inputJSON))
}

// IsDone returns false for action steps.
func (s *ActionReasoningStep) IsDone() bool {
	return false
}

// StepType returns the step type.
func (s *ActionReasoningStep) StepType() ReasoningStepType {
	return ReasoningStepTypeAction
}

// ObservationReasoningStep represents an observation from a tool.
type ObservationReasoningStep struct {
	// Observation is the tool output.
	Observation string `json:"observation"`
	// ReturnDirect indicates if this should be returned directly.
	ReturnDirect bool `json:"return_direct"`
}

// GetContent returns the formatted content.
func (s *ObservationReasoningStep) GetContent() string {
	return fmt.Sprintf("Observation: %s", s.Observation)
}

// IsDone returns true if return_direct is set.
func (s *ObservationReasoningStep) IsDone() bool {
	return s.ReturnDirect
}

// StepType returns the step type.
func (s *ObservationReasoningStep) StepType() ReasoningStepType {
	return ReasoningStepTypeObservation
}

// ResponseReasoningStep represents a final response.
type ResponseReasoningStep struct {
	// Thought is the agent's final reasoning.
	Thought string `json:"thought"`
	// Response is the final answer.
	Response string `json:"response"`
	// IsStreaming indicates if this is a streaming response.
	IsStreaming bool `json:"is_streaming"`
}

// GetContent returns the formatted content.
func (s *ResponseReasoningStep) GetContent() string {
	if s.IsStreaming {
		return fmt.Sprintf("Thought: %s\nAnswer (Starts With): %s ...", s.Thought, s.Response)
	}
	return fmt.Sprintf("Thought: %s\nAnswer: %s", s.Thought, s.Response)
}

// IsDone returns true for response steps.
func (s *ResponseReasoningStep) IsDone() bool {
	return true
}

// StepType returns the step type.
func (s *ResponseReasoningStep) StepType() ReasoningStepType {
	return ReasoningStepTypeResponse
}

// OutputParser is the interface for parsing agent output.
type OutputParser interface {
	// Parse parses the LLM output into a reasoning step.
	Parse(output string, isStreaming bool) (BaseReasoningStep, error)
}

// ReActOutputParser parses ReAct-style output.
type ReActOutputParser struct{}

// NewReActOutputParser creates a new ReActOutputParser.
func NewReActOutputParser() *ReActOutputParser {
	return &ReActOutputParser{}
}

// Parse parses the LLM output into a reasoning step.
//
// Expected formats:
// 1. Tool call:
//
//	Thought: <thought>
//	Action: <action>
//	Action Input: <action_input>
//
// 2. Final answer:
//
//	Thought: <thought>
//	Answer: <answer>
func (p *ReActOutputParser) Parse(output string, isStreaming bool) (BaseReasoningStep, error) {
	// Find positions of keywords
	thoughtMatch := regexp.MustCompile(`(?m)Thought:`).FindStringIndex(output)
	actionMatch := regexp.MustCompile(`(?m)Action:`).FindStringIndex(output)
	answerMatch := regexp.MustCompile(`(?m)Answer:`).FindStringIndex(output)

	var thoughtIdx, actionIdx, answerIdx int = -1, -1, -1
	if thoughtMatch != nil {
		thoughtIdx = thoughtMatch[0]
	}
	if actionMatch != nil {
		actionIdx = actionMatch[0]
	}
	if answerMatch != nil {
		answerIdx = answerMatch[0]
	}

	// If no keywords found, treat the entire output as an implicit answer
	if thoughtIdx == -1 && actionIdx == -1 && answerIdx == -1 {
		return &ResponseReasoningStep{
			Thought:     "(Implicit) I can answer without any more tools!",
			Response:    strings.TrimSpace(output),
			IsStreaming: isStreaming,
		}, nil
	}

	// Action takes priority over Answer if Action comes first
	if actionIdx != -1 && (answerIdx == -1 || actionIdx < answerIdx) {
		return p.parseActionStep(output)
	}

	// Parse as final answer
	if answerIdx != -1 {
		thought, answer, err := extractFinalResponse(output)
		if err != nil {
			return nil, err
		}
		return &ResponseReasoningStep{
			Thought:     thought,
			Response:    answer,
			IsStreaming: isStreaming,
		}, nil
	}

	return nil, fmt.Errorf("could not parse output: %s", output)
}

// parseActionStep parses an action reasoning step.
func (p *ReActOutputParser) parseActionStep(output string) (*ActionReasoningStep, error) {
	thought, action, actionInput, err := extractToolUse(output)
	if err != nil {
		return nil, err
	}

	// Parse the action input JSON
	jsonStr := extractJSONStr(actionInput)
	actionInputDict, err := parseActionInput(jsonStr)
	if err != nil {
		return nil, fmt.Errorf("failed to parse action input: %w", err)
	}

	return &ActionReasoningStep{
		Thought:     thought,
		Action:      action,
		ActionInput: actionInputDict,
	}, nil
}

// extractToolUse extracts thought, action, and action input from text.
func extractToolUse(input string) (thought, action, actionInput string, err error) {
	// Pattern to match:
	// Thought: <thought>
	// Action: <action>
	// Action Input: <action_input>
	pattern := regexp.MustCompile(`(?s)(?:\s*Thought:\s*(.*?)|(.+?))\n+Action:\s*([^\n\(\)\s]+).*?\n+Action Input:\s*(.*)`)

	match := pattern.FindStringSubmatch(input)
	if match == nil {
		return "", "", "", fmt.Errorf("could not extract tool use from input text: %s", input)
	}

	// Thought is in group 1 or 2
	if match[1] != "" {
		thought = strings.TrimSpace(match[1])
	} else {
		thought = strings.TrimSpace(match[2])
	}
	action = strings.TrimSpace(match[3])
	actionInput = strings.TrimSpace(match[4])

	return thought, action, actionInput, nil
}

// extractFinalResponse extracts thought and answer from text.
func extractFinalResponse(input string) (thought, answer string, err error) {
	pattern := regexp.MustCompile(`(?s)\s*Thought:(.*?)Answer:(.*)`)

	match := pattern.FindStringSubmatch(input)
	if match == nil {
		return "", "", fmt.Errorf("could not extract final answer from input text: %s", input)
	}

	thought = strings.TrimSpace(match[1])
	answer = strings.TrimSpace(match[2])
	return thought, answer, nil
}

// extractJSONStr extracts a JSON object from text.
func extractJSONStr(input string) string {
	input = strings.TrimSpace(input)

	// Try to find JSON in code blocks
	codeBlockPattern := regexp.MustCompile("(?s)```(?:json)?\\s*(\\{.*?\\})\\s*```")
	if match := codeBlockPattern.FindStringSubmatch(input); match != nil {
		return match[1]
	}

	// Try to find a JSON object directly
	jsonPattern := regexp.MustCompile(`(?s)(\{.*\})`)
	if match := jsonPattern.FindStringSubmatch(input); match != nil {
		return match[1]
	}

	// Return the input as-is if no JSON found
	return input
}

// parseActionInput parses the action input JSON.
func parseActionInput(jsonStr string) (map[string]interface{}, error) {
	// Handle empty input
	if jsonStr == "" || jsonStr == "{}" {
		return map[string]interface{}{}, nil
	}

	// Try standard JSON parsing first
	var result map[string]interface{}
	if err := json.Unmarshal([]byte(jsonStr), &result); err == nil {
		return result, nil
	}

	// Try to fix common JSON issues (single quotes instead of double quotes)
	fixedJSON := fixJSONQuotes(jsonStr)
	if err := json.Unmarshal([]byte(fixedJSON), &result); err == nil {
		return result, nil
	}

	// Try regex-based parsing as fallback
	return parseActionInputFallback(jsonStr)
}

// fixJSONQuotes replaces single quotes with double quotes in JSON.
func fixJSONQuotes(input string) string {
	// Simple replacement of single quotes with double quotes
	// This is a basic approach that works for most cases
	result := strings.ReplaceAll(input, "'", "\"")
	return result
}

// parseActionInputFallback uses regex to parse malformed JSON.
func parseActionInputFallback(input string) (map[string]interface{}, error) {
	result := make(map[string]interface{})

	// Pattern to match "key": "value" pairs
	pattern := regexp.MustCompile(`"(\w+)":\s*"([^"]*)"`)
	matches := pattern.FindAllStringSubmatch(input, -1)

	for _, match := range matches {
		if len(match) == 3 {
			result[match[1]] = match[2]
		}
	}

	if len(result) == 0 {
		return nil, errors.New("could not parse action input")
	}

	return result, nil
}
