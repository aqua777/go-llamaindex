package agent

import (
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/tools"
)

// ReActSystemHeaderTemplate is the default system header template for ReAct agents.
const ReActSystemHeaderTemplate = `You are designed to help with a variety of tasks, from answering questions to providing summaries to other types of analyses.

## Tools

You have access to a wide variety of tools. You are responsible for using the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools to complete each subtask.

You have access to the following tools:
{tool_desc}
{context_prompt}

## Output Format

Please answer in the same language as the question and use the following format:

` + "```" + `
Thought: The current language of the user is: (user's language). I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {"input": "hello world", "num_beams": 5})
` + "```" + `

Please ALWAYS start with a Thought.

NEVER surround your response with markdown code markers. You may use code markers within your response if you need to.

Please use a valid JSON format for the Action Input. Do NOT do this {'input': 'hello world', 'num_beams': 5}. If you include the "Action:" line, then you MUST include the "Action Input:" line too, even if the tool does not need kwargs, in that case you MUST use "Action Input: {}".

If this format is used, the tool will respond in the following format:

` + "```" + `
Observation: tool response
` + "```" + `

You should keep repeating the above format till you have enough information to answer the question without using any more tools. At that point, you MUST respond in one of the following two formats:

` + "```" + `
Thought: I can answer without using any more tools. I'll use the user's language to answer
Answer: [your answer here (In the same language as the user's question)]
` + "```" + `

` + "```" + `
Thought: I cannot answer the question with the provided tools.
Answer: [your answer here (In the same language as the user's question)]
` + "```" + `

## Current Conversation

Below is the current conversation consisting of interleaving human and assistant messages.
`

// ContextReActSystemHeaderTemplate includes a context section.
const ContextReActSystemHeaderTemplate = `You are designed to help with a variety of tasks, from answering questions to providing summaries to other types of analyses.

## Tools

You have access to a wide variety of tools. You are responsible for using the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools to complete each subtask.

You have access to the following tools:
{tool_desc}

Here is some context to help you answer the question and plan:
{context}

## Output Format

Please answer in the same language as the question and use the following format:

` + "```" + `
Thought: The current language of the user is: (user's language). I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {"input": "hello world", "num_beams": 5})
` + "```" + `

Please ALWAYS start with a Thought.

NEVER surround your response with markdown code markers. You may use code markers within your response if you need to.

Please use a valid JSON format for the Action Input. Do NOT do this {'input': 'hello world', 'num_beams': 5}. If you include the "Action:" line, then you MUST include the "Action Input:" line too, even if the tool does not need kwargs, in that case you MUST use "Action Input: {}".

If this format is used, the tool will respond in the following format:

` + "```" + `
Observation: tool response
` + "```" + `

You should keep repeating the above format till you have enough information to answer the question without using any more tools. At that point, you MUST respond in one of the following two formats:

` + "```" + `
Thought: I can answer without using any more tools. I'll use the user's language to answer
Answer: [your answer here (In the same language as the user's question)]
` + "```" + `

` + "```" + `
Thought: I cannot answer the question with the provided tools.
Answer: [your answer here (In the same language as the user's question)]
` + "```" + `

## Current Conversation

Below is the current conversation consisting of interleaving human and assistant messages.
`

// AgentChatFormatter is the interface for formatting agent chat messages.
type AgentChatFormatter interface {
	// Format formats the chat history and reasoning steps into messages.
	Format(tools []tools.Tool, chatHistory []llm.ChatMessage, currentReasoning []BaseReasoningStep) []llm.ChatMessage
}

// ReActChatFormatter formats chat history for ReAct agents.
type ReActChatFormatter struct {
	// SystemHeader is the system header template.
	SystemHeader string
	// Context is additional context to include.
	Context string
	// ObservationRole is the role for observation messages.
	ObservationRole llm.MessageRole
}

// ReActChatFormatterOption configures a ReActChatFormatter.
type ReActChatFormatterOption func(*ReActChatFormatter)

// WithReActSystemHeader sets the system header template.
func WithReActSystemHeader(header string) ReActChatFormatterOption {
	return func(f *ReActChatFormatter) {
		f.SystemHeader = header
	}
}

// WithReActContext sets the context.
func WithReActContext(context string) ReActChatFormatterOption {
	return func(f *ReActChatFormatter) {
		f.Context = context
	}
}

// WithReActObservationRole sets the observation role.
func WithReActObservationRole(role llm.MessageRole) ReActChatFormatterOption {
	return func(f *ReActChatFormatter) {
		f.ObservationRole = role
	}
}

// NewReActChatFormatter creates a new ReActChatFormatter.
func NewReActChatFormatter(opts ...ReActChatFormatterOption) *ReActChatFormatter {
	f := &ReActChatFormatter{
		SystemHeader:    ReActSystemHeaderTemplate,
		ObservationRole: llm.MessageRoleUser,
	}

	for _, opt := range opts {
		opt(f)
	}

	return f
}

// NewReActChatFormatterFromDefaults creates a ReActChatFormatter with defaults.
func NewReActChatFormatterFromDefaults(systemHeader, context string, observationRole llm.MessageRole) *ReActChatFormatter {
	header := systemHeader
	if header == "" {
		if context != "" {
			header = ContextReActSystemHeaderTemplate
		} else {
			header = ReActSystemHeaderTemplate
		}
	}

	return &ReActChatFormatter{
		SystemHeader:    header,
		Context:         context,
		ObservationRole: observationRole,
	}
}

// Format formats the chat history and reasoning steps into messages.
func (f *ReActChatFormatter) Format(agentTools []tools.Tool, chatHistory []llm.ChatMessage, currentReasoning []BaseReasoningStep) []llm.ChatMessage {
	// Build tool descriptions
	toolDesc := getReActToolDescriptions(agentTools)
	toolNames := getToolNames(agentTools)

	// Format the system header
	formatArgs := map[string]string{
		"tool_desc":  strings.Join(toolDesc, "\n"),
		"tool_names": strings.Join(toolNames, ", "),
	}

	if f.Context != "" {
		formatArgs["context"] = f.Context
		formatArgs["context_prompt"] = fmt.Sprintf("\nHere is some context to help you answer the question and plan:\n%s\n", f.Context)
	} else {
		formatArgs["context_prompt"] = ""
	}

	sysHeader := formatTemplate(f.SystemHeader, formatArgs)

	// Build reasoning history as alternating messages
	reasoningHistory := make([]llm.ChatMessage, 0, len(currentReasoning))
	for _, step := range currentReasoning {
		var msg llm.ChatMessage
		if step.StepType() == ReasoningStepTypeObservation {
			msg = llm.ChatMessage{
				Role:    f.ObservationRole,
				Content: step.GetContent(),
			}
		} else {
			msg = llm.ChatMessage{
				Role:    llm.MessageRoleAssistant,
				Content: step.GetContent(),
			}
		}
		reasoningHistory = append(reasoningHistory, msg)
	}

	// Combine: system message + chat history + reasoning history
	messages := make([]llm.ChatMessage, 0, 1+len(chatHistory)+len(reasoningHistory))
	messages = append(messages, llm.ChatMessage{
		Role:    llm.MessageRoleSystem,
		Content: sysHeader,
	})
	messages = append(messages, chatHistory...)
	messages = append(messages, reasoningHistory...)

	return messages
}

// getReActToolDescriptions returns formatted tool descriptions.
func getReActToolDescriptions(agentTools []tools.Tool) []string {
	descriptions := make([]string, len(agentTools))
	for i, tool := range agentTools {
		meta := tool.Metadata()
		paramsJSON, _ := meta.GetParametersJSON()
		descriptions[i] = fmt.Sprintf("> Tool Name: %s\nTool Description: %s\nTool Args: %s\n",
			meta.Name, meta.Description, paramsJSON)
	}
	return descriptions
}

// getToolNames returns the names of all tools.
func getToolNames(agentTools []tools.Tool) []string {
	names := make([]string, len(agentTools))
	for i, tool := range agentTools {
		names[i] = tool.Metadata().Name
	}
	return names
}

// formatTemplate replaces {key} placeholders with values.
func formatTemplate(template string, args map[string]string) string {
	result := template
	for key, value := range args {
		result = strings.ReplaceAll(result, "{"+key+"}", value)
	}
	return result
}
