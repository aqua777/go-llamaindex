// Package agent provides agent abstractions for autonomous reasoning with tools.
package agent

import (
	"context"
	"encoding/json"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/memory"
	"github.com/aqua777/go-llamaindex/tools"
)

// DefaultMaxIterations is the default maximum number of reasoning iterations.
const DefaultMaxIterations = 10

// AgentState represents the current state of an agent's execution.
type AgentState string

const (
	// AgentStateIdle indicates the agent is not currently executing.
	AgentStateIdle AgentState = "idle"
	// AgentStateRunning indicates the agent is currently executing.
	AgentStateRunning AgentState = "running"
	// AgentStateWaitingForTool indicates the agent is waiting for tool execution.
	AgentStateWaitingForTool AgentState = "waiting_for_tool"
	// AgentStateCompleted indicates the agent has completed execution.
	AgentStateCompleted AgentState = "completed"
	// AgentStateError indicates the agent encountered an error.
	AgentStateError AgentState = "error"
)

// ToolSelection represents a tool selected by the agent to execute.
type ToolSelection struct {
	// ToolID is the unique identifier for this tool call.
	ToolID string `json:"tool_id"`
	// ToolName is the name of the tool to call.
	ToolName string `json:"tool_name"`
	// ToolKwargs are the keyword arguments for the tool.
	ToolKwargs map[string]interface{} `json:"tool_kwargs"`
}

// NewToolSelection creates a new ToolSelection.
func NewToolSelection(toolID, toolName string, kwargs map[string]interface{}) *ToolSelection {
	return &ToolSelection{
		ToolID:     toolID,
		ToolName:   toolName,
		ToolKwargs: kwargs,
	}
}

// ToJSON returns the tool selection as a JSON string.
func (ts *ToolSelection) ToJSON() (string, error) {
	data, err := json.Marshal(ts.ToolKwargs)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// ToolCallResult represents the result of a tool call.
type ToolCallResult struct {
	// ToolName is the name of the tool that was called.
	ToolName string `json:"tool_name"`
	// ToolKwargs are the arguments that were passed to the tool.
	ToolKwargs map[string]interface{} `json:"tool_kwargs"`
	// ToolID is the unique identifier for this tool call.
	ToolID string `json:"tool_id"`
	// ToolOutput is the output from the tool.
	ToolOutput *tools.ToolOutput `json:"tool_output"`
	// ReturnDirect indicates if this result should be returned directly.
	ReturnDirect bool `json:"return_direct"`
}

// NewToolCallResult creates a new ToolCallResult.
func NewToolCallResult(toolName, toolID string, kwargs map[string]interface{}, output *tools.ToolOutput, returnDirect bool) *ToolCallResult {
	return &ToolCallResult{
		ToolName:     toolName,
		ToolKwargs:   kwargs,
		ToolID:       toolID,
		ToolOutput:   output,
		ReturnDirect: returnDirect,
	}
}

// AgentOutput represents the output from an agent step.
type AgentOutput struct {
	// Response is the chat message response from the agent.
	Response llm.ChatMessage `json:"response"`
	// ToolCalls are the tool calls requested by the agent.
	ToolCalls []*ToolSelection `json:"tool_calls,omitempty"`
	// Raw is the raw response data.
	Raw interface{} `json:"raw,omitempty"`
	// CurrentAgentName is the name of the current agent.
	CurrentAgentName string `json:"current_agent_name,omitempty"`
	// RetryMessages are messages to retry with if parsing failed.
	RetryMessages []llm.ChatMessage `json:"retry_messages,omitempty"`
	// StructuredResponse is the structured output if requested.
	StructuredResponse interface{} `json:"structured_response,omitempty"`
}

// NewAgentOutput creates a new AgentOutput.
func NewAgentOutput(response llm.ChatMessage) *AgentOutput {
	return &AgentOutput{
		Response:  response,
		ToolCalls: []*ToolSelection{},
	}
}

// WithToolCalls adds tool calls to the output.
func (o *AgentOutput) WithToolCalls(calls []*ToolSelection) *AgentOutput {
	o.ToolCalls = calls
	return o
}

// WithRaw sets the raw response data.
func (o *AgentOutput) WithRaw(raw interface{}) *AgentOutput {
	o.Raw = raw
	return o
}

// WithAgentName sets the current agent name.
func (o *AgentOutput) WithAgentName(name string) *AgentOutput {
	o.CurrentAgentName = name
	return o
}

// HasToolCalls returns true if the output contains tool calls.
func (o *AgentOutput) HasToolCalls() bool {
	return len(o.ToolCalls) > 0
}

// AgentStep represents a single step in the agent's execution.
type AgentStep struct {
	// StepID is the unique identifier for this step.
	StepID string `json:"step_id"`
	// Input is the input to this step.
	Input string `json:"input"`
	// Output is the output from this step.
	Output *AgentOutput `json:"output,omitempty"`
	// ToolCalls are the tool calls made in this step.
	ToolCalls []*ToolSelection `json:"tool_calls,omitempty"`
	// ToolResults are the results from tool calls.
	ToolResults []*ToolCallResult `json:"tool_results,omitempty"`
	// IsLast indicates if this is the final step.
	IsLast bool `json:"is_last"`
}

// NewAgentStep creates a new AgentStep.
func NewAgentStep(stepID, input string) *AgentStep {
	return &AgentStep{
		StepID:      stepID,
		Input:       input,
		ToolCalls:   []*ToolSelection{},
		ToolResults: []*ToolCallResult{},
	}
}

// AgentChatResponse represents a response from an agent chat.
type AgentChatResponse struct {
	// Response is the text response.
	Response string `json:"response"`
	// ToolCalls are the tool calls made during the response.
	ToolCalls []*ToolCallResult `json:"tool_calls,omitempty"`
	// Sources are the sources used in generating the response.
	Sources []*tools.ToolOutput `json:"sources,omitempty"`
	// Metadata contains additional response metadata.
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// NewAgentChatResponse creates a new AgentChatResponse.
func NewAgentChatResponse(response string) *AgentChatResponse {
	return &AgentChatResponse{
		Response:  response,
		ToolCalls: []*ToolCallResult{},
		Sources:   []*tools.ToolOutput{},
		Metadata:  make(map[string]interface{}),
	}
}

// String returns the response text.
func (r *AgentChatResponse) String() string {
	return r.Response
}

// StreamingAgentChatResponse represents a streaming response from an agent.
type StreamingAgentChatResponse struct {
	// ResponseChan is the channel for streaming response tokens.
	ResponseChan <-chan string
	// ToolCalls are the tool calls made during the response.
	ToolCalls []*ToolCallResult
	// Sources are the sources used in generating the response.
	Sources []*tools.ToolOutput
	// done indicates if streaming is complete.
	done bool
	// fullResponse accumulates the full response.
	fullResponse string
}

// NewStreamingAgentChatResponse creates a new StreamingAgentChatResponse.
func NewStreamingAgentChatResponse(responseChan <-chan string) *StreamingAgentChatResponse {
	return &StreamingAgentChatResponse{
		ResponseChan: responseChan,
		ToolCalls:    []*ToolCallResult{},
		Sources:      []*tools.ToolOutput{},
	}
}

// Response returns the full accumulated response.
func (r *StreamingAgentChatResponse) Response() string {
	return r.fullResponse
}

// IsDone returns whether streaming is complete.
func (r *StreamingAgentChatResponse) IsDone() bool {
	return r.done
}

// Consume reads all tokens from the stream and returns the full response.
func (r *StreamingAgentChatResponse) Consume() string {
	for token := range r.ResponseChan {
		r.fullResponse += token
	}
	r.done = true
	return r.fullResponse
}

// Agent is the interface for autonomous agents that can reason and use tools.
type Agent interface {
	// Chat sends a message and returns a response.
	Chat(ctx context.Context, message string) (*AgentChatResponse, error)

	// ChatWithHistory sends a message with explicit chat history.
	ChatWithHistory(ctx context.Context, message string, chatHistory []llm.ChatMessage) (*AgentChatResponse, error)

	// StreamChat sends a message and returns a streaming response.
	StreamChat(ctx context.Context, message string) (*StreamingAgentChatResponse, error)

	// Reset clears the agent's state.
	Reset(ctx context.Context) error

	// ChatHistory returns the current chat history.
	ChatHistory(ctx context.Context) ([]llm.ChatMessage, error)
}

// AgentWithTools extends Agent with tool management.
type AgentWithTools interface {
	Agent

	// Tools returns the tools available to the agent.
	Tools() []tools.Tool

	// AddTool adds a tool to the agent.
	AddTool(tool tools.Tool)

	// RemoveTool removes a tool from the agent.
	RemoveTool(toolName string)
}

// BaseAgent provides common functionality for agents.
type BaseAgent struct {
	name           string
	description    string
	llm            llm.LLM
	tools          []tools.Tool
	memory         memory.Memory
	systemPrompt   string
	maxIterations  int
	verbose        bool
	state          AgentState
}

// BaseAgentOption configures a BaseAgent.
type BaseAgentOption func(*BaseAgent)

// WithAgentName sets the agent name.
func WithAgentName(name string) BaseAgentOption {
	return func(a *BaseAgent) {
		a.name = name
	}
}

// WithAgentDescription sets the agent description.
func WithAgentDescription(description string) BaseAgentOption {
	return func(a *BaseAgent) {
		a.description = description
	}
}

// WithAgentLLM sets the LLM.
func WithAgentLLM(l llm.LLM) BaseAgentOption {
	return func(a *BaseAgent) {
		a.llm = l
	}
}

// WithAgentTools sets the tools.
func WithAgentTools(t []tools.Tool) BaseAgentOption {
	return func(a *BaseAgent) {
		a.tools = t
	}
}

// WithAgentMemory sets the memory.
func WithAgentMemory(m memory.Memory) BaseAgentOption {
	return func(a *BaseAgent) {
		a.memory = m
	}
}

// WithAgentSystemPrompt sets the system prompt.
func WithAgentSystemPrompt(prompt string) BaseAgentOption {
	return func(a *BaseAgent) {
		a.systemPrompt = prompt
	}
}

// WithAgentMaxIterations sets the maximum iterations.
func WithAgentMaxIterations(max int) BaseAgentOption {
	return func(a *BaseAgent) {
		a.maxIterations = max
	}
}

// WithAgentVerbose sets verbose mode.
func WithAgentVerbose(verbose bool) BaseAgentOption {
	return func(a *BaseAgent) {
		a.verbose = verbose
	}
}

// NewBaseAgent creates a new BaseAgent.
func NewBaseAgent(opts ...BaseAgentOption) *BaseAgent {
	a := &BaseAgent{
		name:          "Agent",
		description:   "An agent that can perform tasks",
		tools:         []tools.Tool{},
		maxIterations: DefaultMaxIterations,
		state:         AgentStateIdle,
	}

	for _, opt := range opts {
		opt(a)
	}

	return a
}

// Name returns the agent name.
func (a *BaseAgent) Name() string {
	return a.name
}

// Description returns the agent description.
func (a *BaseAgent) Description() string {
	return a.description
}

// LLM returns the LLM.
func (a *BaseAgent) LLM() llm.LLM {
	return a.llm
}

// Tools returns the tools.
func (a *BaseAgent) Tools() []tools.Tool {
	return a.tools
}

// AddTool adds a tool to the agent.
func (a *BaseAgent) AddTool(tool tools.Tool) {
	a.tools = append(a.tools, tool)
}

// RemoveTool removes a tool from the agent by name.
func (a *BaseAgent) RemoveTool(toolName string) {
	for i, t := range a.tools {
		if t.Metadata().Name == toolName {
			a.tools = append(a.tools[:i], a.tools[i+1:]...)
			return
		}
	}
}

// Memory returns the memory.
func (a *BaseAgent) Memory() memory.Memory {
	return a.memory
}

// SystemPrompt returns the system prompt.
func (a *BaseAgent) SystemPrompt() string {
	return a.systemPrompt
}

// MaxIterations returns the maximum iterations.
func (a *BaseAgent) MaxIterations() int {
	return a.maxIterations
}

// Verbose returns verbose mode.
func (a *BaseAgent) Verbose() bool {
	return a.verbose
}

// State returns the current agent state.
func (a *BaseAgent) State() AgentState {
	return a.state
}

// SetState sets the agent state.
func (a *BaseAgent) SetState(state AgentState) {
	a.state = state
}

// GetToolByName returns a tool by name.
func (a *BaseAgent) GetToolByName(name string) tools.Tool {
	for _, t := range a.tools {
		if t.Metadata().Name == name {
			return t
		}
	}
	return nil
}

// GetToolMetadata returns metadata for all tools.
func (a *BaseAgent) GetToolMetadata() []*tools.ToolMetadata {
	metadata := make([]*tools.ToolMetadata, len(a.tools))
	for i, t := range a.tools {
		metadata[i] = t.Metadata()
	}
	return metadata
}

// Reset clears the agent's state.
func (a *BaseAgent) Reset(ctx context.Context) error {
	a.state = AgentStateIdle
	if a.memory != nil {
		return a.memory.Reset(ctx)
	}
	return nil
}

// ChatHistory returns the current chat history.
func (a *BaseAgent) ChatHistory(ctx context.Context) ([]llm.ChatMessage, error) {
	if a.memory != nil {
		return a.memory.GetAll(ctx)
	}
	return []llm.ChatMessage{}, nil
}
