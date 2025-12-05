package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/memory"
	"github.com/aqua777/go-llamaindex/tools"
)

// ReActAgent implements the ReAct (Reasoning and Acting) agent pattern.
// It uses a thought-action-observation loop to reason about tasks and use tools.
type ReActAgent struct {
	*BaseAgent
	outputParser     OutputParser
	formatter        *ReActChatFormatter
	currentReasoning []BaseReasoningStep
}

// ReActAgentOption configures a ReActAgent.
type ReActAgentOption func(*ReActAgent)

// WithReActOutputParser sets the output parser.
func WithReActOutputParser(parser OutputParser) ReActAgentOption {
	return func(a *ReActAgent) {
		a.outputParser = parser
	}
}

// WithReActFormatter sets the chat formatter.
func WithReActFormatter(formatter *ReActChatFormatter) ReActAgentOption {
	return func(a *ReActAgent) {
		a.formatter = formatter
	}
}

// NewReActAgent creates a new ReActAgent.
func NewReActAgent(opts ...interface{}) *ReActAgent {
	// Separate base agent options from ReAct-specific options
	var baseOpts []BaseAgentOption
	var reactOpts []ReActAgentOption

	for _, opt := range opts {
		switch o := opt.(type) {
		case BaseAgentOption:
			baseOpts = append(baseOpts, o)
		case ReActAgentOption:
			reactOpts = append(reactOpts, o)
		}
	}

	a := &ReActAgent{
		BaseAgent:        NewBaseAgent(baseOpts...),
		outputParser:     NewReActOutputParser(),
		formatter:        NewReActChatFormatter(),
		currentReasoning: []BaseReasoningStep{},
	}

	for _, opt := range reactOpts {
		opt(a)
	}

	// If system prompt is set, update formatter context
	if a.systemPrompt != "" && a.formatter.Context == "" {
		a.formatter.Context = a.systemPrompt
	}

	return a
}

// NewReActAgentFromDefaults creates a ReActAgent with common defaults.
func NewReActAgentFromDefaults(
	agentLLM llm.LLM,
	agentTools []tools.Tool,
	opts ...interface{},
) *ReActAgent {
	allOpts := []interface{}{
		WithAgentLLM(agentLLM),
		WithAgentTools(agentTools),
	}
	allOpts = append(allOpts, opts...)
	return NewReActAgent(allOpts...)
}

// Chat sends a message and returns a response.
func (a *ReActAgent) Chat(ctx context.Context, message string) (*AgentChatResponse, error) {
	// Get chat history from memory
	var chatHistory []llm.ChatMessage
	if a.memory != nil {
		var err error
		chatHistory, err = a.memory.GetAll(ctx)
		if err != nil {
			return nil, fmt.Errorf("failed to get chat history: %w", err)
		}
	}

	return a.ChatWithHistory(ctx, message, chatHistory)
}

// ChatWithHistory sends a message with explicit chat history.
func (a *ReActAgent) ChatWithHistory(ctx context.Context, message string, chatHistory []llm.ChatMessage) (*AgentChatResponse, error) {
	a.SetState(AgentStateRunning)
	defer a.SetState(AgentStateIdle)

	// Reset reasoning for new conversation turn
	a.currentReasoning = []BaseReasoningStep{}

	// Add user message to history
	userMsg := llm.NewUserMessage(message)
	chatHistory = append(chatHistory, userMsg)

	// Store user message in memory
	if a.memory != nil {
		if err := a.memory.Put(ctx, userMsg); err != nil {
			return nil, fmt.Errorf("failed to store user message: %w", err)
		}
	}

	// Run the reasoning loop
	var finalResponse string
	var allToolCalls []*ToolCallResult

	for iteration := 0; iteration < a.maxIterations; iteration++ {
		// Format messages for LLM
		messages := a.formatter.Format(a.tools, chatHistory, a.currentReasoning)

		if a.verbose {
			fmt.Printf("[ReActAgent] Iteration %d, sending %d messages to LLM\n", iteration+1, len(messages))
		}

		// Get LLM response
		response, err := a.llm.Chat(ctx, messages)
		if err != nil {
			return nil, fmt.Errorf("LLM chat failed: %w", err)
		}

		if a.verbose {
			fmt.Printf("[ReActAgent] LLM response: %s\n", response)
		}

		// Parse the response
		reasoningStep, err := a.outputParser.Parse(response, false)
		if err != nil {
			// If parsing fails, try to recover by asking LLM to fix format
			if a.verbose {
				fmt.Printf("[ReActAgent] Parse error: %v, attempting recovery\n", err)
			}

			// Add error message and retry
			errorMsg := fmt.Sprintf(
				"Error while parsing the output: %v\n\n"+
					"The output should be in one of the following formats:\n"+
					"1. To call a tool:\n"+
					"```\n"+
					"Thought: <thought>\n"+
					"Action: <action>\n"+
					"Action Input: <action_input>\n"+
					"```\n"+
					"2. To answer the question:\n"+
					"```\n"+
					"Thought: <thought>\n"+
					"Answer: <answer>\n"+
					"```\n",
				err,
			)

			// Add the failed response and error as messages
			a.currentReasoning = append(a.currentReasoning, &ResponseReasoningStep{
				Thought:  response,
				Response: "",
			})
			chatHistory = append(chatHistory, llm.NewUserMessage(errorMsg))
			continue
		}

		// Add reasoning step
		a.currentReasoning = append(a.currentReasoning, reasoningStep)

		// Check if we're done
		if reasoningStep.IsDone() {
			if respStep, ok := reasoningStep.(*ResponseReasoningStep); ok {
				finalResponse = respStep.Response
			}
			break
		}

		// Handle action step
		if actionStep, ok := reasoningStep.(*ActionReasoningStep); ok {
			a.SetState(AgentStateWaitingForTool)

			// Execute the tool
			toolResult, err := a.executeTool(ctx, actionStep)
			if err != nil {
				if a.verbose {
					fmt.Printf("[ReActAgent] Tool execution error: %v\n", err)
				}
			}

			allToolCalls = append(allToolCalls, toolResult)

			// Add observation
			observation := &ObservationReasoningStep{
				Observation:  toolResult.ToolOutput.Content,
				ReturnDirect: toolResult.ReturnDirect,
			}
			a.currentReasoning = append(a.currentReasoning, observation)

			// If return_direct, use the tool output as the final response
			if toolResult.ReturnDirect && !toolResult.ToolOutput.IsError {
				finalResponse = toolResult.ToolOutput.Content
				break
			}

			a.SetState(AgentStateRunning)
		}
	}

	// Clean up the response
	finalResponse = CleanResponse(finalResponse)

	// Store assistant response in memory
	if a.memory != nil && finalResponse != "" {
		assistantMsg := llm.NewAssistantMessage(finalResponse)
		if err := a.memory.Put(ctx, assistantMsg); err != nil {
			return nil, fmt.Errorf("failed to store assistant message: %w", err)
		}
	}

	a.SetState(AgentStateCompleted)

	return &AgentChatResponse{
		Response:  finalResponse,
		ToolCalls: allToolCalls,
		Sources:   extractSources(allToolCalls),
		Metadata: map[string]interface{}{
			"iterations": len(a.currentReasoning),
		},
	}, nil
}

// StreamChat sends a message and returns a streaming response.
func (a *ReActAgent) StreamChat(ctx context.Context, message string) (*StreamingAgentChatResponse, error) {
	// For now, implement streaming as a wrapper around non-streaming
	// A full implementation would stream tokens from the LLM
	responseChan := make(chan string, 1)

	go func() {
		defer close(responseChan)

		response, err := a.Chat(ctx, message)
		if err != nil {
			responseChan <- fmt.Sprintf("Error: %v", err)
			return
		}

		responseChan <- response.Response
	}()

	return NewStreamingAgentChatResponse(responseChan), nil
}

// Reset clears the agent's state.
func (a *ReActAgent) Reset(ctx context.Context) error {
	a.currentReasoning = []BaseReasoningStep{}
	return a.BaseAgent.Reset(ctx)
}

// executeTool executes a tool based on an action step.
func (a *ReActAgent) executeTool(ctx context.Context, action *ActionReasoningStep) (*ToolCallResult, error) {
	toolID := GenerateToolID()

	// Find the tool
	tool := a.GetToolByName(action.Action)
	if tool == nil {
		errOutput := tools.NewErrorToolOutput(action.Action, fmt.Errorf("tool not found: %s", action.Action))
		return NewToolCallResult(action.Action, toolID, action.ActionInput, errOutput, false), fmt.Errorf("tool not found: %s", action.Action)
	}

	if a.verbose {
		fmt.Printf("[ReActAgent] Executing tool: %s with input: %v\n", action.Action, action.ActionInput)
	}

	// Execute the tool
	output, err := tool.Call(ctx, action.ActionInput)
	if err != nil {
		errOutput := tools.NewErrorToolOutput(action.Action, err)
		return NewToolCallResult(action.Action, toolID, action.ActionInput, errOutput, tool.Metadata().ReturnDirect), err
	}

	if a.verbose {
		fmt.Printf("[ReActAgent] Tool output: %s\n", output.Content)
	}

	return NewToolCallResult(action.Action, toolID, action.ActionInput, output, tool.Metadata().ReturnDirect), nil
}

// CurrentReasoning returns the current reasoning steps.
func (a *ReActAgent) CurrentReasoning() []BaseReasoningStep {
	return a.currentReasoning
}

// SetCurrentReasoning sets the current reasoning steps.
func (a *ReActAgent) SetCurrentReasoning(steps []BaseReasoningStep) {
	a.currentReasoning = steps
}

// extractSources extracts tool outputs as sources.
func extractSources(toolCalls []*ToolCallResult) []*tools.ToolOutput {
	sources := make([]*tools.ToolOutput, 0, len(toolCalls))
	for _, tc := range toolCalls {
		if tc.ToolOutput != nil {
			sources = append(sources, tc.ToolOutput)
		}
	}
	return sources
}

// FunctionCallingReActAgent is a ReAct agent that uses function calling LLMs.
// It leverages the LLM's native tool calling capabilities instead of text parsing.
type FunctionCallingReActAgent struct {
	*BaseAgent
	currentReasoning []BaseReasoningStep
}

// NewFunctionCallingReActAgent creates a new FunctionCallingReActAgent.
func NewFunctionCallingReActAgent(opts ...BaseAgentOption) *FunctionCallingReActAgent {
	return &FunctionCallingReActAgent{
		BaseAgent:        NewBaseAgent(opts...),
		currentReasoning: []BaseReasoningStep{},
	}
}

// Chat sends a message and returns a response using function calling.
func (a *FunctionCallingReActAgent) Chat(ctx context.Context, message string) (*AgentChatResponse, error) {
	var chatHistory []llm.ChatMessage
	if a.memory != nil {
		var err error
		chatHistory, err = a.memory.GetAll(ctx)
		if err != nil {
			return nil, fmt.Errorf("failed to get chat history: %w", err)
		}
	}

	return a.ChatWithHistory(ctx, message, chatHistory)
}

// ChatWithHistory sends a message with explicit chat history using function calling.
func (a *FunctionCallingReActAgent) ChatWithHistory(ctx context.Context, message string, chatHistory []llm.ChatMessage) (*AgentChatResponse, error) {
	// Check if LLM supports tool calling
	toolLLM, ok := a.llm.(llm.LLMWithToolCalling)
	if !ok {
		return nil, fmt.Errorf("LLM does not support tool calling")
	}

	a.SetState(AgentStateRunning)
	defer a.SetState(AgentStateIdle)

	// Reset reasoning
	a.currentReasoning = []BaseReasoningStep{}

	// Build messages
	messages := chatHistory
	if a.systemPrompt != "" {
		messages = append([]llm.ChatMessage{llm.NewSystemMessage(a.systemPrompt)}, messages...)
	}
	messages = append(messages, llm.NewUserMessage(message))

	// Store user message
	if a.memory != nil {
		if err := a.memory.Put(ctx, llm.NewUserMessage(message)); err != nil {
			return nil, fmt.Errorf("failed to store user message: %w", err)
		}
	}

	// Get tool metadata
	toolMetadata := make([]*llm.ToolMetadata, len(a.tools))
	for i, t := range a.tools {
		meta := t.Metadata()
		toolMetadata[i] = &llm.ToolMetadata{
			Name:        meta.Name,
			Description: meta.Description,
			Parameters:  meta.Parameters,
		}
	}

	var allToolCalls []*ToolCallResult

	// Run the tool calling loop
	for iteration := 0; iteration < a.maxIterations; iteration++ {
		// Call LLM with tools
		response, err := toolLLM.ChatWithTools(ctx, messages, toolMetadata, nil)
		if err != nil {
			return nil, fmt.Errorf("LLM chat with tools failed: %w", err)
		}

		// Check for tool calls
		if response.Message != nil && response.Message.HasToolCalls() {
			toolCalls := response.Message.GetToolCalls()

			// Add assistant message with tool calls
			messages = append(messages, *response.Message)

			// Execute each tool call
			for _, tc := range toolCalls {
				a.SetState(AgentStateWaitingForTool)

				args, _ := tc.ParseArguments()
				tool := a.GetToolByName(tc.Name)

				var output *tools.ToolOutput
				var returnDirect bool

				if tool == nil {
					output = tools.NewErrorToolOutput(tc.Name, fmt.Errorf("tool not found: %s", tc.Name))
				} else {
					output, _ = tool.Call(ctx, args)
					returnDirect = tool.Metadata().ReturnDirect
				}

				result := NewToolCallResult(tc.Name, tc.ID, args, output, returnDirect)
				allToolCalls = append(allToolCalls, result)

				// Add tool result message
				toolMsg := llm.NewToolMessage(tc.ID, output.Content)
				messages = append(messages, toolMsg)

				// If return_direct, return immediately
				if returnDirect && !output.IsError {
					return &AgentChatResponse{
						Response:  output.Content,
						ToolCalls: allToolCalls,
						Sources:   extractSources(allToolCalls),
					}, nil
				}
			}

			a.SetState(AgentStateRunning)
			continue
		}

		// No tool calls, we have a final response
		finalResponse := response.Text
		if response.Message != nil {
			finalResponse = response.Message.GetTextContent()
		}

		// Store assistant response
		if a.memory != nil && finalResponse != "" {
			if err := a.memory.Put(ctx, llm.NewAssistantMessage(finalResponse)); err != nil {
				return nil, fmt.Errorf("failed to store assistant message: %w", err)
			}
		}

		a.SetState(AgentStateCompleted)

		return &AgentChatResponse{
			Response:  finalResponse,
			ToolCalls: allToolCalls,
			Sources:   extractSources(allToolCalls),
		}, nil
	}

	return nil, fmt.Errorf("max iterations (%d) reached", a.maxIterations)
}

// StreamChat sends a message and returns a streaming response.
func (a *FunctionCallingReActAgent) StreamChat(ctx context.Context, message string) (*StreamingAgentChatResponse, error) {
	responseChan := make(chan string, 1)

	go func() {
		defer close(responseChan)

		response, err := a.Chat(ctx, message)
		if err != nil {
			responseChan <- fmt.Sprintf("Error: %v", err)
			return
		}

		responseChan <- response.Response
	}()

	return NewStreamingAgentChatResponse(responseChan), nil
}

// Reset clears the agent's state.
func (a *FunctionCallingReActAgent) Reset(ctx context.Context) error {
	a.currentReasoning = []BaseReasoningStep{}
	return a.BaseAgent.Reset(ctx)
}

// SimpleAgent is a basic agent that directly uses an LLM without tool calling.
type SimpleAgent struct {
	*BaseAgent
}

// NewSimpleAgent creates a new SimpleAgent.
func NewSimpleAgent(opts ...BaseAgentOption) *SimpleAgent {
	return &SimpleAgent{
		BaseAgent: NewBaseAgent(opts...),
	}
}

// NewSimpleAgentFromDefaults creates a SimpleAgent with common defaults.
func NewSimpleAgentFromDefaults(agentLLM llm.LLM, mem memory.Memory, systemPrompt string) *SimpleAgent {
	return NewSimpleAgent(
		WithAgentLLM(agentLLM),
		WithAgentMemory(mem),
		WithAgentSystemPrompt(systemPrompt),
	)
}

// Chat sends a message and returns a response.
func (a *SimpleAgent) Chat(ctx context.Context, message string) (*AgentChatResponse, error) {
	var chatHistory []llm.ChatMessage
	if a.memory != nil {
		var err error
		chatHistory, err = a.memory.GetAll(ctx)
		if err != nil {
			return nil, fmt.Errorf("failed to get chat history: %w", err)
		}
	}

	return a.ChatWithHistory(ctx, message, chatHistory)
}

// ChatWithHistory sends a message with explicit chat history.
func (a *SimpleAgent) ChatWithHistory(ctx context.Context, message string, chatHistory []llm.ChatMessage) (*AgentChatResponse, error) {
	a.SetState(AgentStateRunning)
	defer a.SetState(AgentStateIdle)

	// Build messages
	messages := chatHistory
	if a.systemPrompt != "" {
		messages = append([]llm.ChatMessage{llm.NewSystemMessage(a.systemPrompt)}, messages...)
	}
	messages = append(messages, llm.NewUserMessage(message))

	// Store user message
	if a.memory != nil {
		if err := a.memory.Put(ctx, llm.NewUserMessage(message)); err != nil {
			return nil, fmt.Errorf("failed to store user message: %w", err)
		}
	}

	// Get LLM response
	response, err := a.llm.Chat(ctx, messages)
	if err != nil {
		return nil, fmt.Errorf("LLM chat failed: %w", err)
	}

	// Store assistant response
	if a.memory != nil {
		if err := a.memory.Put(ctx, llm.NewAssistantMessage(response)); err != nil {
			return nil, fmt.Errorf("failed to store assistant message: %w", err)
		}
	}

	a.SetState(AgentStateCompleted)

	return &AgentChatResponse{
		Response: response,
	}, nil
}

// StreamChat sends a message and returns a streaming response.
func (a *SimpleAgent) StreamChat(ctx context.Context, message string) (*StreamingAgentChatResponse, error) {
	responseChan := make(chan string, 1)

	go func() {
		defer close(responseChan)

		response, err := a.Chat(ctx, message)
		if err != nil {
			responseChan <- fmt.Sprintf("Error: %v", err)
			return
		}

		responseChan <- response.Response
	}()

	return NewStreamingAgentChatResponse(responseChan), nil
}

// GetAgentForLLM returns the appropriate agent type based on LLM capabilities.
func GetAgentForLLM(agentLLM llm.LLM, agentTools []tools.Tool, opts ...BaseAgentOption) Agent {
	// Check if LLM supports tool calling
	if toolLLM, ok := agentLLM.(llm.LLMWithToolCalling); ok && toolLLM.SupportsToolCalling() {
		allOpts := append([]BaseAgentOption{
			WithAgentLLM(agentLLM),
			WithAgentTools(agentTools),
		}, opts...)
		return NewFunctionCallingReActAgent(allOpts...)
	}

	// Fall back to text-based ReAct agent
	allOpts := []interface{}{
		WithAgentLLM(agentLLM),
		WithAgentTools(agentTools),
	}
	for _, opt := range opts {
		allOpts = append(allOpts, opt)
	}
	return NewReActAgent(allOpts...)
}

// Ensure agents implement the Agent interface
var (
	_ Agent = (*ReActAgent)(nil)
	_ Agent = (*FunctionCallingReActAgent)(nil)
	_ Agent = (*SimpleAgent)(nil)
)

// Helper function to check if a string contains a substring
func containsSubstring(s, substr string) bool {
	return strings.Contains(s, substr)
}
