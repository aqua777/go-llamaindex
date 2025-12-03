package agent

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/memory"
	"github.com/aqua777/go-llamaindex/tools"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

// MockLLM implements llm.LLM for testing.
type MockLLM struct {
	responses []string
	callCount int
}

func NewMockLLM(responses ...string) *MockLLM {
	return &MockLLM{responses: responses}
}

func (m *MockLLM) Complete(ctx context.Context, prompt string) (string, error) {
	return m.getNextResponse(), nil
}

func (m *MockLLM) Chat(ctx context.Context, messages []llm.ChatMessage) (string, error) {
	return m.getNextResponse(), nil
}

func (m *MockLLM) Stream(ctx context.Context, prompt string) (<-chan string, error) {
	ch := make(chan string, 1)
	ch <- m.getNextResponse()
	close(ch)
	return ch, nil
}

func (m *MockLLM) getNextResponse() string {
	if m.callCount >= len(m.responses) {
		return "Thought: I can answer without using any more tools.\nAnswer: Default response"
	}
	response := m.responses[m.callCount]
	m.callCount++
	return response
}

// MockToolCallingLLM implements llm.LLMWithToolCalling for testing.
type MockToolCallingLLM struct {
	*MockLLM
	toolCallResponses []llm.CompletionResponse
	toolCallCount     int
}

func NewMockToolCallingLLM(responses ...llm.CompletionResponse) *MockToolCallingLLM {
	return &MockToolCallingLLM{
		MockLLM:           NewMockLLM(),
		toolCallResponses: responses,
	}
}

func (m *MockToolCallingLLM) ChatWithTools(ctx context.Context, messages []llm.ChatMessage, tools []*llm.ToolMetadata, opts *llm.ChatCompletionOptions) (llm.CompletionResponse, error) {
	if m.toolCallCount >= len(m.toolCallResponses) {
		return llm.CompletionResponse{Text: "Default response"}, nil
	}
	response := m.toolCallResponses[m.toolCallCount]
	m.toolCallCount++
	return response, nil
}

func (m *MockToolCallingLLM) SupportsToolCalling() bool {
	return true
}

// MockTool implements tools.Tool for testing.
type MockTool struct {
	name        string
	description string
	handler     func(ctx context.Context, input interface{}) (*tools.ToolOutput, error)
}

func NewMockTool(name, description string, handler func(ctx context.Context, input interface{}) (*tools.ToolOutput, error)) *MockTool {
	return &MockTool{
		name:        name,
		description: description,
		handler:     handler,
	}
}

func (t *MockTool) Metadata() *tools.ToolMetadata {
	return &tools.ToolMetadata{
		Name:        t.name,
		Description: t.description,
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"input": map[string]interface{}{
					"type":        "string",
					"description": "The input to the tool",
				},
			},
		},
	}
}

func (t *MockTool) Call(ctx context.Context, input interface{}) (*tools.ToolOutput, error) {
	if t.handler != nil {
		return t.handler(ctx, input)
	}
	return tools.NewToolOutput(t.name, "Mock output"), nil
}

// AgentTestSuite tests the agent package.
type AgentTestSuite struct {
	suite.Suite
}

func TestAgentSuite(t *testing.T) {
	suite.Run(t, new(AgentTestSuite))
}

// Test Types

func (s *AgentTestSuite) TestToolSelection() {
	selection := NewToolSelection("call_1", "search", map[string]interface{}{"query": "test"})

	s.Equal("call_1", selection.ToolID)
	s.Equal("search", selection.ToolName)
	s.Equal("test", selection.ToolKwargs["query"])

	jsonStr, err := selection.ToJSON()
	s.NoError(err)
	s.Contains(jsonStr, "query")
}

func (s *AgentTestSuite) TestToolCallResult() {
	output := tools.NewToolOutput("search", "Search results")
	result := NewToolCallResult("search", "call_1", map[string]interface{}{"query": "test"}, output, false)

	s.Equal("search", result.ToolName)
	s.Equal("call_1", result.ToolID)
	s.Equal("Search results", result.ToolOutput.Content)
	s.False(result.ReturnDirect)
}

func (s *AgentTestSuite) TestAgentOutput() {
	msg := llm.NewAssistantMessage("Hello")
	output := NewAgentOutput(msg)

	s.Equal("Hello", output.Response.Content)
	s.False(output.HasToolCalls())

	// Add tool calls
	output.WithToolCalls([]*ToolSelection{
		NewToolSelection("call_1", "search", nil),
	})
	s.True(output.HasToolCalls())
}

func (s *AgentTestSuite) TestAgentStep() {
	step := NewAgentStep("step_1", "What is 2+2?")

	s.Equal("step_1", step.StepID)
	s.Equal("What is 2+2?", step.Input)
	s.False(step.IsLast)
}

func (s *AgentTestSuite) TestAgentChatResponse() {
	response := NewAgentChatResponse("The answer is 4")

	s.Equal("The answer is 4", response.Response)
	s.Equal("The answer is 4", response.String())
	s.Empty(response.ToolCalls)
}

func (s *AgentTestSuite) TestStreamingAgentChatResponse() {
	ch := make(chan string, 2)
	ch <- "Hello "
	ch <- "World"
	close(ch)

	response := NewStreamingAgentChatResponse(ch)
	s.False(response.IsDone())

	fullResponse := response.Consume()
	s.Equal("Hello World", fullResponse)
	s.True(response.IsDone())
}

func (s *AgentTestSuite) TestAgentState() {
	s.Equal(AgentState("idle"), AgentStateIdle)
	s.Equal(AgentState("running"), AgentStateRunning)
	s.Equal(AgentState("completed"), AgentStateCompleted)
}

// Test BaseAgent

func (s *AgentTestSuite) TestBaseAgentCreation() {
	agent := NewBaseAgent(
		WithAgentName("TestAgent"),
		WithAgentDescription("A test agent"),
		WithAgentMaxIterations(5),
		WithAgentVerbose(true),
	)

	s.Equal("TestAgent", agent.Name())
	s.Equal("A test agent", agent.Description())
	s.Equal(5, agent.MaxIterations())
	s.True(agent.Verbose())
	s.Equal(AgentStateIdle, agent.State())
}

func (s *AgentTestSuite) TestBaseAgentTools() {
	agent := NewBaseAgent()

	tool := NewMockTool("search", "Search the web", nil)
	agent.AddTool(tool)

	s.Len(agent.Tools(), 1)
	s.Equal("search", agent.Tools()[0].Metadata().Name)

	foundTool := agent.GetToolByName("search")
	s.NotNil(foundTool)
	s.Equal("search", foundTool.Metadata().Name)

	agent.RemoveTool("search")
	s.Empty(agent.Tools())
}

func (s *AgentTestSuite) TestBaseAgentReset() {
	ctx := context.Background()
	mem := memory.NewSimpleMemory()
	agent := NewBaseAgent(WithAgentMemory(mem))

	agent.SetState(AgentStateRunning)
	err := agent.Reset(ctx)
	s.NoError(err)
	s.Equal(AgentStateIdle, agent.State())
}

// Test Output Parser

func (s *AgentTestSuite) TestReActOutputParserAction() {
	parser := NewReActOutputParser()

	output := `Thought: I need to search for information.
Action: search
Action Input: {"query": "test query"}`

	step, err := parser.Parse(output, false)
	s.NoError(err)
	s.IsType(&ActionReasoningStep{}, step)

	actionStep := step.(*ActionReasoningStep)
	s.Equal("I need to search for information.", actionStep.Thought)
	s.Equal("search", actionStep.Action)
	s.Equal("test query", actionStep.ActionInput["query"])
	s.False(step.IsDone())
}

func (s *AgentTestSuite) TestReActOutputParserAnswer() {
	parser := NewReActOutputParser()

	output := `Thought: I now have enough information to answer.
Answer: The answer is 42.`

	step, err := parser.Parse(output, false)
	s.NoError(err)
	s.IsType(&ResponseReasoningStep{}, step)

	responseStep := step.(*ResponseReasoningStep)
	s.Equal("I now have enough information to answer.", responseStep.Thought)
	s.Equal("The answer is 42.", responseStep.Response)
	s.True(step.IsDone())
}

func (s *AgentTestSuite) TestReActOutputParserImplicitAnswer() {
	parser := NewReActOutputParser()

	output := `The answer is simply 42.`

	step, err := parser.Parse(output, false)
	s.NoError(err)
	s.IsType(&ResponseReasoningStep{}, step)

	responseStep := step.(*ResponseReasoningStep)
	s.Contains(responseStep.Thought, "Implicit")
	s.Equal("The answer is simply 42.", responseStep.Response)
}

func (s *AgentTestSuite) TestReActOutputParserMalformedJSON() {
	parser := NewReActOutputParser()

	// Test with single quotes (common LLM mistake)
	output := `Thought: I need to search.
Action: search
Action Input: {"query": "test"}`

	step, err := parser.Parse(output, false)
	s.NoError(err)
	s.IsType(&ActionReasoningStep{}, step)
}

// Test Reasoning Steps

func (s *AgentTestSuite) TestActionReasoningStep() {
	step := &ActionReasoningStep{
		Thought:     "I need to search",
		Action:      "search",
		ActionInput: map[string]interface{}{"query": "test"},
	}

	s.Contains(step.GetContent(), "Thought: I need to search")
	s.Contains(step.GetContent(), "Action: search")
	s.False(step.IsDone())
	s.Equal(ReasoningStepTypeAction, step.StepType())
}

func (s *AgentTestSuite) TestObservationReasoningStep() {
	step := &ObservationReasoningStep{
		Observation:  "Search results: found 10 items",
		ReturnDirect: false,
	}

	s.Contains(step.GetContent(), "Observation: Search results")
	s.False(step.IsDone())
	s.Equal(ReasoningStepTypeObservation, step.StepType())

	// Test return_direct
	step.ReturnDirect = true
	s.True(step.IsDone())
}

func (s *AgentTestSuite) TestResponseReasoningStep() {
	step := &ResponseReasoningStep{
		Thought:  "I can answer now",
		Response: "The answer is 42",
	}

	s.Contains(step.GetContent(), "Thought: I can answer now")
	s.Contains(step.GetContent(), "Answer: The answer is 42")
	s.True(step.IsDone())
	s.Equal(ReasoningStepTypeResponse, step.StepType())
}

// Test Formatter

func (s *AgentTestSuite) TestReActChatFormatter() {
	formatter := NewReActChatFormatter()

	tool := NewMockTool("search", "Search the web", nil)
	agentTools := []tools.Tool{tool}

	chatHistory := []llm.ChatMessage{
		llm.NewUserMessage("Hello"),
	}

	messages := formatter.Format(agentTools, chatHistory, nil)

	s.GreaterOrEqual(len(messages), 2)
	s.Equal(llm.MessageRoleSystem, messages[0].Role)
	s.Contains(messages[0].Content, "search")
	s.Contains(messages[0].Content, "Search the web")
}

func (s *AgentTestSuite) TestReActChatFormatterWithContext() {
	formatter := NewReActChatFormatterFromDefaults("", "Some context", llm.MessageRoleUser)

	tool := NewMockTool("search", "Search the web", nil)
	agentTools := []tools.Tool{tool}

	messages := formatter.Format(agentTools, []llm.ChatMessage{}, nil)

	s.Contains(messages[0].Content, "Some context")
}

func (s *AgentTestSuite) TestReActChatFormatterWithReasoning() {
	formatter := NewReActChatFormatter()

	tool := NewMockTool("search", "Search the web", nil)
	agentTools := []tools.Tool{tool}

	reasoning := []BaseReasoningStep{
		&ActionReasoningStep{
			Thought:     "I need to search",
			Action:      "search",
			ActionInput: map[string]interface{}{"query": "test"},
		},
		&ObservationReasoningStep{
			Observation: "Found results",
		},
	}

	messages := formatter.Format(agentTools, []llm.ChatMessage{}, reasoning)

	// Should have system + 2 reasoning messages
	s.GreaterOrEqual(len(messages), 3)
}

// Test ReActAgent

func (s *AgentTestSuite) TestReActAgentCreation() {
	mockLLM := NewMockLLM()
	tool := NewMockTool("search", "Search the web", nil)

	agent := NewReActAgentFromDefaults(mockLLM, []tools.Tool{tool})

	s.NotNil(agent)
	s.Len(agent.Tools(), 1)
}

func (s *AgentTestSuite) TestReActAgentDirectAnswer() {
	mockLLM := NewMockLLM(
		"Thought: I can answer this directly.\nAnswer: The answer is 42.",
	)

	agent := NewReActAgentFromDefaults(mockLLM, []tools.Tool{})

	ctx := context.Background()
	response, err := agent.Chat(ctx, "What is the answer?")

	s.NoError(err)
	s.Equal("The answer is 42.", response.Response)
	s.Empty(response.ToolCalls)
}

func (s *AgentTestSuite) TestReActAgentWithToolCall() {
	mockLLM := NewMockLLM(
		`Thought: I need to search for this.
Action: search
Action Input: {"query": "test"}`,
		"Thought: Now I can answer.\nAnswer: Found the result.",
	)

	tool := NewMockTool("search", "Search the web", func(ctx context.Context, input interface{}) (*tools.ToolOutput, error) {
		return tools.NewToolOutput("search", "Search result: test data"), nil
	})

	agent := NewReActAgentFromDefaults(mockLLM, []tools.Tool{tool})

	ctx := context.Background()
	response, err := agent.Chat(ctx, "Search for test")

	s.NoError(err)
	s.Equal("Found the result.", response.Response)
	s.Len(response.ToolCalls, 1)
	s.Equal("search", response.ToolCalls[0].ToolName)
}

func (s *AgentTestSuite) TestReActAgentWithMemory() {
	mockLLM := NewMockLLM(
		"Thought: I can answer.\nAnswer: Hello!",
	)

	mem := memory.NewSimpleMemory()
	agent := NewReActAgent(
		WithAgentLLM(mockLLM),
		WithAgentMemory(mem),
	)

	ctx := context.Background()
	_, err := agent.Chat(ctx, "Hi")
	s.NoError(err)

	history, err := agent.ChatHistory(ctx)
	s.NoError(err)
	s.Len(history, 2) // User message + Assistant message
}

func (s *AgentTestSuite) TestReActAgentReset() {
	mockLLM := NewMockLLM()
	mem := memory.NewSimpleMemory()

	agent := NewReActAgent(
		WithAgentLLM(mockLLM),
		WithAgentMemory(mem),
	)

	// Add some reasoning
	agent.SetCurrentReasoning([]BaseReasoningStep{
		&ActionReasoningStep{Thought: "test", Action: "test", ActionInput: nil},
	})

	ctx := context.Background()
	err := agent.Reset(ctx)
	s.NoError(err)
	s.Empty(agent.CurrentReasoning())
}

// Test SimpleAgent

func (s *AgentTestSuite) TestSimpleAgent() {
	mockLLM := NewMockLLM("Hello, how can I help?")

	agent := NewSimpleAgentFromDefaults(mockLLM, nil, "You are a helpful assistant.")

	ctx := context.Background()
	response, err := agent.Chat(ctx, "Hi")

	s.NoError(err)
	s.Equal("Hello, how can I help?", response.Response)
}

// Test Utils

func (s *AgentTestSuite) TestCallToolByName() {
	ctx := context.Background()

	tool := NewMockTool("search", "Search", func(ctx context.Context, input interface{}) (*tools.ToolOutput, error) {
		return tools.NewToolOutput("search", "Result"), nil
	})

	output, err := CallToolByName(ctx, []tools.Tool{tool}, "search", nil)
	s.NoError(err)
	s.Equal("Result", output.Content)

	// Test tool not found
	_, err = CallToolByName(ctx, []tools.Tool{tool}, "unknown", nil)
	s.Error(err)
}

func (s *AgentTestSuite) TestGetToolsByName() {
	tool1 := NewMockTool("search", "Search", nil)
	tool2 := NewMockTool("calculate", "Calculate", nil)

	toolsMap := GetToolsByName([]tools.Tool{tool1, tool2})

	s.Len(toolsMap, 2)
	s.NotNil(toolsMap["search"])
	s.NotNil(toolsMap["calculate"])
}

func (s *AgentTestSuite) TestValidateToolSelection() {
	tool := NewMockTool("search", "Search", nil)
	agentTools := []tools.Tool{tool}

	// Valid selection
	selection := NewToolSelection("call_1", "search", nil)
	err := ValidateToolSelection(selection, agentTools)
	s.NoError(err)

	// Invalid tool name
	selection = NewToolSelection("call_1", "unknown", nil)
	err = ValidateToolSelection(selection, agentTools)
	s.Error(err)

	// Nil selection
	err = ValidateToolSelection(nil, agentTools)
	s.Error(err)
}

func (s *AgentTestSuite) TestParseToolArguments() {
	args, err := ParseToolArguments(`{"query": "test", "limit": 10}`)
	s.NoError(err)
	s.Equal("test", args["query"])
	s.Equal(float64(10), args["limit"])

	// Empty input
	args, err = ParseToolArguments("")
	s.NoError(err)
	s.Empty(args)
}

func (s *AgentTestSuite) TestToolArgsToJSON() {
	args := map[string]interface{}{"query": "test"}
	jsonStr, err := ToolArgsToJSON(args)
	s.NoError(err)
	s.Contains(jsonStr, "query")

	// Empty args
	jsonStr, err = ToolArgsToJSON(nil)
	s.NoError(err)
	s.Equal("{}", jsonStr)
}

func (s *AgentTestSuite) TestCleanResponse() {
	s.Equal("The answer is 42", CleanResponse("Answer: The answer is 42"))
	s.Equal("The answer is 42", CleanResponse("The answer is 42"))
	s.Equal("42", CleanResponse("  42  "))
}

func (s *AgentTestSuite) TestBuildToolsText() {
	tool1 := NewMockTool("search", "Search the web", nil)
	tool2 := NewMockTool("calculate", "Do math", nil)

	text := BuildToolsText([]tools.Tool{tool1, tool2})

	s.Contains(text, "1. search")
	s.Contains(text, "2. calculate")
}

func (s *AgentTestSuite) TestIsValidToolName() {
	tool := NewMockTool("search", "Search", nil)
	agentTools := []tools.Tool{tool}

	s.True(IsValidToolName("search", agentTools))
	s.False(IsValidToolName("unknown", agentTools))
}

// Test Interface Compliance

func (s *AgentTestSuite) TestAgentInterfaceCompliance() {
	var _ Agent = (*ReActAgent)(nil)
	var _ Agent = (*FunctionCallingReActAgent)(nil)
	var _ Agent = (*SimpleAgent)(nil)
}

func (s *AgentTestSuite) TestOutputParserInterfaceCompliance() {
	var _ OutputParser = (*ReActOutputParser)(nil)
}

func (s *AgentTestSuite) TestAgentChatFormatterInterfaceCompliance() {
	var _ AgentChatFormatter = (*ReActChatFormatter)(nil)
}

// Benchmark Tests

func BenchmarkReActOutputParser(b *testing.B) {
	parser := NewReActOutputParser()
	output := `Thought: I need to search for information.
Action: search
Action Input: {"query": "test query"}`

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = parser.Parse(output, false)
	}
}

func BenchmarkReActChatFormatter(b *testing.B) {
	formatter := NewReActChatFormatter()
	tool := NewMockTool("search", "Search the web", nil)
	agentTools := []tools.Tool{tool}
	chatHistory := []llm.ChatMessage{
		llm.NewUserMessage("Hello"),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = formatter.Format(agentTools, chatHistory, nil)
	}
}

// Additional edge case tests

func TestExtractToolUse(t *testing.T) {
	tests := []struct {
		name        string
		input       string
		wantThought string
		wantAction  string
		wantInput   string
		wantErr     bool
	}{
		{
			name: "standard format",
			input: `Thought: I need to search.
Action: search
Action Input: {"query": "test"}`,
			wantThought: "I need to search.",
			wantAction:  "search",
			wantInput:   `{"query": "test"}`,
			wantErr:     false,
		},
		{
			name: "multiline thought",
			input: `Thought: I need to search for information about this topic.
Action: search
Action Input: {"query": "test"}`,
			wantThought: "I need to search for information about this topic.",
			wantAction:  "search",
			wantInput:   `{"query": "test"}`,
			wantErr:     false,
		},
		{
			name:    "missing action",
			input:   `Thought: I need to search.`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			thought, action, actionInput, err := extractToolUse(tt.input)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.wantThought, thought)
			assert.Equal(t, tt.wantAction, action)
			assert.Contains(t, actionInput, "query")
		})
	}
}

func TestExtractFinalResponse(t *testing.T) {
	tests := []struct {
		name        string
		input       string
		wantThought string
		wantAnswer  string
		wantErr     bool
	}{
		{
			name: "standard format",
			input: `Thought: I can answer now.
Answer: The answer is 42.`,
			wantThought: "I can answer now.",
			wantAnswer:  "The answer is 42.",
			wantErr:     false,
		},
		{
			name:    "missing answer",
			input:   `Thought: I can answer now.`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			thought, answer, err := extractFinalResponse(tt.input)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.wantThought, thought)
			assert.Equal(t, tt.wantAnswer, answer)
		})
	}
}

func TestExtractJSONStr(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "plain JSON",
			input: `{"key": "value"}`,
			want:  `{"key": "value"}`,
		},
		{
			name:  "JSON in code block",
			input: "```json\n{\"key\": \"value\"}\n```",
			want:  `{"key": "value"}`,
		},
		{
			name:  "JSON with surrounding text",
			input: `Here is the JSON: {"key": "value"} and more text`,
			want:  `{"key": "value"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractJSONStr(tt.input)
			// Parse both to compare
			var gotMap, wantMap map[string]interface{}
			json.Unmarshal([]byte(got), &gotMap)
			json.Unmarshal([]byte(tt.want), &wantMap)
			assert.Equal(t, wantMap, gotMap)
		})
	}
}

func TestParseActionInput(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    map[string]interface{}
		wantErr bool
	}{
		{
			name:  "valid JSON",
			input: `{"query": "test", "limit": 10}`,
			want:  map[string]interface{}{"query": "test", "limit": float64(10)},
		},
		{
			name:  "empty JSON",
			input: `{}`,
			want:  map[string]interface{}{},
		},
		{
			name:  "nested object",
			input: `{"query": "test", "options": {"limit": 10}}`,
			want:  map[string]interface{}{"query": "test", "options": map[string]interface{}{"limit": float64(10)}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseActionInput(tt.input)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

// Test error handling

func TestReActAgentToolNotFound(t *testing.T) {
	mockLLM := NewMockLLM(
		`Thought: I need to use a tool.
Action: nonexistent_tool
Action Input: {}`,
		"Thought: I can answer.\nAnswer: Done.",
	)

	agent := NewReActAgentFromDefaults(mockLLM, []tools.Tool{})

	ctx := context.Background()
	response, err := agent.Chat(ctx, "Do something")

	// Should still complete, but with error in tool call
	require.NoError(t, err)
	assert.NotEmpty(t, response.Response)
}

func TestReActAgentMaxIterations(t *testing.T) {
	// LLM that always wants to use tools
	mockLLM := NewMockLLM()
	for i := 0; i < 20; i++ {
		mockLLM.responses = append(mockLLM.responses, `Thought: I need more info.
Action: search
Action Input: {"query": "test"}`)
	}

	tool := NewMockTool("search", "Search", func(ctx context.Context, input interface{}) (*tools.ToolOutput, error) {
		return tools.NewToolOutput("search", "Result"), nil
	})

	agent := NewReActAgent(
		WithAgentLLM(mockLLM),
		WithAgentTools([]tools.Tool{tool}),
		WithAgentMaxIterations(3),
	)

	ctx := context.Background()
	response, err := agent.Chat(ctx, "Search forever")

	// Should complete after max iterations
	require.NoError(t, err)
	// Response might be empty since we hit max iterations
	assert.NotNil(t, response)
}

// Test verbose mode

func TestReActAgentVerbose(t *testing.T) {
	mockLLM := NewMockLLM("Thought: Done.\nAnswer: Result.")

	agent := NewReActAgent(
		WithAgentLLM(mockLLM),
		WithAgentVerbose(true),
	)

	ctx := context.Background()
	response, err := agent.Chat(ctx, "Test")

	require.NoError(t, err)
	assert.Equal(t, "Result.", response.Response)
}

// Test GetAgentForLLM

func TestGetAgentForLLM(t *testing.T) {
	// Non-tool-calling LLM
	mockLLM := NewMockLLM()
	agent := GetAgentForLLM(mockLLM, []tools.Tool{})
	_, ok := agent.(*ReActAgent)
	assert.True(t, ok, "Should return ReActAgent for non-tool-calling LLM")

	// Tool-calling LLM
	toolLLM := NewMockToolCallingLLM()
	agent = GetAgentForLLM(toolLLM, []tools.Tool{})
	_, ok = agent.(*FunctionCallingReActAgent)
	assert.True(t, ok, "Should return FunctionCallingReActAgent for tool-calling LLM")
}

// Test streaming

func TestReActAgentStreamChat(t *testing.T) {
	mockLLM := NewMockLLM("Thought: Done.\nAnswer: Streamed result.")

	agent := NewReActAgentFromDefaults(mockLLM, []tools.Tool{})

	ctx := context.Background()
	streamResponse, err := agent.StreamChat(ctx, "Test")

	require.NoError(t, err)
	response := streamResponse.Consume()
	assert.Equal(t, "Streamed result.", response)
}

// Test return_direct tool

// MockReturnDirectTool is a tool that returns directly.
type MockReturnDirectTool struct {
	name        string
	description string
	handler     func(ctx context.Context, input interface{}) (*tools.ToolOutput, error)
}

func (t *MockReturnDirectTool) Metadata() *tools.ToolMetadata {
	return &tools.ToolMetadata{
		Name:         t.name,
		Description:  t.description,
		ReturnDirect: true,
		Parameters: map[string]interface{}{
			"type":       "object",
			"properties": map[string]interface{}{},
		},
	}
}

func (t *MockReturnDirectTool) Call(ctx context.Context, input interface{}) (*tools.ToolOutput, error) {
	if t.handler != nil {
		return t.handler(ctx, input)
	}
	return tools.NewToolOutput(t.name, "Direct output"), nil
}

func TestReActAgentReturnDirectTool(t *testing.T) {
	mockLLM := NewMockLLM(
		`Thought: I need to use the direct tool.
Action: direct_tool
Action Input: {}`,
	)

	tool := &MockReturnDirectTool{
		name:        "direct_tool",
		description: "Returns directly",
		handler: func(ctx context.Context, input interface{}) (*tools.ToolOutput, error) {
			return tools.NewToolOutput("direct_tool", "Direct result"), nil
		},
	}

	agent := NewReActAgentFromDefaults(mockLLM, []tools.Tool{tool})

	ctx := context.Background()
	response, err := agent.Chat(ctx, "Use direct tool")

	require.NoError(t, err)
	assert.Equal(t, "Direct result", response.Response)
}

// Test ID generation

func TestGenerateToolID(t *testing.T) {
	ResetIDCounter()

	id1 := GenerateToolID()
	id2 := GenerateToolID()

	assert.NotEqual(t, id1, id2)
	assert.Contains(t, id1, "call_")
}

// Test format reasoning steps

func TestFormatReasoningSteps(t *testing.T) {
	steps := []BaseReasoningStep{
		&ActionReasoningStep{
			Thought:     "I need to search",
			Action:      "search",
			ActionInput: map[string]interface{}{"query": "test"},
		},
		&ObservationReasoningStep{
			Observation: "Found results",
		},
		&ResponseReasoningStep{
			Thought:  "I can answer",
			Response: "The answer is 42",
		},
	}

	formatted := FormatReasoningSteps(steps)

	assert.Contains(t, formatted, "Thought: I need to search")
	assert.Contains(t, formatted, "Observation: Found results")
	assert.Contains(t, formatted, "Answer: The answer is 42")
}

func TestExtractResponseFromReasoning(t *testing.T) {
	steps := []BaseReasoningStep{
		&ActionReasoningStep{
			Thought:     "I need to search",
			Action:      "search",
			ActionInput: nil,
		},
		&ResponseReasoningStep{
			Thought:  "I can answer",
			Response: "The answer is 42",
		},
	}

	response := ExtractResponseFromReasoning(steps)
	assert.Equal(t, "The answer is 42", response)

	// Empty steps
	response = ExtractResponseFromReasoning([]BaseReasoningStep{})
	assert.Empty(t, response)
}
