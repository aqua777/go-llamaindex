// Package main demonstrates implementing an agent using the workflow system.
// This example corresponds to Python's workflow/function_calling_agent.ipynb
// and workflow/react_agent.ipynb
package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"strings"
	"time"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/tools"
	"github.com/aqua777/go-llamaindex/workflow"
)

func main() {
	ctx := context.Background()

	// 1. Create LLM
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("=== Workflow Agent Demo ===")
	fmt.Println("\nLLM initialized")

	separator := strings.Repeat("=", 70)

	// 2. Create tools
	agentTools := createAgentTools()
	fmt.Printf("\nCreated %d tools:\n", len(agentTools))
	for _, tool := range agentTools {
		meta := tool.Metadata()
		fmt.Printf("  - %s: %s\n", meta.Name, truncate(meta.Description, 40))
	}

	// 3. Create workflow-based agent
	fmt.Println("\n" + separator)
	fmt.Println("=== Workflow-Based Agent ===")
	fmt.Println(separator)

	agentWorkflow := NewAgentWorkflow(llmInstance, agentTools)

	// Test queries
	queries := []string{
		"What is 25 * 4?",
		"What's the weather in Tokyo?",
		"Calculate the square root of 144",
	}

	for _, query := range queries {
		fmt.Printf("\nUser: %s\n", query)

		result, err := agentWorkflow.Run(ctx, query)
		if err != nil {
			log.Printf("Workflow error: %v\n", err)
			continue
		}

		fmt.Printf("Agent: %s\n", result.Response)
		if len(result.ToolCalls) > 0 {
			fmt.Printf("Tools used: %v\n", result.ToolCalls)
		}
	}

	// 4. Multi-step workflow
	fmt.Println("\n" + separator)
	fmt.Println("=== Multi-Step Workflow ===")
	fmt.Println(separator)

	complexQuery := "I need to calculate 15% of 200 and then add 50 to the result"
	fmt.Printf("\nUser: %s\n", complexQuery)

	result, err := agentWorkflow.Run(ctx, complexQuery)
	if err != nil {
		log.Printf("Workflow error: %v\n", err)
	} else {
		fmt.Printf("Agent: %s\n", result.Response)
		fmt.Printf("Steps taken: %d\n", result.Steps)
	}

	// 5. Demonstrate workflow events
	fmt.Println("\n" + separator)
	fmt.Println("=== Workflow Event Flow ===")
	fmt.Println(separator)

	verboseWorkflow := NewVerboseAgentWorkflow(llmInstance, agentTools)

	eventQuery := "What time is it in London and what's the weather there?"
	fmt.Printf("\nUser: %s\n\n", eventQuery)

	result, err = verboseWorkflow.Run(ctx, eventQuery)
	if err != nil {
		log.Printf("Workflow error: %v\n", err)
	} else {
		fmt.Printf("\nFinal Response: %s\n", result.Response)
	}

	// 6. Custom workflow with branching
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom Workflow with Branching ===")
	fmt.Println(separator)

	branchingWorkflow := NewBranchingWorkflow(llmInstance, agentTools)

	branchQueries := []string{
		"Calculate 100 / 4",      // Math branch
		"What's the weather?",    // Info branch
		"Hello, how are you?",    // Direct response branch
	}

	for _, query := range branchQueries {
		fmt.Printf("\nUser: %s\n", query)

		result, err := branchingWorkflow.Run(ctx, query)
		if err != nil {
			log.Printf("Workflow error: %v\n", err)
			continue
		}

		fmt.Printf("Agent: %s\n", result.Response)
		fmt.Printf("Branch taken: %s\n", result.Branch)
	}

	fmt.Println("\n=== Workflow Agent Demo Complete ===")
}

// AgentWorkflowResult holds the result of an agent workflow execution.
type AgentWorkflowResult struct {
	Response  string
	ToolCalls []string
	Steps     int
	Branch    string
}

// AgentWorkflow implements an agent using the workflow system.
type AgentWorkflow struct {
	llm      llm.LLM
	tools    []tools.Tool
	toolMap  map[string]tools.Tool
	workflow *workflow.Workflow
}

// NewAgentWorkflow creates a new workflow-based agent.
func NewAgentWorkflow(llmModel llm.LLM, agentTools []tools.Tool) *AgentWorkflow {
	toolMap := make(map[string]tools.Tool)
	for _, t := range agentTools {
		toolMap[t.Metadata().Name] = t
	}

	aw := &AgentWorkflow{
		llm:     llmModel,
		tools:   agentTools,
		toolMap: toolMap,
	}

	// Create workflow
	aw.workflow = workflow.NewWorkflow(
		workflow.WithWorkflowName("agent_workflow"),
		workflow.WithWorkflowTimeout(60*time.Second),
	)

	return aw
}

// Run executes the agent workflow.
func (aw *AgentWorkflow) Run(ctx context.Context, query string) (*AgentWorkflowResult, error) {
	// Build tool descriptions for the prompt
	var toolDescs []string
	for _, t := range aw.tools {
		meta := t.Metadata()
		toolDescs = append(toolDescs, fmt.Sprintf("- %s: %s", meta.Name, meta.Description))
	}

	// Create prompt with tools
	systemPrompt := fmt.Sprintf(`You are a helpful assistant with access to the following tools:

%s

To use a tool, respond with:
TOOL: <tool_name>
INPUT: <tool_input>

If you don't need a tool, just respond directly.`, strings.Join(toolDescs, "\n"))

	messages := []llm.ChatMessage{
		llm.NewSystemMessage(systemPrompt),
		llm.NewUserMessage(query),
	}

	var toolCalls []string
	steps := 0
	maxSteps := 5

	for steps < maxSteps {
		steps++

		// Get LLM response
		response, err := aw.llm.Chat(ctx, messages)
		if err != nil {
			return nil, fmt.Errorf("LLM error: %w", err)
		}

		// Check if response contains tool call
		if strings.Contains(response, "TOOL:") {
			toolName, toolInput := parseToolCall(response)
			if toolName != "" {
				tool, ok := aw.toolMap[toolName]
				if !ok {
					messages = append(messages, llm.NewAssistantMessage(response))
					messages = append(messages, llm.NewUserMessage(fmt.Sprintf("Error: Tool '%s' not found", toolName)))
					continue
				}

				// Execute tool
				output, err := tool.Call(ctx, toolInput)
				if err != nil {
					messages = append(messages, llm.NewAssistantMessage(response))
					messages = append(messages, llm.NewUserMessage(fmt.Sprintf("Tool error: %v", err)))
					continue
				}

				toolCalls = append(toolCalls, toolName)

				// Add tool result to conversation
				messages = append(messages, llm.NewAssistantMessage(response))
				messages = append(messages, llm.NewUserMessage(fmt.Sprintf("Tool result: %s", output.Content)))
				continue
			}
		}

		// No tool call, return response
		return &AgentWorkflowResult{
			Response:  response,
			ToolCalls: toolCalls,
			Steps:     steps,
		}, nil
	}

	return &AgentWorkflowResult{
		Response:  "Max steps reached",
		ToolCalls: toolCalls,
		Steps:     steps,
	}, nil
}

// VerboseAgentWorkflow is an agent workflow that logs events.
type VerboseAgentWorkflow struct {
	*AgentWorkflow
}

// NewVerboseAgentWorkflow creates a verbose workflow-based agent.
func NewVerboseAgentWorkflow(llmModel llm.LLM, agentTools []tools.Tool) *VerboseAgentWorkflow {
	return &VerboseAgentWorkflow{
		AgentWorkflow: NewAgentWorkflow(llmModel, agentTools),
	}
}

// Run executes the verbose agent workflow.
func (vaw *VerboseAgentWorkflow) Run(ctx context.Context, query string) (*AgentWorkflowResult, error) {
	fmt.Println("[Event] StartEvent: Query received")

	// Build tool descriptions
	var toolDescs []string
	for _, t := range vaw.tools {
		meta := t.Metadata()
		toolDescs = append(toolDescs, fmt.Sprintf("- %s: %s", meta.Name, meta.Description))
	}

	systemPrompt := fmt.Sprintf(`You are a helpful assistant with access to the following tools:

%s

To use a tool, respond with:
TOOL: <tool_name>
INPUT: <tool_input>

If you don't need a tool, just respond directly.`, strings.Join(toolDescs, "\n"))

	messages := []llm.ChatMessage{
		llm.NewSystemMessage(systemPrompt),
		llm.NewUserMessage(query),
	}

	var toolCalls []string
	steps := 0
	maxSteps := 5

	for steps < maxSteps {
		steps++
		fmt.Printf("[Event] LLMCallEvent: Step %d\n", steps)

		response, err := vaw.llm.Chat(ctx, messages)
		if err != nil {
			fmt.Printf("[Event] ErrorEvent: %v\n", err)
			return nil, err
		}

		fmt.Printf("[Event] LLMResponseEvent: Got response\n")

		if strings.Contains(response, "TOOL:") {
			toolName, toolInput := parseToolCall(response)
			if toolName != "" {
				fmt.Printf("[Event] ToolCallEvent: %s\n", toolName)

				tool, ok := vaw.toolMap[toolName]
				if !ok {
					fmt.Printf("[Event] ToolErrorEvent: Tool not found\n")
					messages = append(messages, llm.NewAssistantMessage(response))
					messages = append(messages, llm.NewUserMessage(fmt.Sprintf("Error: Tool '%s' not found", toolName)))
					continue
				}

				output, err := tool.Call(ctx, toolInput)
				if err != nil {
					fmt.Printf("[Event] ToolErrorEvent: %v\n", err)
					messages = append(messages, llm.NewAssistantMessage(response))
					messages = append(messages, llm.NewUserMessage(fmt.Sprintf("Tool error: %v", err)))
					continue
				}

				fmt.Printf("[Event] ToolResultEvent: %s\n", truncate(output.Content, 50))
				toolCalls = append(toolCalls, toolName)

				messages = append(messages, llm.NewAssistantMessage(response))
				messages = append(messages, llm.NewUserMessage(fmt.Sprintf("Tool result: %s", output.Content)))
				continue
			}
		}

		fmt.Println("[Event] StopEvent: Final response ready")
		return &AgentWorkflowResult{
			Response:  response,
			ToolCalls: toolCalls,
			Steps:     steps,
		}, nil
	}

	fmt.Println("[Event] StopEvent: Max steps reached")
	return &AgentWorkflowResult{
		Response:  "Max steps reached",
		ToolCalls: toolCalls,
		Steps:     steps,
	}, nil
}

// BranchingWorkflow demonstrates workflow branching based on query type.
type BranchingWorkflow struct {
	*AgentWorkflow
}

// NewBranchingWorkflow creates a branching workflow.
func NewBranchingWorkflow(llmModel llm.LLM, agentTools []tools.Tool) *BranchingWorkflow {
	return &BranchingWorkflow{
		AgentWorkflow: NewAgentWorkflow(llmModel, agentTools),
	}
}

// Run executes the branching workflow.
func (bw *BranchingWorkflow) Run(ctx context.Context, query string) (*AgentWorkflowResult, error) {
	// Classify the query
	branch := classifyQuery(query)

	switch branch {
	case "math":
		return bw.handleMathBranch(ctx, query)
	case "info":
		return bw.handleInfoBranch(ctx, query)
	default:
		return bw.handleDirectBranch(ctx, query)
	}
}

func (bw *BranchingWorkflow) handleMathBranch(ctx context.Context, query string) (*AgentWorkflowResult, error) {
	// Use calculator tool
	for _, t := range bw.tools {
		if t.Metadata().Name == "calculator" {
			// Extract numbers and operation from query
			output, err := t.Call(ctx, query)
			if err != nil {
				return &AgentWorkflowResult{
					Response: fmt.Sprintf("Math error: %v", err),
					Branch:   "math",
				}, nil
			}
			return &AgentWorkflowResult{
				Response:  output.Content,
				ToolCalls: []string{"calculator"},
				Branch:    "math",
			}, nil
		}
	}

	// Fallback to LLM
	response, _ := bw.llm.Chat(ctx, []llm.ChatMessage{
		llm.NewUserMessage(query),
	})
	return &AgentWorkflowResult{
		Response: response,
		Branch:   "math",
	}, nil
}

func (bw *BranchingWorkflow) handleInfoBranch(ctx context.Context, query string) (*AgentWorkflowResult, error) {
	// Use info tools (weather, time)
	var toolCalls []string
	var results []string

	for _, t := range bw.tools {
		name := t.Metadata().Name
		if name == "get_weather" || name == "get_time" {
			output, err := t.Call(ctx, map[string]interface{}{"city": "London", "input": query})
			if err == nil {
				toolCalls = append(toolCalls, name)
				results = append(results, output.Content)
			}
		}
	}

	if len(results) > 0 {
		return &AgentWorkflowResult{
			Response:  strings.Join(results, "\n"),
			ToolCalls: toolCalls,
			Branch:    "info",
		}, nil
	}

	response, _ := bw.llm.Chat(ctx, []llm.ChatMessage{
		llm.NewUserMessage(query),
	})
	return &AgentWorkflowResult{
		Response: response,
		Branch:   "info",
	}, nil
}

func (bw *BranchingWorkflow) handleDirectBranch(ctx context.Context, query string) (*AgentWorkflowResult, error) {
	response, err := bw.llm.Chat(ctx, []llm.ChatMessage{
		llm.NewSystemMessage("You are a helpful assistant. Respond directly without using tools."),
		llm.NewUserMessage(query),
	})
	if err != nil {
		return nil, err
	}
	return &AgentWorkflowResult{
		Response: response,
		Branch:   "direct",
	}, nil
}

// classifyQuery determines the type of query.
func classifyQuery(query string) string {
	queryLower := strings.ToLower(query)

	// Math keywords
	mathKeywords := []string{"calculate", "compute", "math", "+", "-", "*", "/", "sum", "multiply", "divide", "square", "sqrt"}
	for _, kw := range mathKeywords {
		if strings.Contains(queryLower, kw) {
			return "math"
		}
	}

	// Info keywords
	infoKeywords := []string{"weather", "time", "temperature", "forecast"}
	for _, kw := range infoKeywords {
		if strings.Contains(queryLower, kw) {
			return "info"
		}
	}

	return "direct"
}

// parseToolCall extracts tool name and input from a response.
func parseToolCall(response string) (string, string) {
	lines := strings.Split(response, "\n")
	var toolName, toolInput string

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "TOOL:") {
			toolName = strings.TrimSpace(strings.TrimPrefix(line, "TOOL:"))
		} else if strings.HasPrefix(line, "INPUT:") {
			toolInput = strings.TrimSpace(strings.TrimPrefix(line, "INPUT:"))
		}
	}

	return toolName, toolInput
}

// createAgentTools creates tools for the workflow agent.
func createAgentTools() []tools.Tool {
	// Calculator tool
	calcTool, _ := tools.NewFunctionToolFromDefaults(
		func(input string) (string, error) {
			// Simple expression evaluation
			return evaluateExpression(input), nil
		},
		"calculator",
		"Perform mathematical calculations. Input should be a math expression.",
	)

	// Weather tool
	weatherTool, _ := tools.NewFunctionToolFromDefaults(
		func(city string) (string, error) {
			weatherData := map[string]string{
				"london": "Cloudy, 55°F",
				"tokyo":  "Clear, 72°F",
				"paris":  "Sunny, 65°F",
			}
			cityLower := strings.ToLower(city)
			if weather, ok := weatherData[cityLower]; ok {
				return fmt.Sprintf("Weather in %s: %s", city, weather), nil
			}
			return fmt.Sprintf("Weather in %s: Unknown", city), nil
		},
		"get_weather",
		"Get weather for a city.",
	)

	// Time tool
	timeTool, _ := tools.NewFunctionToolFromDefaults(
		func(city string) (string, error) {
			offsets := map[string]int{"london": 0, "tokyo": 9, "paris": 1}
			cityLower := strings.ToLower(city)
			offset := offsets[cityLower]
			t := time.Now().UTC().Add(time.Duration(offset) * time.Hour)
			return fmt.Sprintf("Time in %s: %s", city, t.Format("3:04 PM")), nil
		},
		"get_time",
		"Get current time in a city.",
	)

	// Square root tool
	sqrtTool, _ := tools.NewFunctionToolFromDefaults(
		func(n float64) (string, error) {
			if n < 0 {
				return "Error: Cannot take square root of negative number", nil
			}
			return fmt.Sprintf("√%.0f = %.2f", n, math.Sqrt(n)), nil
		},
		"sqrt",
		"Calculate square root of a number.",
	)

	return []tools.Tool{calcTool, weatherTool, timeTool, sqrtTool}
}

// evaluateExpression evaluates a simple math expression.
func evaluateExpression(expr string) string {
	expr = strings.TrimSpace(expr)

	// Handle simple operations
	ops := []struct {
		op   string
		fn   func(a, b float64) float64
	}{
		{"*", func(a, b float64) float64 { return a * b }},
		{"/", func(a, b float64) float64 { if b != 0 { return a / b }; return 0 }},
		{"+", func(a, b float64) float64 { return a + b }},
		{"-", func(a, b float64) float64 { return a - b }},
	}

	for _, op := range ops {
		if strings.Contains(expr, op.op) {
			parts := strings.SplitN(expr, op.op, 2)
			if len(parts) == 2 {
				a := parseFloat(parts[0])
				b := parseFloat(parts[1])
				result := op.fn(a, b)
				return fmt.Sprintf("%.2f", result)
			}
		}
	}

	return "Could not evaluate: " + expr
}

func parseFloat(s string) float64 {
	s = strings.TrimSpace(s)
	var f float64
	fmt.Sscanf(s, "%f", &f)
	return f
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
