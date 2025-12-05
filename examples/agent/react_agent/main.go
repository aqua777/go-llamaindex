// Package main demonstrates the ReAct agent pattern with calculator tools.
// This example corresponds to Python's agent/react_agent.ipynb
package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"strconv"
	"strings"

	"github.com/aqua777/go-llamaindex/agent"
	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/memory"
	"github.com/aqua777/go-llamaindex/tools"
)

func main() {
	ctx := context.Background()

	// 1. Create LLM
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("=== ReAct Agent Demo ===")
	fmt.Println("\nLLM initialized")

	// 2. Create calculator tools
	calculatorTools := createCalculatorTools()
	fmt.Printf("\nCreated %d tools:\n", len(calculatorTools))
	for _, tool := range calculatorTools {
		meta := tool.Metadata()
		fmt.Printf("  - %s: %s\n", meta.Name, truncate(meta.Description, 50))
	}

	separator := strings.Repeat("=", 70)

	// 3. Create ReAct Agent
	fmt.Println("\n" + separator)
	fmt.Println("=== Basic ReAct Agent ===")
	fmt.Println(separator)

	reactAgent := agent.NewReActAgentFromDefaults(
		llmInstance,
		calculatorTools,
		agent.WithAgentVerbose(true),
		agent.WithAgentMaxIterations(10),
		agent.WithAgentSystemPrompt("You are a helpful math assistant. Use the available tools to solve math problems step by step."),
	)

	// Test with math problems
	mathProblems := []string{
		"What is 25 + 17?",
		"Calculate 144 divided by 12",
		"What is the square root of 256?",
		"Calculate 15% of 200",
	}

	for _, problem := range mathProblems {
		fmt.Printf("\nProblem: %s\n", problem)

		response, err := reactAgent.Chat(ctx, problem)
		if err != nil {
			log.Printf("Agent error: %v\n", err)
			continue
		}

		fmt.Printf("Answer: %s\n", response.Response)
		if len(response.ToolCalls) > 0 {
			fmt.Printf("Tools used: %d\n", len(response.ToolCalls))
			for _, tc := range response.ToolCalls {
				fmt.Printf("  - %s(%v) = %s\n", tc.ToolName, tc.ToolKwargs, truncate(tc.ToolOutput.Content, 30))
			}
		}

		// Reset for next problem
		reactAgent.Reset(ctx)
	}

	// 4. Multi-step calculation
	fmt.Println("\n" + separator)
	fmt.Println("=== Multi-Step Calculation ===")
	fmt.Println(separator)

	complexProblem := "I have 150 dollars. If I spend 35% on food and then add 20 dollars, how much do I have left?"
	fmt.Printf("\nProblem: %s\n", complexProblem)

	response, err := reactAgent.Chat(ctx, complexProblem)
	if err != nil {
		log.Printf("Agent error: %v\n", err)
	} else {
		fmt.Printf("Answer: %s\n", response.Response)
		fmt.Printf("Reasoning steps: %d\n", response.Metadata["iterations"])
	}

	// 5. Agent with memory
	fmt.Println("\n" + separator)
	fmt.Println("=== Agent with Memory ===")
	fmt.Println(separator)

	chatMemory := memory.NewChatMemoryBuffer()
	agentWithMemory := agent.NewReActAgentFromDefaults(
		llmInstance,
		calculatorTools,
		agent.WithAgentMemory(chatMemory),
		agent.WithAgentSystemPrompt("You are a helpful math tutor. Remember previous calculations and help the user learn."),
	)

	// Conversation with memory
	conversation := []string{
		"What is 50 times 4?",
		"Now divide that result by 10",
		"Add 25 to the previous result",
	}

	for _, msg := range conversation {
		fmt.Printf("\nUser: %s\n", msg)

		response, err := agentWithMemory.Chat(ctx, msg)
		if err != nil {
			log.Printf("Agent error: %v\n", err)
			continue
		}

		fmt.Printf("Agent: %s\n", response.Response)
	}

	// 6. Show reasoning steps
	fmt.Println("\n" + separator)
	fmt.Println("=== Detailed Reasoning ===")
	fmt.Println(separator)

	verboseAgent := agent.NewReActAgentFromDefaults(
		llmInstance,
		calculatorTools,
		agent.WithAgentVerbose(true),
	)

	detailedProblem := "Calculate the area of a circle with radius 7 (use pi = 3.14159)"
	fmt.Printf("\nProblem: %s\n\n", detailedProblem)

	response, err = verboseAgent.Chat(ctx, detailedProblem)
	if err != nil {
		log.Printf("Agent error: %v\n", err)
	} else {
		fmt.Printf("\nFinal Answer: %s\n", response.Response)

		// Show reasoning steps
		reasoning := verboseAgent.CurrentReasoning()
		fmt.Printf("\nReasoning Steps: %d\n", len(reasoning))
	}

	fmt.Println("\n=== ReAct Agent Demo Complete ===")
}

// createCalculatorTools creates a set of calculator tools.
func createCalculatorTools() []tools.Tool {
	// Add tool
	addTool, _ := tools.NewFunctionToolFromDefaults(
		func(a, b float64) (float64, error) {
			return a + b, nil
		},
		"add",
		"Add two numbers together. Input should be two numbers: a and b.",
		tools.WithFunctionToolParameters(map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"a": map[string]interface{}{"type": "number", "description": "First number"},
				"b": map[string]interface{}{"type": "number", "description": "Second number"},
			},
			"required": []string{"a", "b"},
		}),
	)

	// Subtract tool
	subtractTool, _ := tools.NewFunctionToolFromDefaults(
		func(a, b float64) (float64, error) {
			return a - b, nil
		},
		"subtract",
		"Subtract the second number from the first. Input should be two numbers: a and b.",
		tools.WithFunctionToolParameters(map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"a": map[string]interface{}{"type": "number", "description": "First number"},
				"b": map[string]interface{}{"type": "number", "description": "Second number to subtract"},
			},
			"required": []string{"a", "b"},
		}),
	)

	// Multiply tool
	multiplyTool, _ := tools.NewFunctionToolFromDefaults(
		func(a, b float64) (float64, error) {
			return a * b, nil
		},
		"multiply",
		"Multiply two numbers together. Input should be two numbers: a and b.",
		tools.WithFunctionToolParameters(map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"a": map[string]interface{}{"type": "number", "description": "First number"},
				"b": map[string]interface{}{"type": "number", "description": "Second number"},
			},
			"required": []string{"a", "b"},
		}),
	)

	// Divide tool
	divideTool, _ := tools.NewFunctionToolFromDefaults(
		func(a, b float64) (float64, error) {
			if b == 0 {
				return 0, fmt.Errorf("cannot divide by zero")
			}
			return a / b, nil
		},
		"divide",
		"Divide the first number by the second. Input should be two numbers: a and b.",
		tools.WithFunctionToolParameters(map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"a": map[string]interface{}{"type": "number", "description": "Dividend (number to divide)"},
				"b": map[string]interface{}{"type": "number", "description": "Divisor (number to divide by)"},
			},
			"required": []string{"a", "b"},
		}),
	)

	// Square root tool
	sqrtTool, _ := tools.NewFunctionToolFromDefaults(
		func(n float64) (float64, error) {
			if n < 0 {
				return 0, fmt.Errorf("cannot take square root of negative number")
			}
			return math.Sqrt(n), nil
		},
		"sqrt",
		"Calculate the square root of a number. Input should be a non-negative number.",
		tools.WithFunctionToolParameters(map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"n": map[string]interface{}{"type": "number", "description": "Number to take square root of"},
			},
			"required": []string{"n"},
		}),
	)

	// Power tool
	powerTool, _ := tools.NewFunctionToolFromDefaults(
		func(base, exponent float64) (float64, error) {
			return math.Pow(base, exponent), nil
		},
		"power",
		"Raise a number to a power. Input should be base and exponent.",
		tools.WithFunctionToolParameters(map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"base":     map[string]interface{}{"type": "number", "description": "Base number"},
				"exponent": map[string]interface{}{"type": "number", "description": "Exponent"},
			},
			"required": []string{"base", "exponent"},
		}),
	)

	// Percentage tool
	percentageTool, _ := tools.NewFunctionToolFromDefaults(
		func(percent, value float64) (float64, error) {
			return (percent / 100) * value, nil
		},
		"percentage",
		"Calculate a percentage of a value. Input should be percent and value.",
		tools.WithFunctionToolParameters(map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"percent": map[string]interface{}{"type": "number", "description": "Percentage (e.g., 15 for 15%)"},
				"value":   map[string]interface{}{"type": "number", "description": "Value to calculate percentage of"},
			},
			"required": []string{"percent", "value"},
		}),
	)

	return []tools.Tool{
		addTool,
		subtractTool,
		multiplyTool,
		divideTool,
		sqrtTool,
		powerTool,
		percentageTool,
	}
}

// parseNumber parses a string to float64.
func parseNumber(s string) (float64, error) {
	s = strings.TrimSpace(s)
	return strconv.ParseFloat(s, 64)
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
