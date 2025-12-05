// Package main demonstrates FunctionTool for wrapping Go functions as tools.
// This example corresponds to Python's tools/function_tool_callback.ipynb
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/aqua777/go-llamaindex/tools"
)

// Calculator functions
func add(a, b float64) float64 {
	return a + b
}

func multiply(a, b float64) float64 {
	return a * b
}

func divide(a, b float64) (float64, error) {
	if b == 0 {
		return 0, fmt.Errorf("division by zero")
	}
	return a / b, nil
}

func power(base, exp float64) float64 {
	return math.Pow(base, exp)
}

// String functions
func reverseString(s string) string {
	runes := []rune(s)
	for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
		runes[i], runes[j] = runes[j], runes[i]
	}
	return string(runes)
}

func wordCount(text string) int {
	words := strings.Fields(text)
	return len(words)
}

// Context-aware function
func fetchData(ctx context.Context, query string) (string, error) {
	// Simulate async operation
	select {
	case <-ctx.Done():
		return "", ctx.Err()
	case <-time.After(100 * time.Millisecond):
		return fmt.Sprintf("Results for: %s", query), nil
	}
}

// Struct-based input
type SearchParams struct {
	Query    string `json:"query"`
	MaxItems int    `json:"max_items"`
	Category string `json:"category,omitempty"`
}

func search(params SearchParams) ([]string, error) {
	results := []string{
		fmt.Sprintf("Result 1 for '%s' in %s", params.Query, params.Category),
		fmt.Sprintf("Result 2 for '%s' in %s", params.Query, params.Category),
	}
	if params.MaxItems > 0 && params.MaxItems < len(results) {
		results = results[:params.MaxItems]
	}
	return results, nil
}

// Weather function with multiple returns
type WeatherInfo struct {
	Location    string  `json:"location"`
	Temperature float64 `json:"temperature"`
	Condition   string  `json:"condition"`
}

func getWeather(location string) (*WeatherInfo, error) {
	// Simulated weather data
	return &WeatherInfo{
		Location:    location,
		Temperature: 72.5,
		Condition:   "Sunny",
	}, nil
}

func main() {
	ctx := context.Background()

	fmt.Println("=== Function Tool Demo ===")
	fmt.Println("\nDemonstrates wrapping Go functions as tools for LLM agents.")

	separator := strings.Repeat("=", 60)

	// 1. Basic function tool
	fmt.Println("\n" + separator)
	fmt.Println("=== Basic Function Tool ===")
	fmt.Println(separator)

	addTool, err := tools.NewFunctionTool(add,
		tools.WithFunctionToolName("add"),
		tools.WithFunctionToolDescription("Add two numbers together"),
	)
	if err != nil {
		fmt.Printf("Error creating tool: %v\n", err)
		return
	}

	fmt.Println("\nTool metadata:")
	fmt.Printf("  Name: %s\n", addTool.Metadata().Name)
	fmt.Printf("  Description: %s\n", addTool.Metadata().Description)
	paramsJSON, _ := json.MarshalIndent(addTool.Metadata().Parameters, "  ", "  ")
	fmt.Printf("  Parameters:\n  %s\n", string(paramsJSON))

	// Call the tool
	output, err := addTool.Call(ctx, map[string]interface{}{
		"arg0": 5.0,
		"arg1": 3.0,
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("\nResult: %s\n", output.Content)
		fmt.Printf("Raw output: %v\n", output.RawOutput)
	}

	// 2. Multiple function tools
	fmt.Println("\n" + separator)
	fmt.Println("=== Multiple Function Tools ===")
	fmt.Println(separator)

	multiplyTool, _ := tools.NewFunctionToolFromDefaults(
		multiply,
		"multiply",
		"Multiply two numbers",
	)

	divideTool, _ := tools.NewFunctionToolFromDefaults(
		divide,
		"divide",
		"Divide first number by second number",
	)

	powerTool, _ := tools.NewFunctionToolFromDefaults(
		power,
		"power",
		"Raise base to the power of exponent",
	)

	calculatorTools := []tools.Tool{addTool, multiplyTool, divideTool, powerTool}

	fmt.Println("\nCalculator tools:")
	for _, tool := range calculatorTools {
		fmt.Printf("  - %s: %s\n", tool.Metadata().Name, tool.Metadata().Description)
	}

	// Test each tool
	fmt.Println("\nTesting tools:")

	output, _ = multiplyTool.Call(ctx, map[string]interface{}{"arg0": 4.0, "arg1": 7.0})
	fmt.Printf("  multiply(4, 7) = %s\n", output.Content)

	output, _ = divideTool.Call(ctx, map[string]interface{}{"arg0": 10.0, "arg1": 2.0})
	fmt.Printf("  divide(10, 2) = %s\n", output.Content)

	output, _ = powerTool.Call(ctx, map[string]interface{}{"arg0": 2.0, "arg1": 8.0})
	fmt.Printf("  power(2, 8) = %s\n", output.Content)

	// Test error handling
	output, err = divideTool.Call(ctx, map[string]interface{}{"arg0": 5.0, "arg1": 0.0})
	fmt.Printf("  divide(5, 0) = Error: %v (IsError: %v)\n", err, output.IsError)

	// 3. String function tools
	fmt.Println("\n" + separator)
	fmt.Println("=== String Function Tools ===")
	fmt.Println(separator)

	reverseTool, _ := tools.NewFunctionToolFromDefaults(
		reverseString,
		"reverse_string",
		"Reverse a string",
	)

	wordCountTool, _ := tools.NewFunctionToolFromDefaults(
		wordCount,
		"word_count",
		"Count words in text",
	)

	// String input (single parameter)
	output, _ = reverseTool.Call(ctx, "Hello World")
	fmt.Printf("\nreverse_string('Hello World') = %s\n", output.Content)

	output, _ = wordCountTool.Call(ctx, "The quick brown fox jumps over the lazy dog")
	fmt.Printf("word_count('The quick...') = %s\n", output.Content)

	// 4. Context-aware function
	fmt.Println("\n" + separator)
	fmt.Println("=== Context-Aware Function ===")
	fmt.Println(separator)

	fetchTool, _ := tools.NewFunctionToolFromDefaults(
		fetchData,
		"fetch_data",
		"Fetch data based on query (supports cancellation)",
	)

	fmt.Println("\nFunction with context.Context support:")
	output, _ = fetchTool.Call(ctx, "latest news")
	fmt.Printf("  Result: %s\n", output.Content)

	// Test with timeout
	timeoutCtx, cancel := context.WithTimeout(ctx, 50*time.Millisecond)
	defer cancel()

	output, err = fetchTool.Call(timeoutCtx, "slow query")
	if err != nil {
		fmt.Printf("  With short timeout: %v\n", err)
	}

	// 5. Struct-based input
	fmt.Println("\n" + separator)
	fmt.Println("=== Struct-Based Input ===")
	fmt.Println(separator)

	searchTool, _ := tools.NewFunctionToolFromDefaults(
		search,
		"search",
		"Search for items with filters",
	)

	fmt.Println("\nTool with struct parameter:")
	fmt.Printf("  Parameters schema:\n")
	paramsJSON, _ = json.MarshalIndent(searchTool.Metadata().Parameters, "    ", "  ")
	fmt.Printf("    %s\n", string(paramsJSON))

	output, _ = searchTool.Call(ctx, map[string]interface{}{
		"arg0": map[string]interface{}{
			"query":     "golang",
			"max_items": 2,
			"category":  "programming",
		},
	})
	fmt.Printf("\n  Result: %s\n", output.Content)

	// 6. Tool with complex return type
	fmt.Println("\n" + separator)
	fmt.Println("=== Complex Return Type ===")
	fmt.Println(separator)

	weatherTool, _ := tools.NewFunctionToolFromDefaults(
		getWeather,
		"get_weather",
		"Get current weather for a location",
	)

	output, _ = weatherTool.Call(ctx, "San Francisco")
	fmt.Printf("\nWeather result: %s\n", output.Content)
	if weather, ok := output.RawOutput.(*WeatherInfo); ok {
		fmt.Printf("  Parsed: Location=%s, Temp=%.1f°F, Condition=%s\n",
			weather.Location, weather.Temperature, weather.Condition)
	}

	// 7. ToolSpec for declarative tool creation
	fmt.Println("\n" + separator)
	fmt.Println("=== ToolSpec - Declarative Tool Creation ===")
	fmt.Println(separator)

	spec := &tools.ToolSpec{
		Name:        "calculate_tip",
		Description: "Calculate tip amount for a bill",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"bill_amount": map[string]interface{}{
					"type":        "number",
					"description": "The bill amount in dollars",
				},
				"tip_percent": map[string]interface{}{
					"type":        "number",
					"description": "Tip percentage (e.g., 15, 20)",
				},
			},
			"required": []string{"bill_amount", "tip_percent"},
		},
		Handler: func(ctx context.Context, input map[string]interface{}) (interface{}, error) {
			billAmount, _ := input["bill_amount"].(float64)
			tipPercent, _ := input["tip_percent"].(float64)
			tip := billAmount * (tipPercent / 100)
			return map[string]interface{}{
				"tip_amount": tip,
				"total":      billAmount + tip,
			}, nil
		},
	}

	tipTool := spec.ToTool()

	fmt.Println("\nToolSpec-based tool:")
	fmt.Printf("  Name: %s\n", tipTool.Metadata().Name)
	fmt.Printf("  Description: %s\n", tipTool.Metadata().Description)

	output, _ = tipTool.Call(ctx, map[string]interface{}{
		"bill_amount": 50.0,
		"tip_percent": 20.0,
	})
	fmt.Printf("  Result: %s\n", output.Content)

	// 8. Tool output details
	fmt.Println("\n" + separator)
	fmt.Println("=== Tool Output Details ===")
	fmt.Println(separator)

	output, _ = addTool.Call(ctx, map[string]interface{}{"arg0": 10.0, "arg1": 20.0})

	fmt.Println("\nToolOutput structure:")
	fmt.Printf("  Content: %s\n", output.Content)
	fmt.Printf("  ToolName: %s\n", output.ToolName)
	fmt.Printf("  RawInput: %v\n", output.RawInput)
	fmt.Printf("  RawOutput: %v\n", output.RawOutput)
	fmt.Printf("  IsError: %v\n", output.IsError)

	// 9. OpenAI tool format
	fmt.Println("\n" + separator)
	fmt.Println("=== OpenAI Tool Format ===")
	fmt.Println(separator)

	fmt.Println("\nConverting tool to OpenAI format:")
	openAIFormat := addTool.Metadata().ToOpenAITool()
	openAIJSON, _ := json.MarshalIndent(openAIFormat, "", "  ")
	fmt.Printf("%s\n", string(openAIJSON))

	// 10. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nFunctionTool Features:")
	fmt.Println("  - Wrap any Go function as a tool")
	fmt.Println("  - Automatic parameter schema generation")
	fmt.Println("  - Context support for cancellation/timeout")
	fmt.Println("  - Error handling with IsError flag")
	fmt.Println("  - Struct-based inputs and outputs")
	fmt.Println()
	fmt.Println("Creation Methods:")
	fmt.Println("  - NewFunctionTool: From function with options")
	fmt.Println("  - NewFunctionToolFromDefaults: With explicit name/description")
	fmt.Println("  - ToolSpec.ToTool(): Declarative specification")
	fmt.Println()
	fmt.Println("Use Cases:")
	fmt.Println("  - Agent tool calling")
	fmt.Println("  - Calculator/utility functions")
	fmt.Println("  - API wrappers")
	fmt.Println("  - Data processing pipelines")

	fmt.Println("\n=== Function Tool Demo Complete ===")
}
