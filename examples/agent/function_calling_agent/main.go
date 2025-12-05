// Package main demonstrates the function calling agent pattern.
// This example corresponds to Python's agent/openai_agent_with_query_engine.ipynb
// and workflow/function_calling_agent.ipynb
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"strings"
	"time"

	"github.com/aqua777/go-llamaindex/agent"
	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/memory"
	"github.com/aqua777/go-llamaindex/tools"
)

func main() {
	ctx := context.Background()

	// 1. Create LLM (OpenAI with function calling support)
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("=== Function Calling Agent Demo ===")
	fmt.Println("\nLLM initialized (OpenAI with function calling)")

	separator := strings.Repeat("=", 70)

	// 2. Create function tools
	functionTools := createFunctionTools()
	fmt.Printf("\nCreated %d function tools:\n", len(functionTools))
	for _, tool := range functionTools {
		meta := tool.Metadata()
		fmt.Printf("  - %s: %s\n", meta.Name, truncate(meta.Description, 50))
	}

	// 3. Create Function Calling Agent
	fmt.Println("\n" + separator)
	fmt.Println("=== Function Calling Agent ===")
	fmt.Println(separator)

	fcAgent := agent.NewFunctionCallingReActAgent(
		agent.WithAgentLLM(llmInstance),
		agent.WithAgentTools(functionTools),
		agent.WithAgentVerbose(true),
		agent.WithAgentMaxIterations(10),
		agent.WithAgentSystemPrompt(`You are a helpful assistant with access to various tools.
Use the appropriate tool to help answer user questions.
Always explain what you're doing and provide clear answers.`),
	)

	// Test queries
	queries := []string{
		"What's the weather like in San Francisco?",
		"Calculate the compound interest on $1000 at 5% for 3 years",
		"What time is it in Tokyo?",
	}

	for _, query := range queries {
		fmt.Printf("\nUser: %s\n", query)

		response, err := fcAgent.Chat(ctx, query)
		if err != nil {
			log.Printf("Agent error: %v\n", err)
			continue
		}

		fmt.Printf("Agent: %s\n", response.Response)
		if len(response.ToolCalls) > 0 {
			fmt.Printf("Function calls made:\n")
			for _, tc := range response.ToolCalls {
				fmt.Printf("  - %s(%v)\n", tc.ToolName, tc.ToolKwargs)
			}
		}

		// Reset for next query
		fcAgent.Reset(ctx)
	}

	// 4. Multi-function query
	fmt.Println("\n" + separator)
	fmt.Println("=== Multi-Function Query ===")
	fmt.Println(separator)

	multiQuery := "I'm planning a trip to London. What's the weather there and what time is it now?"
	fmt.Printf("\nUser: %s\n", multiQuery)

	response, err := fcAgent.Chat(ctx, multiQuery)
	if err != nil {
		log.Printf("Agent error: %v\n", err)
	} else {
		fmt.Printf("Agent: %s\n", response.Response)
		fmt.Printf("Functions called: %d\n", len(response.ToolCalls))
	}

	// 5. Agent with memory for multi-turn conversation
	fmt.Println("\n" + separator)
	fmt.Println("=== Function Calling with Memory ===")
	fmt.Println(separator)

	chatMemory := memory.NewChatMemoryBuffer()
	memoryAgent := agent.NewFunctionCallingReActAgent(
		agent.WithAgentLLM(llmInstance),
		agent.WithAgentTools(functionTools),
		agent.WithAgentMemory(chatMemory),
		agent.WithAgentSystemPrompt("You are a helpful financial advisor assistant."),
	)

	conversation := []string{
		"I have $5000 to invest",
		"Calculate the return if I invest it at 7% annual interest for 5 years with compound interest",
		"What if I increase the rate to 10%?",
	}

	for _, msg := range conversation {
		fmt.Printf("\nUser: %s\n", msg)

		response, err := memoryAgent.Chat(ctx, msg)
		if err != nil {
			log.Printf("Agent error: %v\n", err)
			continue
		}

		fmt.Printf("Agent: %s\n", response.Response)
	}

	// 6. Automatic agent selection
	fmt.Println("\n" + separator)
	fmt.Println("=== Automatic Agent Selection ===")
	fmt.Println(separator)

	// GetAgentForLLM automatically selects the best agent type based on LLM capabilities
	autoAgent := agent.GetAgentForLLM(llmInstance, functionTools,
		agent.WithAgentVerbose(true),
		agent.WithAgentSystemPrompt("You are a helpful assistant."),
	)

	fmt.Printf("Selected agent type: %T\n", autoAgent)

	autoQuery := "What's 15% of 850?"
	fmt.Printf("\nUser: %s\n", autoQuery)

	response, err = autoAgent.Chat(ctx, autoQuery)
	if err != nil {
		log.Printf("Agent error: %v\n", err)
	} else {
		fmt.Printf("Agent: %s\n", response.Response)
	}

	fmt.Println("\n=== Function Calling Agent Demo Complete ===")
}

// createFunctionTools creates a set of function tools for the agent.
func createFunctionTools() []tools.Tool {
	// Weather tool (mock)
	weatherTool, _ := tools.NewFunctionToolFromDefaults(
		func(city string) (string, error) {
			// Mock weather data
			weatherData := map[string]string{
				"san francisco": "Sunny, 68°F (20°C), Humidity: 65%",
				"new york":      "Cloudy, 55°F (13°C), Humidity: 70%",
				"london":        "Rainy, 50°F (10°C), Humidity: 85%",
				"tokyo":         "Clear, 72°F (22°C), Humidity: 55%",
				"paris":         "Partly cloudy, 60°F (16°C), Humidity: 60%",
			}

			cityLower := strings.ToLower(city)
			if weather, ok := weatherData[cityLower]; ok {
				return fmt.Sprintf("Weather in %s: %s", city, weather), nil
			}
			return fmt.Sprintf("Weather in %s: Sunny, 70°F (21°C), Humidity: 50%% (simulated)", city), nil
		},
		"get_weather",
		"Get the current weather for a city. Input should be the city name.",
		tools.WithFunctionToolParameters(map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"city": map[string]interface{}{
					"type":        "string",
					"description": "The city name to get weather for",
				},
			},
			"required": []string{"city"},
		}),
	)

	// Time zone tool
	timezoneTool, _ := tools.NewFunctionToolFromDefaults(
		func(city string) (string, error) {
			// Mock timezone data
			timezones := map[string]int{
				"san francisco": -8,
				"new york":      -5,
				"london":        0,
				"tokyo":         9,
				"paris":         1,
				"sydney":        11,
			}

			cityLower := strings.ToLower(city)
			offset, ok := timezones[cityLower]
			if !ok {
				offset = 0
			}

			utc := time.Now().UTC()
			localTime := utc.Add(time.Duration(offset) * time.Hour)
			return fmt.Sprintf("Current time in %s: %s (UTC%+d)", city, localTime.Format("3:04 PM"), offset), nil
		},
		"get_time",
		"Get the current time in a specific city. Input should be the city name.",
		tools.WithFunctionToolParameters(map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"city": map[string]interface{}{
					"type":        "string",
					"description": "The city name to get time for",
				},
			},
			"required": []string{"city"},
		}),
	)

	// Compound interest calculator
	interestTool, _ := tools.NewFunctionToolFromDefaults(
		func(principal, rate, years float64) (string, error) {
			// A = P(1 + r/n)^(nt) where n=1 for annual compounding
			amount := principal * math.Pow(1+rate/100, years)
			interest := amount - principal
			return fmt.Sprintf("Principal: $%.2f, Rate: %.2f%%, Years: %.0f\nFinal Amount: $%.2f\nInterest Earned: $%.2f",
				principal, rate, years, amount, interest), nil
		},
		"calculate_compound_interest",
		"Calculate compound interest. Input should be principal amount, annual interest rate (as percentage), and number of years.",
		tools.WithFunctionToolParameters(map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"principal": map[string]interface{}{
					"type":        "number",
					"description": "The initial principal amount in dollars",
				},
				"rate": map[string]interface{}{
					"type":        "number",
					"description": "The annual interest rate as a percentage (e.g., 5 for 5%)",
				},
				"years": map[string]interface{}{
					"type":        "number",
					"description": "The number of years",
				},
			},
			"required": []string{"principal", "rate", "years"},
		}),
	)

	// Percentage calculator
	percentTool, _ := tools.NewFunctionToolFromDefaults(
		func(percent, value float64) (string, error) {
			result := (percent / 100) * value
			return fmt.Sprintf("%.2f%% of %.2f = %.2f", percent, value, result), nil
		},
		"calculate_percentage",
		"Calculate a percentage of a value. Input should be the percentage and the value.",
		tools.WithFunctionToolParameters(map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"percent": map[string]interface{}{
					"type":        "number",
					"description": "The percentage (e.g., 15 for 15%)",
				},
				"value": map[string]interface{}{
					"type":        "number",
					"description": "The value to calculate percentage of",
				},
			},
			"required": []string{"percent", "value"},
		}),
	)

	// Web search tool (mock)
	searchTool, _ := tools.NewFunctionToolFromDefaults(
		func(query string) (string, error) {
			// Mock search results
			return fmt.Sprintf("Search results for '%s':\n1. Result about %s from Wikipedia\n2. News article about %s\n3. Blog post discussing %s",
				query, query, query, query), nil
		},
		"web_search",
		"Search the web for information. Input should be the search query.",
		tools.WithFunctionToolParameters(map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"query": map[string]interface{}{
					"type":        "string",
					"description": "The search query",
				},
			},
			"required": []string{"query"},
		}),
	)

	// HTTP request tool (mock for safety)
	httpTool, _ := tools.NewFunctionToolFromDefaults(
		func(url string) (string, error) {
			// For demo, we'll mock this instead of making real requests
			if strings.Contains(url, "api.") {
				return `{"status": "ok", "data": {"message": "API response (mocked)"}}`, nil
			}
			return fmt.Sprintf("HTTP GET %s: Response received (mocked for demo)", url), nil
		},
		"http_get",
		"Make an HTTP GET request to a URL. Input should be the URL.",
		tools.WithFunctionToolParameters(map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"url": map[string]interface{}{
					"type":        "string",
					"description": "The URL to request",
				},
			},
			"required": []string{"url"},
		}),
	)

	return []tools.Tool{
		weatherTool,
		timezoneTool,
		interestTool,
		percentTool,
		searchTool,
		httpTool,
	}
}

// makeHTTPRequest makes an actual HTTP request (not used in demo for safety).
func makeHTTPRequest(url string) (string, error) {
	resp, err := http.Get(url)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return fmt.Sprintf("Status: %s", resp.Status), nil
	}

	data, _ := json.MarshalIndent(result, "", "  ")
	return string(data), nil
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
