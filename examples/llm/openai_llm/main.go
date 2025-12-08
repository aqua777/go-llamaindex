// Package main demonstrates OpenAI LLM usage.
// This example corresponds to Python's customization/llms/SimpleIndexDemo-ChatGPT.ipynb
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
)

func main() {
	ctx := context.Background()

	fmt.Println("=== OpenAI LLM Demo ===")
	fmt.Println("\nDemonstrates OpenAI LLM capabilities.")

	separator := strings.Repeat("=", 60)

	// 1. Basic initialization
	fmt.Println("\n" + separator)
	fmt.Println("=== Basic Initialization ===")
	fmt.Println(separator)

	// Default initialization (uses OPENAI_API_KEY env var)
	llmInstance := llm.NewOpenAILLM("", "", "")

	fmt.Println("\nCreated OpenAI LLM with default settings")
	fmt.Println("  - API Key: from OPENAI_API_KEY environment variable")
	fmt.Println("  - Model: gpt-3.5-turbo (default)")
	fmt.Println("  - Base URL: https://api.openai.com/v1")

	// 2. Custom model initialization
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom Model Initialization ===")
	fmt.Println(separator)

	// Using GPT-4
	gpt4LLM := llm.NewOpenAILLM("", "gpt-4", "")
	fmt.Println("\nCreated GPT-4 LLM")

	// Using GPT-4 Turbo
	gpt4TurboLLM := llm.NewOpenAILLM("", "gpt-4-turbo-preview", "")
	fmt.Println("Created GPT-4 Turbo LLM")

	// Using custom base URL (e.g., for proxies or compatible APIs)
	customLLM := llm.NewOpenAILLM("https://api.example.com/v1", "gpt-3.5-turbo", "")
	_ = customLLM
	fmt.Println("Created LLM with custom base URL")

	// 3. Complete (single prompt)
	fmt.Println("\n" + separator)
	fmt.Println("=== Complete (Single Prompt) ===")
	fmt.Println(separator)

	prompt := "What is Go programming language? Answer in one sentence."
	fmt.Printf("\nPrompt: %s\n", prompt)

	response, err := llmInstance.Complete(ctx, prompt)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", response)
	}

	// 4. Chat (multi-turn conversation)
	fmt.Println("\n" + separator)
	fmt.Println("=== Chat (Multi-turn Conversation) ===")
	fmt.Println(separator)

	messages := []llm.ChatMessage{
		llm.NewSystemMessage("You are a helpful programming assistant. Be concise."),
		llm.NewUserMessage("What is a goroutine?"),
	}

	fmt.Println("\nMessages:")
	for _, msg := range messages {
		fmt.Printf("  [%s]: %s\n", msg.Role, truncate(msg.Content, 60))
	}

	response, err = llmInstance.Chat(ctx, messages)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("\nResponse: %s\n", truncate(response, 200))
	}

	// Continue conversation
	messages = append(messages, llm.NewAssistantMessage(response))
	messages = append(messages, llm.NewUserMessage("How do I create one?"))

	fmt.Println("\nContinuing conversation...")
	fmt.Printf("  [user]: How do I create one?\n")

	response, err = llmInstance.Chat(ctx, messages)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("\nResponse: %s\n", truncate(response, 200))
	}

	// 5. Streaming
	fmt.Println("\n" + separator)
	fmt.Println("=== Streaming ===")
	fmt.Println(separator)

	streamPrompt := "List 3 benefits of using Go for backend development."
	fmt.Printf("\nPrompt: %s\n", streamPrompt)
	fmt.Println("\nStreaming response:")

	streamChan, err := llmInstance.Stream(ctx, streamPrompt)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		for token := range streamChan {
			fmt.Print(token)
		}
		fmt.Println()
	}

	// 6. StreamChat
	fmt.Println("\n" + separator)
	fmt.Println("=== StreamChat ===")
	fmt.Println(separator)

	chatMessages := []llm.ChatMessage{
		llm.NewSystemMessage("You are a helpful assistant."),
		llm.NewUserMessage("Explain channels in Go briefly."),
	}

	fmt.Println("\nStreaming chat response:")

	tokenChan, err := llmInstance.StreamChat(ctx, chatMessages)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		for token := range tokenChan {
			fmt.Print(token.Delta)
		}
		fmt.Println()
	}

	// 7. Model metadata
	fmt.Println("\n" + separator)
	fmt.Println("=== Model Metadata ===")
	fmt.Println(separator)

	metadata := llmInstance.Metadata()
	fmt.Println("\nModel capabilities:")
	fmt.Printf("  Model Name: %s\n", metadata.ModelName)
	fmt.Printf("  Context Window: %d tokens\n", metadata.ContextWindow)
	fmt.Printf("  Max Output Tokens: %d\n", metadata.NumOutputTokens)
	fmt.Printf("  Supports Function Calling: %v\n", metadata.IsFunctionCalling)
	fmt.Printf("  Is Chat Model: %v\n", metadata.IsChat)

	// 8. Tool calling
	fmt.Println("\n" + separator)
	fmt.Println("=== Tool Calling ===")
	fmt.Println(separator)

	if llmInstance.SupportsToolCalling() {
		fmt.Println("\nModel supports tool calling")

		tools := []*llm.ToolMetadata{
			{
				Name:        "get_weather",
				Description: "Get the current weather for a location",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"location": map[string]interface{}{
							"type":        "string",
							"description": "City name",
						},
					},
					"required": []string{"location"},
				},
			},
		}

		toolMessages := []llm.ChatMessage{
			llm.NewUserMessage("What's the weather in San Francisco?"),
		}

		resp, err := llmInstance.ChatWithTools(ctx, toolMessages, tools, nil)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
		} else {
			fmt.Printf("Response text: %s\n", resp.Text)
			if resp.Message != nil && len(resp.Message.Blocks) > 0 {
				fmt.Println("Tool calls:")
				for _, block := range resp.Message.Blocks {
					if block.Type == llm.ContentBlockTypeToolCall {
						tc := block.ToolCall
						fmt.Printf("  - Function: %s\n", tc.Name)
						fmt.Printf("    Arguments: %s\n", tc.Arguments)
					}
				}
			}
		}
	}

	// 9. Structured output (JSON mode)
	fmt.Println("\n" + separator)
	fmt.Println("=== Structured Output (JSON Mode) ===")
	fmt.Println(separator)

	if llmInstance.SupportsStructuredOutput() {
		fmt.Println("\nModel supports structured output")

		jsonMessages := []llm.ChatMessage{
			llm.NewSystemMessage("You are a helpful assistant that outputs JSON."),
			llm.NewUserMessage("List 3 programming languages with their main use cases. Output as JSON array."),
		}

		format := &llm.ResponseFormat{
			Type: "json_object",
		}

		jsonResp, err := llmInstance.ChatWithFormat(ctx, jsonMessages, format)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
		} else {
			fmt.Printf("JSON Response:\n%s\n", jsonResp)

			// Verify it's valid JSON
			var parsed interface{}
			if err := json.Unmarshal([]byte(jsonResp), &parsed); err == nil {
				fmt.Println("(Valid JSON)")
			}
		}
	}

	// 10. Different models comparison
	fmt.Println("\n" + separator)
	fmt.Println("=== Model Comparison ===")
	fmt.Println(separator)

	models := []*llm.OpenAILLM{llmInstance, gpt4LLM, gpt4TurboLLM}
	modelNames := []string{"gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"}

	fmt.Println("\nModel capabilities comparison:")
	for i, model := range models {
		meta := model.Metadata()
		fmt.Printf("\n%s:\n", modelNames[i])
		fmt.Printf("  Context Window: %d\n", meta.ContextWindow)
		fmt.Printf("  Function Calling: %v\n", meta.IsFunctionCalling)
	}

	// 11. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nOpenAI LLM Features:")
	fmt.Println("  - Complete(): Single prompt completion")
	fmt.Println("  - Chat(): Multi-turn conversation")
	fmt.Println("  - Stream(): Streaming prompt completion")
	fmt.Println("  - StreamChat(): Streaming chat completion")
	fmt.Println("  - ChatWithTools(): Function/tool calling")
	fmt.Println("  - ChatWithFormat(): JSON mode output")
	fmt.Println("  - Metadata(): Model capabilities info")
	fmt.Println()
	fmt.Println("Initialization Options:")
	fmt.Println("  - NewOpenAILLM(baseUrl, model, apiKey)")
	fmt.Println("  - NewOpenAILLMWithClient(client, model)")
	fmt.Println()
	fmt.Println("Environment Variables:")
	fmt.Println("  - OPENAI_API_KEY: API key")
	fmt.Println("  - OPENAI_URL: Custom base URL (optional)")

	fmt.Println("\n=== OpenAI LLM Demo Complete ===")
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
