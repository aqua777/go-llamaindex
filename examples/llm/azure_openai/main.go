// Package main demonstrates Azure OpenAI LLM usage.
// This example corresponds to Python's customization/llms/AzureOpenAI.ipynb
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

	fmt.Println("=== Azure OpenAI LLM Demo ===")
	fmt.Println("\nDemonstrates Azure OpenAI LLM capabilities.")

	separator := strings.Repeat("=", 60)

	// 1. Basic initialization (from environment variables)
	fmt.Println("\n" + separator)
	fmt.Println("=== Basic Initialization ===")
	fmt.Println(separator)

	// Default initialization (uses environment variables)
	// Required: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
	llmInstance := llm.NewAzureOpenAILLM()

	fmt.Println("\nCreated Azure OpenAI LLM with default settings")
	fmt.Println("  - Endpoint: from AZURE_OPENAI_ENDPOINT environment variable")
	fmt.Println("  - API Key: from AZURE_OPENAI_API_KEY environment variable")
	fmt.Println("  - Deployment: from AZURE_OPENAI_DEPLOYMENT environment variable")
	fmt.Println("  - API Version: 2024-02-15-preview (default)")

	// 2. Initialization with options
	fmt.Println("\n" + separator)
	fmt.Println("=== Initialization with Options ===")
	fmt.Println(separator)

	// Using options pattern
	llmWithOpts := llm.NewAzureOpenAILLM(
		llm.WithAzureDeployment("gpt-4"),
		llm.WithAzureAPIVersion("2024-02-15-preview"),
	)
	_ = llmWithOpts
	fmt.Println("\nCreated Azure OpenAI LLM with options")
	fmt.Println("  - Custom deployment name")
	fmt.Println("  - Custom API version")

	// 3. Explicit configuration
	fmt.Println("\n" + separator)
	fmt.Println("=== Explicit Configuration ===")
	fmt.Println(separator)

	// Using explicit config (useful for programmatic setup)
	explicitLLM := llm.NewAzureOpenAILLMWithConfig(
		"https://your-resource.openai.azure.com/",
		"your-api-key",
		"gpt-35-turbo",
		"2024-02-15-preview",
	)
	_ = explicitLLM
	fmt.Println("\nCreated Azure OpenAI LLM with explicit config")
	fmt.Println("  - Endpoint, API key, deployment, and version all specified")

	// 4. Complete (single prompt)
	fmt.Println("\n" + separator)
	fmt.Println("=== Complete (Single Prompt) ===")
	fmt.Println(separator)

	prompt := "What is Azure OpenAI Service? Answer in one sentence."
	fmt.Printf("\nPrompt: %s\n", prompt)

	response, err := llmInstance.Complete(ctx, prompt)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", response)
	}

	// 5. Chat (multi-turn conversation)
	fmt.Println("\n" + separator)
	fmt.Println("=== Chat (Multi-turn Conversation) ===")
	fmt.Println(separator)

	messages := []llm.ChatMessage{
		llm.NewSystemMessage("You are a helpful Azure cloud assistant. Be concise."),
		llm.NewUserMessage("What are the benefits of using Azure OpenAI?"),
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
	messages = append(messages, llm.NewUserMessage("How does it differ from OpenAI API?"))

	fmt.Println("\nContinuing conversation...")
	fmt.Printf("  [user]: How does it differ from OpenAI API?\n")

	response, err = llmInstance.Chat(ctx, messages)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("\nResponse: %s\n", truncate(response, 200))
	}

	// 6. Streaming
	fmt.Println("\n" + separator)
	fmt.Println("=== Streaming ===")
	fmt.Println(separator)

	streamPrompt := "List 3 key features of Azure OpenAI Service."
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

	// 7. StreamChat
	fmt.Println("\n" + separator)
	fmt.Println("=== StreamChat ===")
	fmt.Println(separator)

	chatMessages := []llm.ChatMessage{
		llm.NewSystemMessage("You are a helpful assistant."),
		llm.NewUserMessage("Explain Azure content filtering briefly."),
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

	// 8. Model metadata
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

	// 9. Tool calling
	fmt.Println("\n" + separator)
	fmt.Println("=== Tool Calling ===")
	fmt.Println(separator)

	if llmInstance.SupportsToolCalling() {
		fmt.Println("\nModel supports tool calling")

		tools := []*llm.ToolMetadata{
			{
				Name:        "get_azure_status",
				Description: "Get the status of an Azure service",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"service_name": map[string]interface{}{
							"type":        "string",
							"description": "Name of the Azure service",
						},
						"region": map[string]interface{}{
							"type":        "string",
							"description": "Azure region",
						},
					},
					"required": []string{"service_name"},
				},
			},
		}

		toolMessages := []llm.ChatMessage{
			llm.NewUserMessage("What's the status of Azure OpenAI in East US?"),
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

	// 10. Structured output (JSON mode)
	fmt.Println("\n" + separator)
	fmt.Println("=== Structured Output (JSON Mode) ===")
	fmt.Println(separator)

	if llmInstance.SupportsStructuredOutput() {
		fmt.Println("\nModel supports structured output")

		jsonMessages := []llm.ChatMessage{
			llm.NewSystemMessage("You are a helpful assistant that outputs JSON."),
			llm.NewUserMessage("List 3 Azure services with their descriptions. Output as JSON array."),
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

	// 11. Azure-specific considerations
	fmt.Println("\n" + separator)
	fmt.Println("=== Azure-Specific Considerations ===")
	fmt.Println(separator)

	fmt.Println("\nAzure OpenAI vs OpenAI API:")
	fmt.Println("  - Uses deployment names instead of model names")
	fmt.Println("  - Requires Azure endpoint and API key")
	fmt.Println("  - Includes content filtering by default")
	fmt.Println("  - Supports private networking (VNet)")
	fmt.Println("  - Regional deployment for data residency")
	fmt.Println("  - Enterprise security and compliance")

	// 12. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nAzure OpenAI LLM Features:")
	fmt.Println("  - Complete(): Single prompt completion")
	fmt.Println("  - Chat(): Multi-turn conversation")
	fmt.Println("  - Stream(): Streaming prompt completion")
	fmt.Println("  - StreamChat(): Streaming chat completion")
	fmt.Println("  - ChatWithTools(): Function/tool calling")
	fmt.Println("  - ChatWithFormat(): JSON mode output")
	fmt.Println("  - Metadata(): Model capabilities info")
	fmt.Println()
	fmt.Println("Initialization Options:")
	fmt.Println("  - NewAzureOpenAILLM(opts...): From env vars with options")
	fmt.Println("  - NewAzureOpenAILLMWithConfig(): Explicit configuration")
	fmt.Println()
	fmt.Println("Environment Variables:")
	fmt.Println("  - AZURE_OPENAI_ENDPOINT: Azure resource endpoint")
	fmt.Println("  - AZURE_OPENAI_API_KEY: API key")
	fmt.Println("  - AZURE_OPENAI_DEPLOYMENT: Deployment name")

	fmt.Println("\n=== Azure OpenAI LLM Demo Complete ===")
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
