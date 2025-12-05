// Package main demonstrates AWS Bedrock LLM and Embedding usage.
// This example shows how to use AWS Bedrock for chat, completion, streaming,
// tool calling, and embeddings with various models (Claude, Nova, Titan, Cohere).
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/llm/bedrock"
)

func main() {
	ctx := context.Background()

	fmt.Println("=== AWS Bedrock LLM & Embeddings Demo ===")
	fmt.Println("\nDemonstrates AWS Bedrock capabilities for LLM and Embeddings.")
	fmt.Println("Requires AWS credentials (via environment, profile, or IAM role).")

	separator := strings.Repeat("=", 60)

	// ========================================
	// PART 1: LLM USAGE
	// ========================================

	fmt.Println("\n" + separator)
	fmt.Println("=== PART 1: LLM USAGE ===")
	fmt.Println(separator)

	// 1. Basic initialization
	fmt.Println("\n" + separator)
	fmt.Println("=== Basic Initialization ===")
	fmt.Println(separator)

	// Default initialization (uses AWS credentials from environment/profile)
	llmInstance := bedrock.New()

	fmt.Println("\nCreated Bedrock LLM with default settings")
	fmt.Printf("  - Model: %s\n", bedrock.DefaultModel)
	fmt.Println("  - Region: from AWS_REGION or AWS_DEFAULT_REGION env var")
	fmt.Println("  - Credentials: from AWS default credential chain")

	// 2. Custom model initialization
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom Model Initialization ===")
	fmt.Println(separator)

	// Using Claude 3 Haiku (fast and cost-effective)
	haikuLLM := bedrock.New(
		bedrock.WithModel(bedrock.Claude3Haiku),
		bedrock.WithMaxTokens(2048),
		bedrock.WithTemperature(0.7),
	)
	fmt.Printf("\nCreated Claude 3 Haiku LLM: %s\n", bedrock.Claude3Haiku)

	// Using Claude 3.5 Sonnet (balanced performance)
	sonnetLLM := bedrock.New(
		bedrock.WithModel(bedrock.Claude35SonnetV2),
		bedrock.WithRegion("us-east-1"),
	)
	fmt.Printf("Created Claude 3.5 Sonnet LLM: %s\n", bedrock.Claude35SonnetV2)

	// Using Amazon Nova Pro (multimodal)
	novaLLM := bedrock.New(
		bedrock.WithModel(bedrock.NovaProV1),
	)
	fmt.Printf("Created Amazon Nova Pro LLM: %s\n", bedrock.NovaProV1)

	// Using Meta Llama 3.3
	llamaLLM := bedrock.New(
		bedrock.WithModel(bedrock.Llama33_70BInstruct),
	)
	fmt.Printf("Created Meta Llama 3.3 LLM: %s\n", bedrock.Llama33_70BInstruct)

	_ = haikuLLM
	_ = sonnetLLM
	_ = novaLLM
	_ = llamaLLM

	// 3. Complete (single prompt)
	fmt.Println("\n" + separator)
	fmt.Println("=== Complete (Single Prompt) ===")
	fmt.Println(separator)

	prompt := "What is Go programming language? Answer in one sentence."
	fmt.Printf("\nPrompt: %s\n", prompt)

	response, err := llmInstance.Complete(ctx, prompt)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		fmt.Println("(Make sure AWS credentials are configured)")
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
	fmt.Printf("  Is Multi-Modal: %v\n", metadata.IsMultiModal)

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
	} else {
		fmt.Println("\nModel does not support tool calling")
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

	// 10. Model comparison
	fmt.Println("\n" + separator)
	fmt.Println("=== Model Comparison ===")
	fmt.Println(separator)

	modelsToCompare := []struct {
		name  string
		model string
	}{
		{"Claude 3.5 Sonnet V2", bedrock.Claude35SonnetV2},
		{"Claude 3 Haiku", bedrock.Claude3Haiku},
		{"Amazon Nova Pro", bedrock.NovaProV1},
		{"Amazon Nova Micro", bedrock.NovaMicroV1},
		{"Meta Llama 3.3 70B", bedrock.Llama33_70BInstruct},
		{"Mistral 7B", bedrock.Mistral7BInstruct},
	}

	fmt.Println("\nModel capabilities comparison:")
	for _, m := range modelsToCompare {
		testLLM := bedrock.New(bedrock.WithModel(m.model))
		meta := testLLM.Metadata()
		fmt.Printf("\n%s (%s):\n", m.name, m.model)
		fmt.Printf("  Context Window: %d\n", meta.ContextWindow)
		fmt.Printf("  Function Calling: %v\n", meta.IsFunctionCalling)
		fmt.Printf("  Multi-Modal: %v\n", meta.IsMultiModal)
	}

	// ========================================
	// PART 2: EMBEDDINGS USAGE
	// ========================================

	fmt.Println("\n" + separator)
	fmt.Println("=== PART 2: EMBEDDINGS USAGE ===")
	fmt.Println(separator)

	// 11. Basic embedding initialization
	fmt.Println("\n" + separator)
	fmt.Println("=== Embedding Initialization ===")
	fmt.Println(separator)

	// Default (Titan Embed Text V2)
	embInstance := bedrock.NewEmbedding()
	fmt.Printf("\nCreated Bedrock Embedding with default model: %s\n", bedrock.DefaultEmbeddingModel)

	// Titan V2 with custom dimensions
	titanEmb := bedrock.NewEmbedding(
		bedrock.WithEmbeddingModel(bedrock.TitanEmbedTextV2),
		bedrock.WithEmbeddingDimensions(512),
		bedrock.WithEmbeddingNormalize(true),
	)
	fmt.Printf("Created Titan V2 Embedding with 512 dimensions\n")

	// Cohere multilingual
	cohereEmb := bedrock.NewEmbedding(
		bedrock.WithEmbeddingModel(bedrock.CohereEmbedMultilingualV3),
	)
	fmt.Printf("Created Cohere Multilingual Embedding: %s\n", bedrock.CohereEmbedMultilingualV3)

	_ = titanEmb
	_ = cohereEmb

	// 12. Single text embedding
	fmt.Println("\n" + separator)
	fmt.Println("=== Single Text Embedding ===")
	fmt.Println(separator)

	text := "Go is a statically typed, compiled programming language."
	fmt.Printf("\nText: %s\n", text)

	embedding, err := embInstance.GetTextEmbedding(ctx, text)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		fmt.Println("(Make sure AWS credentials are configured)")
	} else {
		fmt.Printf("Embedding dimensions: %d\n", len(embedding))
		fmt.Printf("First 5 values: %v\n", embedding[:min(5, len(embedding))])
	}

	// 13. Query embedding
	fmt.Println("\n" + separator)
	fmt.Println("=== Query Embedding ===")
	fmt.Println(separator)

	query := "What is Go programming language?"
	fmt.Printf("\nQuery: %s\n", query)

	queryEmb, err := embInstance.GetQueryEmbedding(ctx, query)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Query embedding dimensions: %d\n", len(queryEmb))
	}

	// 14. Embedding info
	fmt.Println("\n" + separator)
	fmt.Println("=== Embedding Model Info ===")
	fmt.Println(separator)

	info := embInstance.Info()
	fmt.Println("\nEmbedding model info:")
	fmt.Printf("  Model: %s\n", info.ModelName)
	fmt.Printf("  Dimensions: %d\n", info.Dimensions)
	fmt.Printf("  Max Tokens: %d\n", info.MaxTokens)

	// 15. Batch embeddings
	fmt.Println("\n" + separator)
	fmt.Println("=== Batch Embeddings ===")
	fmt.Println(separator)

	texts := []string{
		"Go is great for backend development.",
		"Python is popular for data science.",
		"Rust focuses on memory safety.",
	}

	fmt.Println("\nTexts:")
	for i, t := range texts {
		fmt.Printf("  %d. %s\n", i+1, t)
	}

	batchEmbs, err := embInstance.GetTextEmbeddingsBatch(ctx, texts, func(current, total int) {
		fmt.Printf("Progress: %d/%d\n", current, total)
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("\nGenerated %d embeddings\n", len(batchEmbs))
		for i, emb := range batchEmbs {
			fmt.Printf("  Text %d: %d dimensions\n", i+1, len(emb))
		}
	}

	// 16. Embedding models comparison
	fmt.Println("\n" + separator)
	fmt.Println("=== Embedding Models Comparison ===")
	fmt.Println(separator)

	embModels := []struct {
		name  string
		model string
	}{
		{"Titan Embed V1", bedrock.TitanEmbedTextV1},
		{"Titan Embed V2", bedrock.TitanEmbedTextV2},
		{"Cohere English V3", bedrock.CohereEmbedEnglishV3},
		{"Cohere Multilingual V3", bedrock.CohereEmbedMultilingualV3},
	}

	fmt.Println("\nEmbedding model comparison:")
	for _, m := range embModels {
		testEmb := bedrock.NewEmbedding(bedrock.WithEmbeddingModel(m.model))
		embInfo := testEmb.Info()
		fmt.Printf("\n%s (%s):\n", m.name, m.model)
		fmt.Printf("  Dimensions: %d\n", embInfo.Dimensions)
		fmt.Printf("  Max Tokens: %d\n", embInfo.MaxTokens)
	}

	// ========================================
	// SUMMARY
	// ========================================

	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nAWS Bedrock LLM Features:")
	fmt.Println("  - Complete(): Single prompt completion")
	fmt.Println("  - Chat(): Multi-turn conversation")
	fmt.Println("  - Stream(): Streaming prompt completion")
	fmt.Println("  - StreamChat(): Streaming chat completion")
	fmt.Println("  - ChatWithTools(): Function/tool calling (Claude 3+, Nova, Llama 3.1+)")
	fmt.Println("  - ChatWithFormat(): Structured output")
	fmt.Println("  - Metadata(): Model capabilities info")

	fmt.Println("\nAWS Bedrock Embedding Features:")
	fmt.Println("  - GetTextEmbedding(): Single text embedding")
	fmt.Println("  - GetQueryEmbedding(): Query embedding")
	fmt.Println("  - GetTextEmbeddingsBatch(): Batch embeddings")
	fmt.Println("  - Info(): Embedding model info")

	fmt.Println("\nSupported LLM Models:")
	fmt.Println("  - Anthropic Claude (Instant, V2, 3, 3.5, 3.7, 4)")
	fmt.Println("  - Amazon Nova (Premier, Pro, Lite, Micro)")
	fmt.Println("  - Amazon Titan Text")
	fmt.Println("  - Meta Llama (2, 3, 3.1, 3.2, 3.3)")
	fmt.Println("  - Mistral (7B, Mixtral, Large)")
	fmt.Println("  - Cohere Command")

	fmt.Println("\nSupported Embedding Models:")
	fmt.Println("  - Amazon Titan Embed (V1, V2, G1)")
	fmt.Println("  - Cohere Embed (English V3, Multilingual V3, V4)")

	fmt.Println("\nEnvironment Variables:")
	fmt.Println("  - AWS_REGION / AWS_DEFAULT_REGION: AWS region")
	fmt.Println("  - AWS_ACCESS_KEY_ID: Access key (optional if using IAM role)")
	fmt.Println("  - AWS_SECRET_ACCESS_KEY: Secret key (optional if using IAM role)")
	fmt.Println("  - AWS_SESSION_TOKEN: Session token (optional)")

	fmt.Println("\n=== AWS Bedrock Demo Complete ===")
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
