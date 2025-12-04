// Package main demonstrates basic streaming with LLM and synthesizers.
// This example corresponds to Python's customization/streaming/SimpleIndexDemo-streaming.ipynb
package main

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/aqua777/go-llamaindex/llm"
)

func main() {
	ctx := context.Background()

	fmt.Println("=== Basic Streaming Demo ===")
	fmt.Println("\nDemonstrates streaming responses from LLM.")

	separator := strings.Repeat("=", 60)

	// Create LLM
	llmInstance := llm.NewOpenAILLM("", "", "")

	// 1. Basic streaming with Stream()
	fmt.Println("\n" + separator)
	fmt.Println("=== Basic Stream() ===")
	fmt.Println(separator)

	prompt := "Explain what Go programming language is in 3 sentences."

	fmt.Printf("\nPrompt: %s\n", prompt)
	fmt.Println("\nStreaming response:")

	streamChan, err := llmInstance.Stream(ctx, prompt)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		var fullResponse strings.Builder
		for token := range streamChan {
			fmt.Print(token)
			fullResponse.WriteString(token)
		}
		fmt.Printf("\n\nFull response length: %d characters\n", fullResponse.Len())
	}

	// 2. StreamChat with messages
	fmt.Println("\n" + separator)
	fmt.Println("=== StreamChat with Messages ===")
	fmt.Println(separator)

	// OpenAILLM implements StreamChat directly
	{
		messages := []llm.ChatMessage{
			llm.NewSystemMessage("You are a helpful programming assistant. Be concise."),
			llm.NewUserMessage("What are goroutines in Go?"),
		}

		fmt.Println("\nMessages:")
		for _, msg := range messages {
			fmt.Printf("  [%s]: %s\n", msg.Role, truncate(msg.Content, 50))
		}

		fmt.Println("\nStreaming response:")

		tokenChan, err := llmInstance.StreamChat(ctx, messages)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
		} else {
			var fullResponse strings.Builder
			for token := range tokenChan {
				fmt.Print(token.Delta)
				fullResponse.WriteString(token.Delta)
				if token.FinishReason != "" {
					fmt.Printf("\n[Finish reason: %s]\n", token.FinishReason)
				}
			}
			fmt.Printf("\nFull response length: %d characters\n", fullResponse.Len())
		}
	}

	// 3. Streaming with token counting
	fmt.Println("\n" + separator)
	fmt.Println("=== Streaming with Token Counting ===")
	fmt.Println(separator)

	prompt2 := "List 5 benefits of using Go for backend development."

	fmt.Printf("\nPrompt: %s\n", prompt2)
	fmt.Println("\nStreaming with token count:")

	streamChan, err = llmInstance.Stream(ctx, prompt2)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		tokenCount := 0
		var fullResponse strings.Builder

		for token := range streamChan {
			fmt.Print(token)
			fullResponse.WriteString(token)
			tokenCount++
		}

		fmt.Printf("\n\nTokens received: %d\n", tokenCount)
		fmt.Printf("Response length: %d characters\n", fullResponse.Len())
	}

	// 4. Streaming with timeout
	fmt.Println("\n" + separator)
	fmt.Println("=== Streaming with Timeout ===")
	fmt.Println(separator)

	// Create a context with timeout
	timeoutCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	prompt3 := "Write a short poem about programming."

	fmt.Printf("\nPrompt: %s\n", prompt3)
	fmt.Println("Timeout: 5 seconds")
	fmt.Println("\nStreaming response:")

	streamChan, err = llmInstance.Stream(timeoutCtx, prompt3)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		for token := range streamChan {
			fmt.Print(token)
		}
		fmt.Println()
	}

	// 5. Streaming with progress indicator
	fmt.Println("\n" + separator)
	fmt.Println("=== Streaming with Progress ===")
	fmt.Println(separator)

	prompt4 := "Explain the concept of channels in Go."

	fmt.Printf("\nPrompt: %s\n", prompt4)
	fmt.Println("\nStreaming with progress indicator:")

	streamChan, err = llmInstance.Stream(ctx, prompt4)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		var fullResponse strings.Builder
		charCount := 0
		startTime := time.Now()

		for token := range streamChan {
			fullResponse.WriteString(token)
			charCount += len(token)

			// Print progress every 100 characters
			if charCount%100 == 0 {
				elapsed := time.Since(startTime)
				fmt.Printf("\r[Progress: %d chars, %.1f chars/sec]", charCount, float64(charCount)/elapsed.Seconds())
			}
		}

		elapsed := time.Since(startTime)
		fmt.Printf("\r[Complete: %d chars in %.2fs, %.1f chars/sec]\n",
			charCount, elapsed.Seconds(), float64(charCount)/elapsed.Seconds())
		fmt.Printf("\nResponse preview: %s\n", truncate(fullResponse.String(), 100))
	}

	// 6. Multi-turn streaming conversation
	fmt.Println("\n" + separator)
	fmt.Println("=== Multi-turn Streaming Conversation ===")
	fmt.Println(separator)

	{
		conversationHistory := []llm.ChatMessage{
			llm.NewSystemMessage("You are a helpful Go programming tutor. Keep responses brief."),
		}

		questions := []string{
			"What is a struct in Go?",
			"How do I create one?",
			"Can structs have methods?",
		}

		for i, question := range questions {
			fmt.Printf("\n--- Turn %d ---\n", i+1)
			fmt.Printf("User: %s\n", question)

			// Add user message
			conversationHistory = append(conversationHistory, llm.NewUserMessage(question))

			fmt.Print("Assistant: ")

			tokenChan, err := llmInstance.StreamChat(ctx, conversationHistory)
			if err != nil {
				fmt.Printf("Error: %v\n", err)
				continue
			}

			var response strings.Builder
			for token := range tokenChan {
				fmt.Print(token.Delta)
				response.WriteString(token.Delta)
			}
			fmt.Println()

			// Add assistant response to history
			conversationHistory = append(conversationHistory, llm.NewAssistantMessage(response.String()))
		}
	}

	// 7. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nStreaming Features:")
	fmt.Println("  - Stream(): Basic prompt streaming")
	fmt.Println("  - StreamChat(): Chat message streaming with StreamToken")
	fmt.Println("  - Token-by-token response delivery")
	fmt.Println("  - Context support for cancellation/timeout")
	fmt.Println()
	fmt.Println("StreamToken Fields:")
	fmt.Println("  - Delta: New content in this token")
	fmt.Println("  - FinishReason: Why generation stopped")
	fmt.Println()
	fmt.Println("Use Cases:")
	fmt.Println("  - Real-time response display")
	fmt.Println("  - Progress indicators")
	fmt.Println("  - Interactive chat interfaces")
	fmt.Println("  - Long-running generations")

	fmt.Println("\n=== Basic Streaming Demo Complete ===")
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
