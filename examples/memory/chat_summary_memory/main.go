// Package main demonstrates chat summary memory buffer for long conversations.
// This example corresponds to Python's memory/ChatSummaryMemoryBuffer.ipynb
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/memory"
)

func main() {
	ctx := context.Background()

	// 1. Create LLM for summarization
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("=== Chat Summary Memory Buffer Demo ===")
	fmt.Println("\nLLM initialized for summarization")

	separator := strings.Repeat("=", 60)

	// 2. Create summary memory buffer
	fmt.Println("\n" + separator)
	fmt.Println("=== Basic Summary Memory ===")
	fmt.Println(separator)

	summaryMemory := memory.NewChatSummaryMemoryBuffer(
		memory.WithSummaryLLM(llmInstance),
		memory.WithSummaryTokenLimit(200), // Low limit to trigger summarization
	)

	fmt.Println("\nSummary memory buffer automatically summarizes older messages")
	fmt.Printf("Token limit: %d\n", summaryMemory.TokenLimit())

	// Simulate a long conversation
	conversation := []llm.ChatMessage{
		llm.NewUserMessage("Hi! I'm planning a trip to Japan next month."),
		llm.NewAssistantMessage("That sounds exciting! Japan is a wonderful destination. What cities are you planning to visit?"),
		llm.NewUserMessage("I'm thinking Tokyo, Kyoto, and maybe Osaka. I have about 10 days."),
		llm.NewAssistantMessage("Great choices! For 10 days, I'd suggest 4-5 days in Tokyo, 3 days in Kyoto, and 2 days in Osaka. Tokyo has amazing food, shopping, and modern attractions. Kyoto is perfect for temples and traditional culture. Osaka is known for street food and nightlife."),
		llm.NewUserMessage("What are the must-see places in Tokyo?"),
		llm.NewAssistantMessage("In Tokyo, don't miss: Senso-ji Temple in Asakusa, Shibuya Crossing, Meiji Shrine, Shinjuku for nightlife, Tsukiji/Toyosu fish market, and Akihabara for electronics and anime. Also consider day trips to Mt. Fuji or Nikko."),
		llm.NewUserMessage("What about food recommendations?"),
		llm.NewAssistantMessage("For food in Tokyo: try ramen at Ichiran or Fuunji, sushi at Tsukiji, tempura at Tsunahachi, yakitori in Yurakucho, and don't miss a conveyor belt sushi experience. For Osaka, try takoyaki and okonomiyaki!"),
	}

	fmt.Printf("\nAdding %d messages to conversation...\n", len(conversation))

	for i, msg := range conversation {
		if err := summaryMemory.Put(ctx, msg); err != nil {
			log.Printf("Error adding message %d: %v", i, err)
		}
	}

	// Get messages (triggers summarization if over limit)
	fmt.Println("\nRetrieving messages (may trigger summarization)...")

	history, err := summaryMemory.Get(ctx, "")
	if err != nil {
		log.Printf("Error getting history: %v", err)
	} else {
		fmt.Printf("\nRetrieved %d messages after summarization:\n", len(history))
		for i, msg := range history {
			fmt.Printf("\n  Message %d [%s]:\n", i+1, msg.Role)
			fmt.Printf("    %s\n", truncate(msg.Content, 100))
		}
	}

	// 3. Custom summarization prompt
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom Summarization Prompt ===")
	fmt.Println(separator)

	customPrompt := `You are a conversation summarizer. Create a brief, bullet-point summary of the key information discussed. Focus on:
- Main topics discussed
- Important decisions or preferences mentioned
- Action items or next steps

Keep the summary concise but informative.`

	customSummaryMemory := memory.NewChatSummaryMemoryBuffer(
		memory.WithSummaryLLM(llmInstance),
		memory.WithSummaryTokenLimit(150),
		memory.WithSummarizePrompt(customPrompt),
	)

	fmt.Println("\nUsing custom summarization prompt for bullet-point summaries")

	// Add a technical conversation
	techConversation := []llm.ChatMessage{
		llm.NewUserMessage("I need help setting up a CI/CD pipeline for my Go project."),
		llm.NewAssistantMessage("I can help with that! Are you using GitHub Actions, GitLab CI, or another platform?"),
		llm.NewUserMessage("GitHub Actions. I want to run tests, build, and deploy to AWS."),
		llm.NewAssistantMessage("Great choice! For a Go project with GitHub Actions deploying to AWS, you'll need: 1) A workflow file in .github/workflows, 2) Go setup action, 3) Test and build steps, 4) AWS credentials as secrets, 5) Deployment action for your AWS service (ECS, Lambda, etc.)"),
		llm.NewUserMessage("I'm using ECS. What about Docker?"),
		llm.NewAssistantMessage("For ECS deployment, you'll need to: 1) Build a Docker image, 2) Push to ECR, 3) Update ECS task definition, 4) Deploy to ECS service. I recommend using the aws-actions/amazon-ecr-login and aws-actions/amazon-ecs-deploy-task-definition actions."),
	}

	for _, msg := range techConversation {
		customSummaryMemory.Put(ctx, msg)
	}

	customHistory, err := customSummaryMemory.Get(ctx, "")
	if err != nil {
		log.Printf("Error: %v", err)
	} else {
		fmt.Printf("\nRetrieved %d messages with custom summarization:\n", len(customHistory))
		for i, msg := range customHistory {
			fmt.Printf("\n  Message %d [%s]:\n", i+1, msg.Role)
			// Print full content for summary
			if msg.Role == llm.MessageRoleSystem {
				fmt.Printf("    Summary:\n%s\n", indent(msg.Content, "      "))
			} else {
				fmt.Printf("    %s\n", truncate(msg.Content, 80))
			}
		}
	}

	// 4. Memory with initial token count
	fmt.Println("\n" + separator)
	fmt.Println("=== Accounting for System Prompt Tokens ===")
	fmt.Println(separator)

	tokenAwareMemory := memory.NewChatSummaryMemoryBuffer(
		memory.WithSummaryLLM(llmInstance),
		memory.WithSummaryTokenLimit(300),
		memory.WithCountInitialTokens(true),
	)

	fmt.Println("\nWhen using with a system prompt, account for its tokens")

	// Add conversation
	for _, msg := range conversation[:4] {
		tokenAwareMemory.Put(ctx, msg)
	}

	// Get with initial token count (simulating system prompt tokens)
	systemPromptTokens := 50 // Approximate tokens in system prompt
	fmt.Printf("System prompt tokens: ~%d\n", systemPromptTokens)

	tokenAwareHistory, err := tokenAwareMemory.GetWithInitialTokenCount(ctx, "", systemPromptTokens)
	if err != nil {
		log.Printf("Error: %v", err)
	} else {
		fmt.Printf("Retrieved %d messages accounting for system prompt\n", len(tokenAwareHistory))
	}

	// 5. Comparing with regular buffer
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary vs Regular Buffer ===")
	fmt.Println(separator)

	// Regular buffer
	regularBuffer := memory.NewChatMemoryBuffer(
		memory.WithTokenLimit(200),
	)

	// Summary buffer
	summaryBuffer := memory.NewChatSummaryMemoryBuffer(
		memory.WithSummaryLLM(llmInstance),
		memory.WithSummaryTokenLimit(200),
	)

	// Add same conversation to both
	for _, msg := range conversation {
		regularBuffer.Put(ctx, msg)
		summaryBuffer.Put(ctx, msg)
	}

	regularHistory, _ := regularBuffer.Get(ctx, "")
	summaryHistory, _ := summaryBuffer.Get(ctx, "")

	fmt.Println("\nWith same token limit (200):")
	fmt.Printf("\nRegular Buffer: %d messages (drops old messages)\n", len(regularHistory))
	for i, msg := range regularHistory {
		fmt.Printf("  %d. [%s] %s\n", i+1, msg.Role, truncate(msg.Content, 40))
	}

	fmt.Printf("\nSummary Buffer: %d messages (summarizes old messages)\n", len(summaryHistory))
	for i, msg := range summaryHistory {
		if msg.Role == llm.MessageRoleSystem && strings.Contains(msg.Content, "Japan") {
			fmt.Printf("  %d. [%s] (SUMMARY) %s\n", i+1, msg.Role, truncate(msg.Content, 60))
		} else {
			fmt.Printf("  %d. [%s] %s\n", i+1, msg.Role, truncate(msg.Content, 40))
		}
	}

	// 6. Using with LLM context window
	fmt.Println("\n" + separator)
	fmt.Println("=== Auto-Configure from LLM ===")
	fmt.Println(separator)

	fmt.Println("\nCreate memory buffer that auto-configures based on LLM context window:")

	autoMemory, err := memory.NewChatSummaryMemoryBufferFromDefaults(
		nil,         // No initial history
		llmInstance, // LLM for context window detection
		0,           // 0 = auto-detect from LLM
	)

	if err != nil {
		log.Printf("Error creating auto memory: %v", err)
	} else {
		fmt.Printf("Auto-configured token limit: %d\n", autoMemory.TokenLimit())
		fmt.Println("(Based on 75% of LLM's context window)")
	}

	// 7. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nChatSummaryMemoryBuffer Features:")
	fmt.Println("  - Automatically summarizes older messages when over token limit")
	fmt.Println("  - Preserves recent messages in full detail")
	fmt.Println("  - Maintains conversation context through summaries")
	fmt.Println("  - Customizable summarization prompts")
	fmt.Println()
	fmt.Println("When to Use:")
	fmt.Println("  - Long conversations that exceed context window")
	fmt.Println("  - When historical context is important")
	fmt.Println("  - Multi-turn conversations with complex topics")
	fmt.Println()
	fmt.Println("Configuration Options:")
	fmt.Println("  - WithSummaryLLM: LLM for generating summaries")
	fmt.Println("  - WithSummaryTokenLimit: Max tokens before summarization")
	fmt.Println("  - WithSummarizePrompt: Custom summarization instructions")
	fmt.Println("  - WithCountInitialTokens: Account for system prompt tokens")

	fmt.Println("\n=== Chat Summary Memory Demo Complete ===")
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// indent adds indentation to each line of a string.
func indent(s string, prefix string) string {
	lines := strings.Split(s, "\n")
	for i, line := range lines {
		lines[i] = prefix + line
	}
	return strings.Join(lines, "\n")
}
