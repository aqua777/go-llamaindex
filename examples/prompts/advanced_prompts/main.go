// Package main demonstrates advanced prompt template features.
// This example corresponds to Python's prompts/advanced_prompts.ipynb
package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/prompts"
)

func main() {
	ctx := context.Background()

	// Create LLM
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("=== Advanced Prompts Demo ===")
	fmt.Println("\nDemonstrates advanced prompt template features and customization.")

	separator := strings.Repeat("=", 60)

	// 1. Basic PromptTemplate
	fmt.Println("\n" + separator)
	fmt.Println("=== Basic PromptTemplate ===")
	fmt.Println(separator)

	basicPrompt := prompts.NewPromptTemplate(
		`You are a helpful assistant. Answer the following question:

Question: {question}

Answer:`,
		prompts.PromptTypeQuestionAnswer,
	)

	fmt.Println("\nTemplate:")
	fmt.Println(basicPrompt.GetTemplate())
	fmt.Printf("\nTemplate Variables: %v\n", basicPrompt.GetTemplateVars())
	fmt.Printf("Prompt Type: %s\n", basicPrompt.GetPromptType())

	// Format the prompt
	formatted := basicPrompt.Format(map[string]string{
		"question": "What is the capital of France?",
	})
	fmt.Printf("\nFormatted Prompt:\n%s\n", formatted)

	// Use with LLM
	response, err := llmInstance.Complete(ctx, formatted)
	if err != nil {
		fmt.Printf("LLM Error: %v\n", err)
	} else {
		fmt.Printf("\nLLM Response: %s\n", truncate(response, 100))
	}

	// 2. ChatPromptTemplate
	fmt.Println("\n" + separator)
	fmt.Println("=== ChatPromptTemplate ===")
	fmt.Println(separator)

	chatPrompt := prompts.NewChatPromptTemplate(
		[]llm.ChatMessage{
			llm.NewSystemMessage("You are a knowledgeable {role} assistant. Be concise and helpful."),
			llm.NewUserMessage("Question: {question}"),
		},
		prompts.PromptTypeQuestionAnswer,
	)

	fmt.Println("\nChat Template Variables:", chatPrompt.GetTemplateVars())

	// Format as messages
	messages := chatPrompt.FormatMessages(map[string]string{
		"role":     "science",
		"question": "What is photosynthesis?",
	})

	fmt.Println("\nFormatted Messages:")
	for i, msg := range messages {
		fmt.Printf("  %d. [%s]: %s\n", i+1, msg.Role, truncate(msg.Content, 60))
	}

	// Use with LLM chat
	chatResponse, err := llmInstance.Chat(ctx, messages)
	if err != nil {
		fmt.Printf("Chat Error: %v\n", err)
	} else {
		fmt.Printf("\nChat Response: %s\n", truncate(chatResponse, 100))
	}

	// 3. Partial Formatting
	fmt.Println("\n" + separator)
	fmt.Println("=== Partial Formatting ===")
	fmt.Println(separator)

	multiVarPrompt := prompts.NewPromptTemplate(
		`You are a {language} programming expert.

Topic: {topic}
Difficulty: {difficulty}

Explain {topic} in {language} at a {difficulty} level.`,
		prompts.PromptTypeCustom,
	)

	fmt.Println("\nOriginal Variables:", multiVarPrompt.GetTemplateVars())

	// Create a partial template with language pre-filled
	partialPrompt := multiVarPrompt.PartialFormat(map[string]string{
		"language":   "Go",
		"difficulty": "beginner",
	})

	fmt.Println("After partial formatting (language=Go, difficulty=beginner):")
	fmt.Println("  Remaining variables to fill: topic")

	// Now only need to provide topic
	finalPrompt := partialPrompt.Format(map[string]string{
		"topic": "goroutines",
	})
	fmt.Printf("\nFinal Prompt:\n%s\n", finalPrompt)

	// 4. Prompt with Metadata
	fmt.Println("\n" + separator)
	fmt.Println("=== Prompt with Metadata ===")
	fmt.Println(separator)

	metadataPrompt := prompts.NewPromptTemplateWithMetadata(
		`Analyze the following text for {analysis_type}:

Text: {text}

Analysis:`,
		prompts.PromptTypeCustom,
		map[string]interface{}{
			"version":     "1.0",
			"author":      "example",
			"description": "Text analysis prompt",
			"tags":        []string{"analysis", "text"},
		},
	)

	fmt.Println("\nPrompt Metadata:")
	for k, v := range metadataPrompt.GetMetadata() {
		fmt.Printf("  %s: %v\n", k, v)
	}

	// 5. Default Prompts
	fmt.Println("\n" + separator)
	fmt.Println("=== Default Prompts ===")
	fmt.Println(separator)

	fmt.Println("\nAvailable default prompts:")

	defaultPrompts := []struct {
		name       string
		promptType prompts.PromptType
	}{
		{"Summary", prompts.PromptTypeSummary},
		{"Text QA", prompts.PromptTypeQuestionAnswer},
		{"Refine", prompts.PromptTypeRefine},
		{"Keyword Extract", prompts.PromptTypeKeywordExtract},
		{"Knowledge Triplet", prompts.PromptTypeKnowledgeTripletExtract},
		{"Choice Select", prompts.PromptTypeChoiceSelect},
	}

	for _, dp := range defaultPrompts {
		prompt := prompts.GetDefaultPrompt(dp.promptType)
		if prompt != nil {
			vars := prompt.GetTemplateVars()
			fmt.Printf("  - %s: variables=%v\n", dp.name, vars)
		}
	}

	// Use default QA prompt
	fmt.Println("\nUsing Default Text QA Prompt:")
	qaPrompt := prompts.DefaultTextQAPrompt
	qaFormatted := qaPrompt.Format(map[string]string{
		"context_str": "Go is a programming language created at Google.",
		"query_str":   "Who created Go?",
	})
	fmt.Printf("%s\n", truncate(qaFormatted, 200))

	// 6. Custom Prompt for RAG
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom RAG Prompt ===")
	fmt.Println(separator)

	ragPrompt := prompts.NewPromptTemplate(
		`You are a helpful assistant that answers questions based on the provided context.

Context Information:
---------------------
{context}
---------------------

Instructions:
- Only use information from the context above
- If the answer is not in the context, say "I don't have enough information"
- Be concise and accurate
- Cite specific parts of the context when relevant

Question: {question}

Answer:`,
		prompts.PromptTypeQuestionAnswer,
	)

	fmt.Println("Custom RAG Prompt Variables:", ragPrompt.GetTemplateVars())

	ragFormatted := ragPrompt.Format(map[string]string{
		"context":  "LlamaIndex is a data framework for LLM applications. It provides tools for data ingestion, indexing, and querying.",
		"question": "What is LlamaIndex?",
	})

	response, err = llmInstance.Complete(ctx, ragFormatted)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("\nRAG Response: %s\n", truncate(response, 150))
	}

	// 7. Multi-turn Chat Template
	fmt.Println("\n" + separator)
	fmt.Println("=== Multi-turn Chat Template ===")
	fmt.Println(separator)

	multiTurnPrompt := prompts.NewChatPromptTemplate(
		[]llm.ChatMessage{
			llm.NewSystemMessage("You are a helpful coding assistant specializing in {language}."),
			llm.NewUserMessage("I'm working on a {project_type} project."),
			llm.NewAssistantMessage("Great! I'd be happy to help with your {project_type} project in {language}. What do you need assistance with?"),
			llm.NewUserMessage("{user_question}"),
		},
		prompts.PromptTypeConversation,
	)

	fmt.Println("Multi-turn Variables:", multiTurnPrompt.GetTemplateVars())

	multiTurnMessages := multiTurnPrompt.FormatMessages(map[string]string{
		"language":      "Python",
		"project_type":  "web scraping",
		"user_question": "How do I handle rate limiting?",
	})

	fmt.Println("\nFormatted Conversation:")
	for _, msg := range multiTurnMessages {
		fmt.Printf("  [%s]: %s\n", msg.Role, truncate(msg.Content, 50))
	}

	// 8. Template Variable Extraction
	fmt.Println("\n" + separator)
	fmt.Println("=== Template Variable Extraction ===")
	fmt.Println(separator)

	testTemplates := []string{
		"Hello {name}, welcome to {place}!",
		"The {animal} jumped over the {obstacle}.",
		"No variables here.",
		"{a} + {b} = {result}, where {a} appears twice",
	}

	fmt.Println("\nExtracting variables from templates:")
	for _, tmpl := range testTemplates {
		vars := prompts.GetTemplateVars(tmpl)
		fmt.Printf("  Template: %s\n", truncate(tmpl, 40))
		fmt.Printf("  Variables: %v\n\n", vars)
	}

	// 9. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nAdvanced Prompt Features:")
	fmt.Println("  - PromptTemplate: Simple string-based templates")
	fmt.Println("  - ChatPromptTemplate: Multi-message chat templates")
	fmt.Println("  - Partial formatting: Pre-fill some variables")
	fmt.Println("  - Metadata: Attach metadata to prompts")
	fmt.Println("  - Default prompts: Built-in prompts for common tasks")
	fmt.Println("  - Variable extraction: Automatic {var} detection")
	fmt.Println()
	fmt.Println("Use Cases:")
	fmt.Println("  - Custom RAG prompts")
	fmt.Println("  - Multi-turn conversations")
	fmt.Println("  - Domain-specific assistants")
	fmt.Println("  - Prompt versioning and management")

	fmt.Println("\n=== Advanced Prompts Demo Complete ===")
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
