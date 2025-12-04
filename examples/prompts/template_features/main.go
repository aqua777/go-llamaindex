// Package main demonstrates rich prompt template features.
// This example corresponds to Python's prompts/rich_prompt_template_features.ipynb
package main

import (
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/prompts"
)

func main() {
	fmt.Println("=== Rich Prompt Template Features Demo ===")
	fmt.Println("\nDemonstrates advanced template features and patterns.")

	separator := strings.Repeat("=", 60)

	// 1. Variable extraction
	fmt.Println("\n" + separator)
	fmt.Println("=== Variable Extraction ===")
	fmt.Println(separator)

	templates := []string{
		"Hello {name}!",
		"The {animal} jumped over the {obstacle}.",
		"{greeting}, {name}! Welcome to {place}.",
		"No variables here.",
		"{var1} and {var2} and {var1} again",
	}

	fmt.Println("\nExtracting variables from templates:")
	for _, tmpl := range templates {
		vars := prompts.GetTemplateVars(tmpl)
		fmt.Printf("  Template: %s\n", tmpl)
		fmt.Printf("  Variables: %v (unique)\n\n", vars)
	}

	// 2. String formatting
	fmt.Println("\n" + separator)
	fmt.Println("=== String Formatting ===")
	fmt.Println(separator)

	template := "Dear {name}, your order #{order_id} for {product} is {status}."

	vars := map[string]string{
		"name":     "Alice",
		"order_id": "12345",
		"product":  "Widget Pro",
		"status":   "shipped",
	}

	formatted := prompts.FormatString(template, vars)
	fmt.Printf("\nTemplate: %s\n", template)
	fmt.Printf("Variables: %v\n", vars)
	fmt.Printf("Result: %s\n", formatted)

	// Partial formatting (only some variables)
	partialVars := map[string]string{
		"name":   "Bob",
		"status": "processing",
	}
	partialFormatted := prompts.FormatString(template, partialVars)
	fmt.Printf("\nPartial formatting with: %v\n", partialVars)
	fmt.Printf("Result: %s\n", partialFormatted)

	// 3. PromptTemplate types
	fmt.Println("\n" + separator)
	fmt.Println("=== Prompt Types ===")
	fmt.Println(separator)

	promptTypes := []prompts.PromptType{
		prompts.PromptTypeSummary,
		prompts.PromptTypeQuestionAnswer,
		prompts.PromptTypeRefine,
		prompts.PromptTypeKeywordExtract,
		prompts.PromptTypeKnowledgeTripletExtract,
		prompts.PromptTypeChoiceSelect,
		prompts.PromptTypeConversation,
		prompts.PromptTypeCustom,
	}

	fmt.Println("\nAvailable prompt types:")
	for _, pt := range promptTypes {
		fmt.Printf("  - %s\n", pt.String())
	}

	// 4. ChatPromptTemplate construction
	fmt.Println("\n" + separator)
	fmt.Println("=== ChatPromptTemplate Construction ===")
	fmt.Println(separator)

	// Method 1: From ChatMessage slice
	chatPrompt1 := prompts.NewChatPromptTemplate(
		[]llm.ChatMessage{
			llm.NewSystemMessage("You are a {role} assistant."),
			llm.NewUserMessage("{user_input}"),
		},
		prompts.PromptTypeConversation,
	)

	fmt.Println("\nMethod 1: From ChatMessage slice")
	fmt.Printf("  Variables: %v\n", chatPrompt1.GetTemplateVars())

	// Method 2: From role-content pairs
	chatPrompt2 := prompts.ChatPromptTemplateFromMessages(
		[]struct {
			Role    llm.MessageRole
			Content string
		}{
			{llm.MessageRoleSystem, "You are an expert in {domain}."},
			{llm.MessageRoleUser, "Question: {question}"},
			{llm.MessageRoleAssistant, "Let me help you with {question}."},
			{llm.MessageRoleUser, "Thanks! Also, {followup}"},
		},
		prompts.PromptTypeConversation,
	)

	fmt.Println("\nMethod 2: From role-content pairs")
	fmt.Printf("  Variables: %v\n", chatPrompt2.GetTemplateVars())

	// Format and display
	messages := chatPrompt2.FormatMessages(map[string]string{
		"domain":   "machine learning",
		"question": "what is gradient descent",
		"followup": "can you give an example",
	})

	fmt.Println("\nFormatted messages:")
	for i, msg := range messages {
		fmt.Printf("  %d. [%s]: %s\n", i+1, msg.Role, truncate(msg.Content, 40))
	}

	// 5. Partial formatting patterns
	fmt.Println("\n" + separator)
	fmt.Println("=== Partial Formatting Patterns ===")
	fmt.Println(separator)

	// Create a template with many variables
	multiVarTemplate := prompts.NewPromptTemplate(
		`System: {system_instruction}
Domain: {domain}
User Level: {user_level}
Language: {language}

Question: {question}

Please provide a {response_style} answer.`,
		prompts.PromptTypeCustom,
	)

	fmt.Println("\nOriginal template variables:", multiVarTemplate.GetTemplateVars())

	// Create specialized versions through partial formatting
	beginnerTemplate := multiVarTemplate.PartialFormat(map[string]string{
		"user_level":     "beginner",
		"response_style": "simple and clear",
	})

	expertTemplate := multiVarTemplate.PartialFormat(map[string]string{
		"user_level":     "expert",
		"response_style": "detailed and technical",
	})

	fmt.Println("\nBeginner template - remaining vars:", beginnerTemplate.GetTemplateVars())
	fmt.Println("Expert template - remaining vars:", expertTemplate.GetTemplateVars())

	// Use the specialized templates
	beginnerFormatted := beginnerTemplate.Format(map[string]string{
		"system_instruction": "Be helpful",
		"domain":             "programming",
		"language":           "English",
		"question":           "What is a variable?",
	})

	fmt.Printf("\nBeginner formatted (preview):\n%s\n", truncate(beginnerFormatted, 100))

	// 6. Template composition
	fmt.Println("\n" + separator)
	fmt.Println("=== Template Composition ===")
	fmt.Println(separator)

	// Build complex prompts from parts
	systemPart := "You are a helpful {role} assistant."
	contextPart := "Context:\n{context}"
	instructionPart := "Instructions: {instructions}"
	queryPart := "Query: {query}"

	composedTemplate := strings.Join([]string{
		systemPart,
		contextPart,
		instructionPart,
		queryPart,
		"Response:",
	}, "\n\n")

	composedPrompt := prompts.NewPromptTemplate(composedTemplate, prompts.PromptTypeQuestionAnswer)

	fmt.Println("\nComposed template variables:", composedPrompt.GetTemplateVars())
	fmt.Printf("Template preview:\n%s\n", truncate(composedTemplate, 150))

	// 7. Metadata usage
	fmt.Println("\n" + separator)
	fmt.Println("=== Metadata Usage ===")
	fmt.Println(separator)

	promptWithMeta := prompts.NewPromptTemplateWithMetadata(
		"Analyze {text} for {analysis_type}.",
		prompts.PromptTypeCustom,
		map[string]interface{}{
			"version":      "2.0",
			"author":       "team-ai",
			"created":      "2024-01-15",
			"tags":         []string{"analysis", "text", "v2"},
			"max_tokens":   500,
			"temperature":  0.7,
			"tested":       true,
			"success_rate": 0.95,
		},
	)

	fmt.Println("\nPrompt with metadata:")
	fmt.Printf("  Template: %s\n", promptWithMeta.GetTemplate())
	fmt.Printf("  Variables: %v\n", promptWithMeta.GetTemplateVars())
	fmt.Println("  Metadata:")
	for k, v := range promptWithMeta.GetMetadata() {
		fmt.Printf("    %s: %v\n", k, v)
	}

	// 8. Default prompts exploration
	fmt.Println("\n" + separator)
	fmt.Println("=== Default Prompts Exploration ===")
	fmt.Println(separator)

	defaultPromptTypes := []struct {
		name string
		pt   prompts.PromptType
	}{
		{"Summary", prompts.PromptTypeSummary},
		{"Tree Summarize", prompts.PromptTypeTreeSummarize},
		{"Text QA", prompts.PromptTypeQuestionAnswer},
		{"Refine", prompts.PromptTypeRefine},
		{"Keyword Extract", prompts.PromptTypeKeywordExtract},
		{"KG Triplet Extract", prompts.PromptTypeKnowledgeTripletExtract},
		{"Choice Select", prompts.PromptTypeChoiceSelect},
	}

	fmt.Println("\nDefault prompts available:")
	for _, dp := range defaultPromptTypes {
		prompt := prompts.GetDefaultPrompt(dp.pt)
		if prompt != nil {
			fmt.Printf("\n  %s:\n", dp.name)
			fmt.Printf("    Variables: %v\n", prompt.GetTemplateVars())
			fmt.Printf("    Preview: %s\n", truncate(prompt.GetTemplate(), 60))
		}
	}

	// 9. Format as string vs messages
	fmt.Println("\n" + separator)
	fmt.Println("=== Format: String vs Messages ===")
	fmt.Println(separator)

	chatTemplate := prompts.NewChatPromptTemplate(
		[]llm.ChatMessage{
			llm.NewSystemMessage("You are {assistant_type}."),
			llm.NewUserMessage("Help me with: {task}"),
		},
		prompts.PromptTypeConversation,
	)

	testVars := map[string]string{
		"assistant_type": "a coding assistant",
		"task":           "writing unit tests",
	}

	// Format as string (concatenated)
	asString := chatTemplate.Format(testVars)
	fmt.Println("\nAs string (concatenated):")
	fmt.Printf("%s\n", asString)

	// Format as messages (structured)
	asMessages := chatTemplate.FormatMessages(testVars)
	fmt.Println("\nAs messages (structured):")
	for _, msg := range asMessages {
		fmt.Printf("  Role: %s\n  Content: %s\n\n", msg.Role, msg.Content)
	}

	// 10. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nRich Template Features:")
	fmt.Println("  - Variable extraction with {var} syntax")
	fmt.Println("  - String and message formatting")
	fmt.Println("  - Partial formatting for template reuse")
	fmt.Println("  - Template composition from parts")
	fmt.Println("  - Metadata attachment for management")
	fmt.Println("  - Multiple prompt types for different use cases")
	fmt.Println("  - ChatPromptTemplate for multi-turn conversations")
	fmt.Println()
	fmt.Println("Best Practices:")
	fmt.Println("  - Use descriptive variable names")
	fmt.Println("  - Add metadata for versioning and tracking")
	fmt.Println("  - Use partial formatting for template variants")
	fmt.Println("  - Choose appropriate prompt types")

	fmt.Println("\n=== Rich Prompt Template Features Demo Complete ===")
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
