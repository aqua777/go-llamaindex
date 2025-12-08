// Package main demonstrates LLMProgram for structured output using text parsing.
// This example corresponds to Python's output_parsing/llm_program.ipynb
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/program"
	"github.com/aqua777/go-llamaindex/prompts"
)

// Recipe represents a cooking recipe.
type Recipe struct {
	Name         string   `json:"name" description:"Name of the recipe"`
	Cuisine      string   `json:"cuisine" description:"Type of cuisine"`
	PrepTime     int      `json:"prep_time" description:"Preparation time in minutes"`
	CookTime     int      `json:"cook_time" description:"Cooking time in minutes"`
	Servings     int      `json:"servings" description:"Number of servings"`
	Ingredients  []string `json:"ingredients" description:"List of ingredients"`
	Instructions []string `json:"instructions" description:"Step-by-step instructions"`
	Difficulty   string   `json:"difficulty" description:"Difficulty level: easy, medium, hard"`
}

// Summary represents a text summary.
type Summary struct {
	Title      string   `json:"title" description:"Title for the summary"`
	MainPoints []string `json:"main_points" description:"Key points from the text"`
	Conclusion string   `json:"conclusion" description:"Brief conclusion"`
	WordCount  int      `json:"word_count" description:"Approximate word count of original"`
}

// CodeReview represents a code review result.
type CodeReview struct {
	Quality     string   `json:"quality" description:"Overall quality: good, needs_improvement, poor"`
	Issues      []string `json:"issues" description:"List of issues found"`
	Suggestions []string `json:"suggestions" description:"Improvement suggestions"`
	Score       int      `json:"score" description:"Score out of 100"`
}

func main() {
	ctx := context.Background()

	// Create LLM
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("=== LLM Program Demo ===")
	fmt.Println("\nDemonstrates structured output using LLM text generation with parsing.")

	separator := strings.Repeat("=", 60)

	// 1. Basic LLMProgram with JSON parser
	fmt.Println("\n" + separator)
	fmt.Println("=== Basic LLMProgram with JSON Parser ===")
	fmt.Println(separator)

	jsonParser := program.NewJSONOutputParser()

	basicProgram := program.NewLLMProgramWithParser(llmInstance, jsonParser).
		WithPrompt(prompts.NewPromptTemplate(
			`Extract the following information as JSON:
- name: person's name
- age: person's age
- city: city they live in

Text: {text}`,
			prompts.PromptTypeCustom,
		))

	fmt.Println("\nUsing JSONOutputParser for generic JSON extraction...")

	output, err := basicProgram.Call(ctx, map[string]interface{}{
		"text": "Sarah Johnson, 28, recently moved to Seattle for her new job.",
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Raw output: %s\n", truncate(output.RawOutput, 100))
		if output.ParsedOutput != nil {
			parsed, _ := json.MarshalIndent(output.ParsedOutput, "", "  ")
			fmt.Printf("Parsed output:\n%s\n", string(parsed))
		}
	}

	// 2. LLMProgram with Pydantic-style parser
	fmt.Println("\n" + separator)
	fmt.Println("=== LLMProgram with Type-Safe Parser ===")
	fmt.Println(separator)

	recipeParser := program.NewPydanticOutputParser(Recipe{})

	fmt.Println("\nFormat instructions generated from struct:")
	fmt.Printf("%s\n", truncate(recipeParser.GetFormatInstructions(), 200))

	recipeProgram := program.NewLLMProgramWithParser(llmInstance, recipeParser).
		WithPrompt(prompts.NewPromptTemplate(
			`Create a recipe based on the following request:

Request: {request}

Provide complete recipe details.`,
			prompts.PromptTypeCustom,
		))

	fmt.Println("\nGenerating a recipe...")

	output, err = recipeProgram.Call(ctx, map[string]interface{}{
		"request": "A simple pasta dish with tomatoes and basil that serves 4 people",
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		if output.ParsedOutput != nil {
			parsed, _ := json.MarshalIndent(output.ParsedOutput, "", "  ")
			fmt.Printf("Generated recipe:\n%s\n", string(parsed))
		}
	}

	// 3. LLMProgram for summarization
	fmt.Println("\n" + separator)
	fmt.Println("=== LLMProgram for Summarization ===")
	fmt.Println(separator)

	summaryProgram := program.NewLLMProgramForType(llmInstance, Summary{}).
		WithPrompt(prompts.NewPromptTemplate(
			`Summarize the following text and provide structured output:

Text:
{text}

Create a comprehensive summary.`,
			prompts.PromptTypeCustom,
		))

	longText := `Artificial intelligence has transformed numerous industries over the past decade. 
In healthcare, AI systems now assist doctors in diagnosing diseases from medical images with remarkable accuracy. 
The financial sector uses AI for fraud detection, algorithmic trading, and customer service automation.
Manufacturing has embraced AI for quality control and predictive maintenance.
However, these advances come with challenges including job displacement concerns, ethical considerations around bias in AI systems, and questions about data privacy.
Experts emphasize the need for responsible AI development and appropriate regulation to ensure these technologies benefit society as a whole.`

	fmt.Println("\nSummarizing text...")
	fmt.Printf("Original text: %s...\n", truncate(longText, 100))

	output, err = summaryProgram.Call(ctx, map[string]interface{}{
		"text": longText,
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		var summary Summary
		if err := output.GetParsedAs(&summary); err == nil {
			fmt.Printf("\nSummary:\n")
			fmt.Printf("  Title: %s\n", summary.Title)
			fmt.Printf("  Main Points:\n")
			for _, point := range summary.MainPoints {
				fmt.Printf("    - %s\n", point)
			}
			fmt.Printf("  Conclusion: %s\n", summary.Conclusion)
			fmt.Printf("  Word Count: ~%d\n", summary.WordCount)
		}
	}

	// 4. LLMProgram for code review
	fmt.Println("\n" + separator)
	fmt.Println("=== LLMProgram for Code Review ===")
	fmt.Println(separator)

	codeReviewProgram := program.NewLLMProgramForType(llmInstance, CodeReview{}).
		WithPrompt(prompts.NewPromptTemplate(
			"Review the following code and provide structured feedback:\n\n"+
				"Language: {language}\n"+
				"Code:\n{code}\n\n"+
				"Analyze for quality, issues, and provide suggestions.",
			prompts.PromptTypeCustom,
		))

	sampleCode := `func processData(data []int) int {
    result := 0
    for i := 0; i < len(data); i++ {
        if data[i] > 0 {
            result = result + data[i]
        }
    }
    return result
}`

	fmt.Println("\nReviewing code...")
	fmt.Printf("Code:\n%s\n", sampleCode)

	output, err = codeReviewProgram.Call(ctx, map[string]interface{}{
		"language": "go",
		"code":     sampleCode,
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		var review CodeReview
		if err := output.GetParsedAs(&review); err == nil {
			fmt.Printf("\nCode Review:\n")
			fmt.Printf("  Quality: %s\n", review.Quality)
			fmt.Printf("  Score: %d/100\n", review.Score)
			fmt.Printf("  Issues:\n")
			for _, issue := range review.Issues {
				fmt.Printf("    - %s\n", issue)
			}
			fmt.Printf("  Suggestions:\n")
			for _, suggestion := range review.Suggestions {
				fmt.Printf("    - %s\n", suggestion)
			}
		}
	}

	// 5. MultiOutputProgram
	fmt.Println("\n" + separator)
	fmt.Println("=== MultiOutputProgram ===")
	fmt.Println(separator)

	ideaSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"title":       map[string]interface{}{"type": "string"},
			"description": map[string]interface{}{"type": "string"},
			"difficulty":  map[string]interface{}{"type": "string"},
		},
	}

	multiParser := program.NewJSONOutputParserWithSchema(ideaSchema)
	multiProgram := program.NewMultiOutputProgram(llmInstance, 3, multiParser)
	multiProgram.Prompt = prompts.NewPromptTemplate(
		`Generate project ideas for: {topic}

Each idea should have a title, description, and difficulty level.`,
		prompts.PromptTypeCustom,
	)

	fmt.Println("\nGenerating 3 project ideas...")

	output, err = multiProgram.Call(ctx, map[string]interface{}{
		"topic": "learning Go programming",
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Raw output: %s\n", truncate(output.RawOutput, 200))
		if output.ParsedOutput != nil {
			parsed, _ := json.MarshalIndent(output.ParsedOutput, "", "  ")
			fmt.Printf("\nParsed ideas:\n%s\n", string(parsed))
		}
	}

	// 6. Custom output parser
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom Output Parser ===")
	fmt.Println(separator)

	// Create a custom schema-based parser
	eventSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"event_name": map[string]interface{}{
				"type":        "string",
				"description": "Name of the event",
			},
			"date": map[string]interface{}{
				"type":        "string",
				"description": "Date in YYYY-MM-DD format",
			},
			"location": map[string]interface{}{
				"type":        "string",
				"description": "Event location",
			},
			"attendees": map[string]interface{}{
				"type":        "integer",
				"description": "Expected number of attendees",
			},
		},
		"required": []string{"event_name", "date", "location"},
	}

	eventParser := program.NewJSONOutputParserWithSchema(eventSchema)

	fmt.Println("\nUsing schema-based parser:")
	fmt.Printf("Format instructions:\n%s\n", truncate(eventParser.GetFormatInstructions(), 150))

	eventProgram := program.NewLLMProgramWithParser(llmInstance, eventParser).
		WithPrompt(prompts.NewPromptTemplate(
			`Extract event information from: {text}`,
			prompts.PromptTypeCustom,
		))

	output, err = eventProgram.Call(ctx, map[string]interface{}{
		"text": "The annual tech conference will be held on 2024-06-15 at the Convention Center. We expect around 500 attendees.",
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		if output.ParsedOutput != nil {
			parsed, _ := json.MarshalIndent(output.ParsedOutput, "", "  ")
			fmt.Printf("\nExtracted event:\n%s\n", string(parsed))
		}
	}

	// 7. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nLLMProgram Features:")
	fmt.Println("  - Text-based structured output (no function calling required)")
	fmt.Println("  - Multiple output parsers (JSON, Pydantic-style)")
	fmt.Println("  - Automatic format instructions")
	fmt.Println("  - Supports structured output mode when available")
	fmt.Println()
	fmt.Println("Output Parsers:")
	fmt.Println("  - JSONOutputParser: Generic JSON parsing")
	fmt.Println("  - JSONOutputParserWithSchema: Schema-validated JSON")
	fmt.Println("  - PydanticOutputParser: Go struct-based parsing")
	fmt.Println()
	fmt.Println("Program Types:")
	fmt.Println("  - LLMProgram: Single structured output")
	fmt.Println("  - MultiOutputProgram: Multiple outputs")
	fmt.Println()
	fmt.Println("Use Cases:")
	fmt.Println("  - Text summarization")
	fmt.Println("  - Information extraction")
	fmt.Println("  - Code review automation")
	fmt.Println("  - Content generation")

	fmt.Println("\n=== LLM Program Demo Complete ===")
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
