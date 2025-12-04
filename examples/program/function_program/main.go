// Package main demonstrates FunctionProgram for structured output using function calling.
// This example corresponds to Python's output_parsing/function_program.ipynb
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

// Movie represents a movie with structured fields.
type Movie struct {
	Title    string   `json:"title" description:"The title of the movie"`
	Director string   `json:"director" description:"The director of the movie"`
	Year     int      `json:"year" description:"The year the movie was released"`
	Genre    string   `json:"genre" description:"The primary genre of the movie"`
	Rating   float64  `json:"rating" description:"Rating out of 10"`
	Cast     []string `json:"cast,omitempty" description:"Main cast members"`
}

// Person represents a person with basic info.
type Person struct {
	Name       string `json:"name" description:"Full name of the person"`
	Age        int    `json:"age" description:"Age in years"`
	Occupation string `json:"occupation" description:"Current occupation"`
	Location   string `json:"location" description:"City or country of residence"`
}

// Analysis represents a text analysis result.
type Analysis struct {
	Sentiment  string   `json:"sentiment" description:"Overall sentiment: positive, negative, or neutral"`
	Topics     []string `json:"topics" description:"Main topics discussed"`
	KeyPoints  []string `json:"key_points" description:"Key points from the text"`
	Confidence float64  `json:"confidence" description:"Confidence score 0-1"`
}

func main() {
	ctx := context.Background()

	// Create LLM
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("=== Function Program Demo ===")
	fmt.Println("\nDemonstrates structured output using LLM function calling.")

	separator := strings.Repeat("=", 60)

	// 1. Basic FunctionProgram with schema
	fmt.Println("\n" + separator)
	fmt.Println("=== Basic FunctionProgram with Schema ===")
	fmt.Println(separator)

	movieSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"title": map[string]interface{}{
				"type":        "string",
				"description": "The title of the movie",
			},
			"director": map[string]interface{}{
				"type":        "string",
				"description": "The director of the movie",
			},
			"year": map[string]interface{}{
				"type":        "integer",
				"description": "The year the movie was released",
			},
			"genre": map[string]interface{}{
				"type":        "string",
				"description": "The primary genre",
			},
		},
		"required": []string{"title", "director", "year", "genre"},
	}

	movieProgram := program.NewFunctionProgramFromSchema(
		llmInstance,
		"extract_movie",
		"Extract movie information from text",
		movieSchema,
	)

	fmt.Println("\nFunction: extract_movie")
	fmt.Println("Description: Extract movie information from text")
	schemaJSON, _ := json.MarshalIndent(movieSchema, "", "  ")
	fmt.Printf("Schema:\n%s\n", string(schemaJSON))

	// Call the program
	fmt.Println("\nCalling with: 'The Matrix was directed by the Wachowskis in 1999. It's a sci-fi action film.'")

	output, err := movieProgram.Call(ctx, map[string]interface{}{
		"input": "The Matrix was directed by the Wachowskis in 1999. It's a sci-fi action film.",
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("\nRaw output: %s\n", truncate(output.RawOutput, 100))
		if output.ParsedOutput != nil {
			parsed, _ := json.MarshalIndent(output.ParsedOutput, "", "  ")
			fmt.Printf("Parsed output:\n%s\n", string(parsed))
		}
	}

	// 2. FunctionProgram from Go type
	fmt.Println("\n" + separator)
	fmt.Println("=== FunctionProgram from Go Type ===")
	fmt.Println(separator)

	personProgram := program.NewFunctionProgramFromType(
		llmInstance,
		"extract_person",
		"Extract person information from text",
		Person{},
	)

	fmt.Println("\nUsing Go struct to define schema:")
	fmt.Println("type Person struct {")
	fmt.Println("    Name       string")
	fmt.Println("    Age        int")
	fmt.Println("    Occupation string")
	fmt.Println("    Location   string")
	fmt.Println("}")

	fmt.Println("\nCalling with: 'John Smith is a 35-year-old software engineer living in San Francisco.'")

	output, err = personProgram.Call(ctx, map[string]interface{}{
		"input": "John Smith is a 35-year-old software engineer living in San Francisco.",
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		if output.ParsedOutput != nil {
			parsed, _ := json.MarshalIndent(output.ParsedOutput, "", "  ")
			fmt.Printf("Parsed output:\n%s\n", string(parsed))
		}
	}

	// 3. FunctionProgram with custom prompt
	fmt.Println("\n" + separator)
	fmt.Println("=== FunctionProgram with Custom Prompt ===")
	fmt.Println(separator)

	analysisSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"sentiment": map[string]interface{}{
				"type": "string",
				"enum": []string{"positive", "negative", "neutral"},
			},
			"topics": map[string]interface{}{
				"type":  "array",
				"items": map[string]interface{}{"type": "string"},
			},
			"key_points": map[string]interface{}{
				"type":  "array",
				"items": map[string]interface{}{"type": "string"},
			},
			"confidence": map[string]interface{}{
				"type":    "number",
				"minimum": 0,
				"maximum": 1,
			},
		},
		"required": []string{"sentiment", "topics", "key_points", "confidence"},
	}

	analysisPrompt := prompts.NewPromptTemplate(
		`Analyze the following text for sentiment, topics, and key points.

Text: {text}

Provide a thorough analysis.`,
		prompts.PromptTypeCustom,
	)

	analysisProgram := program.NewFunctionProgramFromSchema(
		llmInstance,
		"analyze_text",
		"Analyze text for sentiment and topics",
		analysisSchema,
	).WithPrompt(analysisPrompt)

	sampleText := `The new AI features in the latest smartphone are impressive. 
The camera quality has improved significantly, and battery life is excellent. 
However, the price increase is disappointing for many consumers.`

	fmt.Println("\nAnalyzing text with custom prompt...")
	fmt.Printf("Text: %s\n", truncate(sampleText, 80))

	output, err = analysisProgram.Call(ctx, map[string]interface{}{
		"text": sampleText,
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		if output.ParsedOutput != nil {
			parsed, _ := json.MarshalIndent(output.ParsedOutput, "", "  ")
			fmt.Printf("\nAnalysis result:\n%s\n", string(parsed))
		}
	}

	// 4. Using ProgramOutput methods
	fmt.Println("\n" + separator)
	fmt.Println("=== Using ProgramOutput Methods ===")
	fmt.Println(separator)

	movieProgram2 := program.NewFunctionProgramFromType(
		llmInstance,
		"extract_movie_full",
		"Extract complete movie information",
		Movie{},
	)

	output, err = movieProgram2.Call(ctx, map[string]interface{}{
		"input": "Inception (2010) directed by Christopher Nolan stars Leonardo DiCaprio. It's a mind-bending sci-fi thriller rated 8.8/10.",
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Println("\nUsing GetParsedAs to get typed result:")

		var movie Movie
		if err := output.GetParsedAs(&movie); err != nil {
			fmt.Printf("Error parsing: %v\n", err)
		} else {
			fmt.Printf("  Title: %s\n", movie.Title)
			fmt.Printf("  Director: %s\n", movie.Director)
			fmt.Printf("  Year: %d\n", movie.Year)
			fmt.Printf("  Genre: %s\n", movie.Genre)
			fmt.Printf("  Rating: %.1f\n", movie.Rating)
			if len(movie.Cast) > 0 {
				fmt.Printf("  Cast: %v\n", movie.Cast)
			}
		}

		fmt.Printf("\nMetadata: %v\n", output.Metadata)
	}

	// 5. Complex nested schema
	fmt.Println("\n" + separator)
	fmt.Println("=== Complex Nested Schema ===")
	fmt.Println(separator)

	orderSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"order_id": map[string]interface{}{"type": "string"},
			"customer": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"name":  map[string]interface{}{"type": "string"},
					"email": map[string]interface{}{"type": "string"},
				},
			},
			"items": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"product":  map[string]interface{}{"type": "string"},
						"quantity": map[string]interface{}{"type": "integer"},
						"price":    map[string]interface{}{"type": "number"},
					},
				},
			},
			"total": map[string]interface{}{"type": "number"},
		},
	}

	orderProgram := program.NewFunctionProgramFromSchema(
		llmInstance,
		"extract_order",
		"Extract order information from text",
		orderSchema,
	)

	orderText := "Order #12345 for John Doe (john@example.com): 2x Widget Pro at $29.99 each, 1x Gadget Plus at $49.99. Total: $109.97"

	fmt.Println("\nExtracting nested order data...")
	fmt.Printf("Text: %s\n", orderText)

	output, err = orderProgram.Call(ctx, map[string]interface{}{
		"input": orderText,
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		if output.ParsedOutput != nil {
			parsed, _ := json.MarshalIndent(output.ParsedOutput, "", "  ")
			fmt.Printf("\nExtracted order:\n%s\n", string(parsed))
		}
	}

	// 6. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nFunctionProgram Features:")
	fmt.Println("  - Uses LLM function calling for structured output")
	fmt.Println("  - Schema-based output validation")
	fmt.Println("  - Go type to schema conversion")
	fmt.Println("  - Custom prompt templates")
	fmt.Println("  - Typed output parsing with GetParsedAs")
	fmt.Println()
	fmt.Println("Creation Methods:")
	fmt.Println("  - NewFunctionProgram: Basic with options")
	fmt.Println("  - NewFunctionProgramFromSchema: From JSON schema")
	fmt.Println("  - NewFunctionProgramFromType: From Go struct")
	fmt.Println()
	fmt.Println("Use Cases:")
	fmt.Println("  - Information extraction")
	fmt.Println("  - Data transformation")
	fmt.Println("  - Structured API responses")
	fmt.Println("  - Form filling from text")

	fmt.Println("\n=== Function Program Demo Complete ===")
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
