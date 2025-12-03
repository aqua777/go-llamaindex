// Package main demonstrates RAG evaluation metrics.
// This example corresponds to Python's low_level/evaluation.ipynb
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/aqua777/go-llamaindex/evaluation"
	"github.com/aqua777/go-llamaindex/llm"
)

func main() {
	ctx := context.Background()

	// 1. Create LLM for evaluation
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("LLM initialized for evaluation")

	separator := strings.Repeat("=", 60)

	// 2. Faithfulness Evaluation
	fmt.Println("\n" + separator)
	fmt.Println("=== Faithfulness Evaluation ===")
	fmt.Println(separator + "\n")

	faithfulnessEval := evaluation.NewFaithfulnessEvaluator(
		evaluation.WithFaithfulnessLLM(llmInstance),
	)

	// Test case 1: Faithful response
	faithfulInput := evaluation.NewEvaluateInput().
		WithContexts([]string{
			"LlamaIndex is a data framework for LLM applications. It helps connect custom data sources to large language models.",
			"LlamaIndex provides tools for data ingestion, indexing, and querying.",
		}).
		WithResponse("LlamaIndex is a data framework that helps connect custom data to LLMs and provides tools for ingestion and querying.")

	faithfulResult, err := faithfulnessEval.Evaluate(ctx, faithfulInput)
	if err != nil {
		log.Printf("Faithfulness evaluation failed: %v", err)
	} else {
		fmt.Println("Test 1: Faithful Response")
		fmt.Printf("  Response: %s\n", truncate(faithfulInput.Response, 80))
		fmt.Printf("  Passing: %v\n", faithfulResult.IsPassing())
		fmt.Printf("  Score: %.2f\n", faithfulResult.GetScore())
		fmt.Printf("  Feedback: %s\n", faithfulResult.Feedback)
	}

	// Test case 2: Hallucinated response
	hallucinatedInput := evaluation.NewEvaluateInput().
		WithContexts([]string{
			"LlamaIndex is a data framework for LLM applications.",
		}).
		WithResponse("LlamaIndex was founded in 2015 and is headquartered in San Francisco.")

	hallucinatedResult, err := faithfulnessEval.Evaluate(ctx, hallucinatedInput)
	if err != nil {
		log.Printf("Faithfulness evaluation failed: %v", err)
	} else {
		fmt.Println("\nTest 2: Hallucinated Response")
		fmt.Printf("  Response: %s\n", truncate(hallucinatedInput.Response, 80))
		fmt.Printf("  Passing: %v\n", hallucinatedResult.IsPassing())
		fmt.Printf("  Score: %.2f\n", hallucinatedResult.GetScore())
		fmt.Printf("  Feedback: %s\n", hallucinatedResult.Feedback)
	}

	// 3. Relevancy Evaluation
	fmt.Println("\n" + separator)
	fmt.Println("=== Relevancy Evaluation ===")
	fmt.Println(separator + "\n")

	relevancyEval := evaluation.NewRelevancyEvaluator(
		evaluation.WithRelevancyLLM(llmInstance),
	)

	// Test case: Relevant response
	relevantInput := evaluation.NewEvaluateInput().
		WithQuery("What is RAG?").
		WithContexts([]string{
			"RAG stands for Retrieval-Augmented Generation. It combines retrieval and generation to produce better responses.",
		}).
		WithResponse("RAG (Retrieval-Augmented Generation) is a technique that enhances LLM responses by first retrieving relevant information from a knowledge base.")

	relevantResult, err := relevancyEval.Evaluate(ctx, relevantInput)
	if err != nil {
		log.Printf("Relevancy evaluation failed: %v", err)
	} else {
		fmt.Println("Test: Relevant Response")
		fmt.Printf("  Query: %s\n", relevantInput.Query)
		fmt.Printf("  Response: %s\n", truncate(relevantInput.Response, 80))
		fmt.Printf("  Passing: %v\n", relevantResult.IsPassing())
		fmt.Printf("  Score: %.2f\n", relevantResult.GetScore())
	}

	// 4. Correctness Evaluation
	fmt.Println("\n" + separator)
	fmt.Println("=== Correctness Evaluation ===")
	fmt.Println(separator + "\n")

	correctnessEval := evaluation.NewCorrectnessEvaluator(
		evaluation.WithCorrectnessLLM(llmInstance),
	)

	// Test case: Correct response
	correctInput := evaluation.NewEvaluateInput().
		WithQuery("What is the capital of France?").
		WithReference("The capital of France is Paris.").
		WithResponse("Paris is the capital of France.")

	correctResult, err := correctnessEval.Evaluate(ctx, correctInput)
	if err != nil {
		log.Printf("Correctness evaluation failed: %v", err)
	} else {
		fmt.Println("Test 1: Correct Response")
		fmt.Printf("  Query: %s\n", correctInput.Query)
		fmt.Printf("  Reference: %s\n", correctInput.Reference)
		fmt.Printf("  Response: %s\n", correctInput.Response)
		fmt.Printf("  Passing: %v\n", correctResult.IsPassing())
		fmt.Printf("  Score: %.2f\n", correctResult.GetScore())
		fmt.Printf("  Feedback: %s\n", truncate(correctResult.Feedback, 100))
	}

	// Test case: Incorrect response
	incorrectInput := evaluation.NewEvaluateInput().
		WithQuery("What is the capital of France?").
		WithReference("The capital of France is Paris.").
		WithResponse("The capital of France is London.")

	incorrectResult, err := correctnessEval.Evaluate(ctx, incorrectInput)
	if err != nil {
		log.Printf("Correctness evaluation failed: %v", err)
	} else {
		fmt.Println("\nTest 2: Incorrect Response")
		fmt.Printf("  Query: %s\n", incorrectInput.Query)
		fmt.Printf("  Reference: %s\n", incorrectInput.Reference)
		fmt.Printf("  Response: %s\n", incorrectInput.Response)
		fmt.Printf("  Passing: %v\n", incorrectResult.IsPassing())
		fmt.Printf("  Score: %.2f\n", incorrectResult.GetScore())
		fmt.Printf("  Feedback: %s\n", truncate(incorrectResult.Feedback, 100))
	}

	// 5. Context Relevancy Evaluation
	fmt.Println("\n" + separator)
	fmt.Println("=== Context Relevancy Evaluation ===")
	fmt.Println(separator + "\n")

	contextRelevancyEval := evaluation.NewContextRelevancyEvaluator(
		evaluation.WithContextRelevancyLLM(llmInstance),
	)

	contextInput := evaluation.NewEvaluateInput().
		WithQuery("How does photosynthesis work?").
		WithContexts([]string{
			"Photosynthesis is the process by which plants convert sunlight into energy.",
			"Plants use chlorophyll to absorb light energy.",
			"The stock market closed higher today.", // Irrelevant context
		})

	contextResult, err := contextRelevancyEval.Evaluate(ctx, contextInput)
	if err != nil {
		log.Printf("Context relevancy evaluation failed: %v", err)
	} else {
		fmt.Println("Test: Mixed Context Relevancy")
		fmt.Printf("  Query: %s\n", contextInput.Query)
		fmt.Printf("  Contexts: %d provided\n", len(contextInput.Contexts))
		fmt.Printf("  Passing: %v\n", contextResult.IsPassing())
		fmt.Printf("  Score: %.2f (ratio of relevant contexts)\n", contextResult.GetScore())
	}

	// 6. Evaluation Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Evaluation Summary ===")
	fmt.Println(separator + "\n")

	fmt.Println("Evaluation Types:")
	fmt.Println()
	fmt.Println("1. Faithfulness:")
	fmt.Println("   - Checks if response is supported by provided context")
	fmt.Println("   - Detects hallucination")
	fmt.Println()
	fmt.Println("2. Relevancy:")
	fmt.Println("   - Checks if response answers the query")
	fmt.Println("   - Considers both query and context")
	fmt.Println()
	fmt.Println("3. Correctness:")
	fmt.Println("   - Compares response to reference answer")
	fmt.Println("   - Provides 1-5 score")
	fmt.Println()
	fmt.Println("4. Context Relevancy:")
	fmt.Println("   - Evaluates if retrieved contexts are relevant")
	fmt.Println("   - Helps identify retrieval quality issues")

	fmt.Println("\n=== RAG Evaluation Demo Complete ===")
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
