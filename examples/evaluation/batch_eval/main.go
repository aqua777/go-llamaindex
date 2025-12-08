// Package main demonstrates batch evaluation of RAG systems.
// This example corresponds to Python's evaluation/batch_eval.ipynb
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
	fmt.Println("=== Batch Evaluation Demo ===")
	fmt.Println("\nLLM initialized for evaluation")

	separator := strings.Repeat("=", 60)

	// 2. Create evaluators
	faithfulnessEval := evaluation.NewFaithfulnessEvaluator(
		evaluation.WithFaithfulnessLLM(llmInstance),
	)

	relevancyEval := evaluation.NewRelevancyEvaluator(
		evaluation.WithRelevancyLLM(llmInstance),
	)

	correctnessEval := evaluation.NewCorrectnessEvaluator(
		evaluation.WithCorrectnessLLM(llmInstance),
	)

	// 3. Create batch runner with multiple evaluators
	fmt.Println("\n" + separator)
	fmt.Println("=== Setting Up Batch Runner ===")
	fmt.Println(separator)

	evaluators := map[string]evaluation.Evaluator{
		"faithfulness": faithfulnessEval,
		"relevancy":    relevancyEval,
		"correctness":  correctnessEval,
	}

	batchRunner := evaluation.NewBatchEvalRunner(
		evaluators,
		evaluation.WithBatchWorkers(3),
		evaluation.WithBatchShowProgress(true),
	)

	fmt.Printf("\nBatch runner configured with %d evaluators:\n", len(evaluators))
	for name := range evaluators {
		fmt.Printf("  - %s\n", name)
	}

	// 4. Prepare test data
	fmt.Println("\n" + separator)
	fmt.Println("=== Test Data ===")
	fmt.Println(separator)

	queries := []string{
		"What is machine learning?",
		"How does photosynthesis work?",
		"What is the capital of Japan?",
		"Explain quantum computing.",
	}

	responses := []string{
		"Machine learning is a subset of AI that enables systems to learn from data and improve over time without explicit programming.",
		"Photosynthesis is the process by which plants convert sunlight, water, and CO2 into glucose and oxygen using chlorophyll.",
		"The capital of Japan is Tokyo, which is also the largest city in Japan.",
		"Quantum computing uses quantum bits (qubits) that can exist in superposition, enabling parallel computation.",
	}

	contexts := [][]string{
		{"Machine learning is a branch of artificial intelligence. It allows computers to learn from data."},
		{"Plants use photosynthesis to make food. Chlorophyll absorbs light energy."},
		{"Tokyo is Japan's capital city. It has a population of over 13 million."},
		{"Quantum computers use qubits. Qubits can be in multiple states simultaneously."},
	}

	references := []string{
		"Machine learning is an AI technique where computers learn patterns from data.",
		"Photosynthesis converts light energy into chemical energy in plants.",
		"Tokyo is the capital of Japan.",
		"Quantum computing leverages quantum mechanics for computation.",
	}

	fmt.Printf("\nPrepared %d test cases\n", len(queries))
	for i, q := range queries {
		fmt.Printf("  %d. %s\n", i+1, truncate(q, 50))
	}

	// 5. Run batch evaluation
	fmt.Println("\n" + separator)
	fmt.Println("=== Running Batch Evaluation ===")
	fmt.Println(separator)

	fmt.Println("\nEvaluating responses...")

	// Create extra kwargs for references
	extraKwargs := make(map[string][]interface{})
	refInterfaces := make([]interface{}, len(references))
	for i, ref := range references {
		refInterfaces[i] = ref
	}
	extraKwargs["reference"] = refInterfaces

	batchResult, err := batchRunner.EvaluateResponseStrs(
		ctx,
		queries,
		responses,
		contexts,
		extraKwargs,
	)

	if err != nil {
		log.Fatalf("Batch evaluation failed: %v", err)
	}

	// 6. Display results
	fmt.Println("\n" + separator)
	fmt.Println("=== Evaluation Results ===")
	fmt.Println(separator)

	// Per-evaluator results
	for evalName, results := range batchResult.Results {
		fmt.Printf("\n%s Results:\n", strings.Title(evalName))
		for i, result := range results {
			status := "✗"
			if result.IsPassing() {
				status = "✓"
			}
			fmt.Printf("  Query %d: %s Score=%.2f\n", i+1, status, result.GetScore())
		}
	}

	// 7. Summary statistics
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary Statistics ===")
	fmt.Println(separator)

	summary := batchResult.Summary()
	fmt.Println("\nEvaluator Performance:")
	fmt.Printf("%-15s %12s %12s %10s\n", "Evaluator", "Avg Score", "Pass Rate", "Count")
	fmt.Println(strings.Repeat("-", 52))

	for name, stats := range summary {
		fmt.Printf("%-15s %12.2f %11.0f%% %10.0f\n",
			name,
			stats["average_score"],
			stats["passing_rate"]*100,
			stats["total_count"],
		)
	}

	// 8. Detailed per-query analysis
	fmt.Println("\n" + separator)
	fmt.Println("=== Per-Query Analysis ===")
	fmt.Println(separator)

	for i := range queries {
		fmt.Printf("\nQuery %d: %s\n", i+1, truncate(queries[i], 40))
		fmt.Printf("Response: %s\n", truncate(responses[i], 50))

		for evalName, results := range batchResult.Results {
			if i < len(results) {
				result := results[i]
				status := "FAIL"
				if result.IsPassing() {
					status = "PASS"
				}
				fmt.Printf("  %s: %s (%.2f)\n", evalName, status, result.GetScore())
			}
		}
	}

	// 9. Error handling
	if len(batchResult.Errors) > 0 {
		fmt.Println("\n" + separator)
		fmt.Println("=== Errors ===")
		fmt.Println(separator)
		for i, err := range batchResult.Errors {
			fmt.Printf("  Error %d: %v\n", i+1, err)
		}
	}

	// 10. Single input evaluation across all evaluators
	fmt.Println("\n" + separator)
	fmt.Println("=== Single Input Evaluation ===")
	fmt.Println(separator)

	singleInput := evaluation.NewEvaluateInput().
		WithQuery("What is the speed of light?").
		WithResponse("The speed of light is approximately 299,792 kilometers per second in a vacuum.").
		WithContexts([]string{"Light travels at about 300,000 km/s in vacuum."}).
		WithReference("Light travels at 299,792 km/s in vacuum.")

	fmt.Printf("\nQuery: %s\n", singleInput.Query)
	fmt.Printf("Response: %s\n", truncate(singleInput.Response, 60))

	singleResults, err := batchRunner.RunSingle(ctx, singleInput)
	if err != nil {
		log.Printf("Single evaluation failed: %v", err)
	} else {
		fmt.Println("\nResults:")
		for name, result := range singleResults {
			status := "FAIL"
			if result.IsPassing() {
				status = "PASS"
			}
			fmt.Printf("  %s: %s (Score: %.2f)\n", name, status, result.GetScore())
		}
	}

	// 11. Comparing two batch results
	fmt.Println("\n" + separator)
	fmt.Println("=== Comparing Evaluation Results ===")
	fmt.Println(separator)

	// Simulate a second batch with slightly different responses
	responses2 := []string{
		"ML is AI that learns from data.", // Shorter, less complete
		"Plants make food from sunlight.", // Simplified
		"Tokyo is Japan's capital.",       // Correct but brief
		"Quantum computers are fast.",     // Oversimplified
	}

	fmt.Println("\nComparing original responses vs simplified responses...")

	batchResult2, err := batchRunner.EvaluateResponseStrs(
		ctx,
		queries,
		responses2,
		contexts,
		extraKwargs,
	)

	if err != nil {
		log.Printf("Second batch evaluation failed: %v", err)
	} else {
		comparison := evaluation.CompareResults(batchResult, batchResult2)

		fmt.Println("\nComparison (Original - Simplified):")
		fmt.Printf("%-15s %15s %18s\n", "Evaluator", "Score Diff", "Pass Rate Diff")
		fmt.Println(strings.Repeat("-", 50))

		for name, diffs := range comparison {
			fmt.Printf("%-15s %+14.2f %+17.0f%%\n",
				name,
				diffs["score_diff"],
				diffs["passing_rate_diff"]*100,
			)
		}
	}

	// 12. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nBatch Evaluation Features:")
	fmt.Println("  - Run multiple evaluators in parallel")
	fmt.Println("  - Configurable worker count for concurrency")
	fmt.Println("  - Aggregate statistics (avg score, pass rate)")
	fmt.Println("  - Compare results between different runs")
	fmt.Println("  - Handle errors gracefully")
	fmt.Println("  - Support for custom evaluation inputs")

	fmt.Println("\n=== Batch Evaluation Demo Complete ===")
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
