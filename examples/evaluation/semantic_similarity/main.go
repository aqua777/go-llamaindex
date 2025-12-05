// Package main demonstrates semantic similarity evaluation using embeddings.
// This example corresponds to Python's evaluation/semantic_similarity_eval.ipynb
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/evaluation"
)

func main() {
	ctx := context.Background()

	// 1. Create embedding model
	embedModel := embedding.NewOpenAIEmbedding("", "")
	fmt.Println("=== Semantic Similarity Evaluation Demo ===")
	fmt.Println("\nEmbedding model initialized")

	separator := strings.Repeat("=", 60)

	// 2. Create semantic similarity evaluator with default settings
	fmt.Println("\n" + separator)
	fmt.Println("=== Basic Semantic Similarity ===")
	fmt.Println(separator)

	semanticEval := evaluation.NewSemanticSimilarityEvaluator(
		evaluation.WithSemanticSimilarityEmbedModel(embedModel),
		evaluation.WithSemanticSimilarityThreshold(0.8),
	)

	// Test case 1: Very similar responses
	fmt.Println("\nTest 1: Nearly identical responses")
	similarInput := evaluation.NewEvaluateInput().
		WithResponse("The capital of France is Paris, a beautiful city known for the Eiffel Tower.").
		WithReference("Paris is the capital of France, famous for its iconic Eiffel Tower.")

	result, err := semanticEval.Evaluate(ctx, similarInput)
	if err != nil {
		log.Printf("Evaluation error: %v", err)
	} else {
		fmt.Printf("  Response: %s\n", truncate(similarInput.Response, 60))
		fmt.Printf("  Reference: %s\n", truncate(similarInput.Reference, 60))
		fmt.Printf("  Similarity Score: %.4f\n", result.GetScore())
		fmt.Printf("  Passing (>0.8): %v\n", result.IsPassing())
		fmt.Printf("  Feedback: %s\n", result.Feedback)
	}

	// Test case 2: Semantically similar but different wording
	fmt.Println("\nTest 2: Same meaning, different wording")
	paraphrasedInput := evaluation.NewEvaluateInput().
		WithResponse("Machine learning is a subset of artificial intelligence that enables computers to learn from data.").
		WithReference("ML is an AI technique where systems improve through experience and data analysis.")

	result, err = semanticEval.Evaluate(ctx, paraphrasedInput)
	if err != nil {
		log.Printf("Evaluation error: %v", err)
	} else {
		fmt.Printf("  Response: %s\n", truncate(paraphrasedInput.Response, 60))
		fmt.Printf("  Reference: %s\n", truncate(paraphrasedInput.Reference, 60))
		fmt.Printf("  Similarity Score: %.4f\n", result.GetScore())
		fmt.Printf("  Passing (>0.8): %v\n", result.IsPassing())
	}

	// Test case 3: Completely different topics
	fmt.Println("\nTest 3: Different topics")
	differentInput := evaluation.NewEvaluateInput().
		WithResponse("The weather today is sunny with a high of 75 degrees.").
		WithReference("Quantum computing uses qubits to perform complex calculations.")

	result, err = semanticEval.Evaluate(ctx, differentInput)
	if err != nil {
		log.Printf("Evaluation error: %v", err)
	} else {
		fmt.Printf("  Response: %s\n", truncate(differentInput.Response, 60))
		fmt.Printf("  Reference: %s\n", truncate(differentInput.Reference, 60))
		fmt.Printf("  Similarity Score: %.4f\n", result.GetScore())
		fmt.Printf("  Passing (>0.8): %v\n", result.IsPassing())
	}

	// 3. Different similarity modes
	fmt.Println("\n" + separator)
	fmt.Println("=== Different Similarity Modes ===")
	fmt.Println(separator)

	testInput := evaluation.NewEvaluateInput().
		WithResponse("Python is a popular programming language for data science.").
		WithReference("Python is widely used in data science and machine learning.")

	// Cosine similarity (default)
	cosineEval := evaluation.NewSemanticSimilarityEvaluator(
		evaluation.WithSemanticSimilarityEmbedModel(embedModel),
		evaluation.WithSemanticSimilarityMode(evaluation.SimilarityModeCosine),
	)

	result, err = cosineEval.Evaluate(ctx, testInput)
	if err != nil {
		log.Printf("Cosine evaluation error: %v", err)
	} else {
		fmt.Printf("\nCosine Similarity: %.4f\n", result.GetScore())
	}

	// Dot product similarity
	dotProductEval := evaluation.NewSemanticSimilarityEvaluator(
		evaluation.WithSemanticSimilarityEmbedModel(embedModel),
		evaluation.WithSemanticSimilarityMode(evaluation.SimilarityModeDotProduct),
	)

	result, err = dotProductEval.Evaluate(ctx, testInput)
	if err != nil {
		log.Printf("Dot product evaluation error: %v", err)
	} else {
		fmt.Printf("Dot Product: %.4f\n", result.GetScore())
	}

	// Euclidean distance (negative, so higher is better)
	euclideanEval := evaluation.NewSemanticSimilarityEvaluator(
		evaluation.WithSemanticSimilarityEmbedModel(embedModel),
		evaluation.WithSemanticSimilarityMode(evaluation.SimilarityModeEuclidean),
		evaluation.WithSemanticSimilarityThreshold(-1.0), // Negative threshold for euclidean
	)

	result, err = euclideanEval.Evaluate(ctx, testInput)
	if err != nil {
		log.Printf("Euclidean evaluation error: %v", err)
	} else {
		fmt.Printf("Negative Euclidean Distance: %.4f\n", result.GetScore())
	}

	// 4. Custom threshold
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom Thresholds ===")
	fmt.Println(separator)

	thresholds := []float64{0.7, 0.8, 0.9, 0.95}
	testPair := evaluation.NewEvaluateInput().
		WithResponse("Deep learning uses neural networks with multiple layers.").
		WithReference("Neural networks with many layers are the basis of deep learning.")

	fmt.Printf("\nResponse: %s\n", testPair.Response)
	fmt.Printf("Reference: %s\n\n", testPair.Reference)

	for _, threshold := range thresholds {
		thresholdEval := evaluation.NewSemanticSimilarityEvaluator(
			evaluation.WithSemanticSimilarityEmbedModel(embedModel),
			evaluation.WithSemanticSimilarityThreshold(threshold),
		)

		result, err = thresholdEval.Evaluate(ctx, testPair)
		if err != nil {
			log.Printf("Evaluation error: %v", err)
			continue
		}
		fmt.Printf("Threshold %.2f: Score=%.4f, Passing=%v\n",
			threshold, result.GetScore(), result.IsPassing())
	}

	// 5. Batch semantic similarity evaluation
	fmt.Println("\n" + separator)
	fmt.Println("=== Batch Semantic Similarity ===")
	fmt.Println(separator)

	responses := []string{
		"The sun is a star at the center of our solar system.",
		"Water boils at 100 degrees Celsius at sea level.",
		"Shakespeare wrote many famous plays and sonnets.",
	}

	references := []string{
		"Our solar system's central star is the sun.",
		"At sea level, water reaches boiling point at 100°C.",
		"William Shakespeare is known for his plays and poetry.",
	}

	fmt.Println("\nBatch evaluation results:")
	for i := range responses {
		input := evaluation.NewEvaluateInput().
			WithResponse(responses[i]).
			WithReference(references[i])

		result, err = semanticEval.Evaluate(ctx, input)
		if err != nil {
			log.Printf("Evaluation error: %v", err)
			continue
		}
		fmt.Printf("\n  Pair %d:\n", i+1)
		fmt.Printf("    Response: %s\n", truncate(responses[i], 50))
		fmt.Printf("    Reference: %s\n", truncate(references[i], 50))
		fmt.Printf("    Score: %.4f, Passing: %v\n", result.GetScore(), result.IsPassing())
	}

	// 6. Using helper functions directly
	fmt.Println("\n" + separator)
	fmt.Println("=== Direct Similarity Computation ===")
	fmt.Println(separator)

	// Get embeddings directly
	vec1, err := embedModel.GetTextEmbedding(ctx, "Hello world")
	if err != nil {
		log.Printf("Embedding error: %v", err)
	}
	vec2, err := embedModel.GetTextEmbedding(ctx, "Hi there world")
	if err != nil {
		log.Printf("Embedding error: %v", err)
	}

	if vec1 != nil && vec2 != nil {
		cosine := evaluation.CosineSimilarity(vec1, vec2)
		dotProd := evaluation.DotProduct(vec1, vec2)
		euclidean := evaluation.EuclideanDistance(vec1, vec2)

		fmt.Printf("\nDirect computation for 'Hello world' vs 'Hi there world':\n")
		fmt.Printf("  Cosine Similarity: %.4f\n", cosine)
		fmt.Printf("  Dot Product: %.4f\n", dotProd)
		fmt.Printf("  Euclidean Distance: %.4f\n", euclidean)
	}

	// 7. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nSemantic Similarity Evaluation:")
	fmt.Println("  - Uses embeddings to compare response and reference")
	fmt.Println("  - Captures meaning rather than exact word matching")
	fmt.Println("  - Supports multiple similarity modes:")
	fmt.Println("    * Cosine: Measures angle between vectors (0-1)")
	fmt.Println("    * Dot Product: Raw dot product of vectors")
	fmt.Println("    * Euclidean: Negative distance (higher = more similar)")
	fmt.Println("  - Configurable threshold for pass/fail determination")

	fmt.Println("\n=== Semantic Similarity Demo Complete ===")
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
