// Package main demonstrates different response synthesis strategies.
// This example corresponds to Python's low_level/response_synthesis.ipynb
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/prompts"
	"github.com/aqua777/go-llamaindex/rag/synthesizer"
	"github.com/aqua777/go-llamaindex/schema"
)

func main() {
	ctx := context.Background()

	// 1. Create LLM (empty strings use defaults/env vars)
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("LLM initialized")

	// 2. Create sample nodes (simulating retrieved context)
	nodes := createSampleNodes()
	fmt.Printf("Created %d sample nodes\n", len(nodes))

	// Query to answer
	query := "What are the key benefits and challenges of implementing RAG systems?"

	separator := strings.Repeat("=", 60)

	// 3. Simple Synthesizer - Single LLM call with all context
	fmt.Println("\n" + separator)
	fmt.Println("=== Simple Synthesizer ===")
	fmt.Println(separator + "\n")

	simpleSynth := synthesizer.NewSimpleSynthesizer(llmInstance)
	simpleResponse, err := simpleSynth.Synthesize(ctx, query, nodes)
	if err != nil {
		log.Printf("Simple synthesis failed: %v", err)
	} else {
		fmt.Printf("Query: %s\n\n", query)
		fmt.Printf("Response:\n%s\n", simpleResponse.Response)
		fmt.Printf("\nSource nodes used: %d\n", len(simpleResponse.SourceNodes))
	}

	// 4. Refine Synthesizer - Iteratively refines across chunks
	fmt.Println("\n" + separator)
	fmt.Println("=== Refine Synthesizer ===")
	fmt.Println(separator + "\n")

	refineSynth := synthesizer.NewRefineSynthesizer(llmInstance)
	refineResponse, err := refineSynth.Synthesize(ctx, query, nodes)
	if err != nil {
		log.Printf("Refine synthesis failed: %v", err)
	} else {
		fmt.Printf("Query: %s\n\n", query)
		fmt.Printf("Response:\n%s\n", refineResponse.Response)
		fmt.Printf("\nSource nodes used: %d\n", len(refineResponse.SourceNodes))
	}

	// 5. Tree Summarize Synthesizer - Recursive tree summarization
	fmt.Println("\n" + separator)
	fmt.Println("=== Tree Summarize Synthesizer ===")
	fmt.Println(separator + "\n")

	treeSynth := synthesizer.NewTreeSummarizeSynthesizer(llmInstance)
	treeResponse, err := treeSynth.Synthesize(ctx, query, nodes)
	if err != nil {
		log.Printf("Tree summarize synthesis failed: %v", err)
	} else {
		fmt.Printf("Query: %s\n\n", query)
		fmt.Printf("Response:\n%s\n", treeResponse.Response)
		fmt.Printf("\nSource nodes used: %d\n", len(treeResponse.SourceNodes))
	}

	// 6. Custom Prompt Synthesizer
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom Prompt Synthesizer ===")
	fmt.Println(separator + "\n")

	customTemplate := prompts.NewPromptTemplate(
		`You are a technical expert. Based on the following context, provide a detailed and structured answer to the question.

Context:
{context_str}

Question: {query_str}

Please structure your response with:
1. A brief summary
2. Key points
3. Practical recommendations

Response:`,
		prompts.PromptTypeQuestionAnswer,
	)

	customSynth := synthesizer.NewSimpleSynthesizer(
		llmInstance,
		synthesizer.WithTextQATemplate(customTemplate),
	)

	customResponse, err := customSynth.Synthesize(ctx, query, nodes)
	if err != nil {
		log.Printf("Custom synthesis failed: %v", err)
	} else {
		fmt.Printf("Query: %s\n\n", query)
		fmt.Printf("Response:\n%s\n", customResponse.Response)
	}

	// 7. Compare synthesis methods
	fmt.Println("\n" + separator)
	fmt.Println("=== Synthesis Method Comparison ===")
	fmt.Println(separator + "\n")

	fmt.Println("Method Characteristics:")
	fmt.Println()
	fmt.Println("1. SimpleSynthesizer:")
	fmt.Println("   - Single LLM call with all context concatenated")
	fmt.Println("   - Fast but limited by context window")
	fmt.Println("   - Best for: Small to medium context sizes")
	fmt.Println()
	fmt.Println("2. RefineSynthesizer:")
	fmt.Println("   - Iteratively refines response across chunks")
	fmt.Println("   - Multiple LLM calls (one per chunk)")
	fmt.Println("   - Best for: Large context that exceeds window")
	fmt.Println()
	fmt.Println("3. TreeSummarizeSynthesizer:")
	fmt.Println("   - Recursively summarizes in tree structure")
	fmt.Println("   - Efficient for very large contexts")
	fmt.Println("   - Best for: Summarization tasks with many chunks")

	fmt.Println("\n=== Response Synthesis Demo Complete ===")
}

// createSampleNodes creates sample nodes simulating retrieved context.
func createSampleNodes() []schema.NodeWithScore {
	texts := []string{
		`RAG (Retrieval-Augmented Generation) Benefits:
RAG systems offer several key advantages over traditional LLM approaches:
1. Reduced Hallucination: By grounding responses in actual retrieved documents, RAG significantly reduces the tendency of LLMs to generate false information.
2. Up-to-date Information: Unlike static model training, RAG can access current information from updated document stores.
3. Domain Specificity: Organizations can leverage their proprietary data without expensive fine-tuning.
4. Transparency: Source attribution allows users to verify the information sources.`,

		`RAG Implementation Challenges:
Implementing RAG systems comes with several challenges:
1. Retrieval Quality: The system is only as good as its retrieval component. Poor retrieval leads to irrelevant context.
2. Chunk Size Optimization: Finding the right balance between too small (losing context) and too large (noise) chunks.
3. Latency: Additional retrieval step adds latency compared to direct LLM calls.
4. Cost: Multiple components (embeddings, vector store, LLM) increase operational costs.`,

		`RAG Best Practices:
To build effective RAG systems, consider these practices:
1. Hybrid Search: Combine semantic search with keyword matching for better retrieval.
2. Reranking: Use a reranker to improve the quality of retrieved documents.
3. Metadata Filtering: Leverage metadata to narrow down search scope.
4. Evaluation: Regularly evaluate retrieval quality and response accuracy.
5. Chunking Strategy: Experiment with different chunking strategies for your use case.`,

		`RAG Architecture Components:
A typical RAG system consists of:
1. Document Loader: Ingests documents from various sources
2. Text Splitter: Chunks documents into manageable pieces
3. Embedding Model: Converts text to vector representations
4. Vector Store: Stores and indexes embeddings for similarity search
5. Retriever: Finds relevant chunks based on query
6. Synthesizer: Generates final response using retrieved context`,
	}

	nodes := make([]schema.NodeWithScore, len(texts))
	for i, text := range texts {
		nodes[i] = schema.NodeWithScore{
			Node: schema.Node{
				ID:   fmt.Sprintf("node_%d", i),
				Text: text,
				Type: schema.ObjectTypeText,
				Metadata: map[string]interface{}{
					"source": fmt.Sprintf("document_%d", i),
				},
			},
			Score: 0.9 - float64(i)*0.1, // Decreasing relevance scores
		}
	}

	return nodes
}
