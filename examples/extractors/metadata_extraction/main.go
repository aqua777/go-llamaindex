// Package main demonstrates metadata extraction from documents.
// This example corresponds to Python's metadata_extraction/MetadataExtractionSEC.ipynb
// and metadata_extraction/MetadataExtraction_LLMSurvey.ipynb
package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/extractors"
	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/schema"
)

func main() {
	ctx := context.Background()

	// Create LLM for extraction
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("=== Metadata Extraction Demo ===")
	fmt.Println("\nDemonstrates extracting metadata from documents using LLM.")

	separator := strings.Repeat("=", 60)

	// 1. Create sample documents
	fmt.Println("\n" + separator)
	fmt.Println("=== Creating Sample Documents ===")
	fmt.Println(separator)

	// Sample documents simulating SEC filings or research papers
	nodes := []*schema.Node{
		{
			ID:   "node1",
			Text: "Apple Inc. reported quarterly revenue of $89.5 billion for Q4 2023, representing a 1% year-over-year decline. The company's iPhone segment generated $43.8 billion in revenue, while Services reached an all-time high of $22.3 billion. CEO Tim Cook emphasized the company's commitment to innovation and sustainability initiatives.",
			Metadata: map[string]interface{}{
				"source":     "10-K Filing",
				"company":    "Apple Inc.",
				"year":       2023,
				"ref_doc_id": "apple_10k_2023",
			},
		},
		{
			ID:   "node2",
			Text: "Large Language Models (LLMs) have revolutionized natural language processing. Models like GPT-4, Claude, and LLaMA demonstrate remarkable capabilities in text generation, reasoning, and code synthesis. Recent research focuses on improving efficiency through techniques like quantization, pruning, and knowledge distillation.",
			Metadata: map[string]interface{}{
				"source":     "LLM Survey Paper",
				"topic":      "Machine Learning",
				"ref_doc_id": "llm_survey_2024",
			},
		},
		{
			ID:   "node3",
			Text: "Microsoft Corporation announced Azure revenue growth of 29% in fiscal year 2023. The Intelligent Cloud segment contributed $96.8 billion to total revenue. CEO Satya Nadella highlighted the integration of AI capabilities across all Microsoft products, including Copilot features in Office 365 and GitHub.",
			Metadata: map[string]interface{}{
				"source":     "10-K Filing",
				"company":    "Microsoft Corporation",
				"year":       2023,
				"ref_doc_id": "msft_10k_2023",
			},
		},
	}

	fmt.Printf("Created %d sample nodes:\n", len(nodes))
	for _, node := range nodes {
		fmt.Printf("  - %s: %s...\n", node.ID, truncate(node.Text, 50))
	}

	// 2. Title Extraction
	fmt.Println("\n" + separator)
	fmt.Println("=== Title Extraction ===")
	fmt.Println(separator)

	titleExtractor := extractors.NewTitleExtractor(
		extractors.WithTitleLLM(llmInstance),
		extractors.WithTitleNodes(3),
	)

	fmt.Println("\nExtracting document titles...")
	fmt.Printf("Extractor: %s\n", titleExtractor.Name())

	titleMetadata, err := titleExtractor.Extract(ctx, nodes)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Println("\nExtracted titles:")
		for i, meta := range titleMetadata {
			if title, ok := meta["document_title"]; ok {
				fmt.Printf("  Node %d: %s\n", i+1, title)
			}
		}
	}

	// 3. Keywords Extraction
	fmt.Println("\n" + separator)
	fmt.Println("=== Keywords Extraction ===")
	fmt.Println(separator)

	keywordsExtractor := extractors.NewKeywordsExtractor(
		extractors.WithKeywordsLLM(llmInstance),
		extractors.WithKeywordsCount(5),
	)

	fmt.Println("\nExtracting keywords...")
	fmt.Printf("Extractor: %s\n", keywordsExtractor.Name())
	fmt.Printf("Keywords per node: 5\n")

	keywordsMetadata, err := keywordsExtractor.Extract(ctx, nodes)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Println("\nExtracted keywords:")
		for i, meta := range keywordsMetadata {
			if keywords, ok := meta["excerpt_keywords"]; ok {
				fmt.Printf("  Node %d: %s\n", i+1, keywords)
				// Parse into individual keywords
				parsed := extractors.ParseKeywords(keywords.(string))
				fmt.Printf("    Parsed: %v\n", parsed)
			}
		}
	}

	// 4. Summary Extraction
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary Extraction ===")
	fmt.Println(separator)

	summaryExtractor := extractors.NewSummaryExtractor(
		extractors.WithSummaryLLM(llmInstance),
		extractors.WithSummaryTypes(
			extractors.SummaryTypeSelf,
			extractors.SummaryTypePrev,
			extractors.SummaryTypeNext,
		),
	)

	fmt.Println("\nExtracting summaries with context...")
	fmt.Printf("Extractor: %s\n", summaryExtractor.Name())
	fmt.Println("Summary types: self, prev, next")

	summaryMetadata, err := summaryExtractor.Extract(ctx, nodes)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Println("\nExtracted summaries:")
		for i, meta := range summaryMetadata {
			fmt.Printf("\n  Node %d:\n", i+1)
			if summary, ok := meta["section_summary"]; ok {
				fmt.Printf("    Self: %s\n", truncate(summary.(string), 80))
			}
			if prevSummary, ok := meta["prev_section_summary"]; ok {
				fmt.Printf("    Prev: %s\n", truncate(prevSummary.(string), 80))
			}
			if nextSummary, ok := meta["next_section_summary"]; ok {
				fmt.Printf("    Next: %s\n", truncate(nextSummary.(string), 80))
			}
		}
	}

	// 5. Extractor Chain
	fmt.Println("\n" + separator)
	fmt.Println("=== Extractor Chain ===")
	fmt.Println(separator)

	chain := extractors.NewExtractorChain(
		extractors.NewTitleExtractor(extractors.WithTitleLLM(llmInstance)),
		extractors.NewKeywordsExtractor(extractors.WithKeywordsLLM(llmInstance)),
		extractors.NewSummaryExtractor(extractors.WithSummaryLLM(llmInstance)),
	)

	fmt.Println("\nRunning extractor chain...")
	fmt.Printf("Chain contains %d extractors\n", len(chain.Extractors()))

	chainMetadata, err := chain.Extract(ctx, nodes[:1]) // Just first node for demo
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Println("\nCombined metadata for Node 1:")
		for key, value := range chainMetadata[0] {
			fmt.Printf("  %s: %s\n", key, truncate(fmt.Sprintf("%v", value), 60))
		}
	}

	// 6. ProcessNodes - Update nodes in place
	fmt.Println("\n" + separator)
	fmt.Println("=== ProcessNodes - In-Place Update ===")
	fmt.Println(separator)

	// Create fresh nodes for this demo
	testNodes := []*schema.Node{
		{
			ID:   "test1",
			Text: "Retrieval-Augmented Generation (RAG) combines the power of large language models with external knowledge retrieval. This approach reduces hallucinations and enables access to up-to-date information.",
			Metadata: map[string]interface{}{
				"source": "RAG Tutorial",
			},
		},
	}

	fmt.Println("\nBefore extraction:")
	fmt.Printf("  Metadata keys: %v\n", getKeys(testNodes[0].Metadata))

	keywordsExtractor2 := extractors.NewKeywordsExtractor(
		extractors.WithKeywordsLLM(llmInstance),
		extractors.WithKeywordsCount(3),
	)

	processedNodes, err := keywordsExtractor2.ProcessNodes(ctx, testNodes)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Println("\nAfter extraction:")
		fmt.Printf("  Metadata keys: %v\n", getKeys(processedNodes[0].Metadata))
		if keywords, ok := processedNodes[0].Metadata["excerpt_keywords"]; ok {
			fmt.Printf("  excerpt_keywords: %s\n", keywords)
		}
	}

	// 7. Custom extraction template
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom Extraction Template ===")
	fmt.Println(separator)

	customTemplate := `Analyze the following text and extract exactly {keywords} key entities (companies, people, technologies, or metrics).

Text: {context_str}

Entities (comma-separated):`

	customExtractor := extractors.NewKeywordsExtractor(
		extractors.WithKeywordsLLM(llmInstance),
		extractors.WithKeywordsCount(5),
		extractors.WithKeywordsPromptTemplate(customTemplate),
	)

	fmt.Println("\nUsing custom entity extraction template...")

	customMetadata, err := customExtractor.Extract(ctx, nodes[:1])
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		if entities, ok := customMetadata[0]["excerpt_keywords"]; ok {
			fmt.Printf("Extracted entities: %s\n", entities)
		}
	}

	// 8. Concurrent extraction
	fmt.Println("\n" + separator)
	fmt.Println("=== Concurrent Extraction ===")
	fmt.Println(separator)

	// Create more nodes for concurrent demo
	manyNodes := make([]*schema.Node, 5)
	for i := 0; i < 5; i++ {
		manyNodes[i] = &schema.Node{
			ID:   fmt.Sprintf("batch_node_%d", i+1),
			Text: fmt.Sprintf("Document %d discusses important topics in technology and business. Key areas include artificial intelligence, cloud computing, and digital transformation strategies for enterprise organizations.", i+1),
			Metadata: map[string]interface{}{
				"batch_id": i + 1,
			},
		}
	}

	concurrentExtractor := extractors.NewKeywordsExtractor(
		extractors.WithKeywordsLLM(llmInstance),
		extractors.WithKeywordsCount(3),
	)

	fmt.Printf("\nExtracting keywords from %d nodes concurrently...\n", len(manyNodes))

	batchMetadata, err := concurrentExtractor.Extract(ctx, manyNodes)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Println("Results:")
		for i, meta := range batchMetadata {
			if keywords, ok := meta["excerpt_keywords"]; ok {
				fmt.Printf("  Node %d: %s\n", i+1, truncate(keywords.(string), 50))
			}
		}
	}

	// 9. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nMetadata Extractors:")
	fmt.Println("  - TitleExtractor: Extract document titles from content")
	fmt.Println("  - KeywordsExtractor: Extract keywords/entities")
	fmt.Println("  - SummaryExtractor: Generate summaries with context")
	fmt.Println("  - QuestionsAnsweredExtractor: Generate answerable questions")
	fmt.Println()
	fmt.Println("Features:")
	fmt.Println("  - ExtractorChain for combining multiple extractors")
	fmt.Println("  - ProcessNodes for in-place metadata updates")
	fmt.Println("  - Custom prompt templates")
	fmt.Println("  - Concurrent extraction with configurable workers")
	fmt.Println()
	fmt.Println("Use Cases:")
	fmt.Println("  - SEC filing analysis")
	fmt.Println("  - Research paper metadata")
	fmt.Println("  - Document classification")
	fmt.Println("  - Search index enrichment")

	fmt.Println("\n=== Metadata Extraction Demo Complete ===")
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// getKeys returns the keys of a map.
func getKeys(m map[string]interface{}) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}
