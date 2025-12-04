// Package main demonstrates entity extraction from documents.
// This example corresponds to Python's metadata_extraction/EntityExtractionClimate.ipynb
package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/extractors"
	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/schema"
)

// EntityType represents different types of entities to extract.
type EntityType string

const (
	EntityTypePerson       EntityType = "person"
	EntityTypeOrganization EntityType = "organization"
	EntityTypeLocation     EntityType = "location"
	EntityTypeDate         EntityType = "date"
	EntityTypeMetric       EntityType = "metric"
	EntityTypeTechnology   EntityType = "technology"
)

// EntityExtractor is a custom extractor for named entity recognition.
type EntityExtractor struct {
	*extractors.LLMExtractor
	entityTypes    []EntityType
	promptTemplate string
}

// NewEntityExtractor creates a new EntityExtractor.
func NewEntityExtractor(llmInstance llm.LLM, entityTypes []EntityType) *EntityExtractor {
	template := buildEntityPrompt(entityTypes)

	return &EntityExtractor{
		LLMExtractor: extractors.NewLLMExtractor(
			[]extractors.BaseExtractorOption{
				extractors.WithExtractorName("EntityExtractor"),
				extractors.WithTextNodeOnly(true),
				extractors.WithNumWorkers(4),
			},
			extractors.WithLLM(llmInstance),
		),
		entityTypes:    entityTypes,
		promptTemplate: template,
	}
}

// buildEntityPrompt builds the entity extraction prompt.
func buildEntityPrompt(entityTypes []EntityType) string {
	typesList := make([]string, len(entityTypes))
	for i, t := range entityTypes {
		typesList[i] = string(t)
	}

	return fmt.Sprintf(`Extract named entities from the following text.
Entity types to extract: %s

Text: {context_str}

For each entity found, provide the entity name and its type.
Format: entity_name (type)
Entities:`, strings.Join(typesList, ", "))
}

// Extract extracts entities from nodes.
func (e *EntityExtractor) Extract(ctx context.Context, nodes []*schema.Node) ([]extractors.ExtractedMetadata, error) {
	if e.LLM() == nil {
		return nil, fmt.Errorf("LLM must be provided for EntityExtractor")
	}

	result := make([]extractors.ExtractedMetadata, len(nodes))

	for i, node := range nodes {
		content := e.GetNodeContent(node)
		prompt := strings.ReplaceAll(e.promptTemplate, "{context_str}", content)

		response, err := e.LLM().Complete(ctx, prompt)
		if err != nil {
			return nil, fmt.Errorf("failed to extract entities: %w", err)
		}

		result[i] = extractors.ExtractedMetadata{
			"entities":     strings.TrimSpace(response),
			"entity_types": e.entityTypes,
		}
	}

	return result, nil
}

// Name returns the extractor name.
func (e *EntityExtractor) Name() string {
	return "EntityExtractor"
}

func main() {
	ctx := context.Background()

	// Create LLM for extraction
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("=== Entity Extraction Demo ===")
	fmt.Println("\nDemonstrates extracting named entities from climate-related documents.")

	separator := strings.Repeat("=", 60)

	// 1. Create climate-related documents
	fmt.Println("\n" + separator)
	fmt.Println("=== Creating Climate Documents ===")
	fmt.Println(separator)

	climateNodes := []*schema.Node{
		{
			ID: "climate1",
			Text: `The Paris Agreement, adopted in 2015, aims to limit global warming to 1.5°C above pre-industrial levels. 
The Intergovernmental Panel on Climate Change (IPCC) reported that global temperatures have already risen by 1.1°C. 
Key signatories include the United States, China, and the European Union. The agreement requires countries to submit 
Nationally Determined Contributions (NDCs) every five years.`,
			Metadata: map[string]interface{}{
				"source": "Climate Policy Overview",
				"topic":  "Paris Agreement",
			},
		},
		{
			ID: "climate2",
			Text: `Tesla Inc. announced plans to build a new Gigafactory in Nevada, expected to produce 500,000 electric vehicles annually by 2025. 
CEO Elon Musk stated that the facility will be powered entirely by renewable energy. The investment of $3.6 billion 
will create approximately 3,000 jobs. This expansion supports the company's mission to accelerate the world's transition 
to sustainable energy.`,
			Metadata: map[string]interface{}{
				"source": "Business News",
				"topic":  "Electric Vehicles",
			},
		},
		{
			ID: "climate3",
			Text: `The Amazon rainforest, spanning 5.5 million square kilometers across Brazil, Peru, and Colombia, 
stores approximately 150-200 billion tons of carbon. Deforestation rates increased by 22% in 2022 according to 
Brazil's National Institute for Space Research (INPE). Environmental organizations like Greenpeace and WWF 
are calling for stronger protection measures.`,
			Metadata: map[string]interface{}{
				"source": "Environmental Report",
				"topic":  "Deforestation",
			},
		},
	}

	fmt.Printf("Created %d climate-related documents:\n", len(climateNodes))
	for _, node := range climateNodes {
		fmt.Printf("  - %s (%s): %s...\n",
			node.ID,
			node.Metadata["topic"],
			truncate(node.Text, 50))
	}

	// 2. Custom Entity Extraction
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom Entity Extraction ===")
	fmt.Println(separator)

	entityExtractor := NewEntityExtractor(llmInstance, []EntityType{
		EntityTypePerson,
		EntityTypeOrganization,
		EntityTypeLocation,
		EntityTypeDate,
		EntityTypeMetric,
	})

	fmt.Println("\nExtracting entities...")
	fmt.Printf("Entity types: person, organization, location, date, metric\n")

	entityMetadata, err := entityExtractor.Extract(ctx, climateNodes)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Println("\nExtracted entities:")
		for i, meta := range entityMetadata {
			fmt.Printf("\n  Document %d (%s):\n", i+1, climateNodes[i].Metadata["topic"])
			if entities, ok := meta["entities"]; ok {
				// Print each entity on a new line
				lines := strings.Split(entities.(string), "\n")
				for _, line := range lines {
					line = strings.TrimSpace(line)
					if line != "" {
						fmt.Printf("    - %s\n", line)
					}
				}
			}
		}
	}

	// 3. Keywords for climate topics
	fmt.Println("\n" + separator)
	fmt.Println("=== Climate Keywords Extraction ===")
	fmt.Println(separator)

	climateKeywordsTemplate := `{context_str}

Extract {keywords} keywords related to climate change, environmental policy, or sustainability from the above text.
Focus on: scientific terms, policy names, environmental metrics, and key concepts.
Keywords (comma-separated):`

	keywordsExtractor := extractors.NewKeywordsExtractor(
		extractors.WithKeywordsLLM(llmInstance),
		extractors.WithKeywordsCount(7),
		extractors.WithKeywordsPromptTemplate(climateKeywordsTemplate),
	)

	fmt.Println("\nExtracting climate-specific keywords...")

	keywordsMetadata, err := keywordsExtractor.Extract(ctx, climateNodes)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Println("\nClimate keywords:")
		for i, meta := range keywordsMetadata {
			if keywords, ok := meta["excerpt_keywords"]; ok {
				fmt.Printf("  Doc %d: %s\n", i+1, keywords)
			}
		}
	}

	// 4. Summary with environmental focus
	fmt.Println("\n" + separator)
	fmt.Println("=== Environmental Summary Extraction ===")
	fmt.Println(separator)

	envSummaryTemplate := `Here is content about climate and environment:
{context_str}

Provide a brief summary focusing on:
1. Key environmental impacts or metrics
2. Organizations or policies mentioned
3. Actions or recommendations

Environmental Summary:`

	summaryExtractor := extractors.NewSummaryExtractor(
		extractors.WithSummaryLLM(llmInstance),
		extractors.WithSummaryPromptTemplate(envSummaryTemplate),
		extractors.WithSummaryTypes(extractors.SummaryTypeSelf),
	)

	fmt.Println("\nGenerating environmental summaries...")

	summaryMetadata, err := summaryExtractor.Extract(ctx, climateNodes)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Println("\nEnvironmental summaries:")
		for i, meta := range summaryMetadata {
			if summary, ok := meta["section_summary"]; ok {
				fmt.Printf("\n  Document %d (%s):\n", i+1, climateNodes[i].Metadata["topic"])
				fmt.Printf("    %s\n", truncate(summary.(string), 150))
			}
		}
	}

	// 5. Combined extraction chain
	fmt.Println("\n" + separator)
	fmt.Println("=== Combined Extraction Chain ===")
	fmt.Println(separator)

	chain := extractors.NewExtractorChain(
		extractors.NewTitleExtractor(extractors.WithTitleLLM(llmInstance)),
		extractors.NewKeywordsExtractor(
			extractors.WithKeywordsLLM(llmInstance),
			extractors.WithKeywordsCount(5),
		),
		extractors.NewSummaryExtractor(
			extractors.WithSummaryLLM(llmInstance),
			extractors.WithSummaryTypes(extractors.SummaryTypeSelf),
		),
	)

	fmt.Println("\nRunning combined extraction on first document...")

	chainMetadata, err := chain.Extract(ctx, climateNodes[:1])
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Println("\nCombined metadata:")
		for key, value := range chainMetadata[0] {
			fmt.Printf("  %s: %s\n", key, truncate(fmt.Sprintf("%v", value), 80))
		}
	}

	// 6. Process nodes and enrich metadata
	fmt.Println("\n" + separator)
	fmt.Println("=== Enriching Node Metadata ===")
	fmt.Println(separator)

	fmt.Println("\nOriginal metadata for document 1:")
	for k, v := range climateNodes[0].Metadata {
		fmt.Printf("  %s: %v\n", k, v)
	}

	// Process with chain
	enrichedNodes, err := chain.ProcessNodes(ctx, climateNodes[:1])
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Println("\nEnriched metadata for document 1:")
		for k, v := range enrichedNodes[0].Metadata {
			fmt.Printf("  %s: %s\n", k, truncate(fmt.Sprintf("%v", v), 60))
		}
	}

	// 7. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nEntity Extraction Features:")
	fmt.Println("  - Custom entity types (person, org, location, etc.)")
	fmt.Println("  - Domain-specific extraction templates")
	fmt.Println("  - Climate/environmental focus")
	fmt.Println("  - Combined extraction chains")
	fmt.Println()
	fmt.Println("Extracted Metadata Types:")
	fmt.Println("  - document_title: Document titles")
	fmt.Println("  - excerpt_keywords: Key terms and concepts")
	fmt.Println("  - section_summary: Content summaries")
	fmt.Println("  - entities: Named entities with types")
	fmt.Println()
	fmt.Println("Use Cases:")
	fmt.Println("  - Climate research analysis")
	fmt.Println("  - Environmental policy tracking")
	fmt.Println("  - ESG report processing")
	fmt.Println("  - Scientific literature review")

	fmt.Println("\n=== Entity Extraction Demo Complete ===")
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
