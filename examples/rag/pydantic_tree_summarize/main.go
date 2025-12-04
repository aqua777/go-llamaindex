// Package main demonstrates Pydantic-style tree summarization with structured output.
// This example corresponds to Python's response_synthesizers/pydantic_tree_summarize.ipynb
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/program"
	"github.com/aqua777/go-llamaindex/prompts"
	"github.com/aqua777/go-llamaindex/schema"
)

// Summary represents a structured summary output (Go equivalent of Pydantic model).
type Summary struct {
	Title      string   `json:"title" description:"A concise title summarizing the content"`
	MainPoints []string `json:"main_points" description:"List of key points from the text"`
	Conclusion string   `json:"conclusion" description:"A brief concluding statement"`
}

// DetailedSummary is a more complex structured output.
type DetailedSummary struct {
	Title       string    `json:"title" description:"Title of the summary"`
	Overview    string    `json:"overview" description:"Brief overview of the content"`
	KeyFindings []Finding `json:"key_findings" description:"List of key findings"`
	Sentiment   string    `json:"sentiment" description:"Overall sentiment: positive, negative, or neutral"`
	Confidence  float64   `json:"confidence" description:"Confidence score between 0 and 1"`
}

// Finding represents a single finding in the summary.
type Finding struct {
	Topic       string `json:"topic" description:"Topic of the finding"`
	Description string `json:"description" description:"Description of the finding"`
	Importance  string `json:"importance" description:"Importance level: high, medium, or low"`
}

func main() {
	ctx := context.Background()

	fmt.Println("=== Pydantic Tree Summarize Demo ===")
	fmt.Println()
	fmt.Println("This example demonstrates structured output summarization using")
	fmt.Println("Go structs as the equivalent of Python's Pydantic models.")
	fmt.Println()

	separator := strings.Repeat("=", 70)

	// 1. Create LLM
	fmt.Println(separator)
	fmt.Println("=== Setting Up LLM ===")
	fmt.Println(separator)
	fmt.Println()

	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("LLM initialized (OpenAI)")
	fmt.Println()

	// 2. Show struct-to-schema conversion
	fmt.Println(separator)
	fmt.Println("=== Struct to JSON Schema Conversion ===")
	fmt.Println(separator)
	fmt.Println()

	fmt.Println("Go struct definition:")
	fmt.Println(`
type Summary struct {
    Title      string   ` + "`json:\"title\" description:\"A concise title\"`" + `
    MainPoints []string ` + "`json:\"main_points\" description:\"Key points\"`" + `
    Conclusion string   ` + "`json:\"conclusion\" description:\"Concluding statement\"`" + `
}`)
	fmt.Println()

	parser := program.NewPydanticOutputParser(Summary{})
	fmt.Println("Generated JSON Schema:")
	fmt.Println(parser.GetFormatInstructions())
	fmt.Println()

	// 3. Create sample documents
	fmt.Println(separator)
	fmt.Println("=== Sample Documents ===")
	fmt.Println(separator)
	fmt.Println()

	documents := getSampleDocuments()
	fmt.Printf("Created %d sample documents about AI/ML topics\n", len(documents))
	for i, doc := range documents {
		fmt.Printf("  Doc %d: %s...\n", i+1, truncate(doc.Text, 60))
	}
	fmt.Println()

	// 4. Basic Pydantic Tree Summarize
	fmt.Println(separator)
	fmt.Println("=== Basic Pydantic Tree Summarize ===")
	fmt.Println(separator)
	fmt.Println()

	summarizer := NewPydanticTreeSummarizer[Summary](llmInstance)

	query := "Summarize the key aspects of AI and machine learning"
	fmt.Printf("Query: %s\n\n", query)

	fmt.Println("Processing documents through tree summarization...")
	result, err := summarizer.Summarize(ctx, query, documents)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Println("Structured Summary:")
		printJSON(result)
	}
	fmt.Println()

	// 5. Detailed Summary with nested structs
	fmt.Println(separator)
	fmt.Println("=== Detailed Summary (Nested Structs) ===")
	fmt.Println(separator)
	fmt.Println()

	fmt.Println("Using more complex struct with nested types:")
	fmt.Println(`
type DetailedSummary struct {
    Title       string    ` + "`json:\"title\"`" + `
    Overview    string    ` + "`json:\"overview\"`" + `
    KeyFindings []Finding ` + "`json:\"key_findings\"`" + `
    Sentiment   string    ` + "`json:\"sentiment\"`" + `
    Confidence  float64   ` + "`json:\"confidence\"`" + `
}`)
	fmt.Println()

	detailedSummarizer := NewPydanticTreeSummarizer[DetailedSummary](llmInstance)

	detailedResult, err := detailedSummarizer.Summarize(ctx, query, documents)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Println("Detailed Structured Summary:")
		printJSON(detailedResult)
	}
	fmt.Println()

	// 6. Custom prompt template
	fmt.Println(separator)
	fmt.Println("=== Custom Prompt Template ===")
	fmt.Println(separator)
	fmt.Println()

	customPrompt := prompts.NewPromptTemplate(
		`You are an expert analyst. Analyze the following content and provide a structured summary.

Query: {query_str}

Content to analyze:
{context_str}

{format_instructions}

Provide your analysis as a JSON object.`,
		prompts.PromptTypeCustom,
	)

	customSummarizer := NewPydanticTreeSummarizer[Summary](llmInstance)
	customSummarizer.SetPromptTemplate(customPrompt)

	fmt.Println("Using custom prompt template for analysis...")
	customResult, err := customSummarizer.Summarize(ctx, "Analyze the technological implications", documents[:2])
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Println("Custom Analysis Result:")
		printJSON(customResult)
	}
	fmt.Println()

	// 7. Tree summarization explanation
	fmt.Println(separator)
	fmt.Println("=== How Pydantic Tree Summarize Works ===")
	fmt.Println(separator)
	fmt.Println()

	fmt.Println("1. Document Chunking:")
	fmt.Println("   - Documents are split into manageable chunks")
	fmt.Println("   - Each chunk fits within LLM context window")
	fmt.Println()
	fmt.Println("2. Leaf-Level Summarization:")
	fmt.Println("   - Each chunk is summarized into structured output")
	fmt.Println("   - Output conforms to the Go struct schema")
	fmt.Println()
	fmt.Println("3. Tree Aggregation:")
	fmt.Println("   - Summaries are recursively combined")
	fmt.Println("   - Parent nodes summarize child summaries")
	fmt.Println("   - Process continues until single root summary")
	fmt.Println()
	fmt.Println("4. Structured Output:")
	fmt.Println("   - Final output is parsed into Go struct")
	fmt.Println("   - Type-safe access to summary fields")
	fmt.Println()

	// 8. Benefits of structured output
	fmt.Println(separator)
	fmt.Println("=== Benefits of Structured Output ===")
	fmt.Println(separator)
	fmt.Println()

	fmt.Println("1. Type Safety:")
	fmt.Println("   - Compile-time type checking")
	fmt.Println("   - IDE autocompletion support")
	fmt.Println()
	fmt.Println("2. Validation:")
	fmt.Println("   - JSON schema ensures correct format")
	fmt.Println("   - Required fields are enforced")
	fmt.Println()
	fmt.Println("3. Downstream Processing:")
	fmt.Println("   - Easy to integrate with other systems")
	fmt.Println("   - Structured data for databases, APIs")
	fmt.Println()
	fmt.Println("4. Consistency:")
	fmt.Println("   - Uniform output format across calls")
	fmt.Println("   - Predictable structure for parsing")
	fmt.Println()

	fmt.Println("=== Pydantic Tree Summarize Demo Complete ===")
}

// PydanticTreeSummarizer performs tree summarization with structured output.
type PydanticTreeSummarizer[T any] struct {
	llm            llm.LLM
	parser         *program.PydanticOutputParser
	promptTemplate *prompts.PromptTemplate
	maxChunkSize   int
}

// NewPydanticTreeSummarizer creates a new PydanticTreeSummarizer for type T.
func NewPydanticTreeSummarizer[T any](l llm.LLM) *PydanticTreeSummarizer[T] {
	var zero T
	return &PydanticTreeSummarizer[T]{
		llm:          l,
		parser:       program.NewPydanticOutputParser(zero),
		maxChunkSize: 4096,
	}
}

// SetPromptTemplate sets a custom prompt template.
func (s *PydanticTreeSummarizer[T]) SetPromptTemplate(template *prompts.PromptTemplate) {
	s.promptTemplate = template
}

// SetMaxChunkSize sets the maximum chunk size for repacking.
func (s *PydanticTreeSummarizer[T]) SetMaxChunkSize(size int) {
	s.maxChunkSize = size
}

// Summarize performs tree summarization and returns structured output.
func (s *PydanticTreeSummarizer[T]) Summarize(ctx context.Context, query string, nodes []schema.Node) (*T, error) {
	if len(nodes) == 0 {
		return nil, fmt.Errorf("no nodes to summarize")
	}

	// Extract text from nodes
	textChunks := make([]string, len(nodes))
	for i, node := range nodes {
		textChunks[i] = node.Text
	}

	// Perform tree summarization
	rawOutput, err := s.treeSummarize(ctx, query, textChunks)
	if err != nil {
		return nil, err
	}

	// Parse into structured output
	parsed, err := s.parser.Parse(rawOutput)
	if err != nil {
		return nil, fmt.Errorf("failed to parse output: %w", err)
	}

	// Convert to target type
	result, ok := parsed.(T)
	if !ok {
		// Try JSON round-trip for type conversion
		jsonBytes, _ := json.Marshal(parsed)
		var typedResult T
		if err := json.Unmarshal(jsonBytes, &typedResult); err != nil {
			return nil, fmt.Errorf("failed to convert to target type: %w", err)
		}
		return &typedResult, nil
	}

	return &result, nil
}

// treeSummarize recursively summarizes text chunks.
func (s *PydanticTreeSummarizer[T]) treeSummarize(ctx context.Context, query string, textChunks []string) (string, error) {
	if len(textChunks) == 0 {
		return "", fmt.Errorf("no text chunks to summarize")
	}

	// Repack chunks to better utilize context
	repackedChunks := compactTextChunks(textChunks, s.maxChunkSize)

	// Base case: single chunk
	if len(repackedChunks) == 1 {
		return s.summarizeChunk(ctx, query, repackedChunks[0])
	}

	// Recursive case: summarize each chunk, then summarize summaries
	summaries := make([]string, len(repackedChunks))
	for i, chunk := range repackedChunks {
		summary, err := s.summarizeChunk(ctx, query, chunk)
		if err != nil {
			return "", err
		}
		summaries[i] = summary
	}

	// Recursively summarize the summaries
	return s.treeSummarize(ctx, query, summaries)
}

// summarizeChunk summarizes a single chunk with structured output.
func (s *PydanticTreeSummarizer[T]) summarizeChunk(ctx context.Context, query, chunk string) (string, error) {
	formatInstructions := s.parser.GetFormatInstructions()

	var promptText string
	if s.promptTemplate != nil {
		promptText = s.promptTemplate.Format(map[string]string{
			"query_str":           query,
			"context_str":         chunk,
			"format_instructions": formatInstructions,
		})
	} else {
		promptText = fmt.Sprintf(`Summarize the following content in response to the query.

Query: %s

Content:
%s

%s

Respond with a valid JSON object only.`, query, chunk, formatInstructions)
	}

	return s.llm.Complete(ctx, promptText)
}

// compactTextChunks combines small chunks to better utilize context window.
func compactTextChunks(chunks []string, maxSize int) []string {
	if len(chunks) == 0 {
		return chunks
	}

	var result []string
	current := ""

	for _, chunk := range chunks {
		if current == "" {
			current = chunk
		} else if len(current)+len(chunk)+2 <= maxSize {
			current = current + "\n\n" + chunk
		} else {
			result = append(result, current)
			current = chunk
		}
	}

	if current != "" {
		result = append(result, current)
	}

	return result
}

// getSampleDocuments returns sample documents for demonstration.
func getSampleDocuments() []schema.Node {
	texts := []string{
		`Artificial Intelligence (AI) is transforming industries worldwide. Machine learning, 
a subset of AI, enables computers to learn from data without explicit programming. 
Deep learning, using neural networks with multiple layers, has achieved breakthrough 
results in image recognition, natural language processing, and game playing.`,

		`The applications of AI span healthcare, finance, transportation, and entertainment. 
In healthcare, AI assists in diagnosis and drug discovery. Financial institutions use 
AI for fraud detection and algorithmic trading. Self-driving cars rely on AI for 
perception and decision-making.`,

		`Ethical considerations in AI include bias in training data, job displacement, 
privacy concerns, and the need for explainable AI. Researchers and policymakers are 
working on frameworks to ensure AI development benefits society while minimizing risks. 
Responsible AI practices are becoming industry standards.`,

		`The future of AI includes advances in general artificial intelligence, quantum 
machine learning, and neuromorphic computing. Edge AI brings intelligence to devices 
without cloud connectivity. AI is expected to augment human capabilities rather than 
replace them entirely, leading to new forms of human-AI collaboration.`,
	}

	nodes := make([]schema.Node, len(texts))
	for i, text := range texts {
		nodes[i] = schema.Node{
			ID:   fmt.Sprintf("doc-%d", i+1),
			Text: text,
			Type: schema.ObjectTypeText,
			Metadata: map[string]interface{}{
				"source": "sample",
				"index":  i,
			},
		}
	}

	return nodes
}

// printJSON prints a value as formatted JSON.
func printJSON(v interface{}) {
	data, err := json.MarshalIndent(v, "  ", "  ")
	if err != nil {
		fmt.Printf("  Error formatting JSON: %v\n", err)
		return
	}
	fmt.Printf("  %s\n", string(data))
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
