// Package main demonstrates the PromptMixin pattern for prompt management.
// This example corresponds to Python's prompts/prompt_mixin.ipynb
package main

import (
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/prompts"
)

// QueryEngine is a sample component that uses prompts.
type QueryEngine struct {
	*prompts.BasePromptMixin
	name string
}

// NewQueryEngine creates a new QueryEngine with default prompts.
func NewQueryEngine(name string) *QueryEngine {
	qe := &QueryEngine{
		BasePromptMixin: prompts.NewBasePromptMixin(),
		name:            name,
	}

	// Set default prompts
	qe.SetPrompt("qa_prompt", prompts.NewPromptTemplate(
		`Context: {context}

Question: {question}

Answer based on the context above:`,
		prompts.PromptTypeQuestionAnswer,
	))

	qe.SetPrompt("refine_prompt", prompts.NewPromptTemplate(
		`Original question: {question}
Existing answer: {existing_answer}
New context: {new_context}

Refine the answer if needed:`,
		prompts.PromptTypeRefine,
	))

	return qe
}

// Retriever is a sample sub-component that also uses prompts.
type Retriever struct {
	*prompts.BasePromptMixin
	name string
}

// NewRetriever creates a new Retriever with default prompts.
func NewRetriever(name string) *Retriever {
	r := &Retriever{
		BasePromptMixin: prompts.NewBasePromptMixin(),
		name:            name,
	}

	// Set default prompts
	r.SetPrompt("query_transform", prompts.NewPromptTemplate(
		`Transform the following query for better retrieval:

Original query: {query}

Transformed query:`,
		prompts.PromptTypeCustom,
	))

	return r
}

// RAGPipeline is a composite component with sub-modules.
type RAGPipeline struct {
	*prompts.BasePromptMixin
	queryEngine *QueryEngine
	retriever   *Retriever
}

// NewRAGPipeline creates a new RAGPipeline.
func NewRAGPipeline() *RAGPipeline {
	pipeline := &RAGPipeline{
		BasePromptMixin: prompts.NewBasePromptMixin(),
		queryEngine:     NewQueryEngine("main_qe"),
		retriever:       NewRetriever("main_retriever"),
	}

	// Register sub-modules for prompt management
	pipeline.AddModule("query_engine", pipeline.queryEngine)
	pipeline.AddModule("retriever", pipeline.retriever)

	// Set pipeline-level prompts
	pipeline.SetPrompt("system_prompt", prompts.NewPromptTemplate(
		`You are a helpful assistant for {domain} questions.
Always be accurate and cite your sources.`,
		prompts.PromptTypeCustom,
	))

	return pipeline
}

func main() {
	fmt.Println("=== Prompt Mixin Demo ===")
	fmt.Println("\nDemonstrates hierarchical prompt management with PromptMixin.")

	separator := strings.Repeat("=", 60)

	// 1. Basic PromptMixin usage
	fmt.Println("\n" + separator)
	fmt.Println("=== Basic PromptMixin Usage ===")
	fmt.Println(separator)

	qe := NewQueryEngine("basic_qe")

	fmt.Println("\nQueryEngine prompts:")
	for name, prompt := range qe.GetPrompts() {
		fmt.Printf("  - %s: type=%s, vars=%v\n",
			name, prompt.GetPromptType(), prompt.GetTemplateVars())
	}

	// Get a specific prompt
	qaPrompt := qe.GetPrompt("qa_prompt")
	if qaPrompt != nil {
		formatted := qaPrompt.Format(map[string]string{
			"context":  "Go is a programming language.",
			"question": "What is Go?",
		})
		fmt.Printf("\nFormatted QA Prompt:\n%s\n", formatted)
	}

	// 2. Updating prompts
	fmt.Println("\n" + separator)
	fmt.Println("=== Updating Prompts ===")
	fmt.Println(separator)

	fmt.Println("\nBefore update:")
	fmt.Printf("  qa_prompt vars: %v\n", qe.GetPrompt("qa_prompt").GetTemplateVars())

	// Update with a custom prompt
	customQA := prompts.NewPromptTemplate(
		`You are an expert assistant.

Context Information:
{context}

User Question: {question}

Instructions:
- Be concise
- Cite sources
- If unsure, say so

Your Answer:`,
		prompts.PromptTypeQuestionAnswer,
	)

	qe.UpdatePrompts(prompts.PromptDictType{
		"qa_prompt": customQA,
	})

	fmt.Println("\nAfter update:")
	fmt.Printf("  qa_prompt vars: %v\n", qe.GetPrompt("qa_prompt").GetTemplateVars())

	// 3. Hierarchical prompt management
	fmt.Println("\n" + separator)
	fmt.Println("=== Hierarchical Prompt Management ===")
	fmt.Println(separator)

	pipeline := NewRAGPipeline()

	fmt.Println("\nRAGPipeline - All prompts (including sub-modules):")
	allPrompts := pipeline.GetPrompts()
	for name := range allPrompts {
		fmt.Printf("  - %s\n", name)
	}

	fmt.Println("\nPrompt hierarchy:")
	fmt.Println("  RAGPipeline")
	fmt.Println("    └── system_prompt")
	fmt.Println("    └── query_engine:")
	fmt.Println("        └── qa_prompt")
	fmt.Println("        └── refine_prompt")
	fmt.Println("    └── retriever:")
	fmt.Println("        └── query_transform")

	// 4. Updating sub-module prompts
	fmt.Println("\n" + separator)
	fmt.Println("=== Updating Sub-Module Prompts ===")
	fmt.Println(separator)

	fmt.Println("\nUpdating prompts using 'module:prompt_name' syntax...")

	// Update query_engine's qa_prompt through the pipeline
	pipeline.UpdatePrompts(prompts.PromptDictType{
		"query_engine:qa_prompt": prompts.NewPromptTemplate(
			`[UPDATED] Context: {context}
Question: {question}
Answer:`,
			prompts.PromptTypeQuestionAnswer,
		),
		"retriever:query_transform": prompts.NewPromptTemplate(
			`[UPDATED] Rewrite for search: {query}`,
			prompts.PromptTypeCustom,
		),
	})

	// Verify updates
	fmt.Println("\nVerifying updates:")
	updatedQA := pipeline.queryEngine.GetPrompt("qa_prompt")
	fmt.Printf("  query_engine:qa_prompt template starts with: %s\n",
		truncate(updatedQA.GetTemplate(), 30))

	updatedTransform := pipeline.retriever.GetPrompt("query_transform")
	fmt.Printf("  retriever:query_transform template starts with: %s\n",
		truncate(updatedTransform.GetTemplate(), 30))

	// 5. Prompt inspection
	fmt.Println("\n" + separator)
	fmt.Println("=== Prompt Inspection ===")
	fmt.Println(separator)

	fmt.Println("\nInspecting all prompts in pipeline:")
	for name, prompt := range pipeline.GetPrompts() {
		fmt.Printf("\n  Prompt: %s\n", name)
		fmt.Printf("    Type: %s\n", prompt.GetPromptType())
		fmt.Printf("    Variables: %v\n", prompt.GetTemplateVars())
		fmt.Printf("    Template preview: %s\n", truncate(prompt.GetTemplate(), 40))
	}

	// 6. Creating a custom component with PromptMixin
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom Component with PromptMixin ===")
	fmt.Println(separator)

	type Summarizer struct {
		*prompts.BasePromptMixin
	}

	summarizer := &Summarizer{
		BasePromptMixin: prompts.NewBasePromptMixin(),
	}

	// Add multiple prompts
	summarizer.SetPrompt("short_summary", prompts.NewPromptTemplate(
		"Summarize in one sentence: {text}",
		prompts.PromptTypeSummary,
	))
	summarizer.SetPrompt("detailed_summary", prompts.NewPromptTemplate(
		"Provide a detailed summary with key points:\n\n{text}\n\nSummary:",
		prompts.PromptTypeSummary,
	))
	summarizer.SetPrompt("bullet_summary", prompts.NewPromptTemplate(
		"Summarize as bullet points:\n\n{text}\n\nBullet Points:",
		prompts.PromptTypeSummary,
	))

	fmt.Println("\nSummarizer prompts:")
	for name, prompt := range summarizer.GetPrompts() {
		fmt.Printf("  - %s (vars: %v)\n", name, prompt.GetTemplateVars())
	}

	// Use different prompts for different use cases
	text := "Go is a statically typed, compiled language designed at Google."

	fmt.Println("\nUsing different summary prompts:")
	for name, prompt := range summarizer.GetPrompts() {
		formatted := prompt.Format(map[string]string{"text": text})
		fmt.Printf("\n  %s:\n    %s\n", name, truncate(formatted, 60))
	}

	// 7. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nPromptMixin Features:")
	fmt.Println("  - Centralized prompt management")
	fmt.Println("  - Hierarchical prompt organization")
	fmt.Println("  - Sub-module prompt access via 'module:prompt' syntax")
	fmt.Println("  - Easy prompt updates and customization")
	fmt.Println("  - Prompt inspection and debugging")
	fmt.Println()
	fmt.Println("Use Cases:")
	fmt.Println("  - Complex pipelines with multiple components")
	fmt.Println("  - Prompt versioning and A/B testing")
	fmt.Println("  - Runtime prompt customization")
	fmt.Println("  - Debugging and logging prompts")

	fmt.Println("\n=== Prompt Mixin Demo Complete ===")
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
