// Package main demonstrates QueryEngineTool for wrapping query engines as tools.
// This example corresponds to Python's tools/eval_query_engine_tool.ipynb
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/rag/queryengine"
	"github.com/aqua777/go-llamaindex/rag/synthesizer"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/tools"
)

// MockQueryEngine simulates a query engine for demonstration.
type MockQueryEngine struct {
	name      string
	knowledge map[string]string
}

// NewMockQueryEngine creates a mock query engine with predefined knowledge.
func NewMockQueryEngine(name string, knowledge map[string]string) *MockQueryEngine {
	return &MockQueryEngine{
		name:      name,
		knowledge: knowledge,
	}
}

// Query implements queryengine.QueryEngine.
func (m *MockQueryEngine) Query(ctx context.Context, query string) (*synthesizer.Response, error) {
	queryLower := strings.ToLower(query)

	// Find matching knowledge
	var response string
	for keyword, answer := range m.knowledge {
		if strings.Contains(queryLower, strings.ToLower(keyword)) {
			response = answer
			break
		}
	}

	if response == "" {
		response = fmt.Sprintf("No specific information found for: %s", query)
	}

	return synthesizer.NewResponse(response, []schema.NodeWithScore{}), nil
}

// Ensure MockQueryEngine implements QueryEngine.
var _ queryengine.QueryEngine = (*MockQueryEngine)(nil)

func main() {
	ctx := context.Background()

	fmt.Println("=== Query Engine Tool Demo ===")
	fmt.Println("\nDemonstrates wrapping query engines as tools for LLM agents.")

	separator := strings.Repeat("=", 60)

	// 1. Create mock knowledge bases
	fmt.Println("\n" + separator)
	fmt.Println("=== Creating Knowledge Bases ===")
	fmt.Println(separator)

	// Programming knowledge base
	progKnowledge := map[string]string{
		"go":     "Go (Golang) is a statically typed, compiled programming language designed at Google. It features garbage collection, structural typing, and CSP-style concurrency. Go is known for its simplicity and efficiency in building scalable systems.",
		"python": "Python is a high-level, interpreted programming language known for its readability and versatility. It supports multiple paradigms including procedural, object-oriented, and functional programming. Python is widely used in data science, web development, and automation.",
		"rust":   "Rust is a systems programming language focused on safety, speed, and concurrency. It prevents memory errors through its ownership system without using garbage collection. Rust is popular for building reliable and efficient software.",
	}

	// AI/ML knowledge base
	aiKnowledge := map[string]string{
		"llm":        "Large Language Models (LLMs) are neural networks trained on vast amounts of text data. They can generate human-like text, answer questions, and perform various language tasks. Examples include GPT-4, Claude, and LLaMA.",
		"rag":        "Retrieval-Augmented Generation (RAG) combines retrieval systems with generative models. It retrieves relevant documents from a knowledge base and uses them to generate more accurate and grounded responses.",
		"embedding":  "Embeddings are dense vector representations of text that capture semantic meaning. Similar texts have similar embeddings, enabling semantic search and similarity comparisons.",
		"embeddings": "Embeddings are dense vector representations of text that capture semantic meaning. Similar texts have similar embeddings, enabling semantic search and similarity comparisons.",
	}

	fmt.Printf("Created programming knowledge base with %d topics\n", len(progKnowledge))
	fmt.Printf("Created AI/ML knowledge base with %d topics\n", len(aiKnowledge))

	// 2. Create query engines
	fmt.Println("\n" + separator)
	fmt.Println("=== Creating Query Engines ===")
	fmt.Println(separator)

	progQueryEngine := NewMockQueryEngine("programming", progKnowledge)
	aiQueryEngine := NewMockQueryEngine("ai_ml", aiKnowledge)

	fmt.Println("Created programming query engine")
	fmt.Println("Created AI/ML query engine")

	// 3. Create QueryEngineTools
	fmt.Println("\n" + separator)
	fmt.Println("=== Creating Query Engine Tools ===")
	fmt.Println(separator)

	progTool := tools.NewQueryEngineToolFromDefaults(
		progQueryEngine,
		"programming_kb",
		"Query a knowledge base about programming languages including Go, Python, and Rust. Use this for questions about programming language features, syntax, and use cases.",
	)

	aiTool := tools.NewQueryEngineToolFromDefaults(
		aiQueryEngine,
		"ai_ml_kb",
		"Query a knowledge base about AI and machine learning topics including LLMs, RAG, and embeddings. Use this for questions about AI concepts and techniques.",
	)

	fmt.Println("\nCreated tools:")
	fmt.Printf("  1. %s\n", progTool.Metadata().Name)
	fmt.Printf("     Description: %s\n", truncate(progTool.Metadata().Description, 80))
	fmt.Printf("  2. %s\n", aiTool.Metadata().Name)
	fmt.Printf("     Description: %s\n", truncate(aiTool.Metadata().Description, 80))

	// 4. Query the tools
	fmt.Println("\n" + separator)
	fmt.Println("=== Querying Tools ===")
	fmt.Println(separator)

	// Query programming KB
	fmt.Println("\nQuery 1: 'What is Go programming language?'")
	output, err := progTool.Call(ctx, "What is Go programming language?")
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", truncate(output.Content, 200))
	}

	// Query AI KB
	fmt.Println("\nQuery 2: 'Explain RAG and how it works'")
	output, err = aiTool.Call(ctx, "Explain RAG and how it works")
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", truncate(output.Content, 200))
	}

	// 5. Tool with map input
	fmt.Println("\n" + separator)
	fmt.Println("=== Tool with Map Input ===")
	fmt.Println(separator)

	fmt.Println("\nQuery using map input format:")
	output, err = aiTool.Call(ctx, map[string]interface{}{
		"input": "What are embeddings used for?",
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", truncate(output.Content, 200))
	}

	// 6. Tool output details
	fmt.Println("\n" + separator)
	fmt.Println("=== Tool Output Details ===")
	fmt.Println(separator)

	output, _ = progTool.Call(ctx, "Compare Python and Rust")

	fmt.Println("\nToolOutput structure:")
	fmt.Printf("  ToolName: %s\n", output.ToolName)
	fmt.Printf("  Content: %s\n", truncate(output.Content, 100))
	fmt.Printf("  RawInput: %v\n", output.RawInput)
	fmt.Printf("  IsError: %v\n", output.IsError)

	// 7. OpenAI tool format
	fmt.Println("\n" + separator)
	fmt.Println("=== OpenAI Tool Format ===")
	fmt.Println(separator)

	fmt.Println("\nConverting to OpenAI tool format:")
	openAIFormat := progTool.Metadata().ToOpenAITool()
	openAIJSON, _ := json.MarshalIndent(openAIFormat, "", "  ")
	fmt.Printf("%s\n", string(openAIJSON))

	// 8. Multiple tools for agent
	fmt.Println("\n" + separator)
	fmt.Println("=== Multiple Tools for Agent ===")
	fmt.Println(separator)

	allTools := []tools.Tool{progTool, aiTool}

	fmt.Println("\nAvailable tools for agent:")
	for _, tool := range allTools {
		fmt.Printf("\n  Tool: %s\n", tool.Metadata().Name)
		fmt.Printf("  Description: %s\n", truncate(tool.Metadata().Description, 60))
		fmt.Printf("  Parameters: %v\n", tool.Metadata().Parameters["properties"])
	}

	// 9. Tool with return direct
	fmt.Println("\n" + separator)
	fmt.Println("=== Tool with Return Direct ===")
	fmt.Println(separator)

	directTool := tools.NewQueryEngineTool(
		progQueryEngine,
		tools.WithQueryEngineToolName("direct_answer"),
		tools.WithQueryEngineToolDescription("Get a direct answer about programming"),
		tools.WithQueryEngineToolReturnDirect(true),
	)

	fmt.Println("\nTool with ReturnDirect=true:")
	fmt.Printf("  Name: %s\n", directTool.Metadata().Name)
	fmt.Printf("  ReturnDirect: %v\n", directTool.Metadata().ReturnDirect)
	fmt.Println("  (Agent will return this tool's output directly without further processing)")

	// 10. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nQueryEngineTool Features:")
	fmt.Println("  - Wrap any QueryEngine as a tool")
	fmt.Println("  - Automatic input parsing (string or map)")
	fmt.Println("  - Configurable name and description")
	fmt.Println("  - ReturnDirect option for agent control")
	fmt.Println("  - OpenAI tool format conversion")
	fmt.Println()
	fmt.Println("Creation Methods:")
	fmt.Println("  - NewQueryEngineTool: With options")
	fmt.Println("  - NewQueryEngineToolFromDefaults: With explicit config")
	fmt.Println()
	fmt.Println("Use Cases:")
	fmt.Println("  - Multi-knowledge-base agents")
	fmt.Println("  - Domain-specific Q&A tools")
	fmt.Println("  - RAG-powered assistants")
	fmt.Println("  - Specialized search tools")

	fmt.Println("\n=== Query Engine Tool Demo Complete ===")
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
