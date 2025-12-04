// Package main demonstrates different chat engine modes.
// This example corresponds to Python's chat_engine examples.
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/aqua777/go-llamaindex/chatengine"
	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/memory"
	"github.com/aqua777/go-llamaindex/rag/retriever"
	"github.com/aqua777/go-llamaindex/schema"
)

func main() {
	ctx := context.Background()

	// 1. Create LLM
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("LLM initialized")

	separator := strings.Repeat("=", 60)

	// ============================================================
	// SIMPLE CHAT ENGINE - Direct LLM conversation
	// ============================================================
	fmt.Println("\n" + separator)
	fmt.Println("=== Simple Chat Engine ===")
	fmt.Println(separator + "\n")

	simpleChatEngine := chatengine.NewSimpleChatEngine(
		chatengine.WithSimpleChatEngineLLM(llmInstance),
		chatengine.WithSimpleChatEngineMemory(memory.NewChatMemoryBuffer()),
		chatengine.WithSimpleChatEngineSystemPrompt("You are a helpful assistant. Be concise in your responses."),
	)

	// Multi-turn conversation
	simpleConversation := []string{
		"Hello! What's your name?",
		"Can you help me understand what RAG is?",
		"What are its main benefits?",
	}

	fmt.Println("Simple Chat Engine - Direct LLM conversation:")
	for _, msg := range simpleConversation {
		fmt.Printf("\nUser: %s\n", msg)
		response, err := simpleChatEngine.Chat(ctx, msg)
		if err != nil {
			log.Printf("Chat failed: %v", err)
			continue
		}
		fmt.Printf("Assistant: %s\n", truncate(response.Response, 200))
	}

	// Show chat history
	history, _ := simpleChatEngine.ChatHistory(ctx)
	fmt.Printf("\nChat history contains %d messages\n", len(history))

	// Reset conversation
	_ = simpleChatEngine.Reset(ctx)
	fmt.Println("Conversation reset")

	// ============================================================
	// CONTEXT CHAT ENGINE - RAG-enhanced chat
	// ============================================================
	fmt.Println("\n" + separator)
	fmt.Println("=== Context Chat Engine ===")
	fmt.Println(separator + "\n")

	// Create a mock retriever with knowledge base
	mockRetriever := NewMockRetriever(getKnowledgeBase())

	contextChatEngine := chatengine.NewContextChatEngine(
		chatengine.WithContextChatEngineLLM(llmInstance),
		chatengine.WithContextChatEngineMemory(memory.NewChatMemoryBuffer()),
		chatengine.WithContextChatEngineRetriever(mockRetriever),
		chatengine.WithContextChatEngineSystemPrompt("You are a helpful assistant that answers questions based on the provided context. If the context doesn't contain relevant information, say so."),
	)

	// Questions that benefit from context
	contextQuestions := []string{
		"What is LlamaIndex?",
		"How does the ingestion pipeline work?",
		"What vector stores are supported?",
	}

	fmt.Println("Context Chat Engine - RAG-enhanced conversation:")
	for _, question := range contextQuestions {
		fmt.Printf("\nUser: %s\n", question)
		response, err := contextChatEngine.Chat(ctx, question)
		if err != nil {
			log.Printf("Chat failed: %v", err)
			continue
		}
		fmt.Printf("Assistant: %s\n", truncate(response.Response, 200))
		if len(response.SourceNodes) > 0 {
			fmt.Printf("(Retrieved %d source nodes)\n", len(response.SourceNodes))
		}
	}

	// ============================================================
	// CONDENSE PLUS CONTEXT CHAT ENGINE - Query condensation
	// ============================================================
	fmt.Println("\n" + separator)
	fmt.Println("=== Condense Plus Context Chat Engine ===")
	fmt.Println(separator + "\n")

	condenseChatEngine := chatengine.NewCondensePlusContextChatEngine(
		chatengine.WithCondensePlusContextLLM(llmInstance),
		chatengine.WithCondensePlusContextMemory(memory.NewChatMemoryBuffer()),
		chatengine.WithCondensePlusContextRetriever(mockRetriever),
		chatengine.WithCondensePlusContextSystemPrompt("You are a helpful assistant. Answer questions based on the provided context."),
		chatengine.WithCondensePlusContextVerbose(true),
	)

	// Multi-turn conversation with follow-ups
	condenseConversation := []string{
		"Tell me about LlamaIndex",
		"What about its main features?",
		"How does it handle document ingestion?",
	}

	fmt.Println("Condense Plus Context - Query condensation for better retrieval:")
	for _, msg := range condenseConversation {
		fmt.Printf("\nUser: %s\n", msg)
		response, err := condenseChatEngine.Chat(ctx, msg)
		if err != nil {
			log.Printf("Chat failed: %v", err)
			continue
		}
		fmt.Printf("Assistant: %s\n", truncate(response.Response, 200))
	}

	// ============================================================
	// CUSTOM PERSONALITY CHAT ENGINE
	// ============================================================
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom Personality Chat Engine ===")
	fmt.Println(separator + "\n")

	// Create a chat engine with a specific personality
	pirateChatEngine := chatengine.NewSimpleChatEngine(
		chatengine.WithSimpleChatEngineLLM(llmInstance),
		chatengine.WithSimpleChatEngineMemory(memory.NewChatMemoryBuffer()),
		chatengine.WithSimpleChatEngineSystemPrompt(`You are a friendly pirate assistant named Captain Code. 
You speak like a pirate (using "arr", "matey", "ye", etc.) but are still helpful and knowledgeable about programming.
Keep responses concise but fun.`),
	)

	pirateQuestions := []string{
		"What is a function in programming?",
		"How do I fix a bug?",
	}

	fmt.Println("Custom Personality - Pirate-themed assistant:")
	for _, question := range pirateQuestions {
		fmt.Printf("\nUser: %s\n", question)
		response, err := pirateChatEngine.Chat(ctx, question)
		if err != nil {
			log.Printf("Chat failed: %v", err)
			continue
		}
		fmt.Printf("Captain Code: %s\n", truncate(response.Response, 250))
	}

	// ============================================================
	// CHAT ENGINE COMPARISON
	// ============================================================
	fmt.Println("\n" + separator)
	fmt.Println("=== Chat Engine Comparison ===")
	fmt.Println(separator + "\n")

	fmt.Println("Chat Engine Types:")
	fmt.Println()
	fmt.Println("1. SimpleChatEngine:")
	fmt.Println("   - Direct LLM conversation without retrieval")
	fmt.Println("   - Maintains conversation history")
	fmt.Println("   - Best for: General chat, creative tasks")
	fmt.Println()
	fmt.Println("2. ContextChatEngine:")
	fmt.Println("   - Retrieves context for each message")
	fmt.Println("   - Augments responses with relevant documents")
	fmt.Println("   - Best for: Q&A, knowledge-grounded chat")
	fmt.Println()
	fmt.Println("3. CondensePlusContextChatEngine:")
	fmt.Println("   - Condenses history into standalone questions")
	fmt.Println("   - Better retrieval for follow-up questions")
	fmt.Println("   - Best for: Multi-turn document Q&A")
	fmt.Println()
	fmt.Println("4. Custom Personality:")
	fmt.Println("   - Any engine with custom system prompt")
	fmt.Println("   - Defines assistant behavior and tone")
	fmt.Println("   - Best for: Branded/themed assistants")

	fmt.Println("\n=== Chat Engine Demo Complete ===")
}

// MockRetriever simulates a retriever with predefined documents.
type MockRetriever struct {
	*retriever.BaseRetriever
	documents []schema.NodeWithScore
}

// NewMockRetriever creates a new MockRetriever.
func NewMockRetriever(docs []schema.NodeWithScore) *MockRetriever {
	return &MockRetriever{
		BaseRetriever: retriever.NewBaseRetriever(),
		documents:     docs,
	}
}

// Retrieve returns documents matching the query.
func (m *MockRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	queryLower := strings.ToLower(query.QueryString)
	var results []schema.NodeWithScore

	for _, doc := range m.documents {
		textLower := strings.ToLower(doc.Node.Text)
		// Simple keyword matching
		keywords := strings.Fields(queryLower)
		matchCount := 0
		for _, kw := range keywords {
			if len(kw) > 3 && strings.Contains(textLower, kw) {
				matchCount++
			}
		}
		if matchCount > 0 {
			results = append(results, schema.NodeWithScore{
				Node:  doc.Node,
				Score: float64(matchCount) * 0.3,
			})
		}
	}

	// Return top 3 results
	if len(results) > 3 {
		results = results[:3]
	}

	// If no matches, return first document as fallback
	if len(results) == 0 && len(m.documents) > 0 {
		results = []schema.NodeWithScore{m.documents[0]}
	}

	return results, nil
}

// getKnowledgeBase returns sample documents for the mock retriever.
func getKnowledgeBase() []schema.NodeWithScore {
	docs := []struct {
		id    string
		text  string
		score float64
	}{
		{
			"doc-1",
			`LlamaIndex is a data framework for LLM applications. It provides tools to ingest, structure, and access private or domain-specific data. The framework supports various data sources and enables building RAG (Retrieval-Augmented Generation) applications.`,
			0.95,
		},
		{
			"doc-2",
			`The LlamaIndex ingestion pipeline handles document loading, text splitting, and embedding generation. Documents are first loaded using readers, then split into chunks using text splitters, and finally embedded using embedding models. The resulting nodes are stored in vector stores for retrieval.`,
			0.90,
		},
		{
			"doc-3",
			`LlamaIndex supports multiple vector stores including ChromaDB, Pinecone, Weaviate, and Milvus. Each vector store has different characteristics - some are in-memory for development, while others are distributed for production scale.`,
			0.85,
		},
		{
			"doc-4",
			`Chat engines in LlamaIndex provide conversational interfaces. SimpleChatEngine offers direct LLM chat, ContextChatEngine adds retrieval-augmented responses, and CondensePlusContextChatEngine condenses conversation history for better follow-up question handling.`,
			0.88,
		},
		{
			"doc-5",
			`Response synthesis in LlamaIndex can use different strategies: SimpleSynthesizer concatenates context and makes one LLM call, RefineSynthesizer iteratively refines across chunks, and TreeSummarizeSynthesizer uses hierarchical summarization.`,
			0.82,
		},
	}

	results := make([]schema.NodeWithScore, len(docs))
	for i, d := range docs {
		results[i] = schema.NodeWithScore{
			Node: schema.Node{
				ID:   d.id,
				Text: d.text,
				Type: schema.ObjectTypeText,
				Metadata: map[string]interface{}{
					"source": "knowledge_base",
				},
			},
			Score: d.score,
		}
	}
	return results
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
