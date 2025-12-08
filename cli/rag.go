package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/rag/store/chromem"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/textsplitter"
	"github.com/aqua777/krait"
	"github.com/google/uuid"
)

// RAGCommand holds the RAG pipeline components.
type RAGCommand struct {
	cacheDir    string
	verbose     bool
	stream      bool
	embedModel  *embedding.OllamaEmbedding
	llmModel    *llm.OllamaLLM
	vectorStore *chromem.ChromemStore
	splitter    *textsplitter.SentenceSplitter
	topK        int
	chatHistory []llm.ChatMessage
}

// NewRAGCommand creates a new RAG command instance from krait config.
func NewRAGCommand() (*RAGCommand, error) {
	cacheDir := krait.GetString(KeyCacheDir)
	verbose := krait.GetBool(KeyVerbose)
	stream := krait.GetBool(KeyStream)

	// Ensure cache directory exists
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create cache directory: %w", err)
	}

	// Initialize Ollama embedding
	embedModel := embedding.NewOllamaEmbedding(
		embedding.WithOllamaEmbeddingBaseURL(krait.GetString(KeyOllamaURL)),
		embedding.WithOllamaEmbeddingModel(krait.GetString(KeyOllamaEmbedModel)),
	)

	// Initialize Ollama LLM
	llmModel := llm.NewOllamaLLM(
		llm.WithOllamaBaseURL(krait.GetString(KeyOllamaURL)),
		llm.WithOllamaModel(krait.GetString(KeyOllamaModel)),
	)

	// Initialize vector store
	persistPath := ChromemPersistPath(cacheDir)
	collection := krait.GetString(KeyCollection)
	vectorStore, err := chromem.NewChromemStore(persistPath, collection)
	if err != nil {
		return nil, fmt.Errorf("failed to create vector store: %w", err)
	}

	// Initialize text splitter
	chunkSize := krait.GetInt(KeyChunkSize)
	chunkOverlap := krait.GetInt(KeyChunkOverlap)
	splitter := textsplitter.NewSentenceSplitter(chunkSize, chunkOverlap, nil, nil)

	// Load chat history if exists
	var chatHistory []llm.ChatMessage
	historyPath := ChatHistoryPath(cacheDir)
	if data, err := os.ReadFile(historyPath); err == nil {
		_ = json.Unmarshal(data, &chatHistory)
	}

	return &RAGCommand{
		cacheDir:    cacheDir,
		verbose:     verbose,
		stream:      stream,
		embedModel:  embedModel,
		llmModel:    llmModel,
		vectorStore: vectorStore,
		splitter:    splitter,
		topK:        krait.GetInt(KeyTopK),
		chatHistory: chatHistory,
	}, nil
}

// IngestFiles ingests files or directories into the vector store.
func (r *RAGCommand) IngestFiles(ctx context.Context, paths []string) error {
	var allFiles []string

	for _, pattern := range paths {
		// Expand glob patterns
		matches, err := filepath.Glob(pattern)
		if err != nil {
			return fmt.Errorf("invalid glob pattern %s: %w", pattern, err)
		}

		for _, match := range matches {
			info, err := os.Stat(match)
			if err != nil {
				return fmt.Errorf("failed to stat %s: %w", match, err)
			}

			if info.IsDir() {
				// Walk directory for .txt files
				err := filepath.WalkDir(match, func(path string, d fs.DirEntry, err error) error {
					if err != nil {
						return err
					}
					if !d.IsDir() && strings.HasSuffix(path, ".txt") {
						allFiles = append(allFiles, path)
					}
					return nil
				})
				if err != nil {
					return fmt.Errorf("failed to walk directory %s: %w", match, err)
				}
			} else {
				allFiles = append(allFiles, match)
			}
		}
	}

	if len(allFiles) == 0 {
		fmt.Println("No files found to ingest.")
		return nil
	}

	if r.verbose {
		fmt.Printf("Found %d file(s) to ingest\n", len(allFiles))
	}

	// Track ingested files
	historyPath := FilesHistoryPath(r.cacheDir)
	historyFile, err := os.OpenFile(historyPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("failed to open files history: %w", err)
	}
	defer historyFile.Close()

	for _, filePath := range allFiles {
		if r.verbose {
			fmt.Printf("Ingesting: %s\n", filePath)
		}

		content, err := os.ReadFile(filePath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: failed to read %s: %v\n", filePath, err)
			continue
		}

		// Split text into chunks
		chunks := r.splitter.SplitText(string(content))
		if r.verbose {
			fmt.Printf("  Split into %d chunks\n", len(chunks))
		}

		var nodes []schema.Node
		for i, chunk := range chunks {
			// Generate embedding
			emb, err := r.embedModel.GetTextEmbedding(ctx, chunk)
			if err != nil {
				return fmt.Errorf("failed to get embedding for chunk %d of %s: %w", i, filePath, err)
			}

			node := schema.Node{
				ID:        fmt.Sprintf("%s-chunk-%d", filepath.Base(filePath), i),
				Text:      chunk,
				Type:      schema.ObjectTypeText,
				Embedding: emb,
				Metadata: map[string]interface{}{
					"source_id": filePath,
					"filename":  filepath.Base(filePath),
					"chunk_idx": i,
				},
			}
			nodes = append(nodes, node)
		}

		// Add to vector store
		if len(nodes) > 0 {
			_, err = r.vectorStore.Add(ctx, nodes)
			if err != nil {
				return fmt.Errorf("failed to add nodes for %s: %w", filePath, err)
			}
		}

		// Record in history
		absPath, _ := filepath.Abs(filePath)
		fmt.Fprintln(historyFile, absPath)
	}

	fmt.Printf("Successfully ingested %d file(s)\n", len(allFiles))
	return nil
}

// Query performs a one-shot query against the RAG system.
func (r *RAGCommand) Query(ctx context.Context, question string) error {
	if r.verbose {
		fmt.Printf("Querying: %s\n", question)
	}

	// Get query embedding
	queryEmb, err := r.embedModel.GetQueryEmbedding(ctx, question)
	if err != nil {
		return fmt.Errorf("failed to get query embedding: %w", err)
	}

	// Retrieve relevant documents
	results, err := r.vectorStore.Query(ctx, schema.VectorStoreQuery{
		Embedding: queryEmb,
		TopK:      r.topK,
	})
	if err != nil {
		return fmt.Errorf("failed to query vector store: %w", err)
	}

	if len(results) == 0 {
		fmt.Println("No relevant documents found. Try ingesting some files first with --files.")
		return nil
	}

	if r.verbose {
		fmt.Printf("Found %d relevant chunks\n", len(results))
	}

	// Build context from retrieved documents
	var contextParts []string
	for _, result := range results {
		contextParts = append(contextParts, result.Node.Text)
	}
	context := strings.Join(contextParts, "\n\n---\n\n")

	// Build prompt
	systemPrompt := `You are a helpful assistant that answers questions based on the provided context. 
If the context doesn't contain relevant information, say so. Be concise and accurate.`

	userPrompt := fmt.Sprintf(`Context:
%s

Question: %s

Answer:`, context, question)

	messages := []llm.ChatMessage{
		llm.NewSystemMessage(systemPrompt),
		llm.NewUserMessage(userPrompt),
	}

	// Generate response
	if r.stream {
		return r.streamResponse(ctx, messages)
	}

	response, err := r.llmModel.Chat(ctx, messages)
	if err != nil {
		return fmt.Errorf("failed to generate response: %w", err)
	}

	fmt.Println(response)
	return nil
}

// streamResponse streams the LLM response to stdout.
func (r *RAGCommand) streamResponse(ctx context.Context, messages []llm.ChatMessage) error {
	tokenChan, err := r.llmModel.StreamChat(ctx, messages)
	if err != nil {
		return fmt.Errorf("failed to start streaming: %w", err)
	}

	for token := range tokenChan {
		fmt.Print(token.Delta)
	}
	fmt.Println()
	return nil
}

// Chat starts an interactive chat REPL.
func (r *RAGCommand) Chat(ctx context.Context) error {
	fmt.Println("Starting chat mode. Type 'exit' or 'quit' to end, 'clear' to reset history.")
	fmt.Println("---")

	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("\nYou: ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}

		switch strings.ToLower(input) {
		case "exit", "quit":
			fmt.Println("Goodbye!")
			return r.saveChatHistory()
		case "clear":
			r.chatHistory = nil
			fmt.Println("Chat history cleared.")
			continue
		}

		// Get query embedding for RAG
		queryEmb, err := r.embedModel.GetQueryEmbedding(ctx, input)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error getting embedding: %v\n", err)
			continue
		}

		// Retrieve relevant documents
		results, err := r.vectorStore.Query(ctx, schema.VectorStoreQuery{
			Embedding: queryEmb,
			TopK:      r.topK,
		})
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error querying: %v\n", err)
			continue
		}

		// Build context
		var contextParts []string
		for _, result := range results {
			contextParts = append(contextParts, result.Node.Text)
		}

		// Build messages with history
		var messages []llm.ChatMessage

		systemPrompt := `You are a helpful assistant that answers questions based on the provided context and conversation history.
If the context doesn't contain relevant information, say so. Be concise and accurate.`

		if len(contextParts) > 0 {
			systemPrompt += fmt.Sprintf("\n\nRelevant context:\n%s", strings.Join(contextParts, "\n\n---\n\n"))
		}

		messages = append(messages, llm.NewSystemMessage(systemPrompt))
		messages = append(messages, r.chatHistory...)
		messages = append(messages, llm.NewUserMessage(input))

		// Generate response
		fmt.Print("\nAssistant: ")
		var response string

		if r.stream {
			tokenChan, err := r.llmModel.StreamChat(ctx, messages)
			if err != nil {
				fmt.Fprintf(os.Stderr, "\nError: %v\n", err)
				continue
			}
			var sb strings.Builder
			for token := range tokenChan {
				fmt.Print(token.Delta)
				sb.WriteString(token.Delta)
			}
			response = sb.String()
			fmt.Println()
		} else {
			response, err = r.llmModel.Chat(ctx, messages)
			if err != nil {
				fmt.Fprintf(os.Stderr, "\nError: %v\n", err)
				continue
			}
			fmt.Println(response)
		}

		// Update history
		r.chatHistory = append(r.chatHistory, llm.NewUserMessage(input))
		r.chatHistory = append(r.chatHistory, llm.NewAssistantMessage(response))

		// Periodically save history
		if len(r.chatHistory)%4 == 0 {
			_ = r.saveChatHistory()
		}
	}

	return r.saveChatHistory()
}

// saveChatHistory persists chat history to disk.
func (r *RAGCommand) saveChatHistory() error {
	historyPath := ChatHistoryPath(r.cacheDir)
	data, err := json.MarshalIndent(r.chatHistory, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(historyPath, data, 0644)
}

// Clear removes all cached data after confirmation.
func (r *RAGCommand) Clear() error {
	fmt.Printf("Are you sure you want to delete data within %s? [y/N] ", r.cacheDir)

	scanner := bufio.NewScanner(os.Stdin)
	if !scanner.Scan() {
		return nil
	}

	response := strings.TrimSpace(strings.ToLower(scanner.Text()))
	if response != "y" && response != "yes" {
		fmt.Println("Aborted.")
		return nil
	}

	if err := os.RemoveAll(r.cacheDir); err != nil {
		return fmt.Errorf("failed to clear cache: %w", err)
	}

	fmt.Printf("Successfully cleared %s\n", r.cacheDir)
	return nil
}

// Ensure uuid is used (for potential future use)
var _ = uuid.New
