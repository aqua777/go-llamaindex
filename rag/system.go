package rag

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/rag/reader"
	"github.com/aqua777/go-llamaindex/rag/store"
	"github.com/aqua777/go-llamaindex/rag/store/chromem"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/textsplitter"
	"github.com/google/uuid"
	openai "github.com/sashabaranov/go-openai"
)

// RAGConfig holds configuration for the RAG system.
type RAGConfig struct {
	OpenAIKey      string
	OpenAIBaseURL  string // Optional: for using other OpenAI-compatible APIs
	LLMModel       string
	EmbeddingModel string
	ChunkSize      int
	ChunkOverlap   int
	TopK           int
	PersistPath    string   // Path to persist vector store. Empty for in-memory.
	CollectionName string   // Name of the vector store collection.
	FileExtensions []string // File extensions to process (e.g., ".txt", ".md")
}

// RAGSystem encapsulates the RAG pipeline components.
type RAGSystem struct {
	Config      RAGConfig
	Client      *openai.Client
	EmbedModel  embedding.EmbeddingModel
	LLM         llm.LLM
	VectorStore store.VectorStore
	QueryEngine *RetrieverQueryEngine
	Splitter    *textsplitter.SentenceSplitter
}

// NewRAGSystem creates a new RAGSystem with the provided configuration.
func NewRAGSystem(config RAGConfig) (*RAGSystem, error) {
	// Set Defaults
	if config.LLMModel == "" {
		config.LLMModel = "gpt-3.5-turbo"
	}
	if config.EmbeddingModel == "" {
		config.EmbeddingModel = "text-embedding-3-small"
	}
	if config.ChunkSize <= 0 {
		config.ChunkSize = 1024
	}
	if config.ChunkOverlap <= 0 {
		config.ChunkOverlap = 200
	}
	if config.TopK <= 0 {
		config.TopK = 3
	}
	if config.CollectionName == "" {
		config.CollectionName = "documents"
	}
	if len(config.FileExtensions) == 0 {
		config.FileExtensions = []string{".txt", ".md"}
	}

	// If no key is provided, we don't error immediately.
	// We only error if we need to initialize the default OpenAI client and the key is missing.
	var client *openai.Client
	if config.OpenAIKey != "" {
		openaiConfig := openai.DefaultConfig(config.OpenAIKey)
		if config.OpenAIBaseURL != "" {
			openaiConfig.BaseURL = config.OpenAIBaseURL
		}
		client = openai.NewClientWithConfig(openaiConfig)
	}

	// Vector Store
	// ChromemStore implements VectorStore interface
	vectorStore, err := chromem.NewChromemStore(config.PersistPath, config.CollectionName)
	if err != nil {
		return nil, fmt.Errorf("failed to create vector store: %w", err)
	}

	// Splitter
	splitter := textsplitter.NewSentenceSplitter(config.ChunkSize, config.ChunkOverlap, nil, nil)

	sys := &RAGSystem{
		Config:      config,
		Client:      client,
		VectorStore: vectorStore,
		Splitter:    splitter,
	}

	// Initialize defaults if client is available
	if client != nil {
		sys.EmbedModel = embedding.NewOpenAIEmbeddingWithClient(client, config.EmbeddingModel)
		sys.LLM = llm.NewOpenAILLMWithClient(client, config.LLMModel)
	}

	return sys, nil
}

// WithEmbedding allows injecting a custom embedding model.
// This must be called before usage (Ingest/Query) to ensure the pipeline is correctly set up.
func (s *RAGSystem) WithEmbedding(embedModel embedding.EmbeddingModel) *RAGSystem {
	s.EmbedModel = embedModel
	return s
}

// WithLLM allows injecting a custom LLM.
// This must be called before usage (Query) to ensure the pipeline is correctly set up.
func (s *RAGSystem) WithLLM(llmModel llm.LLM) *RAGSystem {
	s.LLM = llmModel
	return s
}

// bootstrap ensures that the QueryEngine and other dependent components are initialized.
// It should be called lazily or explicitly before operations that need them.
func (s *RAGSystem) bootstrap() error {
	if s.EmbedModel == nil {
		return fmt.Errorf("embedding model is not initialized. Provide OpenAIKey in config or use WithEmbedding()")
	}
	if s.LLM == nil {
		return fmt.Errorf("LLM is not initialized. Provide OpenAIKey in config or use WithLLM()")
	}

	// Re-initialize QueryEngine if it doesn't exist or if components changed
	// For simplicity, we just recreate it if it's nil or if we want to be safe.
	// Given the chainable nature, we can't easily know when "configuration" is done.
	// So we'll check if QueryEngine is nil or if we want to force update.
	// Let's just create it if it's nil.
	if s.QueryEngine == nil {
		retriever := NewVectorRetriever(s.VectorStore, s.EmbedModel, s.Config.TopK)
		synthesizer := NewSimpleSynthesizer(s.LLM)
		s.QueryEngine = NewRetrieverQueryEngine(retriever, synthesizer)
	}
	return nil
}

// IngestDirectory loads documents from inputDir, chunks them, embeds them, and stores them in the vector store.
func (s *RAGSystem) IngestDirectory(ctx context.Context, inputDir string) error {
	if err := s.bootstrap(); err != nil {
		return err
	}

	// 1. Load Data
	// We unpack FileExtensions to pass as variadic arguments
	// reader.NewSimpleDirectoryReader expects specific extensions
	// Actually NewSimpleDirectoryReader takes (dir string, ext ...string)
	docReader := reader.NewSimpleDirectoryReader(inputDir, s.Config.FileExtensions...)

	// The SimpleDirectoryReader currently returns []schema.Node (which act as documents).
	// We need to convert them to []schema.Document.
	nodes, err := docReader.LoadData()
	if err != nil {
		return fmt.Errorf("failed to load data: %w", err)
	}

	if len(nodes) == 0 {
		log.Println("No documents found in", inputDir)
		return nil
	}

	// Convert nodes to docs
	var docs []schema.Document
	for _, node := range nodes {
		docs = append(docs, schema.Document{
			ID:       node.ID,
			Text:     node.Text,
			Metadata: node.Metadata,
		})
	}

	return s.ingestDocuments(ctx, docs)
}

// IngestText accepts a raw string of text, creates a document from it, and ingests it.
func (s *RAGSystem) IngestText(ctx context.Context, text string, sourceID string) error {
	if err := s.bootstrap(); err != nil {
		return err
	}

	if sourceID == "" {
		sourceID = uuid.New().String()
	}
	doc := schema.Document{
		ID:   sourceID,
		Text: text,
		Metadata: map[string]interface{}{
			"source_id": sourceID,
			"source":    "text_variable",
		},
	}
	return s.ingestDocuments(ctx, []schema.Document{doc})
}

// IngestFile reads a single file and ingests it.
func (s *RAGSystem) IngestFile(ctx context.Context, filePath string) error {
	if err := s.bootstrap(); err != nil {
		return err
	}

	content, err := os.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("failed to read file %s: %w", filePath, err)
	}

	doc := schema.Document{
		ID:   filePath,
		Text: string(content),
		Metadata: map[string]interface{}{
			"source_id": filePath,
			"filename":  filePath,
		},
	}
	return s.ingestDocuments(ctx, []schema.Document{doc})
}

// ingestDocuments handles the common logic of splitting, embedding, and adding documents to the store.
func (s *RAGSystem) ingestDocuments(ctx context.Context, docs []schema.Document) error {
	// 2. Split and Embed
	var allNodes []schema.Node
	for _, doc := range docs {
		chunks := s.Splitter.SplitText(doc.Text)
		for i, chunk := range chunks {
			// Create node
			node := schema.Node{
				ID:   fmt.Sprintf("%s-chunk-%d", doc.ID, i),
				Text: chunk,
				Type: schema.ObjectTypeText,
				Metadata: map[string]interface{}{
					"source_id": doc.ID,
				},
			}
			// Copy over document metadata if it exists
			if doc.Metadata != nil {
				for k, v := range doc.Metadata {
					node.Metadata[k] = v
				}
			}

			// Generate embedding explicitly
			emb, err := s.EmbedModel.GetTextEmbedding(ctx, chunk)
			if err != nil {
				return fmt.Errorf("failed to get embedding for chunk %d of doc %s: %w", i, doc.ID, err)
			}
			node.Embedding = emb
			allNodes = append(allNodes, node)
		}
	}

	// 3. Ingest
	if len(allNodes) > 0 {
		_, err := s.VectorStore.Add(ctx, allNodes)
		if err != nil {
			return fmt.Errorf("failed to add nodes to vector store: %w", err)
		}
	}

	return nil
}

// Query executes a query against the RAG system and returns the response.
func (s *RAGSystem) Query(ctx context.Context, queryStr string) (string, error) {
	if err := s.bootstrap(); err != nil {
		return "", err
	}

	response, err := s.QueryEngine.Query(ctx, schema.QueryBundle{QueryString: queryStr})
	if err != nil {
		return "", err
	}
	return response.Response, nil
}
