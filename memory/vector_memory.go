package memory

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/rag/store"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/google/uuid"
)

// VectorMemory is memory backed by a vector store.
type VectorMemory struct {
	vectorStore        store.VectorStore
	embedModel         embedding.EmbeddingModel
	retrieverTopK      int
	batchByUserMessage bool
	curBatchNode       *schema.Node
	curBatchMessages   []llm.ChatMessage
}

// VectorMemoryOption configures a VectorMemory.
type VectorMemoryOption func(*VectorMemory)

// WithVectorStore sets the vector store.
func WithVectorStore(vs store.VectorStore) VectorMemoryOption {
	return func(m *VectorMemory) {
		m.vectorStore = vs
	}
}

// WithVectorMemoryEmbedModel sets the embedding model.
func WithVectorMemoryEmbedModel(model embedding.EmbeddingModel) VectorMemoryOption {
	return func(m *VectorMemory) {
		m.embedModel = model
	}
}

// WithRetrieverTopK sets the number of results to retrieve.
func WithRetrieverTopK(k int) VectorMemoryOption {
	return func(m *VectorMemory) {
		m.retrieverTopK = k
	}
}

// WithBatchByUserMessage sets whether to batch messages by user message.
func WithBatchByUserMessage(batch bool) VectorMemoryOption {
	return func(m *VectorMemory) {
		m.batchByUserMessage = batch
	}
}

// NewVectorMemory creates a new VectorMemory.
func NewVectorMemory(opts ...VectorMemoryOption) *VectorMemory {
	m := &VectorMemory{
		vectorStore:        store.NewSimpleVectorStore(),
		retrieverTopK:      5,
		batchByUserMessage: true,
	}

	for _, opt := range opts {
		opt(m)
	}

	// Initialize current batch node
	m.curBatchNode = m.newBatchNode()
	m.curBatchMessages = []llm.ChatMessage{}

	return m
}

// NewVectorMemoryFromDefaults creates a VectorMemory with defaults.
func NewVectorMemoryFromDefaults(
	vectorStore store.VectorStore,
	embedModel embedding.EmbeddingModel,
	opts ...VectorMemoryOption,
) *VectorMemory {
	allOpts := append([]VectorMemoryOption{
		WithVectorStore(vectorStore),
		WithVectorMemoryEmbedModel(embedModel),
	}, opts...)

	return NewVectorMemory(allOpts...)
}

// newBatchNode creates a new batch node.
func (m *VectorMemory) newBatchNode() *schema.Node {
	node := schema.NewTextNode("")
	node.ID = uuid.New().String()
	node.Metadata = map[string]interface{}{
		"sub_dicts": []map[string]interface{}{},
	}
	return node
}

// Get retrieves chat history relevant to the input query.
func (m *VectorMemory) Get(ctx context.Context, input string) ([]llm.ChatMessage, error) {
	if input == "" {
		return []llm.ChatMessage{}, nil
	}

	if m.embedModel == nil {
		return nil, fmt.Errorf("embedding model not configured")
	}

	// Generate query embedding
	queryEmbedding, err := m.embedModel.GetQueryEmbedding(ctx, input)
	if err != nil {
		return nil, err
	}

	// Query vector store
	vsQuery := schema.NewVectorStoreQuery(queryEmbedding, m.retrieverTopK)
	results, err := m.vectorStore.Query(ctx, *vsQuery)
	if err != nil {
		return nil, err
	}

	// Extract messages from results
	var messages []llm.ChatMessage
	for _, result := range results {
		subDicts, ok := result.Node.Metadata["sub_dicts"].([]interface{})
		if !ok {
			continue
		}

		for _, subDict := range subDicts {
			msgMap, ok := subDict.(map[string]interface{})
			if !ok {
				continue
			}

			msg, err := chatMessageFromMap(msgMap)
			if err != nil {
				continue
			}
			messages = append(messages, msg)
		}
	}

	return messages, nil
}

// GetAll is not supported for vector memory.
func (m *VectorMemory) GetAll(ctx context.Context) ([]llm.ChatMessage, error) {
	return nil, fmt.Errorf("vector memory does not support GetAll, can only retrieve based on input")
}

// Put adds a message to the memory.
func (m *VectorMemory) Put(ctx context.Context, message llm.ChatMessage) error {
	// Start new batch on user/system messages if batching is enabled
	if !m.batchByUserMessage || message.Role == llm.MessageRoleUser || message.Role == llm.MessageRoleSystem {
		// Commit current batch if it has content
		if m.curBatchNode.Text != "" {
			if err := m.commitNode(ctx, true); err != nil {
				return err
			}
		}
		m.curBatchNode = m.newBatchNode()
		m.curBatchMessages = []llm.ChatMessage{}
	}

	// Update current batch
	msgMap := chatMessageToMap(message)
	m.curBatchMessages = append(m.curBatchMessages, message)

	// Update node text
	if m.curBatchNode.Text == "" {
		m.curBatchNode.Text = message.Content
	} else {
		m.curBatchNode.Text += " " + message.Content
	}

	// Update metadata
	subDicts := m.curBatchNode.Metadata["sub_dicts"].([]map[string]interface{})
	subDicts = append(subDicts, msgMap)
	m.curBatchNode.Metadata["sub_dicts"] = subDicts

	// Commit the updated node
	return m.commitNode(ctx, true)
}

// PutMessages adds multiple messages to the memory.
func (m *VectorMemory) PutMessages(ctx context.Context, messages []llm.ChatMessage) error {
	for _, msg := range messages {
		if err := m.Put(ctx, msg); err != nil {
			return err
		}
	}
	return nil
}

// Set replaces all messages in memory.
func (m *VectorMemory) Set(ctx context.Context, messages []llm.ChatMessage) error {
	if err := m.Reset(ctx); err != nil {
		return err
	}
	return m.PutMessages(ctx, messages)
}

// Reset clears all memory.
func (m *VectorMemory) Reset(ctx context.Context) error {
	// Note: This requires the vector store to support clearing
	// For now, we just reset the current batch
	m.curBatchNode = m.newBatchNode()
	m.curBatchMessages = []llm.ChatMessage{}
	return nil
}

// commitNode commits the current batch node to the vector store.
func (m *VectorMemory) commitNode(ctx context.Context, overrideLast bool) error {
	if m.curBatchNode.Text == "" {
		return nil
	}

	if m.embedModel == nil {
		return fmt.Errorf("embedding model not configured")
	}

	// Generate embedding for the node
	embedding, err := m.embedModel.GetTextEmbedding(ctx, m.curBatchNode.Text)
	if err != nil {
		return err
	}
	m.curBatchNode.Embedding = embedding

	// Delete old node if overriding
	if overrideLast {
		_ = m.vectorStore.Delete(ctx, m.curBatchNode.ID)
	}

	// Add node to vector store
	_, err = m.vectorStore.Add(ctx, []schema.Node{*m.curBatchNode})
	return err
}

// chatMessageToMap converts a ChatMessage to a map.
func chatMessageToMap(msg llm.ChatMessage) map[string]interface{} {
	return map[string]interface{}{
		"role":    string(msg.Role),
		"content": msg.Content,
	}
}

// chatMessageFromMap creates a ChatMessage from a map.
func chatMessageFromMap(m map[string]interface{}) (llm.ChatMessage, error) {
	role, ok := m["role"].(string)
	if !ok {
		return llm.ChatMessage{}, fmt.Errorf("missing or invalid role")
	}

	content, _ := m["content"].(string)

	return llm.ChatMessage{
		Role:    llm.MessageRole(role),
		Content: content,
	}, nil
}

// serializeMessages serializes messages to JSON for storage.
func serializeMessages(messages []llm.ChatMessage) (string, error) {
	data, err := json.Marshal(messages)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// Ensure VectorMemory implements Memory.
var _ Memory = (*VectorMemory)(nil)
