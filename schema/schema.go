package schema

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sort"
	"strings"

	"github.com/google/uuid"
)

// Default templates for text formatting.
const (
	DefaultTextNodeTemplate  = "{metadata_str}\n\n{content}"
	DefaultMetadataTemplate  = "{key}: {value}"
	DefaultMetadataSeparator = "\n"
)

// NodeType represents the type of the node.
type NodeType string

const (
	// ObjectTypeText represents a text node.
	ObjectTypeText NodeType = "TEXT"
	// ObjectTypeImage represents an image node.
	ObjectTypeImage NodeType = "IMAGE"
	// ObjectTypeIndex represents an index node.
	ObjectTypeIndex NodeType = "INDEX"
	// ObjectTypeDocument represents a document node.
	ObjectTypeDocument NodeType = "DOCUMENT"
	// ObjectTypeMultimodal represents a multimodal node.
	ObjectTypeMultimodal NodeType = "MULTIMODAL"
)

// BaseNode is the interface that all node types must implement.
type BaseNode interface {
	BaseComponent

	// GetID returns the unique identifier of the node.
	GetID() string
	// SetID sets the unique identifier of the node.
	SetID(id string)
	// GetType returns the type of the node.
	GetType() NodeType
	// GetContent returns the content of the node with metadata based on mode.
	GetContent(mode MetadataMode) string
	// SetContent sets the content of the node.
	SetContent(content string)
	// GetMetadata returns the metadata of the node.
	GetMetadata() map[string]interface{}
	// SetMetadata sets the metadata of the node.
	SetMetadata(metadata map[string]interface{})
	// GetMetadataStr returns the metadata as a formatted string based on mode.
	GetMetadataStr(mode MetadataMode) string
	// GetEmbedding returns the embedding of the node.
	GetEmbedding() []float64
	// SetEmbedding sets the embedding of the node.
	SetEmbedding(embedding []float64)
	// GetRelationships returns the relationships of the node.
	GetRelationships() NodeRelationships
	// SetRelationships sets the relationships of the node.
	SetRelationships(relationships NodeRelationships)
	// GetHash returns the hash of the node content.
	GetHash() string
	// GenerateHash generates and returns a new hash for the node.
	GenerateHash() string
	// AsRelatedNodeInfo returns the node as a RelatedNodeInfo.
	AsRelatedNodeInfo() RelatedNodeInfo
}

// Node represents a chunk of data.
// It matches the BaseNode/TextNode concept in LlamaIndex.
type Node struct {
	ID                        string                 `json:"id"`
	Text                      string                 `json:"text"`
	Type                      NodeType               `json:"type"`
	Metadata                  map[string]interface{} `json:"metadata,omitempty"`
	Embedding                 []float64              `json:"embedding,omitempty"`
	Relationships             NodeRelationships      `json:"relationships,omitempty"`
	Hash                      string                 `json:"hash,omitempty"`
	ExcludedEmbedMetadataKeys []string               `json:"excluded_embed_metadata_keys,omitempty"`
	ExcludedLLMMetadataKeys   []string               `json:"excluded_llm_metadata_keys,omitempty"`
	MetadataTemplate          string                 `json:"metadata_template,omitempty"`
	MetadataSeparator         string                 `json:"metadata_separator,omitempty"`
	TextTemplate              string                 `json:"text_template,omitempty"`
	StartCharIdx              *int                   `json:"start_char_idx,omitempty"`
	EndCharIdx                *int                   `json:"end_char_idx,omitempty"`
	MimeType                  string                 `json:"mimetype,omitempty"`
}

// NewNode creates a new Node with default values.
func NewNode() *Node {
	return &Node{
		ID:                uuid.New().String(),
		Type:              ObjectTypeText,
		Metadata:          make(map[string]interface{}),
		Relationships:     make(NodeRelationships),
		MetadataTemplate:  DefaultMetadataTemplate,
		MetadataSeparator: DefaultMetadataSeparator,
		TextTemplate:      DefaultTextNodeTemplate,
		MimeType:          "text/plain",
	}
}

// NewTextNode creates a new text node with the given text.
func NewTextNode(text string) *Node {
	node := NewNode()
	node.Text = text
	node.Hash = node.GenerateHash()
	return node
}

// ClassName returns the class name for serialization.
func (n *Node) ClassName() string {
	return "TextNode"
}

// ToDict converts the node to a map representation.
func (n *Node) ToDict() map[string]interface{} {
	result := map[string]interface{}{
		"class_name": n.ClassName(),
		"id":         n.ID,
		"text":       n.Text,
		"type":       string(n.Type),
	}
	if len(n.Metadata) > 0 {
		result["metadata"] = n.Metadata
	}
	if len(n.Embedding) > 0 {
		result["embedding"] = n.Embedding
	}
	if n.Hash != "" {
		result["hash"] = n.Hash
	}
	return result
}

// ToJSON converts the node to a JSON string.
func (n *Node) ToJSON() (string, error) {
	bytes, err := json.Marshal(n.ToDict())
	if err != nil {
		return "", err
	}
	return string(bytes), nil
}

// GetID returns the node ID.
func (n *Node) GetID() string {
	return n.ID
}

// SetID sets the node ID.
func (n *Node) SetID(id string) {
	n.ID = id
}

// GetType returns the node type.
func (n *Node) GetType() NodeType {
	return n.Type
}

// GetContent returns the content with metadata based on mode.
func (n *Node) GetContent(mode MetadataMode) string {
	metadataStr := strings.TrimSpace(n.GetMetadataStr(mode))
	if mode == MetadataModeNone || metadataStr == "" {
		return n.Text
	}
	// Use text template to format content
	template := n.TextTemplate
	if template == "" {
		template = DefaultTextNodeTemplate
	}
	result := strings.ReplaceAll(template, "{metadata_str}", metadataStr)
	result = strings.ReplaceAll(result, "{content}", n.Text)
	return strings.TrimSpace(result)
}

// SetContent sets the text content.
func (n *Node) SetContent(content string) {
	n.Text = content
	n.Hash = n.GenerateHash()
}

// GetMetadata returns the metadata.
func (n *Node) GetMetadata() map[string]interface{} {
	return n.Metadata
}

// SetMetadata sets the metadata.
func (n *Node) SetMetadata(metadata map[string]interface{}) {
	n.Metadata = metadata
}

// GetMetadataStr returns metadata as a formatted string based on mode.
func (n *Node) GetMetadataStr(mode MetadataMode) string {
	if mode == MetadataModeNone {
		return ""
	}

	// Determine which keys to exclude
	excludedKeys := make(map[string]bool)
	switch mode {
	case MetadataModeLLM:
		for _, key := range n.ExcludedLLMMetadataKeys {
			excludedKeys[key] = true
		}
	case MetadataModeEmbed:
		for _, key := range n.ExcludedEmbedMetadataKeys {
			excludedKeys[key] = true
		}
	}

	// Get sorted keys for consistent output
	keys := make([]string, 0, len(n.Metadata))
	for key := range n.Metadata {
		if !excludedKeys[key] {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)

	// Format metadata
	template := n.MetadataTemplate
	if template == "" {
		template = DefaultMetadataTemplate
	}
	separator := n.MetadataSeparator
	if separator == "" {
		separator = DefaultMetadataSeparator
	}

	var parts []string
	for _, key := range keys {
		value := n.Metadata[key]
		formatted := strings.ReplaceAll(template, "{key}", key)
		formatted = strings.ReplaceAll(formatted, "{value}", formatValue(value))
		parts = append(parts, formatted)
	}

	return strings.Join(parts, separator)
}

// formatValue converts a value to string for metadata formatting.
func formatValue(v interface{}) string {
	switch val := v.(type) {
	case string:
		return val
	case []byte:
		return string(val)
	default:
		bytes, _ := json.Marshal(val)
		return string(bytes)
	}
}

// GetEmbedding returns the embedding.
func (n *Node) GetEmbedding() []float64 {
	return n.Embedding
}

// SetEmbedding sets the embedding.
func (n *Node) SetEmbedding(embedding []float64) {
	n.Embedding = embedding
}

// GetRelationships returns the relationships.
func (n *Node) GetRelationships() NodeRelationships {
	if n.Relationships == nil {
		n.Relationships = make(NodeRelationships)
	}
	return n.Relationships
}

// SetRelationships sets the relationships.
func (n *Node) SetRelationships(relationships NodeRelationships) {
	n.Relationships = relationships
}

// GetHash returns the hash.
func (n *Node) GetHash() string {
	if n.Hash == "" {
		n.Hash = n.GenerateHash()
	}
	return n.Hash
}

// GenerateHash generates a SHA256 hash of the node content.
func (n *Node) GenerateHash() string {
	h := sha256.New()
	h.Write([]byte("type=" + string(n.Type)))
	if n.StartCharIdx != nil && n.EndCharIdx != nil {
		h.Write([]byte(strings.Join([]string{
			"startCharIdx=", string(rune(*n.StartCharIdx)),
			" endCharIdx=", string(rune(*n.EndCharIdx)),
		}, "")))
	}
	h.Write([]byte(n.GetContent(MetadataModeAll)))
	return hex.EncodeToString(h.Sum(nil))
}

// AsRelatedNodeInfo returns the node as a RelatedNodeInfo.
func (n *Node) AsRelatedNodeInfo() RelatedNodeInfo {
	return RelatedNodeInfo{
		NodeID:   n.ID,
		NodeType: n.Type,
		Metadata: n.Metadata,
		Hash:     n.GetHash(),
	}
}

// GetNodeInfo returns start and end character indices.
func (n *Node) GetNodeInfo() map[string]interface{} {
	return map[string]interface{}{
		"start": n.StartCharIdx,
		"end":   n.EndCharIdx,
	}
}

// GetText returns the text content without metadata.
func (n *Node) GetText() string {
	return n.GetContent(MetadataModeNone)
}

// Document represents a document.
type Document struct {
	ID       string                 `json:"id"`
	Text     string                 `json:"text"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// NodeWithScore represents a node with a similarity score.
type NodeWithScore struct {
	Node  Node    `json:"node"`
	Score float64 `json:"score"`
}

// FilterOperator represents the operator for a metadata filter.
type FilterOperator string

const (
	FilterOperatorEq  FilterOperator = "=="
	FilterOperatorGt  FilterOperator = ">"
	FilterOperatorLt  FilterOperator = "<"
	FilterOperatorNe  FilterOperator = "!="
	FilterOperatorGte FilterOperator = ">="
	FilterOperatorLte FilterOperator = "<="
	FilterOperatorIn  FilterOperator = "in"
	FilterOperatorNin FilterOperator = "nin"
)

// MetadataFilter represents a single metadata filter.
type MetadataFilter struct {
	Key      string         `json:"key"`
	Value    interface{}    `json:"value"`
	Operator FilterOperator `json:"operator"`
}

// MetadataFilters represents a list of metadata filters.
type MetadataFilters struct {
	Filters []MetadataFilter `json:"filters"`
}

// QueryBundle encapsulates the query string and potential metadata.
// In the future, this can support image queries or multiple modal queries.
type QueryBundle struct {
	QueryString string           `json:"query_string"`
	Filters     *MetadataFilters `json:"filters,omitempty"`
	// Image string or []byte could be added here
}

// EngineResponse encapsulates the generated response and source nodes.
type EngineResponse struct {
	Response    string          `json:"response"`
	SourceNodes []NodeWithScore `json:"source_nodes,omitempty"`
}

// StreamingEngineResponse encapsulates the streaming response and source nodes.
type StreamingEngineResponse struct {
	ResponseStream <-chan string   `json:"-"`
	SourceNodes    []NodeWithScore `json:"source_nodes,omitempty"`
}

// VectorStoreQuery represents a query to the vector store.
type VectorStoreQuery struct {
	Embedding []float64        `json:"embedding"`
	TopK      int              `json:"top_k"`
	Filters   *MetadataFilters `json:"filters,omitempty"`
}
