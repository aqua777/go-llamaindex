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

// VectorStoreQueryMode represents the query mode for vector store queries.
type VectorStoreQueryMode string

const (
	// QueryModeDefault is the default vector similarity search.
	QueryModeDefault VectorStoreQueryMode = "default"
	// QueryModeSparse uses sparse vector search (e.g., BM25).
	QueryModeSparse VectorStoreQueryMode = "sparse"
	// QueryModeHybrid combines dense and sparse search.
	QueryModeHybrid VectorStoreQueryMode = "hybrid"
	// QueryModeTextSearch uses full-text search.
	QueryModeTextSearch VectorStoreQueryMode = "text_search"
	// QueryModeSemanticHybrid combines semantic and keyword search.
	QueryModeSemanticHybrid VectorStoreQueryMode = "semantic_hybrid"
	// QueryModeMMR uses Maximum Marginal Relevance for diversity.
	QueryModeMMR VectorStoreQueryMode = "mmr"
	// QueryModeSVM uses SVM-based retrieval.
	QueryModeSVM VectorStoreQueryMode = "svm"
)

// FilterOperator represents the operator for a metadata filter.
type FilterOperator string

const (
	// Basic comparison operators
	FilterOperatorEq  FilterOperator = "==" // Equal (string, int, float)
	FilterOperatorGt  FilterOperator = ">"  // Greater than (int, float)
	FilterOperatorLt  FilterOperator = "<"  // Less than (int, float)
	FilterOperatorNe  FilterOperator = "!=" // Not equal (string, int, float)
	FilterOperatorGte FilterOperator = ">=" // Greater than or equal (int, float)
	FilterOperatorLte FilterOperator = "<=" // Less than or equal (int, float)

	// Array operators
	FilterOperatorIn  FilterOperator = "in"  // Value in array
	FilterOperatorNin FilterOperator = "nin" // Value not in array
	FilterOperatorAny FilterOperator = "any" // Array contains any of values
	FilterOperatorAll FilterOperator = "all" // Array contains all of values

	// Text operators
	FilterOperatorTextMatch            FilterOperator = "text_match"             // Full text match
	FilterOperatorTextMatchInsensitive FilterOperator = "text_match_insensitive" // Case-insensitive text match

	// Special operators
	FilterOperatorContains FilterOperator = "contains" // Metadata array contains value
	FilterOperatorIsEmpty  FilterOperator = "is_empty" // Field is empty or doesn't exist
)

// FilterCondition represents how multiple filters are combined.
type FilterCondition string

const (
	// FilterConditionAnd combines filters with AND logic.
	FilterConditionAnd FilterCondition = "and"
	// FilterConditionOr combines filters with OR logic.
	FilterConditionOr FilterCondition = "or"
	// FilterConditionNot negates the filter condition.
	FilterConditionNot FilterCondition = "not"
)

// MetadataFilter represents a single metadata filter.
type MetadataFilter struct {
	Key      string         `json:"key"`
	Value    interface{}    `json:"value"`
	Operator FilterOperator `json:"operator"`
}

// NewMetadataFilter creates a new metadata filter with the EQ operator.
func NewMetadataFilter(key string, value interface{}) MetadataFilter {
	return MetadataFilter{
		Key:      key,
		Value:    value,
		Operator: FilterOperatorEq,
	}
}

// NewMetadataFilterWithOp creates a new metadata filter with a specific operator.
func NewMetadataFilterWithOp(key string, value interface{}, op FilterOperator) MetadataFilter {
	return MetadataFilter{
		Key:      key,
		Value:    value,
		Operator: op,
	}
}

// MetadataFilters represents a collection of metadata filters with a condition.
type MetadataFilters struct {
	Filters   []MetadataFilter `json:"filters"`
	Condition FilterCondition  `json:"condition,omitempty"`
	// Nested allows for complex nested filter conditions.
	Nested []*MetadataFilters `json:"nested,omitempty"`
}

// NewMetadataFilters creates a new MetadataFilters with AND condition.
func NewMetadataFilters(filters ...MetadataFilter) *MetadataFilters {
	return &MetadataFilters{
		Filters:   filters,
		Condition: FilterConditionAnd,
	}
}

// NewMetadataFiltersWithCondition creates a new MetadataFilters with a specific condition.
func NewMetadataFiltersWithCondition(condition FilterCondition, filters ...MetadataFilter) *MetadataFilters {
	return &MetadataFilters{
		Filters:   filters,
		Condition: condition,
	}
}

// And adds filters with AND condition.
func (mf *MetadataFilters) And(filters ...MetadataFilter) *MetadataFilters {
	if mf.Condition == "" {
		mf.Condition = FilterConditionAnd
	}
	mf.Filters = append(mf.Filters, filters...)
	return mf
}

// Or creates a nested OR condition.
func (mf *MetadataFilters) Or(filters ...MetadataFilter) *MetadataFilters {
	orFilters := &MetadataFilters{
		Filters:   filters,
		Condition: FilterConditionOr,
	}
	mf.Nested = append(mf.Nested, orFilters)
	return mf
}

// AddNested adds a nested filter group.
func (mf *MetadataFilters) AddNested(nested *MetadataFilters) *MetadataFilters {
	mf.Nested = append(mf.Nested, nested)
	return mf
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
	// QueryEmbedding is the embedding vector for similarity search.
	QueryEmbedding []float64 `json:"query_embedding,omitempty"`
	// Embedding is an alias for QueryEmbedding (for backward compatibility).
	Embedding []float64 `json:"embedding,omitempty"`
	// SimilarityTopK is the number of top results to return.
	SimilarityTopK int `json:"similarity_top_k"`
	// TopK is an alias for SimilarityTopK (for backward compatibility).
	TopK int `json:"top_k,omitempty"`
	// QueryStr is the original query string (for text search modes).
	QueryStr string `json:"query_str,omitempty"`
	// Mode is the query mode (default, sparse, hybrid, mmr, etc.).
	Mode VectorStoreQueryMode `json:"mode,omitempty"`
	// Filters are metadata filters to apply.
	Filters *MetadataFilters `json:"filters,omitempty"`
	// DocIDs limits the search to specific document IDs.
	DocIDs []string `json:"doc_ids,omitempty"`
	// NodeIDs limits the search to specific node IDs.
	NodeIDs []string `json:"node_ids,omitempty"`
	// Alpha is the weight for hybrid search (0 = BM25, 1 = vector).
	Alpha *float64 `json:"alpha,omitempty"`
	// MMRThreshold is the diversity threshold for MMR mode.
	MMRThreshold *float64 `json:"mmr_threshold,omitempty"`
	// SparseTopK is the number of sparse results for hybrid search.
	SparseTopK *int `json:"sparse_top_k,omitempty"`
	// HybridTopK is the final number of results from hybrid search.
	HybridTopK *int `json:"hybrid_top_k,omitempty"`
	// OutputFields specifies which fields to return.
	OutputFields []string `json:"output_fields,omitempty"`
	// EmbeddingField specifies which embedding field to use.
	EmbeddingField string `json:"embedding_field,omitempty"`
}

// NewVectorStoreQuery creates a new VectorStoreQuery with defaults.
func NewVectorStoreQuery(embedding []float64, topK int) *VectorStoreQuery {
	return &VectorStoreQuery{
		QueryEmbedding: embedding,
		Embedding:      embedding,
		SimilarityTopK: topK,
		TopK:           topK,
		Mode:           QueryModeDefault,
	}
}

// WithMode sets the query mode.
func (q *VectorStoreQuery) WithMode(mode VectorStoreQueryMode) *VectorStoreQuery {
	q.Mode = mode
	return q
}

// WithFilters sets the metadata filters.
func (q *VectorStoreQuery) WithFilters(filters *MetadataFilters) *VectorStoreQuery {
	q.Filters = filters
	return q
}

// WithAlpha sets the alpha for hybrid search.
func (q *VectorStoreQuery) WithAlpha(alpha float64) *VectorStoreQuery {
	q.Alpha = &alpha
	return q
}

// WithMMRThreshold sets the MMR threshold.
func (q *VectorStoreQuery) WithMMRThreshold(threshold float64) *VectorStoreQuery {
	q.MMRThreshold = &threshold
	return q
}

// GetEmbedding returns the query embedding (prefers QueryEmbedding over Embedding).
func (q *VectorStoreQuery) GetEmbedding() []float64 {
	if len(q.QueryEmbedding) > 0 {
		return q.QueryEmbedding
	}
	return q.Embedding
}

// GetTopK returns the top-k value (prefers SimilarityTopK over TopK).
func (q *VectorStoreQuery) GetTopK() int {
	if q.SimilarityTopK > 0 {
		return q.SimilarityTopK
	}
	if q.TopK > 0 {
		return q.TopK
	}
	return 10 // default
}

// VectorStoreQueryResult represents the result of a vector store query.
type VectorStoreQueryResult struct {
	Nodes        []BaseNode `json:"nodes,omitempty"`
	Similarities []float64  `json:"similarities,omitempty"`
	IDs          []string   `json:"ids,omitempty"`
}
