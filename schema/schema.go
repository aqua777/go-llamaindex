package schema

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
)

// Node represents a chunk of data.
// It matches the BaseNode/TextNode concept in LlamaIndex.
type Node struct {
	ID        string                 `json:"id"`
	Text      string                 `json:"text"`
	Type      NodeType               `json:"type"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	Embedding []float64              `json:"embedding,omitempty"`
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
