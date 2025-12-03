// Package indexstore provides index store interfaces and implementations.
package indexstore

import (
	"context"
	"encoding/json"

	"github.com/google/uuid"
)

// DefaultNamespace is the default namespace for index stores.
const DefaultNamespace = "index_store"

// IndexStructType represents the type of index structure.
type IndexStructType string

const (
	// IndexStructTypeTree represents a tree-structured index.
	IndexStructTypeTree IndexStructType = "tree"
	// IndexStructTypeList represents a list index.
	IndexStructTypeList IndexStructType = "list"
	// IndexStructTypeKeywordTable represents a keyword table index.
	IndexStructTypeKeywordTable IndexStructType = "keyword_table"
	// IndexStructTypeVectorStore represents a vector store index.
	IndexStructTypeVectorStore IndexStructType = "vector_store"
	// IndexStructTypeEmpty represents an empty index.
	IndexStructTypeEmpty IndexStructType = "empty"
	// IndexStructTypeKG represents a knowledge graph index.
	IndexStructTypeKG IndexStructType = "kg"
	// IndexStructTypeLPG represents a labeled property graph index.
	IndexStructTypeLPG IndexStructType = "simple_lpg"
)

// IndexStruct represents a base index structure.
type IndexStruct struct {
	IndexID string          `json:"index_id"`
	Summary string          `json:"summary,omitempty"`
	Type    IndexStructType `json:"type"`

	// For IndexDict (vector store index)
	NodesDict map[string]string `json:"nodes_dict,omitempty"`

	// For IndexList
	Nodes []string `json:"nodes,omitempty"`

	// For KeywordTable and KG
	Table map[string][]string `json:"table,omitempty"`

	// For IndexGraph (tree index)
	AllNodes            map[int]string      `json:"all_nodes,omitempty"`
	RootNodes           map[int]string      `json:"root_nodes,omitempty"`
	NodeIDToChildrenIDs map[string][]string `json:"node_id_to_children_ids,omitempty"`

	// For KG index - stores embeddings for triplets
	EmbeddingDict map[string][]float64 `json:"embedding_dict,omitempty"`
}

// NewIndexStruct creates a new IndexStruct with a generated ID.
func NewIndexStruct(structType IndexStructType) *IndexStruct {
	return &IndexStruct{
		IndexID:       uuid.New().String(),
		Type:          structType,
		NodesDict:     make(map[string]string),
		Nodes:         make([]string, 0),
		Table:         make(map[string][]string),
		EmbeddingDict: make(map[string][]float64),
	}
}

// NewVectorStoreIndex creates a new vector store index struct.
func NewVectorStoreIndex() *IndexStruct {
	return NewIndexStruct(IndexStructTypeVectorStore)
}

// NewListIndex creates a new list index struct.
func NewListIndex() *IndexStruct {
	return NewIndexStruct(IndexStructTypeList)
}

// NewKeywordTableIndex creates a new keyword table index struct.
func NewKeywordTableIndex() *IndexStruct {
	return NewIndexStruct(IndexStructTypeKeywordTable)
}

// NewTreeIndex creates a new tree index struct.
func NewTreeIndex() *IndexStruct {
	is := NewIndexStruct(IndexStructTypeTree)
	is.AllNodes = make(map[int]string)
	is.RootNodes = make(map[int]string)
	is.NodeIDToChildrenIDs = make(map[string][]string)
	return is
}

// GetType returns the index struct type.
func (is *IndexStruct) GetType() IndexStructType {
	return is.Type
}

// GetSummary returns the summary, or an error if not set.
func (is *IndexStruct) GetSummary() (string, error) {
	if is.Summary == "" {
		return "", ErrSummaryNotSet
	}
	return is.Summary, nil
}

// AddNode adds a node to the index (for vector store index).
func (is *IndexStruct) AddNode(nodeID string, textID string) string {
	if textID == "" {
		textID = nodeID
	}
	is.NodesDict[textID] = nodeID
	return textID
}

// DeleteNode removes a node from the index (for vector store index).
func (is *IndexStruct) DeleteNode(textID string) {
	delete(is.NodesDict, textID)
}

// AddToList adds a node to the list (for list index).
func (is *IndexStruct) AddToList(nodeID string) {
	is.Nodes = append(is.Nodes, nodeID)
}

// AddToTable adds a node to the table with keywords (for keyword table index).
func (is *IndexStruct) AddToTable(keywords []string, nodeID string) {
	for _, keyword := range keywords {
		if _, ok := is.Table[keyword]; !ok {
			is.Table[keyword] = make([]string, 0)
		}
		is.Table[keyword] = append(is.Table[keyword], nodeID)
	}
}

// ToJSON converts the index struct to a JSON map.
func (is *IndexStruct) ToJSON() map[string]interface{} {
	result := map[string]interface{}{
		"index_id": is.IndexID,
		"type":     string(is.Type),
	}
	if is.Summary != "" {
		result["summary"] = is.Summary
	}
	if len(is.NodesDict) > 0 {
		result["nodes_dict"] = is.NodesDict
	}
	if len(is.Nodes) > 0 {
		result["nodes"] = is.Nodes
	}
	if len(is.Table) > 0 {
		result["table"] = is.Table
	}
	if len(is.AllNodes) > 0 {
		result["all_nodes"] = is.AllNodes
		result["root_nodes"] = is.RootNodes
		result["node_id_to_children_ids"] = is.NodeIDToChildrenIDs
	}
	return result
}

// FromJSON creates an IndexStruct from a JSON map.
func FromJSON(data map[string]interface{}) (*IndexStruct, error) {
	is := &IndexStruct{
		NodesDict: make(map[string]string),
		Nodes:     make([]string, 0),
		Table:     make(map[string][]string),
	}

	if indexID, ok := data["index_id"].(string); ok {
		is.IndexID = indexID
	}
	if summary, ok := data["summary"].(string); ok {
		is.Summary = summary
	}
	if structType, ok := data["type"].(string); ok {
		is.Type = IndexStructType(structType)
	}

	// Parse nodes_dict - handle both map[string]interface{} (from JSON) and map[string]string (direct)
	switch nodesDict := data["nodes_dict"].(type) {
	case map[string]interface{}:
		for k, v := range nodesDict {
			if strV, ok := v.(string); ok {
				is.NodesDict[k] = strV
			}
		}
	case map[string]string:
		for k, v := range nodesDict {
			is.NodesDict[k] = v
		}
	}

	// Parse nodes - handle both []interface{} (from JSON) and []string (direct)
	switch nodes := data["nodes"].(type) {
	case []interface{}:
		for _, n := range nodes {
			if strN, ok := n.(string); ok {
				is.Nodes = append(is.Nodes, strN)
			}
		}
	case []string:
		is.Nodes = append(is.Nodes, nodes...)
	}

	// Parse table
	if table, ok := data["table"].(map[string]interface{}); ok {
		for k, v := range table {
			if arr, ok := v.([]interface{}); ok {
				is.Table[k] = make([]string, 0, len(arr))
				for _, item := range arr {
					if strItem, ok := item.(string); ok {
						is.Table[k] = append(is.Table[k], strItem)
					}
				}
			}
		}
	}

	// Parse tree-specific fields
	if allNodes, ok := data["all_nodes"].(map[string]interface{}); ok {
		is.AllNodes = make(map[int]string)
		for k, v := range allNodes {
			var idx int
			json.Unmarshal([]byte(k), &idx)
			if strV, ok := v.(string); ok {
				is.AllNodes[idx] = strV
			}
		}
	}
	if rootNodes, ok := data["root_nodes"].(map[string]interface{}); ok {
		is.RootNodes = make(map[int]string)
		for k, v := range rootNodes {
			var idx int
			json.Unmarshal([]byte(k), &idx)
			if strV, ok := v.(string); ok {
				is.RootNodes[idx] = strV
			}
		}
	}
	if nodeIDToChildren, ok := data["node_id_to_children_ids"].(map[string]interface{}); ok {
		is.NodeIDToChildrenIDs = make(map[string][]string)
		for k, v := range nodeIDToChildren {
			if arr, ok := v.([]interface{}); ok {
				is.NodeIDToChildrenIDs[k] = make([]string, 0, len(arr))
				for _, item := range arr {
					if strItem, ok := item.(string); ok {
						is.NodeIDToChildrenIDs[k] = append(is.NodeIDToChildrenIDs[k], strItem)
					}
				}
			}
		}
	}

	return is, nil
}

// IndexStore is the interface for index stores.
type IndexStore interface {
	// IndexStructs returns all index structs in the store.
	IndexStructs(ctx context.Context) ([]*IndexStruct, error)

	// AddIndexStruct adds an index struct to the store.
	AddIndexStruct(ctx context.Context, indexStruct *IndexStruct) error

	// DeleteIndexStruct removes an index struct from the store.
	DeleteIndexStruct(ctx context.Context, key string) error

	// GetIndexStruct retrieves an index struct by ID.
	// If structID is empty, returns the only index struct (errors if multiple exist).
	GetIndexStruct(ctx context.Context, structID string) (*IndexStruct, error)
}

// Errors
var (
	ErrSummaryNotSet        = newError("summary field not set")
	ErrMultipleIndexStructs = newError("multiple index structs found, specify struct_id")
	ErrIndexStructNotFound  = newError("index struct not found")
)

type indexStoreError struct {
	msg string
}

func (e *indexStoreError) Error() string {
	return e.msg
}

func newError(msg string) error {
	return &indexStoreError{msg: msg}
}
