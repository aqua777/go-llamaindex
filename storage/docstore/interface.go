// Package docstore provides document store interfaces and implementations.
package docstore

import (
	"context"

	"github.com/aqua777/go-llamaindex/schema"
)

// DefaultNamespace is the default namespace for document stores.
const DefaultNamespace = "docstore"

// RefDocInfo represents information about a reference document and its associated nodes.
type RefDocInfo struct {
	// NodeIDs contains the IDs of nodes that belong to this reference document.
	NodeIDs []string `json:"node_ids"`
	// Metadata contains additional metadata about the reference document.
	Metadata map[string]interface{} `json:"metadata"`
}

// NewRefDocInfo creates a new RefDocInfo with empty slices and maps.
func NewRefDocInfo() *RefDocInfo {
	return &RefDocInfo{
		NodeIDs:  make([]string, 0),
		Metadata: make(map[string]interface{}),
	}
}

// ToMap converts RefDocInfo to a map for storage.
func (r *RefDocInfo) ToMap() map[string]interface{} {
	return map[string]interface{}{
		"node_ids": r.NodeIDs,
		"metadata": r.Metadata,
	}
}

// RefDocInfoFromMap creates a RefDocInfo from a map.
func RefDocInfoFromMap(m map[string]interface{}) *RefDocInfo {
	info := NewRefDocInfo()

	// Handle node_ids - can be []interface{} (from JSON) or []string (direct)
	switch nodeIDs := m["node_ids"].(type) {
	case []interface{}:
		for _, id := range nodeIDs {
			if strID, ok := id.(string); ok {
				info.NodeIDs = append(info.NodeIDs, strID)
			}
		}
	case []string:
		info.NodeIDs = append(info.NodeIDs, nodeIDs...)
	}

	if metadata, ok := m["metadata"].(map[string]interface{}); ok {
		info.Metadata = metadata
	}

	return info
}

// DocStore is the interface for document stores.
type DocStore interface {
	// Docs returns all documents in the store.
	Docs(ctx context.Context) (map[string]schema.BaseNode, error)

	// AddDocuments adds documents to the store.
	// If allowUpdate is true, existing documents will be updated.
	AddDocuments(ctx context.Context, docs []schema.BaseNode, allowUpdate bool) error

	// GetDocument retrieves a document by ID.
	// If raiseError is true, returns an error if the document is not found.
	GetDocument(ctx context.Context, docID string, raiseError bool) (schema.BaseNode, error)

	// DeleteDocument removes a document from the store.
	// If raiseError is true, returns an error if the document is not found.
	DeleteDocument(ctx context.Context, docID string, raiseError bool) error

	// DocumentExists checks if a document exists in the store.
	DocumentExists(ctx context.Context, docID string) (bool, error)

	// SetDocumentHash sets the hash for a document.
	SetDocumentHash(ctx context.Context, docID string, docHash string) error

	// GetDocumentHash retrieves the hash for a document.
	GetDocumentHash(ctx context.Context, docID string) (string, error)

	// GetAllDocumentHashes returns all document hashes.
	// Returns a map of hash -> docID.
	GetAllDocumentHashes(ctx context.Context) (map[string]string, error)

	// GetRefDocInfo retrieves reference document info by ID.
	GetRefDocInfo(ctx context.Context, refDocID string) (*RefDocInfo, error)

	// GetAllRefDocInfo returns all reference document info.
	GetAllRefDocInfo(ctx context.Context) (map[string]*RefDocInfo, error)

	// DeleteRefDoc deletes a reference document and all its associated nodes.
	// If raiseError is true, returns an error if the ref doc is not found.
	DeleteRefDoc(ctx context.Context, refDocID string, raiseError bool) error
}

// GetNodes retrieves multiple nodes by their IDs.
func GetNodes(ctx context.Context, store DocStore, nodeIDs []string, raiseError bool) ([]schema.BaseNode, error) {
	nodes := make([]schema.BaseNode, 0, len(nodeIDs))
	for _, nodeID := range nodeIDs {
		node, err := store.GetDocument(ctx, nodeID, raiseError)
		if err != nil {
			if raiseError {
				return nil, err
			}
			continue
		}
		if node != nil {
			nodes = append(nodes, node)
		}
	}
	return nodes, nil
}

// GetNode retrieves a single node by ID.
func GetNode(ctx context.Context, store DocStore, nodeID string, raiseError bool) (schema.BaseNode, error) {
	return store.GetDocument(ctx, nodeID, raiseError)
}
