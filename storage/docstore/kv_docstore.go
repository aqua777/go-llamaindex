package docstore

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/storage/kvstore"
)

const (
	// Collection suffixes for different data types.
	nodeCollectionSuffix     = "/data"
	refDocCollectionSuffix   = "/ref_doc_info"
	metadataCollectionSuffix = "/metadata"

	// Keys for document serialization.
	typeKey = "__type__"
	dataKey = "__data__"
)

// KVDocumentStore is a document store backed by a KVStore.
type KVDocumentStore struct {
	kvstore            kvstore.KVStore
	namespace          string
	nodeCollection     string
	refDocCollection   string
	metadataCollection string
}

// KVDocumentStoreOption is a functional option for KVDocumentStore.
type KVDocumentStoreOption func(*KVDocumentStore)

// WithNamespace sets the namespace for the document store.
func WithNamespace(namespace string) KVDocumentStoreOption {
	return func(s *KVDocumentStore) {
		s.namespace = namespace
	}
}

// NewKVDocumentStore creates a new KVDocumentStore.
func NewKVDocumentStore(kv kvstore.KVStore, opts ...KVDocumentStoreOption) *KVDocumentStore {
	store := &KVDocumentStore{
		kvstore:   kv,
		namespace: DefaultNamespace,
	}

	for _, opt := range opts {
		opt(store)
	}

	// Set collection names based on namespace
	store.nodeCollection = store.namespace + nodeCollectionSuffix
	store.refDocCollection = store.namespace + refDocCollectionSuffix
	store.metadataCollection = store.namespace + metadataCollectionSuffix

	return store
}

// docToJSON converts a node to a JSON-serializable map.
func docToJSON(node schema.BaseNode) map[string]interface{} {
	return map[string]interface{}{
		typeKey: string(node.GetType()),
		dataKey: node.ToDict(),
	}
}

// jsonToDoc converts a JSON map back to a node.
func jsonToDoc(docDict map[string]interface{}) (schema.BaseNode, error) {
	docType, ok := docDict[typeKey].(string)
	if !ok {
		return nil, fmt.Errorf("missing or invalid %s field", typeKey)
	}

	dataDict, ok := docDict[dataKey].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("missing or invalid %s field", dataKey)
	}

	node := schema.NewNode()

	// Restore node fields from dataDict
	if id, ok := dataDict["id"].(string); ok {
		node.ID = id
	}
	if text, ok := dataDict["text"].(string); ok {
		node.Text = text
	}
	if hash, ok := dataDict["hash"].(string); ok {
		node.Hash = hash
	}
	if metadata, ok := dataDict["metadata"].(map[string]interface{}); ok {
		node.Metadata = metadata
	}
	if embedding, ok := dataDict["embedding"].([]interface{}); ok {
		node.Embedding = make([]float64, len(embedding))
		for i, v := range embedding {
			if f, ok := v.(float64); ok {
				node.Embedding[i] = f
			}
		}
	}

	// Set node type
	node.Type = schema.NodeType(docType)

	// Restore relationships if present
	if relationships, ok := dataDict["relationships"].(map[string]interface{}); ok {
		node.Relationships = make(schema.NodeRelationships)
		for relType, relInfo := range relationships {
			relationship := schema.NodeRelationship(relType)
			if relInfoMap, ok := relInfo.(map[string]interface{}); ok {
				info := schema.RelatedNodeInfo{}
				if nodeID, ok := relInfoMap["node_id"].(string); ok {
					info.NodeID = nodeID
				}
				if nodeType, ok := relInfoMap["node_type"].(string); ok {
					info.NodeType = schema.NodeType(nodeType)
				}
				if metadata, ok := relInfoMap["metadata"].(map[string]interface{}); ok {
					info.Metadata = metadata
				}
				if hash, ok := relInfoMap["hash"].(string); ok {
					info.Hash = hash
				}
				// Use appropriate wrapper based on relationship type
				if relationship == schema.RelationshipChild {
					node.Relationships.AddChild(info)
				} else {
					node.Relationships[relationship] = schema.SingleRelatedNode{Info: info}
				}
			}
		}
	}

	return node, nil
}

// isValidDocJSON checks if the map contains valid document JSON.
func isValidDocJSON(docDict map[string]interface{}) bool {
	if docDict == nil {
		return false
	}
	_, hasType := docDict[typeKey]
	_, hasData := docDict[dataKey]
	return hasType && hasData
}

// Docs returns all documents in the store.
func (s *KVDocumentStore) Docs(ctx context.Context) (map[string]schema.BaseNode, error) {
	jsonDict, err := s.kvstore.GetAll(ctx, s.nodeCollection)
	if err != nil {
		return nil, err
	}

	docs := make(map[string]schema.BaseNode)
	for key, value := range jsonDict {
		docDict := storedValueToMap(value)
		if isValidDocJSON(docDict) {
			doc, err := jsonToDoc(docDict)
			if err != nil {
				continue // Skip invalid documents
			}
			docs[key] = doc
		}
	}
	return docs, nil
}

// AddDocuments adds documents to the store.
func (s *KVDocumentStore) AddDocuments(ctx context.Context, docs []schema.BaseNode, allowUpdate bool) error {
	for _, doc := range docs {
		nodeID := doc.GetID()
		if nodeID == "" {
			return fmt.Errorf("doc_id not set")
		}

		if !allowUpdate {
			exists, err := s.DocumentExists(ctx, nodeID)
			if err != nil {
				return err
			}
			if exists {
				return fmt.Errorf("doc_id %s already exists. Set allowUpdate to true to overwrite", nodeID)
			}
		}

		// Store the document
		data := docToJSON(doc)
		if err := s.kvstore.Put(ctx, nodeID, mapToStoredValue(data), s.nodeCollection); err != nil {
			return err
		}

		// Store metadata with doc hash
		metadata := kvstore.StoredValue{
			"doc_hash": doc.GetHash(),
		}

		// Handle reference document tracking
		relationships := doc.GetRelationships()
		if sourceNode := relationships.GetSource(); sourceNode != nil {
			refDocID := sourceNode.NodeID
			if refDocID != "" {
				// Get or create ref doc info
				refDocInfo, err := s.GetRefDocInfo(ctx, refDocID)
				if err != nil {
					return err
				}
				if refDocInfo == nil {
					refDocInfo = NewRefDocInfo()
				}

				// Add this node to the ref doc's node list
				if !contains(refDocInfo.NodeIDs, nodeID) {
					refDocInfo.NodeIDs = append(refDocInfo.NodeIDs, nodeID)
				}

				// Store ref doc info
				if err := s.kvstore.Put(ctx, refDocID, mapToStoredValue(refDocInfo.ToMap()), s.refDocCollection); err != nil {
					return err
				}

				metadata["ref_doc_id"] = refDocID
			}
		}

		// Store metadata
		if err := s.kvstore.Put(ctx, nodeID, metadata, s.metadataCollection); err != nil {
			return err
		}
	}

	return nil
}

// GetDocument retrieves a document by ID.
func (s *KVDocumentStore) GetDocument(ctx context.Context, docID string, raiseError bool) (schema.BaseNode, error) {
	value, err := s.kvstore.Get(ctx, docID, s.nodeCollection)
	if err != nil {
		return nil, err
	}

	if value == nil {
		if raiseError {
			return nil, fmt.Errorf("doc_id %s not found", docID)
		}
		return nil, nil
	}

	docDict := storedValueToMap(value)
	if !isValidDocJSON(docDict) {
		return nil, fmt.Errorf("invalid JSON for doc_id %s", docID)
	}

	return jsonToDoc(docDict)
}

// DeleteDocument removes a document from the store.
func (s *KVDocumentStore) DeleteDocument(ctx context.Context, docID string, raiseError bool) error {
	// Remove from ref doc if applicable
	if err := s.removeFromRefDocNode(ctx, docID); err != nil {
		return err
	}

	// Delete from node collection
	deleted, err := s.kvstore.Delete(ctx, docID, s.nodeCollection)
	if err != nil {
		return err
	}

	// Delete from metadata collection
	if _, err := s.kvstore.Delete(ctx, docID, s.metadataCollection); err != nil {
		return err
	}

	if !deleted && raiseError {
		return fmt.Errorf("doc_id %s not found", docID)
	}

	return nil
}

// DocumentExists checks if a document exists in the store.
func (s *KVDocumentStore) DocumentExists(ctx context.Context, docID string) (bool, error) {
	value, err := s.kvstore.Get(ctx, docID, s.nodeCollection)
	if err != nil {
		return false, err
	}
	return value != nil, nil
}

// SetDocumentHash sets the hash for a document.
func (s *KVDocumentStore) SetDocumentHash(ctx context.Context, docID string, docHash string) error {
	metadata := kvstore.StoredValue{
		"doc_hash": docHash,
	}
	return s.kvstore.Put(ctx, docID, metadata, s.metadataCollection)
}

// GetDocumentHash retrieves the hash for a document.
func (s *KVDocumentStore) GetDocumentHash(ctx context.Context, docID string) (string, error) {
	metadata, err := s.kvstore.Get(ctx, docID, s.metadataCollection)
	if err != nil {
		return "", err
	}
	if metadata == nil {
		return "", nil
	}
	if hash, ok := metadata["doc_hash"].(string); ok {
		return hash, nil
	}
	return "", nil
}

// GetAllDocumentHashes returns all document hashes.
func (s *KVDocumentStore) GetAllDocumentHashes(ctx context.Context) (map[string]string, error) {
	metadataDocs, err := s.kvstore.GetAll(ctx, s.metadataCollection)
	if err != nil {
		return nil, err
	}

	hashes := make(map[string]string)
	for docID, metadata := range metadataDocs {
		if hash, ok := metadata["doc_hash"].(string); ok && hash != "" {
			hashes[hash] = docID
		}
	}
	return hashes, nil
}

// GetRefDocInfo retrieves reference document info by ID.
func (s *KVDocumentStore) GetRefDocInfo(ctx context.Context, refDocID string) (*RefDocInfo, error) {
	value, err := s.kvstore.Get(ctx, refDocID, s.refDocCollection)
	if err != nil {
		return nil, err
	}
	if value == nil {
		return nil, nil
	}
	return RefDocInfoFromMap(value), nil
}

// GetAllRefDocInfo returns all reference document info.
func (s *KVDocumentStore) GetAllRefDocInfo(ctx context.Context) (map[string]*RefDocInfo, error) {
	refDocInfos, err := s.kvstore.GetAll(ctx, s.refDocCollection)
	if err != nil {
		return nil, err
	}

	result := make(map[string]*RefDocInfo)
	for docID, info := range refDocInfos {
		result[docID] = RefDocInfoFromMap(info)
	}
	return result, nil
}

// DeleteRefDoc deletes a reference document and all its associated nodes.
func (s *KVDocumentStore) DeleteRefDoc(ctx context.Context, refDocID string, raiseError bool) error {
	refDocInfo, err := s.GetRefDocInfo(ctx, refDocID)
	if err != nil {
		return err
	}

	if refDocInfo == nil {
		if raiseError {
			return fmt.Errorf("ref_doc_id %s not found", refDocID)
		}
		return nil
	}

	// Delete all associated nodes
	nodeIDs := make([]string, len(refDocInfo.NodeIDs))
	copy(nodeIDs, refDocInfo.NodeIDs)

	for _, docID := range nodeIDs {
		if err := s.deleteDocumentWithoutRefDocUpdate(ctx, docID); err != nil {
			// Continue deleting other nodes even if one fails
			continue
		}
	}

	// Delete ref doc entries
	if _, err := s.kvstore.Delete(ctx, refDocID, s.refDocCollection); err != nil {
		return err
	}
	if _, err := s.kvstore.Delete(ctx, refDocID, s.metadataCollection); err != nil {
		return err
	}
	if _, err := s.kvstore.Delete(ctx, refDocID, s.nodeCollection); err != nil {
		return err
	}

	return nil
}

// removeFromRefDocNode removes a node from its reference document's node list.
func (s *KVDocumentStore) removeFromRefDocNode(ctx context.Context, docID string) error {
	metadata, err := s.kvstore.Get(ctx, docID, s.metadataCollection)
	if err != nil {
		return err
	}
	if metadata == nil {
		return nil
	}

	refDocID, ok := metadata["ref_doc_id"].(string)
	if !ok || refDocID == "" {
		return nil
	}

	refDocInfo, err := s.GetRefDocInfo(ctx, refDocID)
	if err != nil {
		return err
	}
	if refDocInfo == nil {
		return nil
	}

	// Remove docID from node list
	refDocInfo.NodeIDs = removeFromSlice(refDocInfo.NodeIDs, docID)

	if len(refDocInfo.NodeIDs) > 0 {
		// Update ref doc info
		if err := s.kvstore.Put(ctx, refDocID, mapToStoredValue(refDocInfo.ToMap()), s.refDocCollection); err != nil {
			return err
		}
	} else {
		// Delete ref doc if no more nodes
		if _, err := s.kvstore.Delete(ctx, refDocID, s.refDocCollection); err != nil {
			return err
		}
		if _, err := s.kvstore.Delete(ctx, refDocID, s.metadataCollection); err != nil {
			return err
		}
		if _, err := s.kvstore.Delete(ctx, refDocID, s.nodeCollection); err != nil {
			return err
		}
	}

	return nil
}

// deleteDocumentWithoutRefDocUpdate deletes a document without updating ref doc info.
func (s *KVDocumentStore) deleteDocumentWithoutRefDocUpdate(ctx context.Context, docID string) error {
	if _, err := s.kvstore.Delete(ctx, docID, s.nodeCollection); err != nil {
		return err
	}
	if _, err := s.kvstore.Delete(ctx, docID, s.metadataCollection); err != nil {
		return err
	}
	return nil
}

// Helper functions

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func removeFromSlice(slice []string, item string) []string {
	result := make([]string, 0, len(slice))
	for _, s := range slice {
		if s != item {
			result = append(result, s)
		}
	}
	return result
}

func storedValueToMap(sv kvstore.StoredValue) map[string]interface{} {
	if sv == nil {
		return nil
	}
	return map[string]interface{}(sv)
}

func mapToStoredValue(m map[string]interface{}) kvstore.StoredValue {
	if m == nil {
		return nil
	}
	// Deep copy using JSON to ensure proper type handling
	data, err := json.Marshal(m)
	if err != nil {
		return kvstore.StoredValue(m)
	}
	var result kvstore.StoredValue
	if err := json.Unmarshal(data, &result); err != nil {
		return kvstore.StoredValue(m)
	}
	return result
}

// Ensure KVDocumentStore implements DocStore.
var _ DocStore = (*KVDocumentStore)(nil)
