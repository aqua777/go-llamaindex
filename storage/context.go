// Package storage provides unified storage management for LlamaIndex.
package storage

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"

	"github.com/aqua777/go-llamaindex/rag/store"
	"github.com/aqua777/go-llamaindex/storage/docstore"
	"github.com/aqua777/go-llamaindex/storage/indexstore"
	"github.com/aqua777/go-llamaindex/storage/kvstore"
)

const (
	// DefaultPersistDir is the default directory for persistence.
	DefaultPersistDir = "./storage"
	// DocStoreFilename is the default filename for document store.
	DocStoreFilename = "docstore.json"
	// IndexStoreFilename is the default filename for index store.
	IndexStoreFilename = "index_store.json"
	// VectorStoreFilename is the default filename for vector store.
	VectorStoreFilename = "vector_store.json"

	// Storage context keys for serialization.
	DocStoreKey    = "docstore"
	IndexStoreKey  = "index_store"
	VectorStoreKey = "vector_store"

	// DefaultVectorStore is the default namespace for vector stores.
	DefaultVectorStore = "default"
	// NamespaceSeparator separates namespace from filename.
	NamespaceSeparator = "__"
)

// StorageContext is a unified container for storage components.
// It manages document store, index store, and vector stores together.
type StorageContext struct {
	// DocStore stores document nodes and their metadata.
	DocStore docstore.DocStore
	// IndexStore stores index structures.
	IndexStore indexstore.IndexStore
	// VectorStores maps namespace to vector store instances.
	VectorStores map[string]store.VectorStore
}

// StorageContextOptions configures StorageContext creation.
type StorageContextOptions struct {
	// DocStore is an optional pre-configured document store.
	DocStore docstore.DocStore
	// IndexStore is an optional pre-configured index store.
	IndexStore indexstore.IndexStore
	// VectorStore is an optional pre-configured vector store (uses default namespace).
	VectorStore store.VectorStore
	// VectorStores is an optional map of namespaced vector stores.
	VectorStores map[string]store.VectorStore
	// PersistDir is the directory to load persisted data from.
	PersistDir string
}

// NewStorageContext creates a new StorageContext with default stores.
func NewStorageContext() *StorageContext {
	return &StorageContext{
		DocStore:     docstore.NewSimpleDocumentStore(),
		IndexStore:   indexstore.NewSimpleIndexStore(),
		VectorStores: make(map[string]store.VectorStore),
	}
}

// NewStorageContextFromOptions creates a StorageContext from options.
func NewStorageContextFromOptions(ctx context.Context, opts StorageContextOptions) (*StorageContext, error) {
	sc := &StorageContext{
		VectorStores: make(map[string]store.VectorStore),
	}

	if opts.PersistDir != "" {
		// Load from persist directory
		return StorageContextFromPersistDir(ctx, opts.PersistDir)
	}

	// Use provided stores or create defaults
	if opts.DocStore != nil {
		sc.DocStore = opts.DocStore
	} else {
		sc.DocStore = docstore.NewSimpleDocumentStore()
	}

	if opts.IndexStore != nil {
		sc.IndexStore = opts.IndexStore
	} else {
		sc.IndexStore = indexstore.NewSimpleIndexStore()
	}

	// Handle vector stores
	if opts.VectorStores != nil {
		sc.VectorStores = opts.VectorStores
	} else if opts.VectorStore != nil {
		sc.VectorStores[DefaultVectorStore] = opts.VectorStore
	}

	return sc, nil
}

// StorageContextFromPersistDir loads a StorageContext from a persist directory.
func StorageContextFromPersistDir(ctx context.Context, persistDir string) (*StorageContext, error) {
	sc := &StorageContext{
		VectorStores: make(map[string]store.VectorStore),
	}

	// Load document store
	docStorePath := filepath.Join(persistDir, DocStoreFilename)
	docStore, err := docstore.FromPersistPath(ctx, docStorePath)
	if err != nil {
		// If file doesn't exist, create new store
		docStore = docstore.NewSimpleDocumentStore()
	}
	sc.DocStore = docStore

	// Load index store
	indexStorePath := filepath.Join(persistDir, IndexStoreFilename)
	indexStore, err := indexstore.SimpleIndexStoreFromPersistPath(ctx, indexStorePath)
	if err != nil {
		// If file doesn't exist, create new store
		indexStore = indexstore.NewSimpleIndexStore()
	}
	sc.IndexStore = indexStore

	// Note: Vector stores are not automatically loaded as they may use different backends
	// Users should provide vector stores explicitly or use ToDict/FromDict for simple stores

	return sc, nil
}

// VectorStore returns the default vector store.
// Returns nil if no default vector store is configured.
func (sc *StorageContext) VectorStore() store.VectorStore {
	return sc.VectorStores[DefaultVectorStore]
}

// SetVectorStore sets the default vector store.
func (sc *StorageContext) SetVectorStore(vs store.VectorStore) {
	if sc.VectorStores == nil {
		sc.VectorStores = make(map[string]store.VectorStore)
	}
	sc.VectorStores[DefaultVectorStore] = vs
}

// AddVectorStore adds a vector store with a namespace.
func (sc *StorageContext) AddVectorStore(namespace string, vs store.VectorStore) {
	if sc.VectorStores == nil {
		sc.VectorStores = make(map[string]store.VectorStore)
	}
	sc.VectorStores[namespace] = vs
}

// GetVectorStore retrieves a vector store by namespace.
func (sc *StorageContext) GetVectorStore(namespace string) store.VectorStore {
	return sc.VectorStores[namespace]
}

// Persist saves the storage context to disk.
func (sc *StorageContext) Persist(ctx context.Context, persistDir string) error {
	if persistDir == "" {
		persistDir = DefaultPersistDir
	}

	// Create directory if it doesn't exist
	if err := os.MkdirAll(persistDir, 0755); err != nil {
		return err
	}

	// Persist document store if it supports persistence
	if simpleDocStore, ok := sc.DocStore.(*docstore.SimpleDocumentStore); ok {
		docStorePath := filepath.Join(persistDir, DocStoreFilename)
		if err := simpleDocStore.Persist(ctx, docStorePath); err != nil {
			return err
		}
	}

	// Persist index store if it supports persistence
	if simpleIndexStore, ok := sc.IndexStore.(*indexstore.SimpleIndexStore); ok {
		indexStorePath := filepath.Join(persistDir, IndexStoreFilename)
		if err := simpleIndexStore.Persist(ctx, indexStorePath); err != nil {
			return err
		}
	}

	// Note: Vector stores need to be persisted separately as they may have different backends
	// For SimpleVectorStore, users can call Persist directly on the store

	return nil
}

// StorageContextData represents serializable storage context data.
type StorageContextData struct {
	DocStore   map[string]interface{} `json:"docstore,omitempty"`
	IndexStore map[string]interface{} `json:"index_store,omitempty"`
}

// ToDict converts the storage context to a dictionary representation.
// Only works with simple stores that support ToDict.
func (sc *StorageContext) ToDict(ctx context.Context) (map[string]interface{}, error) {
	result := make(map[string]interface{})

	// Convert document store
	if simpleDocStore, ok := sc.DocStore.(*docstore.SimpleDocumentStore); ok {
		result[DocStoreKey] = simpleDocStore.ToDict()
	}

	// Convert index store
	if simpleIndexStore, ok := sc.IndexStore.(*indexstore.SimpleIndexStore); ok {
		result[IndexStoreKey] = simpleIndexStore.ToDict()
	}

	return result, nil
}

// StorageContextFromDict creates a StorageContext from a dictionary.
func StorageContextFromDict(ctx context.Context, data map[string]interface{}) (*StorageContext, error) {
	sc := &StorageContext{
		VectorStores: make(map[string]store.VectorStore),
	}

	// Load document store
	if docStoreData, ok := data[DocStoreKey].(map[string]interface{}); ok {
		// Convert map[string]interface{} to kvstore.DataType
		kvData := convertToKVDataType(docStoreData)
		sc.DocStore = docstore.FromDict(kvData)
	} else {
		sc.DocStore = docstore.NewSimpleDocumentStore()
	}

	// Load index store
	if indexStoreData, ok := data[IndexStoreKey].(map[string]interface{}); ok {
		// Convert map[string]interface{} to kvstore.DataType
		kvData := convertToKVDataType(indexStoreData)
		sc.IndexStore = indexstore.SimpleIndexStoreFromDict(kvData)
	} else {
		sc.IndexStore = indexstore.NewSimpleIndexStore()
	}

	return sc, nil
}

// convertToKVDataType converts map[string]interface{} to kvstore.DataType.
func convertToKVDataType(data map[string]interface{}) kvstore.DataType {
	result := make(kvstore.DataType)
	for collection, collData := range data {
		if collMap, ok := collData.(map[string]interface{}); ok {
			result[collection] = make(map[string]kvstore.StoredValue)
			for key, value := range collMap {
				if valueMap, ok := value.(map[string]interface{}); ok {
					result[collection][key] = valueMap
				}
			}
		}
	}
	return result
}

// ToJSON converts the storage context to JSON.
func (sc *StorageContext) ToJSON(ctx context.Context) ([]byte, error) {
	data, err := sc.ToDict(ctx)
	if err != nil {
		return nil, err
	}
	return json.MarshalIndent(data, "", "  ")
}

// StorageContextFromJSON creates a StorageContext from JSON.
func StorageContextFromJSON(ctx context.Context, jsonData []byte) (*StorageContext, error) {
	var data map[string]interface{}
	if err := json.Unmarshal(jsonData, &data); err != nil {
		return nil, err
	}
	return StorageContextFromDict(ctx, data)
}
