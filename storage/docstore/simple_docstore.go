package docstore

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"

	"github.com/aqua777/go-llamaindex/storage/kvstore"
)

const (
	// DefaultPersistDir is the default directory for persistence.
	DefaultPersistDir = "./storage"
	// DefaultPersistFilename is the default filename for persistence.
	DefaultPersistFilename = "docstore.json"
)

// SimpleDocumentStore is an in-memory document store with optional persistence.
// It wraps a KVDocumentStore backed by a SimpleKVStore.
type SimpleDocumentStore struct {
	*KVDocumentStore
	kvstore     *kvstore.SimpleKVStore
	persistPath string
}

// SimpleDocumentStoreOption is a functional option for SimpleDocumentStore.
type SimpleDocumentStoreOption func(*SimpleDocumentStore)

// WithSimpleDocStoreNamespace sets the namespace for the document store.
func WithSimpleDocStoreNamespace(namespace string) SimpleDocumentStoreOption {
	return func(s *SimpleDocumentStore) {
		s.KVDocumentStore = NewKVDocumentStore(s.kvstore, WithNamespace(namespace))
	}
}

// NewSimpleDocumentStore creates a new SimpleDocumentStore.
func NewSimpleDocumentStore(opts ...SimpleDocumentStoreOption) *SimpleDocumentStore {
	kv := kvstore.NewSimpleKVStore()
	store := &SimpleDocumentStore{
		kvstore:         kv,
		KVDocumentStore: NewKVDocumentStore(kv),
	}

	for _, opt := range opts {
		opt(store)
	}

	return store
}

// Persist saves the document store to disk.
func (s *SimpleDocumentStore) Persist(ctx context.Context, persistPath string) error {
	if persistPath == "" {
		persistPath = filepath.Join(DefaultPersistDir, DefaultPersistFilename)
	}

	dirPath := filepath.Dir(persistPath)
	if err := os.MkdirAll(dirPath, 0755); err != nil {
		return err
	}

	data := s.kvstore.ToDict()
	jsonData, err := json.Marshal(data)
	if err != nil {
		return err
	}

	return os.WriteFile(persistPath, jsonData, 0644)
}

// FromPersistPath loads a SimpleDocumentStore from a persist path.
func FromPersistPath(ctx context.Context, persistPath string) (*SimpleDocumentStore, error) {
	if persistPath == "" {
		persistPath = filepath.Join(DefaultPersistDir, DefaultPersistFilename)
	}

	dirPath := filepath.Dir(persistPath)
	if err := os.MkdirAll(dirPath, 0755); err != nil {
		return nil, err
	}

	store := NewSimpleDocumentStore()
	store.persistPath = persistPath

	// Check if file exists
	if _, err := os.Stat(persistPath); os.IsNotExist(err) {
		// File doesn't exist, return empty store
		return store, nil
	}

	data, err := os.ReadFile(persistPath)
	if err != nil {
		return nil, err
	}

	var kvData kvstore.DataType
	if err := json.Unmarshal(data, &kvData); err != nil {
		return nil, err
	}

	store.kvstore = kvstore.FromDict(kvData)
	store.KVDocumentStore = NewKVDocumentStore(store.kvstore)

	return store, nil
}

// ToDict returns a copy of the internal data.
func (s *SimpleDocumentStore) ToDict() kvstore.DataType {
	return s.kvstore.ToDict()
}

// FromDict creates a SimpleDocumentStore from a dictionary.
func FromDict(data kvstore.DataType) *SimpleDocumentStore {
	kv := kvstore.FromDict(data)
	return &SimpleDocumentStore{
		kvstore:         kv,
		KVDocumentStore: NewKVDocumentStore(kv),
	}
}

// Ensure SimpleDocumentStore implements DocStore.
var _ DocStore = (*SimpleDocumentStore)(nil)
