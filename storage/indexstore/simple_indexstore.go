package indexstore

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
	DefaultPersistFilename = "index_store.json"
)

// SimpleIndexStore is an in-memory index store with optional persistence.
// It wraps a KVIndexStore backed by a SimpleKVStore.
type SimpleIndexStore struct {
	*KVIndexStore
	kvstore     *kvstore.SimpleKVStore
	persistPath string
}

// SimpleIndexStoreOption is a functional option for SimpleIndexStore.
type SimpleIndexStoreOption func(*SimpleIndexStore)

// WithSimpleIndexStoreNamespace sets the namespace for the index store.
func WithSimpleIndexStoreNamespace(namespace string) SimpleIndexStoreOption {
	return func(s *SimpleIndexStore) {
		s.KVIndexStore = NewKVIndexStore(s.kvstore, WithIndexStoreNamespace(namespace))
	}
}

// NewSimpleIndexStore creates a new SimpleIndexStore.
func NewSimpleIndexStore(opts ...SimpleIndexStoreOption) *SimpleIndexStore {
	kv := kvstore.NewSimpleKVStore()
	store := &SimpleIndexStore{
		kvstore:      kv,
		KVIndexStore: NewKVIndexStore(kv),
	}

	for _, opt := range opts {
		opt(store)
	}

	return store
}

// Persist saves the index store to disk.
func (s *SimpleIndexStore) Persist(ctx context.Context, persistPath string) error {
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

// SimpleIndexStoreFromPersistPath loads a SimpleIndexStore from a persist path.
func SimpleIndexStoreFromPersistPath(ctx context.Context, persistPath string) (*SimpleIndexStore, error) {
	if persistPath == "" {
		persistPath = filepath.Join(DefaultPersistDir, DefaultPersistFilename)
	}

	dirPath := filepath.Dir(persistPath)
	if err := os.MkdirAll(dirPath, 0755); err != nil {
		return nil, err
	}

	store := NewSimpleIndexStore()
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
	store.KVIndexStore = NewKVIndexStore(store.kvstore)

	return store, nil
}

// ToDict returns a copy of the internal data.
func (s *SimpleIndexStore) ToDict() kvstore.DataType {
	return s.kvstore.ToDict()
}

// SimpleIndexStoreFromDict creates a SimpleIndexStore from a dictionary.
func SimpleIndexStoreFromDict(data kvstore.DataType) *SimpleIndexStore {
	kv := kvstore.FromDict(data)
	return &SimpleIndexStore{
		kvstore:      kv,
		KVIndexStore: NewKVIndexStore(kv),
	}
}

// Ensure SimpleIndexStore implements IndexStore.
var _ IndexStore = (*SimpleIndexStore)(nil)
