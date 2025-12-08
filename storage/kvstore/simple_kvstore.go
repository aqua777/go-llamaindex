package kvstore

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
)

// DataType represents the internal data structure for SimpleKVStore.
// It maps collection names to their key-value pairs.
type DataType map[string]map[string]StoredValue

// SimpleKVStore is an in-memory key-value store implementation.
// It is thread-safe and supports optional persistence.
type SimpleKVStore struct {
	mu          sync.RWMutex
	data        DataType
	persistPath string
}

// NewSimpleKVStore creates a new SimpleKVStore.
func NewSimpleKVStore() *SimpleKVStore {
	return &SimpleKVStore{
		data: make(DataType),
	}
}

// NewSimpleKVStoreWithData creates a new SimpleKVStore with initial data.
func NewSimpleKVStoreWithData(data DataType) *SimpleKVStore {
	if data == nil {
		data = make(DataType)
	}
	// Deep copy the data
	copiedData := make(DataType)
	for collection, kvPairs := range data {
		copiedData[collection] = make(map[string]StoredValue)
		for k, v := range kvPairs {
			copiedData[collection][k] = copyStoredValue(v)
		}
	}
	return &SimpleKVStore{
		data: copiedData,
	}
}

// Put stores a key-value pair in the specified collection.
func (s *SimpleKVStore) Put(ctx context.Context, key string, val StoredValue, collection string) error {
	if collection == "" {
		collection = DefaultCollection
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.data[collection]; !exists {
		s.data[collection] = make(map[string]StoredValue)
	}
	s.data[collection][key] = copyStoredValue(val)

	// Auto-persist if persistPath is set
	if s.persistPath != "" {
		return s.persistLocked(ctx, s.persistPath)
	}
	return nil
}

// Get retrieves a value by key from the specified collection.
// Returns nil if the key does not exist.
func (s *SimpleKVStore) Get(ctx context.Context, key string, collection string) (StoredValue, error) {
	if collection == "" {
		collection = DefaultCollection
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	collectionData, exists := s.data[collection]
	if !exists {
		return nil, nil
	}

	val, exists := collectionData[key]
	if !exists {
		return nil, nil
	}

	return copyStoredValue(val), nil
}

// GetAll retrieves all key-value pairs from the specified collection.
func (s *SimpleKVStore) GetAll(ctx context.Context, collection string) (map[string]StoredValue, error) {
	if collection == "" {
		collection = DefaultCollection
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	collectionData, exists := s.data[collection]
	if !exists {
		return make(map[string]StoredValue), nil
	}

	// Deep copy the collection data
	result := make(map[string]StoredValue)
	for k, v := range collectionData {
		result[k] = copyStoredValue(v)
	}
	return result, nil
}

// Delete removes a key-value pair from the specified collection.
// Returns true if the key was deleted, false if it did not exist.
func (s *SimpleKVStore) Delete(ctx context.Context, key string, collection string) (bool, error) {
	if collection == "" {
		collection = DefaultCollection
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	collectionData, exists := s.data[collection]
	if !exists {
		return false, nil
	}

	if _, exists := collectionData[key]; !exists {
		return false, nil
	}

	delete(collectionData, key)

	// Auto-persist if persistPath is set
	if s.persistPath != "" {
		if err := s.persistLocked(ctx, s.persistPath); err != nil {
			return true, err
		}
	}
	return true, nil
}

// Persist saves the store to the specified path.
func (s *SimpleKVStore) Persist(ctx context.Context, persistPath string) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.persistLocked(ctx, persistPath)
}

// persistLocked saves the store to the specified path (caller must hold lock).
func (s *SimpleKVStore) persistLocked(ctx context.Context, persistPath string) error {
	dirPath := filepath.Dir(persistPath)
	if err := os.MkdirAll(dirPath, 0755); err != nil {
		return err
	}

	data, err := json.Marshal(s.data)
	if err != nil {
		return err
	}

	return os.WriteFile(persistPath, data, 0644)
}

// FromPersistPath loads a SimpleKVStore from a persist path.
func FromPersistPath(ctx context.Context, persistPath string) (*SimpleKVStore, error) {
	dirPath := filepath.Dir(persistPath)
	if err := os.MkdirAll(dirPath, 0755); err != nil {
		return nil, err
	}

	store := NewSimpleKVStore()
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

	if err := json.Unmarshal(data, &store.data); err != nil {
		return nil, err
	}

	return store, nil
}

// ToDict returns a copy of the internal data.
func (s *SimpleKVStore) ToDict() DataType {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make(DataType)
	for collection, kvPairs := range s.data {
		result[collection] = make(map[string]StoredValue)
		for k, v := range kvPairs {
			result[collection][k] = copyStoredValue(v)
		}
	}
	return result
}

// FromDict creates a SimpleKVStore from a dictionary.
func FromDict(data DataType) *SimpleKVStore {
	return NewSimpleKVStoreWithData(data)
}

// copyStoredValue creates a deep copy of a StoredValue.
func copyStoredValue(val StoredValue) StoredValue {
	if val == nil {
		return nil
	}
	// Use JSON marshal/unmarshal for deep copy
	data, err := json.Marshal(val)
	if err != nil {
		// Fallback to shallow copy if marshal fails
		result := make(StoredValue)
		for k, v := range val {
			result[k] = v
		}
		return result
	}
	var result StoredValue
	if err := json.Unmarshal(data, &result); err != nil {
		// Fallback to shallow copy if unmarshal fails
		result = make(StoredValue)
		for k, v := range val {
			result[k] = v
		}
	}
	return result
}

// Ensure SimpleKVStore implements the interfaces.
var (
	_ KVStore            = (*SimpleKVStore)(nil)
	_ PersistableKVStore = (*SimpleKVStore)(nil)
)
