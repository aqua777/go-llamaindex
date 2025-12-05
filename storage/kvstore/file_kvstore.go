package kvstore

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
)

// FileKVStore is a file-based key-value store implementation.
// It persists data to disk on every write operation.
type FileKVStore struct {
	mu          sync.RWMutex
	data        DataType
	persistPath string
}

// NewFileKVStore creates a new FileKVStore with the specified persist path.
// If the file exists, it loads the data from disk.
func NewFileKVStore(persistPath string) (*FileKVStore, error) {
	store := &FileKVStore{
		data:        make(DataType),
		persistPath: persistPath,
	}

	// Ensure directory exists
	dirPath := filepath.Dir(persistPath)
	if err := os.MkdirAll(dirPath, 0755); err != nil {
		return nil, err
	}

	// Load existing data if file exists
	if _, err := os.Stat(persistPath); err == nil {
		data, err := os.ReadFile(persistPath)
		if err != nil {
			return nil, err
		}
		if len(data) > 0 {
			if err := json.Unmarshal(data, &store.data); err != nil {
				return nil, err
			}
		}
	}

	return store, nil
}

// Put stores a key-value pair in the specified collection and persists to disk.
func (s *FileKVStore) Put(ctx context.Context, key string, val StoredValue, collection string) error {
	if collection == "" {
		collection = DefaultCollection
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.data[collection]; !exists {
		s.data[collection] = make(map[string]StoredValue)
	}
	s.data[collection][key] = copyStoredValue(val)

	return s.persistLocked()
}

// Get retrieves a value by key from the specified collection.
// Returns nil if the key does not exist.
func (s *FileKVStore) Get(ctx context.Context, key string, collection string) (StoredValue, error) {
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
func (s *FileKVStore) GetAll(ctx context.Context, collection string) (map[string]StoredValue, error) {
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

// Delete removes a key-value pair from the specified collection and persists to disk.
// Returns true if the key was deleted, false if it did not exist.
func (s *FileKVStore) Delete(ctx context.Context, key string, collection string) (bool, error) {
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

	if err := s.persistLocked(); err != nil {
		return true, err
	}
	return true, nil
}

// Persist saves the store to disk.
func (s *FileKVStore) Persist(ctx context.Context, persistPath string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Update persist path if different
	if persistPath != "" && persistPath != s.persistPath {
		s.persistPath = persistPath
		dirPath := filepath.Dir(persistPath)
		if err := os.MkdirAll(dirPath, 0755); err != nil {
			return err
		}
	}

	return s.persistLocked()
}

// persistLocked saves the store to disk (caller must hold lock).
func (s *FileKVStore) persistLocked() error {
	data, err := json.Marshal(s.data)
	if err != nil {
		return err
	}

	return os.WriteFile(s.persistPath, data, 0644)
}

// GetPersistPath returns the current persist path.
func (s *FileKVStore) GetPersistPath() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.persistPath
}

// ToDict returns a copy of the internal data.
func (s *FileKVStore) ToDict() DataType {
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

// Ensure FileKVStore implements the interfaces.
var (
	_ KVStore            = (*FileKVStore)(nil)
	_ PersistableKVStore = (*FileKVStore)(nil)
)
