// Package kvstore provides key-value store interfaces and implementations.
package kvstore

import "context"

// DefaultCollection is the default collection name for key-value stores.
const DefaultCollection = "data"

// StoredValue represents a value that can be stored in the KVStore.
// It is a map of string keys to arbitrary values.
type StoredValue map[string]interface{}

// KVStore is the interface for key-value stores.
// It provides basic CRUD operations with optional collection support.
type KVStore interface {
	// Put stores a key-value pair in the specified collection.
	Put(ctx context.Context, key string, val StoredValue, collection string) error

	// Get retrieves a value by key from the specified collection.
	// Returns nil if the key does not exist.
	Get(ctx context.Context, key string, collection string) (StoredValue, error)

	// GetAll retrieves all key-value pairs from the specified collection.
	GetAll(ctx context.Context, collection string) (map[string]StoredValue, error)

	// Delete removes a key-value pair from the specified collection.
	// Returns true if the key was deleted, false if it did not exist.
	Delete(ctx context.Context, key string, collection string) (bool, error)
}

// PersistableKVStore extends KVStore with persistence capabilities.
type PersistableKVStore interface {
	KVStore

	// Persist saves the store to the specified path.
	Persist(ctx context.Context, persistPath string) error
}
