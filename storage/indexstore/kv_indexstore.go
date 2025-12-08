package indexstore

import (
	"context"
	"fmt"

	"github.com/aqua777/go-llamaindex/storage/kvstore"
)

const (
	// Collection suffix for index data.
	collectionSuffix = "/data"

	// Keys for serialization.
	typeKey = "__type__"
	dataKey = "__data__"
)

// KVIndexStore is an index store backed by a KVStore.
type KVIndexStore struct {
	kvstore    kvstore.KVStore
	namespace  string
	collection string
}

// KVIndexStoreOption is a functional option for KVIndexStore.
type KVIndexStoreOption func(*KVIndexStore)

// WithIndexStoreNamespace sets the namespace for the index store.
func WithIndexStoreNamespace(namespace string) KVIndexStoreOption {
	return func(s *KVIndexStore) {
		s.namespace = namespace
	}
}

// NewKVIndexStore creates a new KVIndexStore.
func NewKVIndexStore(kv kvstore.KVStore, opts ...KVIndexStoreOption) *KVIndexStore {
	store := &KVIndexStore{
		kvstore:   kv,
		namespace: DefaultNamespace,
	}

	for _, opt := range opts {
		opt(store)
	}

	store.collection = store.namespace + collectionSuffix

	return store
}

// indexStructToJSON converts an IndexStruct to a JSON-serializable map.
func indexStructToJSON(is *IndexStruct) map[string]interface{} {
	return map[string]interface{}{
		typeKey: string(is.Type),
		dataKey: is.ToJSON(),
	}
}

// jsonToIndexStruct converts a JSON map back to an IndexStruct.
func jsonToIndexStruct(data map[string]interface{}) (*IndexStruct, error) {
	dataDict, ok := data[dataKey].(map[string]interface{})
	if !ok {
		// Try direct parsing (for backward compatibility)
		return FromJSON(data)
	}
	return FromJSON(dataDict)
}

// AddIndexStruct adds an index struct to the store.
func (s *KVIndexStore) AddIndexStruct(ctx context.Context, indexStruct *IndexStruct) error {
	key := indexStruct.IndexID
	data := indexStructToJSON(indexStruct)
	return s.kvstore.Put(ctx, key, mapToStoredValue(data), s.collection)
}

// DeleteIndexStruct removes an index struct from the store.
func (s *KVIndexStore) DeleteIndexStruct(ctx context.Context, key string) error {
	_, err := s.kvstore.Delete(ctx, key, s.collection)
	return err
}

// GetIndexStruct retrieves an index struct by ID.
func (s *KVIndexStore) GetIndexStruct(ctx context.Context, structID string) (*IndexStruct, error) {
	if structID == "" {
		structs, err := s.IndexStructs(ctx)
		if err != nil {
			return nil, err
		}
		if len(structs) == 0 {
			return nil, ErrIndexStructNotFound
		}
		if len(structs) > 1 {
			return nil, ErrMultipleIndexStructs
		}
		return structs[0], nil
	}

	value, err := s.kvstore.Get(ctx, structID, s.collection)
	if err != nil {
		return nil, err
	}
	if value == nil {
		return nil, ErrIndexStructNotFound
	}

	return jsonToIndexStruct(storedValueToMap(value))
}

// IndexStructs returns all index structs in the store.
func (s *KVIndexStore) IndexStructs(ctx context.Context) ([]*IndexStruct, error) {
	jsons, err := s.kvstore.GetAll(ctx, s.collection)
	if err != nil {
		return nil, err
	}

	structs := make([]*IndexStruct, 0, len(jsons))
	for _, jsonData := range jsons {
		is, err := jsonToIndexStruct(storedValueToMap(jsonData))
		if err != nil {
			continue // Skip invalid entries
		}
		structs = append(structs, is)
	}

	return structs, nil
}

// Helper functions

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
	return kvstore.StoredValue(m)
}

// Ensure KVIndexStore implements IndexStore.
var _ IndexStore = (*KVIndexStore)(nil)

// GetIndexStructByType retrieves an index struct by type.
func (s *KVIndexStore) GetIndexStructByType(ctx context.Context, structType IndexStructType) (*IndexStruct, error) {
	structs, err := s.IndexStructs(ctx)
	if err != nil {
		return nil, err
	}

	for _, is := range structs {
		if is.Type == structType {
			return is, nil
		}
	}

	return nil, fmt.Errorf("index struct with type %s not found", structType)
}
