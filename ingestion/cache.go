// Package ingestion provides document ingestion pipeline functionality.
package ingestion

import (
	"encoding/json"
	"os"
	"sync"

	"github.com/aqua777/go-llamaindex/schema"
)

// DefaultCacheName is the default cache collection name.
const DefaultCacheName = "llama_cache"

// IngestionCache provides caching for ingestion pipeline transformations.
type IngestionCache struct {
	collection string
	nodesKey   string
	cache      map[string]map[string]interface{}
	mu         sync.RWMutex
}

// IngestionCacheOption configures an IngestionCache.
type IngestionCacheOption func(*IngestionCache)

// WithCacheCollection sets the cache collection name.
func WithCacheCollection(collection string) IngestionCacheOption {
	return func(c *IngestionCache) {
		c.collection = collection
	}
}

// NewIngestionCache creates a new IngestionCache.
func NewIngestionCache(opts ...IngestionCacheOption) *IngestionCache {
	c := &IngestionCache{
		collection: DefaultCacheName,
		nodesKey:   "nodes",
		cache:      make(map[string]map[string]interface{}),
	}

	for _, opt := range opts {
		opt(c)
	}

	// Initialize the default collection
	if _, ok := c.cache[c.collection]; !ok {
		c.cache[c.collection] = make(map[string]interface{})
	}

	return c
}

// Put stores nodes in the cache.
func (c *IngestionCache) Put(key string, nodes []schema.Node, collection string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if collection == "" {
		collection = c.collection
	}

	if _, ok := c.cache[collection]; !ok {
		c.cache[collection] = make(map[string]interface{})
	}

	// Serialize nodes
	nodeData := make([]map[string]interface{}, len(nodes))
	for i, node := range nodes {
		nodeData[i] = map[string]interface{}{
			"id":       node.ID,
			"text":     node.Text,
			"metadata": node.Metadata,
		}
	}

	c.cache[collection][key] = map[string]interface{}{
		c.nodesKey: nodeData,
	}
}

// Get retrieves nodes from the cache.
func (c *IngestionCache) Get(key string, collection string) ([]schema.Node, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if collection == "" {
		collection = c.collection
	}

	collectionCache, ok := c.cache[collection]
	if !ok {
		return nil, false
	}

	data, ok := collectionCache[key]
	if !ok {
		return nil, false
	}

	dataMap, ok := data.(map[string]interface{})
	if !ok {
		return nil, false
	}

	nodeDataList, ok := dataMap[c.nodesKey].([]map[string]interface{})
	if !ok {
		return nil, false
	}

	nodes := make([]schema.Node, len(nodeDataList))
	for i, nodeData := range nodeDataList {
		nodes[i] = schema.Node{
			ID:   nodeData["id"].(string),
			Text: nodeData["text"].(string),
		}
		if metadata, ok := nodeData["metadata"].(map[string]interface{}); ok {
			nodes[i].Metadata = metadata
		}
	}

	return nodes, true
}

// Clear clears the cache for a collection.
func (c *IngestionCache) Clear(collection string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if collection == "" {
		collection = c.collection
	}

	c.cache[collection] = make(map[string]interface{})
}

// Persist saves the cache to a file.
func (c *IngestionCache) Persist(path string) error {
	c.mu.RLock()
	defer c.mu.RUnlock()

	data, err := json.MarshalIndent(c.cache, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)
}

// LoadFromPath loads the cache from a file.
func (c *IngestionCache) LoadFromPath(path string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	return json.Unmarshal(data, &c.cache)
}

// NewIngestionCacheFromPath creates an IngestionCache from a persist path.
func NewIngestionCacheFromPath(path string, collection string) (*IngestionCache, error) {
	if collection == "" {
		collection = DefaultCacheName
	}

	cache := NewIngestionCache(WithCacheCollection(collection))
	if err := cache.LoadFromPath(path); err != nil {
		return nil, err
	}

	return cache, nil
}

// Collection returns the current collection name.
func (c *IngestionCache) Collection() string {
	return c.collection
}

// HasKey checks if a key exists in the cache.
func (c *IngestionCache) HasKey(key string, collection string) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if collection == "" {
		collection = c.collection
	}

	collectionCache, ok := c.cache[collection]
	if !ok {
		return false
	}

	_, ok = collectionCache[key]
	return ok
}
