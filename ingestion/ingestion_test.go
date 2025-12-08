package ingestion

import (
	"context"
	"os"
	"testing"

	"github.com/aqua777/go-llamaindex/schema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// MockTransform is a mock transformation for testing.
type MockTransform struct {
	name      string
	transform func([]schema.Node) []schema.Node
}

func (m *MockTransform) Transform(ctx context.Context, nodes []schema.Node) ([]schema.Node, error) {
	if m.transform != nil {
		return m.transform(nodes), nil
	}
	return nodes, nil
}

func (m *MockTransform) Name() string {
	return m.name
}

// MockDocStore is a mock document store for testing.
type MockDocStore struct {
	hashes    map[string]string
	documents map[string]schema.Node
}

func NewMockDocStore() *MockDocStore {
	return &MockDocStore{
		hashes:    make(map[string]string),
		documents: make(map[string]schema.Node),
	}
}

func (m *MockDocStore) GetDocumentHash(docID string) (string, bool) {
	hash, ok := m.hashes[docID]
	return hash, ok
}

func (m *MockDocStore) SetDocumentHash(docID string, hash string) {
	m.hashes[docID] = hash
}

func (m *MockDocStore) GetAllDocumentHashes() map[string]string {
	return m.hashes
}

func (m *MockDocStore) AddDocuments(nodes []schema.Node) error {
	for _, node := range nodes {
		m.documents[node.ID] = node
	}
	return nil
}

func (m *MockDocStore) DeleteDocument(docID string) error {
	delete(m.documents, docID)
	delete(m.hashes, docID)
	return nil
}

func (m *MockDocStore) DeleteRefDoc(refDocID string) error {
	delete(m.documents, refDocID)
	delete(m.hashes, refDocID)
	return nil
}

// MockVectorStore is a mock vector store for testing.
type MockVectorStore struct {
	nodes map[string]schema.Node
}

func NewMockVectorStore() *MockVectorStore {
	return &MockVectorStore{
		nodes: make(map[string]schema.Node),
	}
}

func (m *MockVectorStore) Add(ctx context.Context, nodes []schema.Node) error {
	for _, node := range nodes {
		m.nodes[node.ID] = node
	}
	return nil
}

func (m *MockVectorStore) Delete(ctx context.Context, refDocID string) error {
	delete(m.nodes, refDocID)
	return nil
}

// TestIngestionCache tests the IngestionCache.
func TestIngestionCache(t *testing.T) {
	t.Run("NewIngestionCache", func(t *testing.T) {
		cache := NewIngestionCache()
		assert.NotNil(t, cache)
		assert.Equal(t, DefaultCacheName, cache.Collection())
	})

	t.Run("WithCacheCollection", func(t *testing.T) {
		cache := NewIngestionCache(WithCacheCollection("custom"))
		assert.Equal(t, "custom", cache.Collection())
	})

	t.Run("Put and Get", func(t *testing.T) {
		cache := NewIngestionCache()

		nodes := []schema.Node{
			{ID: "1", Text: "Hello"},
			{ID: "2", Text: "World"},
		}

		cache.Put("key1", nodes, "")

		retrieved, found := cache.Get("key1", "")
		require.True(t, found)
		assert.Len(t, retrieved, 2)
		assert.Equal(t, "1", retrieved[0].ID)
		assert.Equal(t, "Hello", retrieved[0].Text)
	})

	t.Run("Get non-existent key", func(t *testing.T) {
		cache := NewIngestionCache()

		_, found := cache.Get("nonexistent", "")
		assert.False(t, found)
	})

	t.Run("HasKey", func(t *testing.T) {
		cache := NewIngestionCache()

		nodes := []schema.Node{{ID: "1", Text: "Test"}}
		cache.Put("key1", nodes, "")

		assert.True(t, cache.HasKey("key1", ""))
		assert.False(t, cache.HasKey("key2", ""))
	})

	t.Run("Clear", func(t *testing.T) {
		cache := NewIngestionCache()

		nodes := []schema.Node{{ID: "1", Text: "Test"}}
		cache.Put("key1", nodes, "")

		cache.Clear("")

		assert.False(t, cache.HasKey("key1", ""))
	})

	t.Run("Persist and Load", func(t *testing.T) {
		cache := NewIngestionCache()

		nodes := []schema.Node{
			{ID: "1", Text: "Hello"},
		}
		cache.Put("key1", nodes, "")

		// Persist
		tmpFile, err := os.CreateTemp("", "cache_test_*.json")
		require.NoError(t, err)
		defer os.Remove(tmpFile.Name())
		tmpFile.Close()

		err = cache.Persist(tmpFile.Name())
		require.NoError(t, err)

		// Load into new cache
		newCache := NewIngestionCache()
		err = newCache.LoadFromPath(tmpFile.Name())
		require.NoError(t, err)

		// Verify data
		assert.True(t, newCache.HasKey("key1", DefaultCacheName))
	})
}

// TestIngestionPipeline tests the IngestionPipeline.
func TestIngestionPipeline(t *testing.T) {
	ctx := context.Background()

	t.Run("NewIngestionPipeline", func(t *testing.T) {
		pipeline := NewIngestionPipeline()
		assert.NotNil(t, pipeline)
		assert.Equal(t, "default", pipeline.Name())
	})

	t.Run("WithPipelineName", func(t *testing.T) {
		pipeline := NewIngestionPipeline(WithPipelineName("custom"))
		assert.Equal(t, "custom", pipeline.Name())
	})

	t.Run("AddTransformation", func(t *testing.T) {
		pipeline := NewIngestionPipeline()
		transform := &MockTransform{name: "test"}

		pipeline.AddTransformation(transform)

		assert.Len(t, pipeline.Transformations(), 1)
	})

	t.Run("Run with documents", func(t *testing.T) {
		pipeline := NewIngestionPipeline(WithDisableCache(true))

		docs := []schema.Document{
			{ID: "doc1", Text: "Hello World"},
		}

		nodes, err := pipeline.Run(ctx, docs, nil)
		require.NoError(t, err)
		assert.Len(t, nodes, 1)
		assert.Equal(t, "doc1", nodes[0].ID)
	})

	t.Run("Run with nodes", func(t *testing.T) {
		pipeline := NewIngestionPipeline(WithDisableCache(true))

		inputNodes := []schema.Node{
			{ID: "node1", Text: "Test"},
		}

		nodes, err := pipeline.Run(ctx, nil, inputNodes)
		require.NoError(t, err)
		assert.Len(t, nodes, 1)
	})

	t.Run("Run with transformation", func(t *testing.T) {
		transform := &MockTransform{
			name: "uppercase",
			transform: func(nodes []schema.Node) []schema.Node {
				result := make([]schema.Node, len(nodes))
				for i, n := range nodes {
					result[i] = schema.Node{
						ID:   n.ID,
						Text: n.Text + " transformed",
					}
				}
				return result
			},
		}

		pipeline := NewIngestionPipeline(
			WithTransformations([]TransformComponent{transform}),
			WithDisableCache(true),
		)

		docs := []schema.Document{
			{ID: "doc1", Text: "Hello"},
		}

		nodes, err := pipeline.Run(ctx, docs, nil)
		require.NoError(t, err)
		assert.Len(t, nodes, 1)
		assert.Contains(t, nodes[0].Text, "transformed")
	})

	t.Run("Run with cache", func(t *testing.T) {
		callCount := 0
		transform := &MockTransform{
			name: "counter",
			transform: func(nodes []schema.Node) []schema.Node {
				callCount++
				return nodes
			},
		}

		cache := NewIngestionCache()
		pipeline := NewIngestionPipeline(
			WithTransformations([]TransformComponent{transform}),
			WithPipelineCache(cache),
		)

		docs := []schema.Document{
			{ID: "doc1", Text: "Hello"},
		}

		// First run
		_, err := pipeline.Run(ctx, docs, nil)
		require.NoError(t, err)
		assert.Equal(t, 1, callCount)

		// Second run - should use cache
		_, err = pipeline.Run(ctx, docs, nil)
		require.NoError(t, err)
		assert.Equal(t, 1, callCount) // Should not increase
	})

	t.Run("Run with docstore deduplication", func(t *testing.T) {
		docstore := NewMockDocStore()
		pipeline := NewIngestionPipeline(
			WithDocstore(docstore),
			WithDocstoreStrategy(DocstoreStrategyDuplicatesOnly),
			WithDisableCache(true),
		)

		docs := []schema.Document{
			{ID: "doc1", Text: "Hello"},
		}

		// First run
		nodes, err := pipeline.Run(ctx, docs, nil)
		require.NoError(t, err)
		assert.Len(t, nodes, 1)

		// Second run - should be deduplicated
		nodes, err = pipeline.Run(ctx, docs, nil)
		require.NoError(t, err)
		assert.Len(t, nodes, 0) // Duplicate, so no nodes to process
	})

	t.Run("Run with vector store", func(t *testing.T) {
		vectorStore := NewMockVectorStore()
		pipeline := NewIngestionPipeline(
			WithVectorStore(vectorStore),
			WithDisableCache(true),
		)

		// Create nodes with embeddings
		inputNodes := []schema.Node{
			{ID: "node1", Text: "Test", Embedding: []float64{0.1, 0.2, 0.3}},
		}

		_, err := pipeline.Run(ctx, nil, inputNodes)
		require.NoError(t, err)

		// Check vector store
		assert.Len(t, vectorStore.nodes, 1)
	})
}

// TestDocstoreStrategy tests the docstore strategies.
func TestDocstoreStrategy(t *testing.T) {
	ctx := context.Background()

	t.Run("Upserts strategy", func(t *testing.T) {
		docstore := NewMockDocStore()
		vectorStore := NewMockVectorStore()

		pipeline := NewIngestionPipeline(
			WithDocstore(docstore),
			WithVectorStore(vectorStore),
			WithDocstoreStrategy(DocstoreStrategyUpserts),
			WithDisableCache(true),
		)

		// First run
		docs := []schema.Document{
			{ID: "doc1", Text: "Hello"},
		}
		_, err := pipeline.Run(ctx, docs, nil)
		require.NoError(t, err)

		// Update document
		docs = []schema.Document{
			{ID: "doc1", Text: "Hello Updated"},
		}
		nodes, err := pipeline.Run(ctx, docs, nil)
		require.NoError(t, err)
		assert.Len(t, nodes, 1) // Should process updated doc
	})

	t.Run("Duplicates only strategy", func(t *testing.T) {
		docstore := NewMockDocStore()

		pipeline := NewIngestionPipeline(
			WithDocstore(docstore),
			WithDocstoreStrategy(DocstoreStrategyDuplicatesOnly),
			WithDisableCache(true),
		)

		docs := []schema.Document{
			{ID: "doc1", Text: "Hello"},
		}

		// First run
		nodes, err := pipeline.Run(ctx, docs, nil)
		require.NoError(t, err)
		assert.Len(t, nodes, 1)

		// Second run with same content
		nodes, err = pipeline.Run(ctx, docs, nil)
		require.NoError(t, err)
		assert.Len(t, nodes, 0) // Should be deduplicated
	})
}

// TestRunTransformations tests the standalone RunTransformations function.
func TestRunTransformations(t *testing.T) {
	ctx := context.Background()

	t.Run("Run without cache", func(t *testing.T) {
		transform := &MockTransform{
			name: "test",
			transform: func(nodes []schema.Node) []schema.Node {
				return nodes
			},
		}

		nodes := []schema.Node{{ID: "1", Text: "Test"}}
		result, err := RunTransformations(ctx, nodes, []TransformComponent{transform}, nil, "")
		require.NoError(t, err)
		assert.Len(t, result, 1)
	})

	t.Run("Run with cache", func(t *testing.T) {
		callCount := 0
		transform := &MockTransform{
			name: "counter",
			transform: func(nodes []schema.Node) []schema.Node {
				callCount++
				return nodes
			},
		}

		cache := NewIngestionCache()
		nodes := []schema.Node{{ID: "1", Text: "Test"}}

		// First run
		_, err := RunTransformations(ctx, nodes, []TransformComponent{transform}, cache, "")
		require.NoError(t, err)
		assert.Equal(t, 1, callCount)

		// Second run - should use cache
		_, err = RunTransformations(ctx, nodes, []TransformComponent{transform}, cache, "")
		require.NoError(t, err)
		assert.Equal(t, 1, callCount)
	})
}

// TestGetTransformationHash tests the hash generation.
func TestGetTransformationHash(t *testing.T) {
	t.Run("Same input produces same hash", func(t *testing.T) {
		nodes := []schema.Node{{ID: "1", Text: "Test"}}
		transform := &MockTransform{name: "test"}

		hash1 := getTransformationHash(nodes, transform)
		hash2 := getTransformationHash(nodes, transform)

		assert.Equal(t, hash1, hash2)
	})

	t.Run("Different input produces different hash", func(t *testing.T) {
		nodes1 := []schema.Node{{ID: "1", Text: "Test1"}}
		nodes2 := []schema.Node{{ID: "1", Text: "Test2"}}
		transform := &MockTransform{name: "test"}

		hash1 := getTransformationHash(nodes1, transform)
		hash2 := getTransformationHash(nodes2, transform)

		assert.NotEqual(t, hash1, hash2)
	})

	t.Run("Different transform produces different hash", func(t *testing.T) {
		nodes := []schema.Node{{ID: "1", Text: "Test"}}
		transform1 := &MockTransform{name: "test1"}
		transform2 := &MockTransform{name: "test2"}

		hash1 := getTransformationHash(nodes, transform1)
		hash2 := getTransformationHash(nodes, transform2)

		assert.NotEqual(t, hash1, hash2)
	})
}
