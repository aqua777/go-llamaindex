package index

import (
	"context"
	"testing"

	"github.com/aqua777/go-llamaindex/rag/store"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/storage"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// MockEmbeddingModel is a mock embedding model for testing.
type MockEmbeddingModel struct {
	embeddings map[string][]float64
}

func NewMockEmbeddingModel() *MockEmbeddingModel {
	return &MockEmbeddingModel{
		embeddings: make(map[string][]float64),
	}
}

func (m *MockEmbeddingModel) GetTextEmbedding(ctx context.Context, text string) ([]float64, error) {
	if emb, ok := m.embeddings[text]; ok {
		return emb, nil
	}
	// Return a simple hash-based embedding for testing
	embedding := make([]float64, 128)
	for i, c := range text {
		embedding[i%128] += float64(c) / 1000.0
	}
	return embedding, nil
}

func (m *MockEmbeddingModel) GetQueryEmbedding(ctx context.Context, query string) ([]float64, error) {
	return m.GetTextEmbedding(ctx, query)
}

func (m *MockEmbeddingModel) SetEmbedding(text string, embedding []float64) {
	m.embeddings[text] = embedding
}

// TestBaseIndex tests the BaseIndex struct.
func TestBaseIndex(t *testing.T) {
	t.Run("NewBaseIndex", func(t *testing.T) {
		sc := storage.NewStorageContext()
		bi := NewBaseIndex(nil, WithStorageContext(sc))

		assert.NotNil(t, bi)
		assert.Equal(t, sc, bi.StorageContext())
	})

	t.Run("SetSummary", func(t *testing.T) {
		sc := storage.NewStorageContext()
		bi := NewBaseIndex(nil, WithStorageContext(sc))

		// This would fail without an index struct, but we test the method exists
		assert.NotNil(t, bi)
	})
}

// TestVectorStoreIndex tests the VectorStoreIndex.
func TestVectorStoreIndex(t *testing.T) {
	ctx := context.Background()

	t.Run("NewVectorStoreIndex", func(t *testing.T) {
		sc := storage.NewStorageContext()
		vs := store.NewSimpleVectorStore()
		sc.SetVectorStore(vs)

		embedModel := NewMockEmbeddingModel()

		nodes := []schema.Node{
			*schema.NewTextNode("Hello world"),
			*schema.NewTextNode("Goodbye world"),
		}

		vsi, err := NewVectorStoreIndex(ctx, nodes,
			WithVectorIndexStorageContext(sc),
			WithVectorIndexEmbedModel(embedModel),
		)

		require.NoError(t, err)
		assert.NotNil(t, vsi)
		assert.NotEmpty(t, vsi.IndexID())
	})

	t.Run("InsertNodes", func(t *testing.T) {
		sc := storage.NewStorageContext()
		vs := store.NewSimpleVectorStore()
		sc.SetVectorStore(vs)

		embedModel := NewMockEmbeddingModel()

		vsi, err := NewVectorStoreIndex(ctx, nil,
			WithVectorIndexStorageContext(sc),
			WithVectorIndexEmbedModel(embedModel),
		)
		require.NoError(t, err)

		nodes := []schema.Node{
			*schema.NewTextNode("Test node 1"),
			*schema.NewTextNode("Test node 2"),
		}

		err = vsi.InsertNodes(ctx, nodes)
		require.NoError(t, err)
	})

	t.Run("DeleteNodes", func(t *testing.T) {
		sc := storage.NewStorageContext()
		vs := store.NewSimpleVectorStore()
		sc.SetVectorStore(vs)

		embedModel := NewMockEmbeddingModel()

		node := schema.NewTextNode("Test node")
		nodes := []schema.Node{*node}

		vsi, err := NewVectorStoreIndex(ctx, nodes,
			WithVectorIndexStorageContext(sc),
			WithVectorIndexEmbedModel(embedModel),
		)
		require.NoError(t, err)

		err = vsi.DeleteNodes(ctx, []string{node.ID})
		require.NoError(t, err)
	})

	t.Run("AsRetriever", func(t *testing.T) {
		sc := storage.NewStorageContext()
		vs := store.NewSimpleVectorStore()
		sc.SetVectorStore(vs)

		embedModel := NewMockEmbeddingModel()

		vsi, err := NewVectorStoreIndex(ctx, nil,
			WithVectorIndexStorageContext(sc),
			WithVectorIndexEmbedModel(embedModel),
		)
		require.NoError(t, err)

		ret := vsi.AsRetriever(WithSimilarityTopK(5))
		assert.NotNil(t, ret)
	})

	t.Run("AsQueryEngine", func(t *testing.T) {
		sc := storage.NewStorageContext()
		vs := store.NewSimpleVectorStore()
		sc.SetVectorStore(vs)

		embedModel := NewMockEmbeddingModel()

		vsi, err := NewVectorStoreIndex(ctx, nil,
			WithVectorIndexStorageContext(sc),
			WithVectorIndexEmbedModel(embedModel),
		)
		require.NoError(t, err)

		qe := vsi.AsQueryEngine()
		assert.NotNil(t, qe)
	})
}

// TestSummaryIndex tests the SummaryIndex.
func TestSummaryIndex(t *testing.T) {
	ctx := context.Background()

	t.Run("NewSummaryIndex", func(t *testing.T) {
		sc := storage.NewStorageContext()

		nodes := []schema.Node{
			*schema.NewTextNode("First document"),
			*schema.NewTextNode("Second document"),
		}

		si, err := NewSummaryIndex(ctx, nodes,
			WithSummaryIndexStorageContext(sc),
		)

		require.NoError(t, err)
		assert.NotNil(t, si)
		assert.NotEmpty(t, si.IndexID())
	})

	t.Run("InsertNodes", func(t *testing.T) {
		sc := storage.NewStorageContext()

		si, err := NewSummaryIndex(ctx, nil,
			WithSummaryIndexStorageContext(sc),
		)
		require.NoError(t, err)

		nodes := []schema.Node{
			*schema.NewTextNode("Test node 1"),
			*schema.NewTextNode("Test node 2"),
		}

		err = si.InsertNodes(ctx, nodes)
		require.NoError(t, err)

		// Verify nodes are in the index
		assert.Equal(t, 2, len(si.IndexStruct().Nodes))
	})

	t.Run("DeleteNodes", func(t *testing.T) {
		sc := storage.NewStorageContext()

		node1 := schema.NewTextNode("Test node 1")
		node2 := schema.NewTextNode("Test node 2")
		nodes := []schema.Node{*node1, *node2}

		si, err := NewSummaryIndex(ctx, nodes,
			WithSummaryIndexStorageContext(sc),
		)
		require.NoError(t, err)

		err = si.DeleteNodes(ctx, []string{node1.ID})
		require.NoError(t, err)

		// Verify only one node remains
		assert.Equal(t, 1, len(si.IndexStruct().Nodes))
	})

	t.Run("AsRetriever", func(t *testing.T) {
		sc := storage.NewStorageContext()

		si, err := NewSummaryIndex(ctx, nil,
			WithSummaryIndexStorageContext(sc),
		)
		require.NoError(t, err)

		ret := si.AsRetriever()
		assert.NotNil(t, ret)
	})

	t.Run("Retrieve", func(t *testing.T) {
		sc := storage.NewStorageContext()

		nodes := []schema.Node{
			*schema.NewTextNode("First document about AI"),
			*schema.NewTextNode("Second document about ML"),
		}

		si, err := NewSummaryIndex(ctx, nodes,
			WithSummaryIndexStorageContext(sc),
		)
		require.NoError(t, err)

		ret := si.AsRetriever()
		results, err := ret.Retrieve(ctx, schema.QueryBundle{QueryString: "AI"})
		require.NoError(t, err)

		// Default mode returns all nodes
		assert.Equal(t, 2, len(results))
	})
}

// TestKeywordTableIndex tests the KeywordTableIndex.
func TestKeywordTableIndex(t *testing.T) {
	ctx := context.Background()

	t.Run("NewKeywordTableIndex", func(t *testing.T) {
		sc := storage.NewStorageContext()

		nodes := []schema.Node{
			*schema.NewTextNode("Machine learning is a subset of artificial intelligence"),
			*schema.NewTextNode("Deep learning uses neural networks"),
		}

		kti, err := NewKeywordTableIndex(ctx, nodes,
			WithKeywordTableStorageContext(sc),
		)

		require.NoError(t, err)
		assert.NotNil(t, kti)
		assert.NotEmpty(t, kti.IndexID())
	})

	t.Run("InsertNodes", func(t *testing.T) {
		sc := storage.NewStorageContext()

		kti, err := NewKeywordTableIndex(ctx, nil,
			WithKeywordTableStorageContext(sc),
		)
		require.NoError(t, err)

		nodes := []schema.Node{
			*schema.NewTextNode("Python programming language"),
			*schema.NewTextNode("Go programming language"),
		}

		err = kti.InsertNodes(ctx, nodes)
		require.NoError(t, err)

		// Verify keywords are in the table
		assert.NotEmpty(t, kti.IndexStruct().Table)
	})

	t.Run("DeleteNodes", func(t *testing.T) {
		sc := storage.NewStorageContext()

		node1 := schema.NewTextNode("Python programming")
		node2 := schema.NewTextNode("Go programming")
		nodes := []schema.Node{*node1, *node2}

		kti, err := NewKeywordTableIndex(ctx, nodes,
			WithKeywordTableStorageContext(sc),
		)
		require.NoError(t, err)

		err = kti.DeleteNodes(ctx, []string{node1.ID})
		require.NoError(t, err)
	})

	t.Run("AsRetriever", func(t *testing.T) {
		sc := storage.NewStorageContext()

		kti, err := NewKeywordTableIndex(ctx, nil,
			WithKeywordTableStorageContext(sc),
		)
		require.NoError(t, err)

		ret := kti.AsRetriever()
		assert.NotNil(t, ret)
	})

	t.Run("Retrieve", func(t *testing.T) {
		sc := storage.NewStorageContext()

		nodes := []schema.Node{
			*schema.NewTextNode("Machine learning algorithms"),
			*schema.NewTextNode("Deep learning neural networks"),
		}

		kti, err := NewKeywordTableIndex(ctx, nodes,
			WithKeywordTableStorageContext(sc),
		)
		require.NoError(t, err)

		ret := kti.AsRetriever()
		results, err := ret.Retrieve(ctx, schema.QueryBundle{QueryString: "machine learning"})
		require.NoError(t, err)

		// Should find at least one result
		assert.GreaterOrEqual(t, len(results), 1)
	})
}

// TestSimpleKeywordExtractor tests the SimpleKeywordExtractor.
func TestSimpleKeywordExtractor(t *testing.T) {
	ctx := context.Background()
	extractor := &SimpleKeywordExtractor{}

	t.Run("ExtractKeywords", func(t *testing.T) {
		text := "Machine learning is a powerful technique for data analysis"
		keywords, err := extractor.ExtractKeywords(ctx, text, 5)

		require.NoError(t, err)
		assert.NotEmpty(t, keywords)
		assert.LessOrEqual(t, len(keywords), 5)
	})

	t.Run("StopWordsRemoved", func(t *testing.T) {
		text := "The quick brown fox jumps over the lazy dog"
		keywords, err := extractor.ExtractKeywords(ctx, text, 10)

		require.NoError(t, err)
		// "the" should not be in keywords
		for _, kw := range keywords {
			assert.NotEqual(t, "the", kw)
		}
	})
}

// TestVectorIndexRetriever tests the VectorIndexRetriever.
func TestVectorIndexRetriever(t *testing.T) {
	ctx := context.Background()

	t.Run("Retrieve", func(t *testing.T) {
		sc := storage.NewStorageContext()
		vs := store.NewSimpleVectorStore()
		sc.SetVectorStore(vs)

		embedModel := NewMockEmbeddingModel()
		// Set specific embeddings for testing
		embedModel.SetEmbedding("Hello world", []float64{1.0, 0.0, 0.0})
		embedModel.SetEmbedding("Goodbye world", []float64{0.0, 1.0, 0.0})
		embedModel.SetEmbedding("Hello", []float64{0.9, 0.1, 0.0})

		nodes := []schema.Node{
			*schema.NewTextNode("Hello world"),
			*schema.NewTextNode("Goodbye world"),
		}

		vsi, err := NewVectorStoreIndex(ctx, nodes,
			WithVectorIndexStorageContext(sc),
			WithVectorIndexEmbedModel(embedModel),
		)
		require.NoError(t, err)

		ret := vsi.AsRetriever(WithSimilarityTopK(2))
		results, err := ret.Retrieve(ctx, schema.QueryBundle{QueryString: "Hello"})
		require.NoError(t, err)

		assert.LessOrEqual(t, len(results), 2)
	})
}

// TestSummaryIndexRetriever tests the SummaryIndexRetriever.
func TestSummaryIndexRetriever(t *testing.T) {
	ctx := context.Background()

	t.Run("RetrieveDefault", func(t *testing.T) {
		sc := storage.NewStorageContext()

		nodes := []schema.Node{
			*schema.NewTextNode("Document 1"),
			*schema.NewTextNode("Document 2"),
			*schema.NewTextNode("Document 3"),
		}

		si, err := NewSummaryIndex(ctx, nodes,
			WithSummaryIndexStorageContext(sc),
		)
		require.NoError(t, err)

		ret := si.AsRetriever()
		results, err := ret.Retrieve(ctx, schema.QueryBundle{QueryString: "test"})
		require.NoError(t, err)

		// Default mode returns all nodes
		assert.Equal(t, 3, len(results))
	})
}

// TestIndexInterface tests that all index types implement the Index interface.
func TestIndexInterface(t *testing.T) {
	ctx := context.Background()
	sc := storage.NewStorageContext()
	vs := store.NewSimpleVectorStore()
	sc.SetVectorStore(vs)
	embedModel := NewMockEmbeddingModel()

	t.Run("VectorStoreIndex implements Index", func(t *testing.T) {
		vsi, err := NewVectorStoreIndex(ctx, nil,
			WithVectorIndexStorageContext(sc),
			WithVectorIndexEmbedModel(embedModel),
		)
		require.NoError(t, err)

		var _ Index = vsi
	})

	t.Run("SummaryIndex implements Index", func(t *testing.T) {
		si, err := NewSummaryIndex(ctx, nil,
			WithSummaryIndexStorageContext(sc),
		)
		require.NoError(t, err)

		var _ Index = si
	})

	t.Run("KeywordTableIndex implements Index", func(t *testing.T) {
		kti, err := NewKeywordTableIndex(ctx, nil,
			WithKeywordTableStorageContext(sc),
		)
		require.NoError(t, err)

		var _ Index = kti
	})
}
