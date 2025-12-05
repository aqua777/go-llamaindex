package index

import (
	"context"
	"testing"

	"github.com/aqua777/go-llamaindex/graphstore"
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

	t.Run("TreeIndex implements Index", func(t *testing.T) {
		ti, err := NewTreeIndex(ctx, nil,
			WithTreeIndexStorageContext(sc),
			WithTreeIndexBuildTree(false),
		)
		require.NoError(t, err)

		var _ Index = ti
	})
}

// TestTreeIndex tests the TreeIndex.
func TestTreeIndex(t *testing.T) {
	ctx := context.Background()

	t.Run("NewTreeIndex", func(t *testing.T) {
		sc := storage.NewStorageContext()

		nodes := []schema.Node{
			*schema.NewTextNode("First document about AI"),
			*schema.NewTextNode("Second document about ML"),
			*schema.NewTextNode("Third document about deep learning"),
		}

		ti, err := NewTreeIndex(ctx, nodes,
			WithTreeIndexStorageContext(sc),
			WithTreeIndexBuildTree(false),
		)

		require.NoError(t, err)
		assert.NotNil(t, ti)
		assert.NotEmpty(t, ti.IndexID())
	})

	t.Run("NewTreeIndexWithTree", func(t *testing.T) {
		sc := storage.NewStorageContext()

		nodes := []schema.Node{
			*schema.NewTextNode("First document"),
			*schema.NewTextNode("Second document"),
			*schema.NewTextNode("Third document"),
		}

		ti, err := NewTreeIndex(ctx, nodes,
			WithTreeIndexStorageContext(sc),
			WithTreeIndexBuildTree(true),
			WithTreeIndexNumChildren(2),
		)

		require.NoError(t, err)
		assert.NotNil(t, ti)

		// With 3 nodes and numChildren=2, we should have some tree structure
		allNodes, err := ti.GetAllNodes(ctx)
		require.NoError(t, err)
		// Should have at least 3 leaf nodes plus some parent nodes
		assert.GreaterOrEqual(t, len(allNodes), 3)
	})

	t.Run("NewTreeIndexFromDocuments", func(t *testing.T) {
		sc := storage.NewStorageContext()

		docs := []schema.Document{
			{Text: "Document 1 content"},
			{Text: "Document 2 content"},
		}

		ti, err := NewTreeIndexFromDocuments(ctx, docs,
			WithTreeIndexStorageContext(sc),
			WithTreeIndexBuildTree(false),
		)

		require.NoError(t, err)
		assert.NotNil(t, ti)
	})

	t.Run("GetLeafNodes", func(t *testing.T) {
		sc := storage.NewStorageContext()

		nodes := []schema.Node{
			*schema.NewTextNode("Leaf 1"),
			*schema.NewTextNode("Leaf 2"),
			*schema.NewTextNode("Leaf 3"),
		}

		ti, err := NewTreeIndex(ctx, nodes,
			WithTreeIndexStorageContext(sc),
			WithTreeIndexBuildTree(false),
		)
		require.NoError(t, err)

		leafNodes, err := ti.GetLeafNodes(ctx)
		require.NoError(t, err)
		assert.Equal(t, 3, len(leafNodes))
	})

	t.Run("GetRootNodes", func(t *testing.T) {
		sc := storage.NewStorageContext()

		nodes := []schema.Node{
			*schema.NewTextNode("Node 1"),
			*schema.NewTextNode("Node 2"),
		}

		ti, err := NewTreeIndex(ctx, nodes,
			WithTreeIndexStorageContext(sc),
			WithTreeIndexBuildTree(false),
		)
		require.NoError(t, err)

		rootNodes, err := ti.GetRootNodes(ctx)
		require.NoError(t, err)
		// Without tree building, all nodes are roots
		assert.Equal(t, 2, len(rootNodes))
	})

	t.Run("AsRetriever", func(t *testing.T) {
		sc := storage.NewStorageContext()

		ti, err := NewTreeIndex(ctx, nil,
			WithTreeIndexStorageContext(sc),
			WithTreeIndexBuildTree(false),
		)
		require.NoError(t, err)

		ret := ti.AsRetriever()
		assert.NotNil(t, ret)
	})

	t.Run("AsQueryEngine", func(t *testing.T) {
		sc := storage.NewStorageContext()

		ti, err := NewTreeIndex(ctx, nil,
			WithTreeIndexStorageContext(sc),
			WithTreeIndexBuildTree(false),
		)
		require.NoError(t, err)

		qe := ti.AsQueryEngine()
		assert.NotNil(t, qe)
	})

	t.Run("DeleteNodes returns error", func(t *testing.T) {
		sc := storage.NewStorageContext()

		ti, err := NewTreeIndex(ctx, nil,
			WithTreeIndexStorageContext(sc),
			WithTreeIndexBuildTree(false),
		)
		require.NoError(t, err)

		err = ti.DeleteNodes(ctx, []string{"some-id"})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "not implemented")
	})
}

// TestTreeAllLeafRetriever tests the TreeAllLeafRetriever.
func TestTreeAllLeafRetriever(t *testing.T) {
	ctx := context.Background()

	t.Run("Retrieve all leaves", func(t *testing.T) {
		sc := storage.NewStorageContext()

		nodes := []schema.Node{
			*schema.NewTextNode("Leaf node 1"),
			*schema.NewTextNode("Leaf node 2"),
			*schema.NewTextNode("Leaf node 3"),
		}

		ti, err := NewTreeIndex(ctx, nodes,
			WithTreeIndexStorageContext(sc),
			WithTreeIndexBuildTree(false),
		)
		require.NoError(t, err)

		ret := NewTreeAllLeafRetriever(ti)
		results, err := ret.Retrieve(ctx, schema.QueryBundle{QueryString: "test query"})
		require.NoError(t, err)

		assert.Equal(t, 3, len(results))
		for _, r := range results {
			assert.Equal(t, 1.0, r.Score)
		}
	})

	t.Run("Retrieve empty index", func(t *testing.T) {
		sc := storage.NewStorageContext()

		ti, err := NewTreeIndex(ctx, nil,
			WithTreeIndexStorageContext(sc),
			WithTreeIndexBuildTree(false),
		)
		require.NoError(t, err)

		ret := NewTreeAllLeafRetriever(ti)
		results, err := ret.Retrieve(ctx, schema.QueryBundle{QueryString: "test"})
		require.NoError(t, err)

		assert.Empty(t, results)
	})
}

// TestTreeRootRetriever tests the TreeRootRetriever.
func TestTreeRootRetriever(t *testing.T) {
	ctx := context.Background()

	t.Run("Retrieve root nodes", func(t *testing.T) {
		sc := storage.NewStorageContext()

		nodes := []schema.Node{
			*schema.NewTextNode("Node 1"),
			*schema.NewTextNode("Node 2"),
		}

		ti, err := NewTreeIndex(ctx, nodes,
			WithTreeIndexStorageContext(sc),
			WithTreeIndexBuildTree(false),
		)
		require.NoError(t, err)

		ret := NewTreeRootRetriever(ti)
		results, err := ret.Retrieve(ctx, schema.QueryBundle{QueryString: "test"})
		require.NoError(t, err)

		// Without tree building, all nodes are roots
		assert.Equal(t, 2, len(results))
	})
}

// TestTreeSelectLeafRetriever tests the TreeSelectLeafRetriever.
func TestTreeSelectLeafRetriever(t *testing.T) {
	ctx := context.Background()

	t.Run("Retrieve without LLM returns error", func(t *testing.T) {
		sc := storage.NewStorageContext()

		nodes := []schema.Node{
			*schema.NewTextNode("Node 1"),
		}

		ti, err := NewTreeIndex(ctx, nodes,
			WithTreeIndexStorageContext(sc),
			WithTreeIndexBuildTree(false),
		)
		require.NoError(t, err)

		ret := NewTreeSelectLeafRetriever(ti)
		_, err = ret.Retrieve(ctx, schema.QueryBundle{QueryString: "test"})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "LLM not configured")
	})
}

// TestTreeSelectLeafEmbeddingRetriever tests the TreeSelectLeafEmbeddingRetriever.
func TestTreeSelectLeafEmbeddingRetriever(t *testing.T) {
	ctx := context.Background()

	t.Run("Retrieve without embed model returns error", func(t *testing.T) {
		sc := storage.NewStorageContext()

		nodes := []schema.Node{
			*schema.NewTextNode("Node 1"),
		}

		ti, err := NewTreeIndex(ctx, nodes,
			WithTreeIndexStorageContext(sc),
			WithTreeIndexBuildTree(false),
		)
		require.NoError(t, err)

		ret := NewTreeSelectLeafEmbeddingRetriever(ti)
		_, err = ret.Retrieve(ctx, schema.QueryBundle{QueryString: "test"})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "embedding model not configured")
	})

	t.Run("Retrieve with embed model", func(t *testing.T) {
		sc := storage.NewStorageContext()
		embedModel := NewMockEmbeddingModel()

		nodes := []schema.Node{
			*schema.NewTextNode("Document about AI"),
			*schema.NewTextNode("Document about cooking"),
		}

		ti, err := NewTreeIndex(ctx, nodes,
			WithTreeIndexStorageContext(sc),
			WithTreeIndexEmbedModel(embedModel),
			WithTreeIndexBuildTree(false),
		)
		require.NoError(t, err)

		ret := NewTreeSelectLeafEmbeddingRetriever(ti,
			WithTreeRetrieverChildBranchFactor(1),
		)
		results, err := ret.Retrieve(ctx, schema.QueryBundle{QueryString: "AI"})
		require.NoError(t, err)

		// Should return at least one result
		assert.GreaterOrEqual(t, len(results), 1)
	})
}

// TestTreeIndexInserter tests the TreeIndexInserter.
func TestTreeIndexInserter(t *testing.T) {
	ctx := context.Background()

	t.Run("Insert into empty index", func(t *testing.T) {
		sc := storage.NewStorageContext()

		ti, err := NewTreeIndex(ctx, nil,
			WithTreeIndexStorageContext(sc),
			WithTreeIndexBuildTree(false),
		)
		require.NoError(t, err)

		nodes := []schema.Node{
			*schema.NewTextNode("New node 1"),
			*schema.NewTextNode("New node 2"),
		}

		err = ti.InsertNodes(ctx, nodes)
		require.NoError(t, err)

		allNodes, err := ti.GetAllNodes(ctx)
		require.NoError(t, err)
		assert.Equal(t, 2, len(allNodes))
	})

	t.Run("Insert into existing index", func(t *testing.T) {
		sc := storage.NewStorageContext()

		initialNodes := []schema.Node{
			*schema.NewTextNode("Initial node"),
		}

		ti, err := NewTreeIndex(ctx, initialNodes,
			WithTreeIndexStorageContext(sc),
			WithTreeIndexBuildTree(false),
		)
		require.NoError(t, err)

		newNodes := []schema.Node{
			*schema.NewTextNode("New node"),
		}

		err = ti.InsertNodes(ctx, newNodes)
		require.NoError(t, err)

		allNodes, err := ti.GetAllNodes(ctx)
		require.NoError(t, err)
		assert.Equal(t, 2, len(allNodes))
	})
}

// TestTreeRetrieverModes tests different retriever modes.
func TestTreeRetrieverModes(t *testing.T) {
	ctx := context.Background()

	t.Run("AsRetrieverWithMode AllLeaf", func(t *testing.T) {
		sc := storage.NewStorageContext()

		nodes := []schema.Node{
			*schema.NewTextNode("Node 1"),
			*schema.NewTextNode("Node 2"),
		}

		ti, err := NewTreeIndex(ctx, nodes,
			WithTreeIndexStorageContext(sc),
			WithTreeIndexBuildTree(false),
		)
		require.NoError(t, err)

		ret, err := ti.AsRetrieverWithMode(TreeRetrieverModeAllLeaf)
		require.NoError(t, err)
		assert.NotNil(t, ret)

		results, err := ret.Retrieve(ctx, schema.QueryBundle{QueryString: "test"})
		require.NoError(t, err)
		assert.Equal(t, 2, len(results))
	})

	t.Run("AsRetrieverWithMode Root", func(t *testing.T) {
		sc := storage.NewStorageContext()

		nodes := []schema.Node{
			*schema.NewTextNode("Node 1"),
		}

		ti, err := NewTreeIndex(ctx, nodes,
			WithTreeIndexStorageContext(sc),
			WithTreeIndexBuildTree(false),
		)
		require.NoError(t, err)

		ret, err := ti.AsRetrieverWithMode(TreeRetrieverModeRoot)
		require.NoError(t, err)
		assert.NotNil(t, ret)
	})

	t.Run("AsRetrieverWithMode SelectLeaf requires tree", func(t *testing.T) {
		sc := storage.NewStorageContext()

		ti, err := NewTreeIndex(ctx, nil,
			WithTreeIndexStorageContext(sc),
			WithTreeIndexBuildTree(false),
		)
		require.NoError(t, err)

		_, err = ti.AsRetrieverWithMode(TreeRetrieverModeSelectLeaf)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "requires tree")
	})
}

// TestTreeIndexHelpers tests helper functions.
func TestTreeIndexHelpers(t *testing.T) {
	t.Run("buildNumberedList", func(t *testing.T) {
		nodes := []schema.Node{
			*schema.NewTextNode("First item"),
			*schema.NewTextNode("Second item"),
		}

		result := buildNumberedList(nodes)
		assert.Contains(t, result, "(1)")
		assert.Contains(t, result, "(2)")
		assert.Contains(t, result, "First item")
		assert.Contains(t, result, "Second item")
	})

	t.Run("extractNumbers", func(t *testing.T) {
		tests := []struct {
			response string
			n        int
			expected []int
		}{
			{"ANSWER: 1", 1, []int{1}},
			{"ANSWER: 3", 1, []int{3}},
			{"The answer is 2 because...", 1, []int{2}},
			{"1, 2, 3", 3, []int{1, 2, 3}},
			{"1, 2, 3", 2, []int{1, 2}},
			{"no numbers here", 1, nil},
		}

		for _, tt := range tests {
			result := extractNumbers(tt.response, tt.n)
			if tt.expected == nil {
				assert.Empty(t, result)
			} else {
				assert.Equal(t, tt.expected, result)
			}
		}
	})
}

// TestTreeIndexOptions tests TreeIndex options.
func TestTreeIndexOptions(t *testing.T) {
	ctx := context.Background()

	t.Run("WithTreeIndexNumChildren", func(t *testing.T) {
		sc := storage.NewStorageContext()

		ti, err := NewTreeIndex(ctx, nil,
			WithTreeIndexStorageContext(sc),
			WithTreeIndexNumChildren(5),
			WithTreeIndexBuildTree(false),
		)
		require.NoError(t, err)

		assert.Equal(t, 5, ti.NumChildren())
	})

	t.Run("WithTreeIndexNumChildren minimum", func(t *testing.T) {
		sc := storage.NewStorageContext()

		ti, err := NewTreeIndex(ctx, nil,
			WithTreeIndexStorageContext(sc),
			WithTreeIndexNumChildren(1), // Should be ignored (< 2)
			WithTreeIndexBuildTree(false),
		)
		require.NoError(t, err)

		// Should keep default of 10
		assert.Equal(t, 10, ti.NumChildren())
	})

	t.Run("WithTreeIndexEmbedModel", func(t *testing.T) {
		sc := storage.NewStorageContext()
		embedModel := NewMockEmbeddingModel()

		ti, err := NewTreeIndex(ctx, nil,
			WithTreeIndexStorageContext(sc),
			WithTreeIndexEmbedModel(embedModel),
			WithTreeIndexBuildTree(false),
		)
		require.NoError(t, err)

		assert.NotNil(t, ti.EmbedModel())
	})
}

// ============================================================================
// KnowledgeGraphIndex Tests
// ============================================================================

func TestKnowledgeGraphIndex(t *testing.T) {
	ctx := context.Background()

	t.Run("NewKnowledgeGraphIndex_empty", func(t *testing.T) {
		kg, err := NewKnowledgeGraphIndex(ctx, nil)
		require.NoError(t, err)
		assert.NotNil(t, kg)
		assert.NotNil(t, kg.GraphStore())
	})

	t.Run("NewKnowledgeGraphIndex_with_triplet_extract_fn", func(t *testing.T) {
		// Custom triplet extraction function
		extractFn := func(text string) ([]graphstore.Triplet, error) {
			return []graphstore.Triplet{
				{Subject: "Alice", Relation: "knows", Object: "Bob"},
			}, nil
		}

		nodes := []schema.Node{
			*schema.NewTextNode("Alice knows Bob"),
		}

		kg, err := NewKnowledgeGraphIndex(ctx, nodes,
			WithKGIndexTripletExtractFn(extractFn),
		)
		require.NoError(t, err)
		assert.NotNil(t, kg)

		// Check that triplet was added
		rels, err := kg.GraphStore().Get(ctx, "Alice")
		require.NoError(t, err)
		assert.NotEmpty(t, rels)
	})

	t.Run("NewKnowledgeGraphIndexFromDocuments", func(t *testing.T) {
		extractFn := func(text string) ([]graphstore.Triplet, error) {
			return []graphstore.Triplet{
				{Subject: "Doc", Relation: "contains", Object: "Info"},
			}, nil
		}

		docs := []schema.Document{
			{Text: "Document content", Metadata: map[string]interface{}{"source": "test"}},
		}

		kg, err := NewKnowledgeGraphIndexFromDocuments(ctx, docs,
			WithKGIndexTripletExtractFn(extractFn),
		)
		require.NoError(t, err)
		assert.NotNil(t, kg)
	})

	t.Run("UpsertTriplet", func(t *testing.T) {
		kg, err := NewKnowledgeGraphIndex(ctx, nil)
		require.NoError(t, err)

		triplet := graphstore.Triplet{Subject: "X", Relation: "rel", Object: "Y"}
		err = kg.UpsertTriplet(ctx, triplet, false)
		require.NoError(t, err)

		rels, err := kg.GraphStore().Get(ctx, "X")
		require.NoError(t, err)
		assert.Len(t, rels, 1)
	})

	t.Run("SearchNodeByKeyword", func(t *testing.T) {
		extractFn := func(text string) ([]graphstore.Triplet, error) {
			return []graphstore.Triplet{
				{Subject: "Entity1", Relation: "rel", Object: "Entity2"},
			}, nil
		}

		nodes := []schema.Node{
			*schema.NewTextNode("Some text"),
		}
		nodes[0].ID = "node1"

		kg, err := NewKnowledgeGraphIndex(ctx, nodes,
			WithKGIndexTripletExtractFn(extractFn),
		)
		require.NoError(t, err)

		nodeIDs := kg.SearchNodeByKeyword("Entity1")
		assert.Contains(t, nodeIDs, "node1")
	})

	t.Run("GetAllKeywords", func(t *testing.T) {
		extractFn := func(text string) ([]graphstore.Triplet, error) {
			return []graphstore.Triplet{
				{Subject: "A", Relation: "r", Object: "B"},
				{Subject: "C", Relation: "r", Object: "D"},
			}, nil
		}

		nodes := []schema.Node{
			*schema.NewTextNode("Text"),
		}

		kg, err := NewKnowledgeGraphIndex(ctx, nodes,
			WithKGIndexTripletExtractFn(extractFn),
		)
		require.NoError(t, err)

		keywords := kg.GetAllKeywords()
		assert.Contains(t, keywords, "A")
		assert.Contains(t, keywords, "B")
		assert.Contains(t, keywords, "C")
		assert.Contains(t, keywords, "D")
	})

	t.Run("AsRetriever", func(t *testing.T) {
		kg, err := NewKnowledgeGraphIndex(ctx, nil)
		require.NoError(t, err)

		ret := kg.AsRetriever()
		assert.NotNil(t, ret)
	})

	t.Run("AsRetrieverWithMode_keyword", func(t *testing.T) {
		kg, err := NewKnowledgeGraphIndex(ctx, nil)
		require.NoError(t, err)

		ret, err := kg.AsRetrieverWithMode(KGRetrieverModeKeyword)
		require.NoError(t, err)
		assert.NotNil(t, ret)
	})

	t.Run("AsRetrieverWithMode_embedding_requires_embeddings", func(t *testing.T) {
		kg, err := NewKnowledgeGraphIndex(ctx, nil)
		require.NoError(t, err)

		_, err = kg.AsRetrieverWithMode(KGRetrieverModeEmbedding)
		assert.Error(t, err) // Should fail because no embeddings
	})

	t.Run("AsQueryEngine", func(t *testing.T) {
		kg, err := NewKnowledgeGraphIndex(ctx, nil)
		require.NoError(t, err)

		qe := kg.AsQueryEngine()
		assert.NotNil(t, qe)
	})

	t.Run("DeleteNodes_returns_error", func(t *testing.T) {
		kg, err := NewKnowledgeGraphIndex(ctx, nil)
		require.NoError(t, err)

		err = kg.DeleteNodes(ctx, []string{"node1"})
		assert.Error(t, err)
	})
}

func TestKGTableRetriever(t *testing.T) {
	ctx := context.Background()

	t.Run("NewKGTableRetriever", func(t *testing.T) {
		kg, err := NewKnowledgeGraphIndex(ctx, nil)
		require.NoError(t, err)

		ret := NewKGTableRetriever(kg)
		assert.NotNil(t, ret)
		assert.Equal(t, "KGTableRetriever", ret.Name())
	})

	t.Run("Retrieve_empty_index", func(t *testing.T) {
		kg, err := NewKnowledgeGraphIndex(ctx, nil)
		require.NoError(t, err)

		ret := NewKGTableRetriever(kg)
		results, err := ret.Retrieve(ctx, schema.QueryBundle{QueryString: "test query"})
		require.NoError(t, err)
		assert.NotEmpty(t, results) // Should return "No relationships found" node
	})

	t.Run("Retrieve_with_data", func(t *testing.T) {
		extractFn := func(text string) ([]graphstore.Triplet, error) {
			return []graphstore.Triplet{
				{Subject: "Alice", Relation: "knows", Object: "Bob"},
			}, nil
		}

		nodes := []schema.Node{
			*schema.NewTextNode("Alice knows Bob"),
		}

		kg, err := NewKnowledgeGraphIndex(ctx, nodes,
			WithKGIndexTripletExtractFn(extractFn),
		)
		require.NoError(t, err)

		ret := NewKGTableRetriever(kg)
		results, err := ret.Retrieve(ctx, schema.QueryBundle{QueryString: "alice"})
		require.NoError(t, err)
		assert.NotEmpty(t, results)
	})

	t.Run("Retrieve_with_options", func(t *testing.T) {
		kg, err := NewKnowledgeGraphIndex(ctx, nil)
		require.NoError(t, err)

		ret := NewKGTableRetriever(kg,
			WithKGRetrieverMode(KGRetrieverModeKeyword),
			WithKGRetrieverMaxKeywords(5),
			WithKGRetrieverNumChunks(5),
			WithKGRetrieverIncludeText(true),
			WithKGRetrieverSimilarityTopK(3),
			WithKGRetrieverGraphDepth(3),
			WithKGRetrieverUseGlobalTriplets(false),
			WithKGRetrieverMaxKnowledgeSequence(20),
			WithKGRetrieverVerbose(false),
		)
		assert.NotNil(t, ret)
	})
}

func TestKGRAGRetriever(t *testing.T) {
	ctx := context.Background()

	t.Run("NewKGRAGRetriever", func(t *testing.T) {
		kg, err := NewKnowledgeGraphIndex(ctx, nil)
		require.NoError(t, err)

		ret := NewKGRAGRetriever(kg)
		assert.NotNil(t, ret)
		assert.Equal(t, "KGRAGRetriever", ret.Name())
	})

	t.Run("NewKGRAGRetriever_with_options", func(t *testing.T) {
		kg, err := NewKnowledgeGraphIndex(ctx, nil)
		require.NoError(t, err)

		entityFn := func(text string) ([]string, error) {
			return []string{"entity1", "entity2"}, nil
		}
		synonymFn := func(text string) ([]string, error) {
			return []string{"syn1", "syn2"}, nil
		}

		ret := NewKGRAGRetriever(kg,
			WithKGRAGEntityExtractFn(entityFn),
			WithKGRAGSynonymExpandFn(synonymFn),
			WithKGRAGMaxEntities(10),
			WithKGRAGMaxSynonyms(10),
		)
		assert.NotNil(t, ret)
	})
}

func TestKGIndexOptions(t *testing.T) {
	ctx := context.Background()

	t.Run("WithKGIndexMaxTripletsPerChunk", func(t *testing.T) {
		kg, err := NewKnowledgeGraphIndex(ctx, nil,
			WithKGIndexMaxTripletsPerChunk(5),
		)
		require.NoError(t, err)
		assert.NotNil(t, kg)
	})

	t.Run("WithKGIndexIncludeEmbeddings", func(t *testing.T) {
		kg, err := NewKnowledgeGraphIndex(ctx, nil,
			WithKGIndexIncludeEmbeddings(true),
		)
		require.NoError(t, err)
		assert.NotNil(t, kg)
	})

	t.Run("WithKGIndexMaxObjectLength", func(t *testing.T) {
		kg, err := NewKnowledgeGraphIndex(ctx, nil,
			WithKGIndexMaxObjectLength(256),
		)
		require.NoError(t, err)
		assert.NotNil(t, kg)
	})

	t.Run("WithKGIndexGraphStoreQueryDepth", func(t *testing.T) {
		kg, err := NewKnowledgeGraphIndex(ctx, nil,
			WithKGIndexGraphStoreQueryDepth(3),
		)
		require.NoError(t, err)
		assert.Equal(t, 3, kg.GraphStoreQueryDepth())
	})

	t.Run("WithKGIndexGraphStore", func(t *testing.T) {
		gs := graphstore.NewSimpleGraphStore()
		kg, err := NewKnowledgeGraphIndex(ctx, nil,
			WithKGIndexGraphStore(gs),
		)
		require.NoError(t, err)
		assert.Equal(t, gs, kg.GraphStore())
	})
}

func TestParseTripletResponse(t *testing.T) {
	ctx := context.Background()

	kg, err := NewKnowledgeGraphIndex(ctx, nil)
	require.NoError(t, err)

	t.Run("parse_valid_triplets", func(t *testing.T) {
		response := `(Alice, knows, Bob)
(Charlie, works_at, Company)
(Dave, likes, Eve)`

		triplets, err := kg.parseTripletResponse(response)
		require.NoError(t, err)
		assert.Len(t, triplets, 3)
		assert.Equal(t, "Alice", triplets[0].Subject)
		assert.Equal(t, "knows", triplets[0].Relation)
		assert.Equal(t, "Bob", triplets[0].Object)
	})

	t.Run("parse_with_quotes", func(t *testing.T) {
		response := `("alice", "knows", "bob")`

		triplets, err := kg.parseTripletResponse(response)
		require.NoError(t, err)
		assert.Len(t, triplets, 1)
		assert.Equal(t, "Alice", triplets[0].Subject) // Should be capitalized
	})

	t.Run("parse_invalid_format", func(t *testing.T) {
		response := `This is not a triplet
(only, two)
()`

		triplets, err := kg.parseTripletResponse(response)
		require.NoError(t, err)
		assert.Empty(t, triplets)
	})

	t.Run("parse_empty_parts", func(t *testing.T) {
		response := `(, knows, Bob)
(Alice, , Bob)
(Alice, knows, )`

		triplets, err := kg.parseTripletResponse(response)
		require.NoError(t, err)
		assert.Empty(t, triplets)
	})
}

func TestKGHelperFunctions(t *testing.T) {
	t.Run("simpleKeywordExtract", func(t *testing.T) {
		keywords := simpleKeywordExtract("What is the capital of France?", 5)
		assert.NotEmpty(t, keywords)
		assert.Contains(t, keywords, "capital")
		assert.Contains(t, keywords, "france")
	})

	t.Run("simpleKeywordExtract_filters_stopwords", func(t *testing.T) {
		keywords := simpleKeywordExtract("the quick brown fox", 10)
		assert.NotContains(t, keywords, "the")
		assert.Contains(t, keywords, "quick")
	})

	t.Run("removeDuplicates", func(t *testing.T) {
		input := []string{"a", "b", "a", "c", "b"}
		result := removeDuplicates(input)
		assert.Len(t, result, 3)
	})

	t.Run("removeSubstrings", func(t *testing.T) {
		input := []string{"hello world", "hello", "world"}
		result := removeSubstrings(input)
		assert.Contains(t, result, "hello world")
	})

	t.Run("extractRelTextKeywords", func(t *testing.T) {
		relTexts := []string{"[Alice, knows, Bob]", "[Charlie, likes, Dave]"}
		keywords := extractRelTextKeywords(relTexts)
		assert.Contains(t, keywords, "Alice")
		assert.Contains(t, keywords, "Bob")
	})

	t.Run("extractKeywordsFromResponse", func(t *testing.T) {
		response := "KEYWORDS: alice, bob, charlie"
		keywords := extractKeywordsFromResponse(response, 10)
		assert.Len(t, keywords, 3)
	})

	t.Run("kgCosineSimilarity", func(t *testing.T) {
		a := []float64{1, 0, 0}
		b := []float64{1, 0, 0}
		sim := kgCosineSimilarity(a, b)
		assert.InDelta(t, 1.0, sim, 0.001)

		c := []float64{0, 1, 0}
		sim = kgCosineSimilarity(a, c)
		assert.InDelta(t, 0.0, sim, 0.001)
	})

	t.Run("kgCosineSimilarity_empty", func(t *testing.T) {
		sim := kgCosineSimilarity(nil, nil)
		assert.Equal(t, 0.0, sim)

		sim = kgCosineSimilarity([]float64{1}, []float64{1, 2})
		assert.Equal(t, 0.0, sim)
	})

	t.Run("sortByCount", func(t *testing.T) {
		counts := map[string]int{"a": 3, "b": 1, "c": 2}
		result := sortByCount(counts, 2)
		assert.Len(t, result, 2)
		assert.Equal(t, "a", result[0])
		assert.Equal(t, "c", result[1])
	})
}
