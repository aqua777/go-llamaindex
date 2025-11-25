package chromem

import (
	"context"
	"os"
	"testing"

	"github.com/aqua777/go-llamaindex/schema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChromemStore(t *testing.T) {
	// Create a temporary directory for persistence test
	tmpDir, err := os.MkdirTemp("", "chromem_test")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	ctx := context.Background()
	collectionName := "test-collection"

	// 1. Initialize Store (Persistent)
	store, err := NewChromemStore(tmpDir, collectionName)
	require.NoError(t, err)
	require.NotNil(t, store)

	// 2. Add Documents with Embeddings
	nodes := []schema.Node{
		{
			ID:   "1",
			Text: "Apple is a fruit.",
			Type: schema.ObjectTypeText,
			Metadata: map[string]interface{}{
				"category": "fruit",
			},
			Embedding: []float64{1.0, 0.0, 0.0},
		},
		{
			ID:   "2",
			Text: "Car is a vehicle.",
			Type: schema.ObjectTypeText,
			Metadata: map[string]interface{}{
				"category": "vehicle",
			},
			Embedding: []float64{0.0, 1.0, 0.0},
		},
	}

	ids, err := store.Add(ctx, nodes)
	require.NoError(t, err)
	assert.Len(t, ids, 2)

	// 3. Query (Exact match for Apple)
	queryVec := []float64{1.0, 0.0, 0.0}
	query := schema.VectorStoreQuery{
		Embedding: queryVec,
		TopK:      1,
	}
	results, err := store.Query(ctx, query)
	require.NoError(t, err)
	require.Len(t, results, 1)

	assert.Equal(t, "1", results[0].Node.ID)
	assert.Equal(t, "Apple is a fruit.", results[0].Node.Text)
	assert.Equal(t, "fruit", results[0].Node.Metadata["category"])
	// Similarity should be close to 1.0
	assert.InDelta(t, 1.0, results[0].Score, 0.0001)

	// 4. Query (Exact match for Car)
	queryVecCar := []float64{0.0, 1.0, 0.0}
	queryCar := schema.VectorStoreQuery{
		Embedding: queryVecCar,
		TopK:      1,
	}
	resultsCar, err := store.Query(ctx, queryCar)
	require.NoError(t, err)
	require.Len(t, resultsCar, 1)
	assert.Equal(t, "2", resultsCar[0].Node.ID)

	// 5. Test Persistence (Re-open store)
	// Re-initialize store pointing to same dir
	store2, err := NewChromemStore(tmpDir, collectionName)
	require.NoError(t, err)

	// Query again
	resultsReopen, err := store2.Query(ctx, query)
	require.NoError(t, err)
	require.Len(t, resultsReopen, 1)
	assert.Equal(t, "1", resultsReopen[0].Node.ID)
	assert.Equal(t, "Apple is a fruit.", resultsReopen[0].Node.Text)
}

func TestChromemStore_InMemory(t *testing.T) {
	ctx := context.Background()
	// Empty path = in-memory
	store, err := NewChromemStore("", "mem-collection")
	require.NoError(t, err)

	nodes := []schema.Node{{ID: "A", Text: "Alpha", Embedding: []float64{0.5}}}

	_, err = store.Add(ctx, nodes)
	require.NoError(t, err)

	query := schema.VectorStoreQuery{
		Embedding: []float64{0.5},
		TopK:      1,
	}
	res, err := store.Query(ctx, query)
	require.NoError(t, err)
	assert.Len(t, res, 1)
	assert.Equal(t, "A", res[0].Node.ID)
}
