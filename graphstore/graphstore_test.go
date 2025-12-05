package graphstore

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTriplet(t *testing.T) {
	t.Run("String", func(t *testing.T) {
		triplet := Triplet{Subject: "Alice", Relation: "knows", Object: "Bob"}
		assert.Equal(t, "(Alice, knows, Bob)", triplet.String())
	})

	t.Run("MarshalJSON", func(t *testing.T) {
		triplet := Triplet{Subject: "Alice", Relation: "knows", Object: "Bob"}
		data, err := json.Marshal(triplet)
		require.NoError(t, err)
		assert.Equal(t, `["Alice","knows","Bob"]`, string(data))
	})

	t.Run("UnmarshalJSON", func(t *testing.T) {
		var triplet Triplet
		err := json.Unmarshal([]byte(`["Alice","knows","Bob"]`), &triplet)
		require.NoError(t, err)
		assert.Equal(t, "Alice", triplet.Subject)
		assert.Equal(t, "knows", triplet.Relation)
		assert.Equal(t, "Bob", triplet.Object)
	})

	t.Run("UnmarshalJSON_invalid", func(t *testing.T) {
		var triplet Triplet
		err := json.Unmarshal([]byte(`["Alice","knows"]`), &triplet)
		assert.Error(t, err)
	})
}

func TestEntityNode(t *testing.T) {
	t.Run("NewEntityNode", func(t *testing.T) {
		node := NewEntityNode("Alice")
		assert.Equal(t, "Alice", node.Name)
		assert.Equal(t, "entity", node.Label)
		assert.NotNil(t, node.Properties)
	})

	t.Run("ID", func(t *testing.T) {
		node := NewEntityNode("Alice")
		assert.Equal(t, "Alice", node.ID())
	})

	t.Run("String_without_properties", func(t *testing.T) {
		node := NewEntityNode("Alice")
		assert.Equal(t, "Alice", node.String())
	})

	t.Run("String_with_properties", func(t *testing.T) {
		node := NewEntityNode("Alice")
		node.Properties["age"] = 30
		assert.Contains(t, node.String(), "Alice")
		assert.Contains(t, node.String(), "age")
	})
}

func TestRelation(t *testing.T) {
	t.Run("NewRelation", func(t *testing.T) {
		rel := NewRelation("knows", "Alice", "Bob")
		assert.Equal(t, "knows", rel.Label)
		assert.Equal(t, "Alice", rel.SourceID)
		assert.Equal(t, "Bob", rel.TargetID)
		assert.NotNil(t, rel.Properties)
	})

	t.Run("ID", func(t *testing.T) {
		rel := NewRelation("knows", "Alice", "Bob")
		assert.Equal(t, "knows", rel.ID())
	})

	t.Run("String_without_properties", func(t *testing.T) {
		rel := NewRelation("knows", "Alice", "Bob")
		assert.Equal(t, "knows", rel.String())
	})

	t.Run("String_with_properties", func(t *testing.T) {
		rel := NewRelation("knows", "Alice", "Bob")
		rel.Properties["since"] = 2020
		assert.Contains(t, rel.String(), "knows")
		assert.Contains(t, rel.String(), "since")
	})
}

func TestGraphStoreData(t *testing.T) {
	t.Run("NewGraphStoreData", func(t *testing.T) {
		data := NewGraphStoreData()
		assert.NotNil(t, data.GraphDict)
		assert.Empty(t, data.GraphDict)
	})

	t.Run("GetRelMap_empty", func(t *testing.T) {
		data := NewGraphStoreData()
		relMap := data.GetRelMap(nil, 2, 30)
		assert.Empty(t, relMap)
	})

	t.Run("GetRelMap_with_data", func(t *testing.T) {
		data := NewGraphStoreData()
		data.GraphDict["Alice"] = [][]string{{"knows", "Bob"}, {"likes", "Charlie"}}
		data.GraphDict["Bob"] = [][]string{{"works_at", "Company"}}

		relMap := data.GetRelMap([]string{"Alice"}, 2, 30)
		assert.Contains(t, relMap, "Alice")
		assert.NotEmpty(t, relMap["Alice"])
	})

	t.Run("GetRelMap_depth_limit", func(t *testing.T) {
		data := NewGraphStoreData()
		data.GraphDict["A"] = [][]string{{"r1", "B"}}
		data.GraphDict["B"] = [][]string{{"r2", "C"}}
		data.GraphDict["C"] = [][]string{{"r3", "D"}}

		// Depth 1 should only get A->B
		relMap := data.GetRelMap([]string{"A"}, 1, 30)
		assert.Len(t, relMap["A"], 1)

		// Depth 2 should get A->B and B->C
		relMap = data.GetRelMap([]string{"A"}, 2, 30)
		assert.True(t, len(relMap["A"]) >= 1)
	})

	t.Run("ToJSON_and_FromJSON", func(t *testing.T) {
		data := NewGraphStoreData()
		data.GraphDict["Alice"] = [][]string{{"knows", "Bob"}}

		jsonData, err := data.ToJSON()
		require.NoError(t, err)

		restored, err := FromJSON(jsonData)
		require.NoError(t, err)
		assert.Equal(t, data.GraphDict, restored.GraphDict)
	})
}

func TestSimpleGraphStore(t *testing.T) {
	ctx := context.Background()

	t.Run("NewSimpleGraphStore", func(t *testing.T) {
		store := NewSimpleGraphStore()
		assert.NotNil(t, store)
		assert.Equal(t, 0, store.Size())
	})

	t.Run("UpsertTriplet", func(t *testing.T) {
		store := NewSimpleGraphStore()
		err := store.UpsertTriplet(ctx, "Alice", "knows", "Bob")
		require.NoError(t, err)
		assert.Equal(t, 1, store.Size())
		assert.Equal(t, 1, store.TripletCount())
	})

	t.Run("UpsertTriplet_duplicate", func(t *testing.T) {
		store := NewSimpleGraphStore()
		err := store.UpsertTriplet(ctx, "Alice", "knows", "Bob")
		require.NoError(t, err)
		err = store.UpsertTriplet(ctx, "Alice", "knows", "Bob")
		require.NoError(t, err)
		assert.Equal(t, 1, store.TripletCount()) // Should not duplicate
	})

	t.Run("Get", func(t *testing.T) {
		store := NewSimpleGraphStore()
		_ = store.UpsertTriplet(ctx, "Alice", "knows", "Bob")
		_ = store.UpsertTriplet(ctx, "Alice", "likes", "Charlie")

		rels, err := store.Get(ctx, "Alice")
		require.NoError(t, err)
		assert.Len(t, rels, 2)
	})

	t.Run("Get_not_found", func(t *testing.T) {
		store := NewSimpleGraphStore()
		rels, err := store.Get(ctx, "Unknown")
		require.NoError(t, err)
		assert.Nil(t, rels)
	})

	t.Run("Delete", func(t *testing.T) {
		store := NewSimpleGraphStore()
		_ = store.UpsertTriplet(ctx, "Alice", "knows", "Bob")
		_ = store.UpsertTriplet(ctx, "Alice", "likes", "Charlie")

		err := store.Delete(ctx, "Alice", "knows", "Bob")
		require.NoError(t, err)

		rels, _ := store.Get(ctx, "Alice")
		assert.Len(t, rels, 1)
	})

	t.Run("Delete_last_relation", func(t *testing.T) {
		store := NewSimpleGraphStore()
		_ = store.UpsertTriplet(ctx, "Alice", "knows", "Bob")

		err := store.Delete(ctx, "Alice", "knows", "Bob")
		require.NoError(t, err)

		assert.Equal(t, 0, store.Size())
	})

	t.Run("GetRelMap", func(t *testing.T) {
		store := NewSimpleGraphStore()
		_ = store.UpsertTriplet(ctx, "Alice", "knows", "Bob")
		_ = store.UpsertTriplet(ctx, "Bob", "works_at", "Company")

		relMap, err := store.GetRelMap(ctx, []string{"Alice"}, 2, 30)
		require.NoError(t, err)
		assert.Contains(t, relMap, "Alice")
	})

	t.Run("GetAllSubjects", func(t *testing.T) {
		store := NewSimpleGraphStore()
		_ = store.UpsertTriplet(ctx, "Alice", "knows", "Bob")
		_ = store.UpsertTriplet(ctx, "Charlie", "likes", "Dave")

		subjects, err := store.GetAllSubjects(ctx)
		require.NoError(t, err)
		assert.Len(t, subjects, 2)
		assert.Contains(t, subjects, "Alice")
		assert.Contains(t, subjects, "Charlie")
	})

	t.Run("GetTriplets", func(t *testing.T) {
		store := NewSimpleGraphStore()
		_ = store.UpsertTriplet(ctx, "Alice", "knows", "Bob")
		_ = store.UpsertTriplet(ctx, "Charlie", "likes", "Dave")

		triplets, err := store.GetTriplets(ctx)
		require.NoError(t, err)
		assert.Len(t, triplets, 2)
	})

	t.Run("Clear", func(t *testing.T) {
		store := NewSimpleGraphStore()
		_ = store.UpsertTriplet(ctx, "Alice", "knows", "Bob")
		store.Clear()
		assert.Equal(t, 0, store.Size())
	})

	t.Run("GetSchema_not_supported", func(t *testing.T) {
		store := NewSimpleGraphStore()
		_, err := store.GetSchema(ctx, false)
		assert.Error(t, err)
	})

	t.Run("Query_not_supported", func(t *testing.T) {
		store := NewSimpleGraphStore()
		_, err := store.Query(ctx, "MATCH (n) RETURN n", nil)
		assert.Error(t, err)
	})

	t.Run("ToDict_and_FromDict", func(t *testing.T) {
		store := NewSimpleGraphStore()
		_ = store.UpsertTriplet(ctx, "Alice", "knows", "Bob")

		dict := store.ToDict()
		restored, err := FromDict(dict)
		require.NoError(t, err)
		assert.Equal(t, store.Size(), restored.Size())
	})

	t.Run("MarshalJSON_and_UnmarshalJSON", func(t *testing.T) {
		store := NewSimpleGraphStore()
		_ = store.UpsertTriplet(ctx, "Alice", "knows", "Bob")

		data, err := json.Marshal(store)
		require.NoError(t, err)

		var restored SimpleGraphStore
		err = json.Unmarshal(data, &restored)
		require.NoError(t, err)
		assert.Equal(t, store.Size(), restored.Size())
	})
}

func TestSimpleGraphStore_Persistence(t *testing.T) {
	ctx := context.Background()

	t.Run("Persist_and_Load", func(t *testing.T) {
		tmpDir := t.TempDir()
		path := filepath.Join(tmpDir, "graph_store.json")

		// Create and persist
		store := NewSimpleGraphStore()
		_ = store.UpsertTriplet(ctx, "Alice", "knows", "Bob")
		_ = store.UpsertTriplet(ctx, "Bob", "works_at", "Company")

		err := store.Persist(ctx, path)
		require.NoError(t, err)

		// Load
		loaded, err := NewSimpleGraphStoreFromFile(path)
		require.NoError(t, err)
		assert.Equal(t, store.Size(), loaded.Size())
		assert.Equal(t, store.TripletCount(), loaded.TripletCount())
	})

	t.Run("Load_nonexistent_file", func(t *testing.T) {
		store, err := NewSimpleGraphStoreFromFile("/nonexistent/path/file.json")
		require.NoError(t, err) // Should return empty store
		assert.Equal(t, 0, store.Size())
	})

	t.Run("Persist_creates_directory", func(t *testing.T) {
		tmpDir := t.TempDir()
		path := filepath.Join(tmpDir, "subdir", "graph_store.json")

		store := NewSimpleGraphStore()
		_ = store.UpsertTriplet(ctx, "Alice", "knows", "Bob")

		err := store.Persist(ctx, path)
		require.NoError(t, err)

		// Verify file exists
		_, err = os.Stat(path)
		assert.NoError(t, err)
	})
}

func TestSimpleGraphStore_WithOptions(t *testing.T) {
	t.Run("WithGraphStoreData", func(t *testing.T) {
		data := NewGraphStoreData()
		data.GraphDict["Alice"] = [][]string{{"knows", "Bob"}}

		store := NewSimpleGraphStore(WithGraphStoreData(data))
		assert.Equal(t, 1, store.Size())
	})
}
