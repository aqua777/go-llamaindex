package indexstore

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/aqua777/go-llamaindex/storage/kvstore"
)

func TestNewIndexStruct(t *testing.T) {
	is := NewVectorStoreIndex()
	if is.IndexID == "" {
		t.Error("IndexID should not be empty")
	}
	if is.Type != IndexStructTypeVectorStore {
		t.Errorf("Expected type %s, got %s", IndexStructTypeVectorStore, is.Type)
	}
}

func TestIndexStructAddNode(t *testing.T) {
	is := NewVectorStoreIndex()

	// Add node with auto-generated text ID
	textID := is.AddNode("node1", "")
	if textID != "node1" {
		t.Errorf("Expected textID node1, got %s", textID)
	}
	if is.NodesDict["node1"] != "node1" {
		t.Error("Node not added correctly")
	}

	// Add node with custom text ID
	textID = is.AddNode("node2", "custom_id")
	if textID != "custom_id" {
		t.Errorf("Expected textID custom_id, got %s", textID)
	}
	if is.NodesDict["custom_id"] != "node2" {
		t.Error("Node not added correctly with custom ID")
	}
}

func TestIndexStructDeleteNode(t *testing.T) {
	is := NewVectorStoreIndex()
	is.AddNode("node1", "text1")

	is.DeleteNode("text1")
	if _, ok := is.NodesDict["text1"]; ok {
		t.Error("Node should have been deleted")
	}
}

func TestIndexStructAddToList(t *testing.T) {
	is := NewListIndex()
	is.AddToList("node1")
	is.AddToList("node2")

	if len(is.Nodes) != 2 {
		t.Errorf("Expected 2 nodes, got %d", len(is.Nodes))
	}
	if is.Nodes[0] != "node1" || is.Nodes[1] != "node2" {
		t.Error("Nodes not added in correct order")
	}
}

func TestIndexStructAddToTable(t *testing.T) {
	is := NewKeywordTableIndex()
	is.AddToTable([]string{"keyword1", "keyword2"}, "node1")
	is.AddToTable([]string{"keyword1"}, "node2")

	if len(is.Table["keyword1"]) != 2 {
		t.Errorf("Expected 2 nodes for keyword1, got %d", len(is.Table["keyword1"]))
	}
	if len(is.Table["keyword2"]) != 1 {
		t.Errorf("Expected 1 node for keyword2, got %d", len(is.Table["keyword2"]))
	}
}

func TestIndexStructToJSONAndFromJSON(t *testing.T) {
	is := NewVectorStoreIndex()
	is.Summary = "test summary"
	is.AddNode("node1", "text1")

	jsonData := is.ToJSON()
	restored, err := FromJSON(jsonData)
	if err != nil {
		t.Fatalf("FromJSON failed: %v", err)
	}

	if restored.IndexID != is.IndexID {
		t.Errorf("IndexID mismatch: expected %s, got %s", is.IndexID, restored.IndexID)
	}
	if restored.Summary != is.Summary {
		t.Errorf("Summary mismatch: expected %s, got %s", is.Summary, restored.Summary)
	}
	if restored.Type != is.Type {
		t.Errorf("Type mismatch: expected %s, got %s", is.Type, restored.Type)
	}
	if restored.NodesDict["text1"] != "node1" {
		t.Error("NodesDict not restored correctly")
	}
}

func TestSimpleIndexStoreBasic(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleIndexStore()

	is := NewVectorStoreIndex()
	is.AddNode("node1", "text1")

	// Test AddIndexStruct
	err := store.AddIndexStruct(ctx, is)
	if err != nil {
		t.Fatalf("AddIndexStruct failed: %v", err)
	}

	// Test GetIndexStruct
	retrieved, err := store.GetIndexStruct(ctx, is.IndexID)
	if err != nil {
		t.Fatalf("GetIndexStruct failed: %v", err)
	}
	if retrieved.IndexID != is.IndexID {
		t.Errorf("IndexID mismatch: expected %s, got %s", is.IndexID, retrieved.IndexID)
	}
}

func TestSimpleIndexStoreGetWithoutID(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleIndexStore()

	is := NewVectorStoreIndex()
	err := store.AddIndexStruct(ctx, is)
	if err != nil {
		t.Fatalf("AddIndexStruct failed: %v", err)
	}

	// Get without ID should return the only struct
	retrieved, err := store.GetIndexStruct(ctx, "")
	if err != nil {
		t.Fatalf("GetIndexStruct failed: %v", err)
	}
	if retrieved.IndexID != is.IndexID {
		t.Errorf("IndexID mismatch: expected %s, got %s", is.IndexID, retrieved.IndexID)
	}
}

func TestSimpleIndexStoreGetWithoutIDMultiple(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleIndexStore()

	is1 := NewVectorStoreIndex()
	is2 := NewListIndex()

	err := store.AddIndexStruct(ctx, is1)
	if err != nil {
		t.Fatalf("AddIndexStruct failed: %v", err)
	}
	err = store.AddIndexStruct(ctx, is2)
	if err != nil {
		t.Fatalf("AddIndexStruct failed: %v", err)
	}

	// Get without ID should error when multiple structs exist
	_, err = store.GetIndexStruct(ctx, "")
	if err != ErrMultipleIndexStructs {
		t.Errorf("Expected ErrMultipleIndexStructs, got %v", err)
	}
}

func TestSimpleIndexStoreDelete(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleIndexStore()

	is := NewVectorStoreIndex()
	err := store.AddIndexStruct(ctx, is)
	if err != nil {
		t.Fatalf("AddIndexStruct failed: %v", err)
	}

	// Delete
	err = store.DeleteIndexStruct(ctx, is.IndexID)
	if err != nil {
		t.Fatalf("DeleteIndexStruct failed: %v", err)
	}

	// Verify deletion
	_, err = store.GetIndexStruct(ctx, is.IndexID)
	if err != ErrIndexStructNotFound {
		t.Errorf("Expected ErrIndexStructNotFound, got %v", err)
	}
}

func TestSimpleIndexStoreIndexStructs(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleIndexStore()

	is1 := NewVectorStoreIndex()
	is2 := NewListIndex()

	err := store.AddIndexStruct(ctx, is1)
	if err != nil {
		t.Fatalf("AddIndexStruct failed: %v", err)
	}
	err = store.AddIndexStruct(ctx, is2)
	if err != nil {
		t.Fatalf("AddIndexStruct failed: %v", err)
	}

	// Get all structs
	structs, err := store.IndexStructs(ctx)
	if err != nil {
		t.Fatalf("IndexStructs failed: %v", err)
	}
	if len(structs) != 2 {
		t.Errorf("Expected 2 structs, got %d", len(structs))
	}
}

func TestSimpleIndexStorePersist(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	persistPath := filepath.Join(tmpDir, "index_store.json")

	store := NewSimpleIndexStore()
	is := NewVectorStoreIndex()
	is.AddNode("node1", "text1")

	err := store.AddIndexStruct(ctx, is)
	if err != nil {
		t.Fatalf("AddIndexStruct failed: %v", err)
	}

	// Persist
	err = store.Persist(ctx, persistPath)
	if err != nil {
		t.Fatalf("Persist failed: %v", err)
	}

	// Verify file exists
	if _, err := os.Stat(persistPath); os.IsNotExist(err) {
		t.Error("Persist file was not created")
	}

	// Load from persist path
	loadedStore, err := SimpleIndexStoreFromPersistPath(ctx, persistPath)
	if err != nil {
		t.Fatalf("SimpleIndexStoreFromPersistPath failed: %v", err)
	}

	// Verify loaded data
	retrieved, err := loadedStore.GetIndexStruct(ctx, is.IndexID)
	if err != nil {
		t.Fatalf("GetIndexStruct failed: %v", err)
	}
	if retrieved.IndexID != is.IndexID {
		t.Errorf("IndexID mismatch after load")
	}
}

func TestSimpleIndexStoreDict(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleIndexStore()

	is := NewVectorStoreIndex()
	err := store.AddIndexStruct(ctx, is)
	if err != nil {
		t.Fatalf("AddIndexStruct failed: %v", err)
	}

	// Test ToDict and FromDict
	data := store.ToDict()
	loadedStore := SimpleIndexStoreFromDict(data)

	retrieved, err := loadedStore.GetIndexStruct(ctx, is.IndexID)
	if err != nil {
		t.Fatalf("GetIndexStruct failed: %v", err)
	}
	if retrieved.IndexID != is.IndexID {
		t.Error("IndexID mismatch after FromDict")
	}
}

func TestKVIndexStoreWithNamespace(t *testing.T) {
	ctx := context.Background()
	kv := kvstore.NewSimpleKVStore()

	store1 := NewKVIndexStore(kv, WithIndexStoreNamespace("namespace1"))
	store2 := NewKVIndexStore(kv, WithIndexStoreNamespace("namespace2"))

	is1 := NewVectorStoreIndex()
	is2 := NewListIndex()

	// Add to different namespaces
	err := store1.AddIndexStruct(ctx, is1)
	if err != nil {
		t.Fatalf("AddIndexStruct to store1 failed: %v", err)
	}

	err = store2.AddIndexStruct(ctx, is2)
	if err != nil {
		t.Fatalf("AddIndexStruct to store2 failed: %v", err)
	}

	// Verify isolation
	structs1, err := store1.IndexStructs(ctx)
	if err != nil {
		t.Fatalf("IndexStructs from store1 failed: %v", err)
	}
	if len(structs1) != 1 {
		t.Errorf("Expected 1 struct in store1, got %d", len(structs1))
	}

	structs2, err := store2.IndexStructs(ctx)
	if err != nil {
		t.Fatalf("IndexStructs from store2 failed: %v", err)
	}
	if len(structs2) != 1 {
		t.Errorf("Expected 1 struct in store2, got %d", len(structs2))
	}
}

func TestGetIndexStructNotFound(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleIndexStore()

	_, err := store.GetIndexStruct(ctx, "non_existent")
	if err != ErrIndexStructNotFound {
		t.Errorf("Expected ErrIndexStructNotFound, got %v", err)
	}
}

func TestGetIndexStructEmptyStore(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleIndexStore()

	_, err := store.GetIndexStruct(ctx, "")
	if err != ErrIndexStructNotFound {
		t.Errorf("Expected ErrIndexStructNotFound for empty store, got %v", err)
	}
}

func TestTreeIndex(t *testing.T) {
	is := NewTreeIndex()
	if is.AllNodes == nil {
		t.Error("AllNodes should not be nil")
	}
	if is.RootNodes == nil {
		t.Error("RootNodes should not be nil")
	}
	if is.NodeIDToChildrenIDs == nil {
		t.Error("NodeIDToChildrenIDs should not be nil")
	}
	if is.Type != IndexStructTypeTree {
		t.Errorf("Expected type %s, got %s", IndexStructTypeTree, is.Type)
	}
}
