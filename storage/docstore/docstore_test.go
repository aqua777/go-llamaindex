package docstore

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/storage/kvstore"
)

func createTestNode(id, text string) *schema.Node {
	node := schema.NewTextNode(text)
	node.ID = id
	return node
}

func TestSimpleDocumentStoreBasic(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleDocumentStore()

	node := createTestNode("test_id", "test content")

	// Test AddDocuments and GetDocument
	err := store.AddDocuments(ctx, []schema.BaseNode{node}, true)
	if err != nil {
		t.Fatalf("AddDocuments failed: %v", err)
	}

	doc, err := store.GetDocument(ctx, "test_id", true)
	if err != nil {
		t.Fatalf("GetDocument failed: %v", err)
	}
	if doc == nil {
		t.Fatal("GetDocument returned nil")
	}
	if doc.GetID() != "test_id" {
		t.Errorf("Expected ID test_id, got %s", doc.GetID())
	}
}

func TestSimpleDocumentStoreDocumentExists(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleDocumentStore()

	node := createTestNode("test_id", "test content")
	err := store.AddDocuments(ctx, []schema.BaseNode{node}, true)
	if err != nil {
		t.Fatalf("AddDocuments failed: %v", err)
	}

	// Test DocumentExists
	exists, err := store.DocumentExists(ctx, "test_id")
	if err != nil {
		t.Fatalf("DocumentExists failed: %v", err)
	}
	if !exists {
		t.Error("Expected document to exist")
	}

	exists, err = store.DocumentExists(ctx, "non_existent")
	if err != nil {
		t.Fatalf("DocumentExists failed: %v", err)
	}
	if exists {
		t.Error("Expected document to not exist")
	}
}

func TestSimpleDocumentStoreDelete(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleDocumentStore()

	node := createTestNode("test_id", "test content")
	err := store.AddDocuments(ctx, []schema.BaseNode{node}, true)
	if err != nil {
		t.Fatalf("AddDocuments failed: %v", err)
	}

	// Test DeleteDocument
	err = store.DeleteDocument(ctx, "test_id", true)
	if err != nil {
		t.Fatalf("DeleteDocument failed: %v", err)
	}

	// Verify deletion
	exists, err := store.DocumentExists(ctx, "test_id")
	if err != nil {
		t.Fatalf("DocumentExists failed: %v", err)
	}
	if exists {
		t.Error("Expected document to be deleted")
	}

	// Test delete non-existent with raiseError=false
	err = store.DeleteDocument(ctx, "non_existent", false)
	if err != nil {
		t.Errorf("DeleteDocument with raiseError=false should not error: %v", err)
	}

	// Test delete non-existent with raiseError=true
	err = store.DeleteDocument(ctx, "non_existent", true)
	if err == nil {
		t.Error("DeleteDocument with raiseError=true should error for non-existent doc")
	}
}

func TestSimpleDocumentStoreDocs(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleDocumentStore()

	node1 := createTestNode("id1", "content 1")
	node2 := createTestNode("id2", "content 2")

	err := store.AddDocuments(ctx, []schema.BaseNode{node1, node2}, true)
	if err != nil {
		t.Fatalf("AddDocuments failed: %v", err)
	}

	// Test Docs
	docs, err := store.Docs(ctx)
	if err != nil {
		t.Fatalf("Docs failed: %v", err)
	}
	if len(docs) != 2 {
		t.Errorf("Expected 2 docs, got %d", len(docs))
	}
}

func TestSimpleDocumentStoreHash(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleDocumentStore()

	node := createTestNode("test_id", "test content")
	err := store.AddDocuments(ctx, []schema.BaseNode{node}, true)
	if err != nil {
		t.Fatalf("AddDocuments failed: %v", err)
	}

	// Test GetDocumentHash
	hash, err := store.GetDocumentHash(ctx, "test_id")
	if err != nil {
		t.Fatalf("GetDocumentHash failed: %v", err)
	}
	if hash == "" {
		t.Error("Expected non-empty hash")
	}

	// Test SetDocumentHash
	err = store.SetDocumentHash(ctx, "test_id", "custom_hash")
	if err != nil {
		t.Fatalf("SetDocumentHash failed: %v", err)
	}

	hash, err = store.GetDocumentHash(ctx, "test_id")
	if err != nil {
		t.Fatalf("GetDocumentHash failed: %v", err)
	}
	if hash != "custom_hash" {
		t.Errorf("Expected custom_hash, got %s", hash)
	}

	// Test GetAllDocumentHashes
	hashes, err := store.GetAllDocumentHashes(ctx)
	if err != nil {
		t.Fatalf("GetAllDocumentHashes failed: %v", err)
	}
	if len(hashes) != 1 {
		t.Errorf("Expected 1 hash, got %d", len(hashes))
	}
}

func TestSimpleDocumentStorePersist(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	persistPath := filepath.Join(tmpDir, "docstore.json")

	store := NewSimpleDocumentStore()
	node := createTestNode("test_id", "test content")

	err := store.AddDocuments(ctx, []schema.BaseNode{node}, true)
	if err != nil {
		t.Fatalf("AddDocuments failed: %v", err)
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
	loadedStore, err := FromPersistPath(ctx, persistPath)
	if err != nil {
		t.Fatalf("FromPersistPath failed: %v", err)
	}

	// Verify loaded data
	doc, err := loadedStore.GetDocument(ctx, "test_id", true)
	if err != nil {
		t.Fatalf("GetDocument failed: %v", err)
	}
	if doc == nil {
		t.Fatal("Document was not persisted")
	}
}

func TestSimpleDocumentStoreDict(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleDocumentStore()

	node := createTestNode("test_id", "test content")
	err := store.AddDocuments(ctx, []schema.BaseNode{node}, true)
	if err != nil {
		t.Fatalf("AddDocuments failed: %v", err)
	}

	// Test ToDict and FromDict
	data := store.ToDict()
	loadedStore := FromDict(data)

	doc, err := loadedStore.GetDocument(ctx, "test_id", true)
	if err != nil {
		t.Fatalf("GetDocument failed: %v", err)
	}
	if doc == nil {
		t.Fatal("Document was not loaded from dict")
	}
}

func TestSimpleDocumentStoreAllowUpdate(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleDocumentStore()

	node := createTestNode("test_id", "original content")
	err := store.AddDocuments(ctx, []schema.BaseNode{node}, true)
	if err != nil {
		t.Fatalf("AddDocuments failed: %v", err)
	}

	// Try to add same ID with allowUpdate=false
	node2 := createTestNode("test_id", "updated content")
	err = store.AddDocuments(ctx, []schema.BaseNode{node2}, false)
	if err == nil {
		t.Error("Expected error when adding duplicate with allowUpdate=false")
	}

	// Add with allowUpdate=true should succeed
	err = store.AddDocuments(ctx, []schema.BaseNode{node2}, true)
	if err != nil {
		t.Fatalf("AddDocuments with allowUpdate=true failed: %v", err)
	}
}

func TestKVDocumentStoreWithNamespace(t *testing.T) {
	ctx := context.Background()
	kv := kvstore.NewSimpleKVStore()

	store1 := NewKVDocumentStore(kv, WithNamespace("namespace1"))
	store2 := NewKVDocumentStore(kv, WithNamespace("namespace2"))

	node1 := createTestNode("test_id", "content 1")
	node2 := createTestNode("test_id", "content 2")

	// Add to different namespaces
	err := store1.AddDocuments(ctx, []schema.BaseNode{node1}, true)
	if err != nil {
		t.Fatalf("AddDocuments to store1 failed: %v", err)
	}

	err = store2.AddDocuments(ctx, []schema.BaseNode{node2}, true)
	if err != nil {
		t.Fatalf("AddDocuments to store2 failed: %v", err)
	}

	// Verify isolation
	doc1, err := store1.GetDocument(ctx, "test_id", true)
	if err != nil {
		t.Fatalf("GetDocument from store1 failed: %v", err)
	}

	doc2, err := store2.GetDocument(ctx, "test_id", true)
	if err != nil {
		t.Fatalf("GetDocument from store2 failed: %v", err)
	}

	// Both should exist but be different
	if doc1 == nil || doc2 == nil {
		t.Fatal("Documents should exist in both namespaces")
	}
}

func TestRefDocInfo(t *testing.T) {
	info := NewRefDocInfo()
	if info.NodeIDs == nil {
		t.Error("NodeIDs should not be nil")
	}
	if info.Metadata == nil {
		t.Error("Metadata should not be nil")
	}

	info.NodeIDs = append(info.NodeIDs, "node1", "node2")
	info.Metadata["key"] = "value"

	// Test ToMap and FromMap
	m := info.ToMap()
	restored := RefDocInfoFromMap(m)

	if len(restored.NodeIDs) != 2 {
		t.Errorf("Expected 2 node IDs, got %d", len(restored.NodeIDs))
	}
	if restored.Metadata["key"] != "value" {
		t.Error("Metadata not restored correctly")
	}
}

func TestGetDocumentNotFound(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleDocumentStore()

	// Test with raiseError=true
	_, err := store.GetDocument(ctx, "non_existent", true)
	if err == nil {
		t.Error("Expected error for non-existent document with raiseError=true")
	}

	// Test with raiseError=false
	doc, err := store.GetDocument(ctx, "non_existent", false)
	if err != nil {
		t.Errorf("Expected no error with raiseError=false: %v", err)
	}
	if doc != nil {
		t.Error("Expected nil document for non-existent ID")
	}
}

func TestGetNodes(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleDocumentStore()

	node1 := createTestNode("id1", "content 1")
	node2 := createTestNode("id2", "content 2")

	err := store.AddDocuments(ctx, []schema.BaseNode{node1, node2}, true)
	if err != nil {
		t.Fatalf("AddDocuments failed: %v", err)
	}

	// Test GetNodes helper
	nodes, err := GetNodes(ctx, store, []string{"id1", "id2"}, true)
	if err != nil {
		t.Fatalf("GetNodes failed: %v", err)
	}
	if len(nodes) != 2 {
		t.Errorf("Expected 2 nodes, got %d", len(nodes))
	}

	// Test with non-existent ID and raiseError=false
	nodes, err = GetNodes(ctx, store, []string{"id1", "non_existent"}, false)
	if err != nil {
		t.Fatalf("GetNodes with raiseError=false failed: %v", err)
	}
	if len(nodes) != 1 {
		t.Errorf("Expected 1 node, got %d", len(nodes))
	}
}
