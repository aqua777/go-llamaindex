package kvstore

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestSimpleKVStoreBasic(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleKVStore()

	testKey := "test_key"
	testBlob := StoredValue{"test_obj_key": "test_obj_val"}

	// Test Put and Get
	err := store.Put(ctx, testKey, testBlob, DefaultCollection)
	if err != nil {
		t.Fatalf("Put failed: %v", err)
	}

	blob, err := store.Get(ctx, testKey, DefaultCollection)
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if blob == nil {
		t.Fatal("Get returned nil")
	}
	if blob["test_obj_key"] != "test_obj_val" {
		t.Errorf("Expected test_obj_val, got %v", blob["test_obj_key"])
	}

	// Test Get from non-existent collection
	blob, err = store.Get(ctx, testKey, "non_existent")
	if err != nil {
		t.Fatalf("Get from non-existent collection failed: %v", err)
	}
	if blob != nil {
		t.Error("Expected nil for non-existent collection")
	}
}

func TestSimpleKVStoreGetAll(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleKVStore()

	// Put multiple items
	err := store.Put(ctx, "key1", StoredValue{"val": "1"}, DefaultCollection)
	if err != nil {
		t.Fatalf("Put failed: %v", err)
	}
	err = store.Put(ctx, "key2", StoredValue{"val": "2"}, DefaultCollection)
	if err != nil {
		t.Fatalf("Put failed: %v", err)
	}

	// Test GetAll
	all, err := store.GetAll(ctx, DefaultCollection)
	if err != nil {
		t.Fatalf("GetAll failed: %v", err)
	}
	if len(all) != 2 {
		t.Errorf("Expected 2 items, got %d", len(all))
	}
}

func TestSimpleKVStoreDelete(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleKVStore()

	testKey := "test_key"
	testBlob := StoredValue{"test_obj_key": "test_obj_val"}

	err := store.Put(ctx, testKey, testBlob, DefaultCollection)
	if err != nil {
		t.Fatalf("Put failed: %v", err)
	}

	// Test Delete
	deleted, err := store.Delete(ctx, testKey, DefaultCollection)
	if err != nil {
		t.Fatalf("Delete failed: %v", err)
	}
	if !deleted {
		t.Error("Expected delete to return true")
	}

	// Verify deletion
	blob, err := store.Get(ctx, testKey, DefaultCollection)
	if err != nil {
		t.Fatalf("Get after delete failed: %v", err)
	}
	if blob != nil {
		t.Error("Expected nil after delete")
	}

	// Test Delete non-existent key
	deleted, err = store.Delete(ctx, "non_existent", DefaultCollection)
	if err != nil {
		t.Fatalf("Delete non-existent failed: %v", err)
	}
	if deleted {
		t.Error("Expected delete to return false for non-existent key")
	}
}

func TestSimpleKVStorePersist(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	persistPath := filepath.Join(tmpDir, "kvstore.json")

	store := NewSimpleKVStore()
	testKey := "test_key"
	testBlob := StoredValue{"test_obj_key": "test_obj_val"}

	err := store.Put(ctx, testKey, testBlob, DefaultCollection)
	if err != nil {
		t.Fatalf("Put failed: %v", err)
	}

	// Persist
	err = store.Persist(ctx, persistPath)
	if err != nil {
		t.Fatalf("Persist failed: %v", err)
	}

	// Load from persist path
	loadedStore, err := FromPersistPath(ctx, persistPath)
	if err != nil {
		t.Fatalf("FromPersistPath failed: %v", err)
	}

	all, err := loadedStore.GetAll(ctx, DefaultCollection)
	if err != nil {
		t.Fatalf("GetAll failed: %v", err)
	}
	if len(all) != 1 {
		t.Errorf("Expected 1 item, got %d", len(all))
	}
}

func TestSimpleKVStoreDict(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleKVStore()

	testKey := "test_key"
	testBlob := StoredValue{"test_obj_key": "test_obj_val"}

	err := store.Put(ctx, testKey, testBlob, DefaultCollection)
	if err != nil {
		t.Fatalf("Put failed: %v", err)
	}

	// Test ToDict and FromDict
	saveDict := store.ToDict()
	loadedStore := FromDict(saveDict)

	all, err := loadedStore.GetAll(ctx, DefaultCollection)
	if err != nil {
		t.Fatalf("GetAll failed: %v", err)
	}
	if len(all) != 1 {
		t.Errorf("Expected 1 item, got %d", len(all))
	}
}

func TestSimpleKVStoreCollections(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleKVStore()

	// Put in different collections
	err := store.Put(ctx, "key1", StoredValue{"val": "1"}, "collection1")
	if err != nil {
		t.Fatalf("Put failed: %v", err)
	}
	err = store.Put(ctx, "key2", StoredValue{"val": "2"}, "collection2")
	if err != nil {
		t.Fatalf("Put failed: %v", err)
	}

	// Verify isolation
	all1, err := store.GetAll(ctx, "collection1")
	if err != nil {
		t.Fatalf("GetAll failed: %v", err)
	}
	if len(all1) != 1 {
		t.Errorf("Expected 1 item in collection1, got %d", len(all1))
	}

	all2, err := store.GetAll(ctx, "collection2")
	if err != nil {
		t.Fatalf("GetAll failed: %v", err)
	}
	if len(all2) != 1 {
		t.Errorf("Expected 1 item in collection2, got %d", len(all2))
	}

	// Key from collection1 should not exist in collection2
	val, err := store.Get(ctx, "key1", "collection2")
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if val != nil {
		t.Error("Expected nil for key from different collection")
	}
}

func TestFileKVStoreBasic(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	persistPath := filepath.Join(tmpDir, "kvstore.json")

	store, err := NewFileKVStore(persistPath)
	if err != nil {
		t.Fatalf("NewFileKVStore failed: %v", err)
	}

	testKey := "test_key"
	testBlob := StoredValue{"test_obj_key": "test_obj_val"}

	// Test Put and Get
	err = store.Put(ctx, testKey, testBlob, DefaultCollection)
	if err != nil {
		t.Fatalf("Put failed: %v", err)
	}

	blob, err := store.Get(ctx, testKey, DefaultCollection)
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if blob == nil {
		t.Fatal("Get returned nil")
	}
	if blob["test_obj_key"] != "test_obj_val" {
		t.Errorf("Expected test_obj_val, got %v", blob["test_obj_key"])
	}

	// Verify file was created
	if _, err := os.Stat(persistPath); os.IsNotExist(err) {
		t.Error("Persist file was not created")
	}
}

func TestFileKVStorePersistence(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	persistPath := filepath.Join(tmpDir, "kvstore.json")

	// Create and populate store
	store, err := NewFileKVStore(persistPath)
	if err != nil {
		t.Fatalf("NewFileKVStore failed: %v", err)
	}

	testKey := "test_key"
	testBlob := StoredValue{"test_obj_key": "test_obj_val"}

	err = store.Put(ctx, testKey, testBlob, DefaultCollection)
	if err != nil {
		t.Fatalf("Put failed: %v", err)
	}

	// Create new store from same path (simulating restart)
	store2, err := NewFileKVStore(persistPath)
	if err != nil {
		t.Fatalf("NewFileKVStore (reload) failed: %v", err)
	}

	blob, err := store2.Get(ctx, testKey, DefaultCollection)
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if blob == nil {
		t.Fatal("Data was not persisted")
	}
	if blob["test_obj_key"] != "test_obj_val" {
		t.Errorf("Expected test_obj_val, got %v", blob["test_obj_key"])
	}
}

func TestFileKVStoreDelete(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	persistPath := filepath.Join(tmpDir, "kvstore.json")

	store, err := NewFileKVStore(persistPath)
	if err != nil {
		t.Fatalf("NewFileKVStore failed: %v", err)
	}

	testKey := "test_key"
	testBlob := StoredValue{"test_obj_key": "test_obj_val"}

	err = store.Put(ctx, testKey, testBlob, DefaultCollection)
	if err != nil {
		t.Fatalf("Put failed: %v", err)
	}

	// Delete
	deleted, err := store.Delete(ctx, testKey, DefaultCollection)
	if err != nil {
		t.Fatalf("Delete failed: %v", err)
	}
	if !deleted {
		t.Error("Expected delete to return true")
	}

	// Verify deletion persisted
	store2, err := NewFileKVStore(persistPath)
	if err != nil {
		t.Fatalf("NewFileKVStore (reload) failed: %v", err)
	}

	blob, err := store2.Get(ctx, testKey, DefaultCollection)
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if blob != nil {
		t.Error("Delete was not persisted")
	}
}

func TestValueIsolation(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleKVStore()

	testKey := "test_key"
	testBlob := StoredValue{"test_obj_key": "test_obj_val"}

	err := store.Put(ctx, testKey, testBlob, DefaultCollection)
	if err != nil {
		t.Fatalf("Put failed: %v", err)
	}

	// Modify original blob
	testBlob["test_obj_key"] = "modified"

	// Get should return original value
	blob, err := store.Get(ctx, testKey, DefaultCollection)
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if blob["test_obj_key"] != "test_obj_val" {
		t.Error("Store value was modified by external change")
	}

	// Modify retrieved blob
	blob["test_obj_key"] = "also_modified"

	// Get again should still return original
	blob2, err := store.Get(ctx, testKey, DefaultCollection)
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if blob2["test_obj_key"] != "test_obj_val" {
		t.Error("Store value was modified by modifying retrieved value")
	}
}

func TestEmptyCollection(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleKVStore()

	// GetAll on empty collection should return empty map
	all, err := store.GetAll(ctx, DefaultCollection)
	if err != nil {
		t.Fatalf("GetAll failed: %v", err)
	}
	if all == nil {
		t.Error("GetAll returned nil instead of empty map")
	}
	if len(all) != 0 {
		t.Errorf("Expected empty map, got %d items", len(all))
	}
}

func TestDefaultCollection(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleKVStore()

	testKey := "test_key"
	testBlob := StoredValue{"val": "test"}

	// Put with empty collection should use default
	err := store.Put(ctx, testKey, testBlob, "")
	if err != nil {
		t.Fatalf("Put failed: %v", err)
	}

	// Get with empty collection should use default
	blob, err := store.Get(ctx, testKey, "")
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if blob == nil {
		t.Fatal("Get returned nil")
	}

	// Should also be accessible via explicit default collection
	blob, err = store.Get(ctx, testKey, DefaultCollection)
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if blob == nil {
		t.Fatal("Get with explicit default collection returned nil")
	}
}
