package storage

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/storage/indexstore"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewStorageContext(t *testing.T) {
	sc := NewStorageContext()

	assert.NotNil(t, sc.DocStore)
	assert.NotNil(t, sc.IndexStore)
	assert.NotNil(t, sc.VectorStores)
	assert.Empty(t, sc.VectorStores)
}

func TestNewStorageContextFromOptions(t *testing.T) {
	ctx := context.Background()

	sc, err := NewStorageContextFromOptions(ctx, StorageContextOptions{})
	require.NoError(t, err)

	assert.NotNil(t, sc.DocStore)
	assert.NotNil(t, sc.IndexStore)
}

func TestStorageContextVectorStore(t *testing.T) {
	sc := NewStorageContext()

	// Initially nil
	assert.Nil(t, sc.VectorStore())

	// After adding, should be retrievable
	// Note: We can't easily test with a real VectorStore without more setup
	// This tests the accessor methods work correctly
	assert.Nil(t, sc.GetVectorStore("nonexistent"))
}

func TestStorageContextPersist(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()

	sc := NewStorageContext()

	// Add some data to document store
	node := schema.NewTextNode("Test content")
	err := sc.DocStore.AddDocuments(ctx, []schema.BaseNode{node}, true)
	require.NoError(t, err)

	// Add an index struct
	indexStruct := indexstore.NewVectorStoreIndex()
	indexStruct.IndexID = "test-index"
	err = sc.IndexStore.AddIndexStruct(ctx, indexStruct)
	require.NoError(t, err)

	// Persist
	err = sc.Persist(ctx, tmpDir)
	require.NoError(t, err)

	// Verify files exist
	docStorePath := filepath.Join(tmpDir, DocStoreFilename)
	indexStorePath := filepath.Join(tmpDir, IndexStoreFilename)

	_, err = os.Stat(docStorePath)
	assert.NoError(t, err, "DocStore file should exist")

	_, err = os.Stat(indexStorePath)
	assert.NoError(t, err, "IndexStore file should exist")
}

func TestStorageContextFromPersistDir(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()

	// Create and persist a storage context
	sc1 := NewStorageContext()

	node := schema.NewTextNode("Persisted content")
	err := sc1.DocStore.AddDocuments(ctx, []schema.BaseNode{node}, true)
	require.NoError(t, err)

	indexStruct := indexstore.NewVectorStoreIndex()
	indexStruct.IndexID = "persisted-index"
	err = sc1.IndexStore.AddIndexStruct(ctx, indexStruct)
	require.NoError(t, err)

	err = sc1.Persist(ctx, tmpDir)
	require.NoError(t, err)

	// Load from persist directory
	sc2, err := StorageContextFromPersistDir(ctx, tmpDir)
	require.NoError(t, err)

	// Verify document was loaded
	exists, err := sc2.DocStore.DocumentExists(ctx, node.GetID())
	require.NoError(t, err)
	assert.True(t, exists, "Document should exist after loading")

	// Verify index struct was loaded
	loadedIndex, err := sc2.IndexStore.GetIndexStruct(ctx, "persisted-index")
	require.NoError(t, err)
	assert.NotNil(t, loadedIndex)
	assert.Equal(t, "persisted-index", loadedIndex.IndexID)
}

func TestStorageContextFromPersistDirNonexistent(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	nonexistentDir := filepath.Join(tmpDir, "nonexistent")

	// Should create empty stores when directory doesn't exist
	sc, err := StorageContextFromPersistDir(ctx, nonexistentDir)
	require.NoError(t, err)
	assert.NotNil(t, sc.DocStore)
	assert.NotNil(t, sc.IndexStore)
}

func TestStorageContextToDict(t *testing.T) {
	ctx := context.Background()
	sc := NewStorageContext()

	node := schema.NewTextNode("Dict test content")
	err := sc.DocStore.AddDocuments(ctx, []schema.BaseNode{node}, true)
	require.NoError(t, err)

	data, err := sc.ToDict(ctx)
	require.NoError(t, err)

	assert.Contains(t, data, DocStoreKey)
	assert.Contains(t, data, IndexStoreKey)
}

func TestStorageContextToAndFromJSON(t *testing.T) {
	ctx := context.Background()
	sc1 := NewStorageContext()

	node := schema.NewTextNode("JSON test content")
	err := sc1.DocStore.AddDocuments(ctx, []schema.BaseNode{node}, true)
	require.NoError(t, err)

	// Convert to JSON
	jsonData, err := sc1.ToJSON(ctx)
	require.NoError(t, err)
	assert.NotEmpty(t, jsonData)

	// Load from JSON
	sc2, err := StorageContextFromJSON(ctx, jsonData)
	require.NoError(t, err)

	// Verify document was loaded
	exists, err := sc2.DocStore.DocumentExists(ctx, node.GetID())
	require.NoError(t, err)
	assert.True(t, exists, "Document should exist after JSON round-trip")
}

func TestStorageContextAddVectorStore(t *testing.T) {
	sc := NewStorageContext()

	// Initially empty
	assert.Empty(t, sc.VectorStores)

	// Note: We can't easily add a real VectorStore without more setup
	// This tests the map operations work correctly
	assert.Nil(t, sc.GetVectorStore("custom"))
}

func TestStorageContextDefaultPersistDir(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()

	// Change to temp directory to test default persist
	originalWd, _ := os.Getwd()
	defer os.Chdir(originalWd)
	os.Chdir(tmpDir)

	sc := NewStorageContext()
	err := sc.Persist(ctx, "")
	require.NoError(t, err)

	// Verify default directory was created
	_, err = os.Stat(DefaultPersistDir)
	assert.NoError(t, err, "Default persist directory should be created")
}
