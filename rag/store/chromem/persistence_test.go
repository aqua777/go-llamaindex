package chromem

import (
	"context"
	"os"
	"testing"

	"github.com/aqua777/go-llamaindex/schema"
)

func TestChromemPersistence(t *testing.T) {
	// Create a temporary directory for the test
	tmpDir, err := os.MkdirTemp("", "chromem_test")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// 1. Create store with persistence
	store, err := NewChromemStore(tmpDir, "test_collection")
	if err != nil {
		t.Fatalf("failed to create persistent store: %v", err)
	}

	// 2. Add a node
	node := schema.Node{
		ID:        "1",
		Text:      "Hello persistence",
		Type:      schema.ObjectTypeText,
		Metadata:  map[string]interface{}{"foo": "bar"},
		Embedding: []float64{0.1, 0.2, 0.3},
	}
	_, err = store.Add(context.Background(), []schema.Node{node})
	if err != nil {
		t.Fatalf("failed to add node: %v", err)
	}

	// 3. Close the store (simulating app restart) - Chromem doesn't have Close(), 
	// but NewPersistentDB loads from disk.
	// So we just create a new instance pointing to the same dir.

	// 4. Create a NEW store instance pointing to the SAME directory
	store2, err := NewChromemStore(tmpDir, "test_collection")
	if err != nil {
		t.Fatalf("failed to create second persistent store: %v", err)
	}

	// 5. Query from the new instance
	// We use the same embedding to find it
	query := schema.VectorStoreQuery{
		Embedding: []float64{0.1, 0.2, 0.3},
		TopK:      1,
	}
	results, err := store2.Query(context.Background(), query)
	if err != nil {
		t.Fatalf("failed to query second store: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}

	if results[0].Node.Text != "Hello persistence" {
		t.Errorf("expected text 'Hello persistence', got '%s'", results[0].Node.Text)
	}

	if results[0].Node.ID != "1" {
		t.Errorf("expected ID '1', got '%s'", results[0].Node.ID)
	}
}
