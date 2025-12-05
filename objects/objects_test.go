package objects

import (
	"context"
	"testing"

	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/tools"
)

// TestObject is a simple test object.
type TestObject struct {
	ID          string
	Name        string
	Description string
}

func (o TestObject) GetID() string          { return o.ID }
func (o TestObject) GetName() string        { return o.Name }
func (o TestObject) GetDescription() string { return o.Description }

// MockTool implements tools.Tool for testing.
type MockTool struct {
	name        string
	description string
}

func NewMockTool(name, description string) *MockTool {
	return &MockTool{name: name, description: description}
}

func (t *MockTool) Metadata() *tools.ToolMetadata {
	return &tools.ToolMetadata{
		Name:        t.name,
		Description: t.description,
	}
}

func (t *MockTool) Call(ctx context.Context, input interface{}) (*tools.ToolOutput, error) {
	return tools.NewToolOutput(t.name, "mock output"), nil
}

// MockEmbeddingModel implements embedding.EmbeddingModel for testing.
type MockEmbeddingModel struct {
	embeddings map[string][]float64
}

func NewMockEmbeddingModel() *MockEmbeddingModel {
	return &MockEmbeddingModel{
		embeddings: make(map[string][]float64),
	}
}

func (m *MockEmbeddingModel) GetTextEmbedding(ctx context.Context, text string) ([]float64, error) {
	// Return a simple hash-based embedding for testing
	emb := make([]float64, 8)
	for i, c := range text {
		if i >= len(emb) {
			break
		}
		emb[i] = float64(c) / 255.0
	}
	return emb, nil
}

func (m *MockEmbeddingModel) GetQueryEmbedding(ctx context.Context, query string) ([]float64, error) {
	return m.GetTextEmbedding(ctx, query)
}

func TestBaseObjectNodeMapping(t *testing.T) {
	t.Run("create mapping", func(t *testing.T) {
		mapping := NewBaseObjectNodeMapping()
		if mapping == nil {
			t.Fatal("expected non-nil mapping")
		}
	})

	t.Run("add object", func(t *testing.T) {
		mapping := NewBaseObjectNodeMapping()
		obj := TestObject{ID: "test-1", Name: "Test", Description: "A test object"}

		err := mapping.AddObject(obj)
		if err != nil {
			t.Fatalf("AddObject() error = %v", err)
		}

		objects := mapping.GetObjects()
		if len(objects) != 1 {
			t.Errorf("expected 1 object, got %d", len(objects))
		}
	})

	t.Run("get object by ID", func(t *testing.T) {
		mapping := NewBaseObjectNodeMapping()
		obj := TestObject{ID: "test-1", Name: "Test", Description: "A test object"}
		mapping.AddObject(obj)

		retrieved, err := mapping.GetObjectByID("test-1")
		if err != nil {
			t.Fatalf("GetObjectByID() error = %v", err)
		}

		testObj, ok := retrieved.(TestObject)
		if !ok {
			t.Fatalf("expected TestObject, got %T", retrieved)
		}

		if testObj.Name != "Test" {
			t.Errorf("expected name 'Test', got %s", testObj.Name)
		}
	})

	t.Run("get nodes", func(t *testing.T) {
		mapping := NewBaseObjectNodeMapping()
		obj := TestObject{ID: "test-1", Name: "Test", Description: "A test object"}
		mapping.AddObject(obj)

		nodes := mapping.GetNodes()
		if len(nodes) != 1 {
			t.Errorf("expected 1 node, got %d", len(nodes))
		}

		if nodes[0].ID != "test-1" {
			t.Errorf("expected node ID 'test-1', got %s", nodes[0].ID)
		}
	})

	t.Run("custom ToNode function", func(t *testing.T) {
		customToNode := func(obj interface{}) (*schema.Node, error) {
			testObj := obj.(TestObject)
			node := schema.NewTextNode("custom: " + testObj.Description)
			node.ID = testObj.ID
			return node, nil
		}

		mapping := NewBaseObjectNodeMapping(WithToNodeFunc(customToNode))
		obj := TestObject{ID: "test-1", Name: "Test", Description: "A test object"}
		mapping.AddObject(obj)

		nodes := mapping.GetNodes()
		if nodes[0].Text != "custom: A test object" {
			t.Errorf("expected custom text, got %s", nodes[0].Text)
		}
	})
}

func TestSimpleObjectNodeMapping(t *testing.T) {
	t.Run("add multiple objects", func(t *testing.T) {
		mapping := NewSimpleObjectNodeMapping()

		obj1 := TestObject{ID: "1", Name: "First", Description: "First object"}
		obj2 := TestObject{ID: "2", Name: "Second", Description: "Second object"}

		err := mapping.AddObjects(obj1, obj2)
		if err != nil {
			t.Fatalf("AddObjects() error = %v", err)
		}

		objects := mapping.GetObjects()
		if len(objects) != 2 {
			t.Errorf("expected 2 objects, got %d", len(objects))
		}
	})
}

func TestToolNodeMapping(t *testing.T) {
	t.Run("create mapping", func(t *testing.T) {
		mapping := NewToolNodeMapping()
		if mapping == nil {
			t.Fatal("expected non-nil mapping")
		}
	})

	t.Run("add tool", func(t *testing.T) {
		mapping := NewToolNodeMapping()
		tool := NewMockTool("search", "Search for information")

		err := mapping.AddTool(tool)
		if err != nil {
			t.Fatalf("AddTool() error = %v", err)
		}

		tools := mapping.GetTools()
		if len(tools) != 1 {
			t.Errorf("expected 1 tool, got %d", len(tools))
		}
	})

	t.Run("add multiple tools", func(t *testing.T) {
		mapping := NewToolNodeMapping()
		tool1 := NewMockTool("search", "Search for information")
		tool2 := NewMockTool("calculate", "Perform calculations")

		err := mapping.AddTools(tool1, tool2)
		if err != nil {
			t.Fatalf("AddTools() error = %v", err)
		}

		tools := mapping.GetTools()
		if len(tools) != 2 {
			t.Errorf("expected 2 tools, got %d", len(tools))
		}
	})

	t.Run("get tool by name", func(t *testing.T) {
		mapping := NewToolNodeMapping()
		tool := NewMockTool("search", "Search for information")
		mapping.AddTool(tool)

		retrieved, err := mapping.GetTool("search")
		if err != nil {
			t.Fatalf("GetTool() error = %v", err)
		}

		if retrieved.Metadata().Name != "search" {
			t.Errorf("expected tool name 'search', got %s", retrieved.Metadata().Name)
		}
	})

	t.Run("get tool names", func(t *testing.T) {
		mapping := NewToolNodeMapping()
		mapping.AddTool(NewMockTool("search", "Search"))
		mapping.AddTool(NewMockTool("calculate", "Calculate"))

		names := mapping.GetToolNames()
		if len(names) != 2 {
			t.Errorf("expected 2 names, got %d", len(names))
		}
	})

	t.Run("tool nodes have correct metadata", func(t *testing.T) {
		mapping := NewToolNodeMapping()
		tool := NewMockTool("search", "Search for information")
		mapping.AddTool(tool)

		nodes := mapping.GetNodes()
		if len(nodes) != 1 {
			t.Fatalf("expected 1 node, got %d", len(nodes))
		}

		node := nodes[0]
		if node.Metadata["name"] != "search" {
			t.Errorf("expected name 'search' in metadata")
		}
		if node.Metadata["object_type"] != "tool" {
			t.Errorf("expected object_type 'tool' in metadata")
		}
	})
}

func TestToolRetriever(t *testing.T) {
	t.Run("build index", func(t *testing.T) {
		mapping := NewToolNodeMapping()
		mapping.AddTool(NewMockTool("search", "Search for information"))
		mapping.AddTool(NewMockTool("calculate", "Perform calculations"))

		embedModel := NewMockEmbeddingModel()
		retriever := NewToolRetriever(mapping, embedModel)

		err := retriever.BuildIndex(context.Background())
		if err != nil {
			t.Fatalf("BuildIndex() error = %v", err)
		}
	})

	t.Run("retrieve tools", func(t *testing.T) {
		mapping := NewToolNodeMapping()
		mapping.AddTool(NewMockTool("search", "Search for information on the web"))
		mapping.AddTool(NewMockTool("calculate", "Perform mathematical calculations"))

		embedModel := NewMockEmbeddingModel()
		retriever := NewToolRetriever(mapping, embedModel, WithToolRetrieverTopK(1))

		err := retriever.BuildIndex(context.Background())
		if err != nil {
			t.Fatalf("BuildIndex() error = %v", err)
		}

		tools, err := retriever.RetrieveTools(context.Background(), "search")
		if err != nil {
			t.Fatalf("RetrieveTools() error = %v", err)
		}

		if len(tools) == 0 {
			t.Error("expected at least 1 tool")
		}
	})

	t.Run("retrieve objects interface", func(t *testing.T) {
		mapping := NewToolNodeMapping()
		mapping.AddTool(NewMockTool("search", "Search"))

		embedModel := NewMockEmbeddingModel()
		retriever := NewToolRetriever(mapping, embedModel)
		retriever.BuildIndex(context.Background())

		objects, err := retriever.RetrieveObjects(context.Background(), "search")
		if err != nil {
			t.Fatalf("RetrieveObjects() error = %v", err)
		}

		if len(objects) == 0 {
			t.Error("expected at least 1 object")
		}
	})
}

func TestTypedObjectNodeMapping(t *testing.T) {
	t.Run("create typed mapping", func(t *testing.T) {
		mapping := NewTypedObjectNodeMapping[TestObject]()
		if mapping == nil {
			t.Fatal("expected non-nil mapping")
		}
	})

	t.Run("add typed object", func(t *testing.T) {
		mapping := NewTypedObjectNodeMapping[TestObject](
			WithTextExtractor(func(o TestObject) string { return o.Description }),
			WithIDExtractor(func(o TestObject) string { return o.ID }),
		)

		obj := TestObject{ID: "test-1", Name: "Test", Description: "A test object"}
		err := mapping.AddTypedObject(obj)
		if err != nil {
			t.Fatalf("AddTypedObject() error = %v", err)
		}

		objects := mapping.GetTypedObjects()
		if len(objects) != 1 {
			t.Errorf("expected 1 object, got %d", len(objects))
		}
	})

	t.Run("get typed object", func(t *testing.T) {
		mapping := NewTypedObjectNodeMapping[TestObject](
			WithIDExtractor(func(o TestObject) string { return o.ID }),
		)

		obj := TestObject{ID: "test-1", Name: "Test", Description: "A test object"}
		mapping.AddTypedObject(obj)

		retrieved, err := mapping.GetTypedObject("test-1")
		if err != nil {
			t.Fatalf("GetTypedObject() error = %v", err)
		}

		if retrieved.Name != "Test" {
			t.Errorf("expected name 'Test', got %s", retrieved.Name)
		}
	})

	t.Run("add multiple typed objects", func(t *testing.T) {
		mapping := NewTypedObjectNodeMapping[TestObject](
			WithIDExtractor(func(o TestObject) string { return o.ID }),
		)

		obj1 := TestObject{ID: "1", Name: "First", Description: "First"}
		obj2 := TestObject{ID: "2", Name: "Second", Description: "Second"}

		err := mapping.AddTypedObjects(obj1, obj2)
		if err != nil {
			t.Fatalf("AddTypedObjects() error = %v", err)
		}

		objects := mapping.GetTypedObjects()
		if len(objects) != 2 {
			t.Errorf("expected 2 objects, got %d", len(objects))
		}
	})
}

func TestTypedObjectRetriever(t *testing.T) {
	t.Run("retrieve typed objects", func(t *testing.T) {
		mapping := NewTypedObjectNodeMapping[TestObject](
			WithTextExtractor(func(o TestObject) string { return o.Description }),
			WithIDExtractor(func(o TestObject) string { return o.ID }),
		)

		mapping.AddTypedObject(TestObject{ID: "1", Name: "Search", Description: "Search tool"})
		mapping.AddTypedObject(TestObject{ID: "2", Name: "Calculate", Description: "Calculator"})

		embedModel := NewMockEmbeddingModel()
		retriever := NewTypedObjectRetriever(mapping, embedModel, WithTypedRetrieverTopK[TestObject](1))

		err := retriever.BuildIndex(context.Background())
		if err != nil {
			t.Fatalf("BuildIndex() error = %v", err)
		}

		objects, err := retriever.RetrieveTypedObjects(context.Background(), "search")
		if err != nil {
			t.Fatalf("RetrieveTypedObjects() error = %v", err)
		}

		if len(objects) == 0 {
			t.Error("expected at least 1 object")
		}
	})
}

func TestObjectIndex(t *testing.T) {
	t.Run("create and use index", func(t *testing.T) {
		embedModel := NewMockEmbeddingModel()
		index := NewObjectIndex[TestObject](
			embedModel,
			WithTextExtractor(func(o TestObject) string { return o.Description }),
			WithIDExtractor(func(o TestObject) string { return o.ID }),
		)

		obj1 := TestObject{ID: "1", Name: "Search", Description: "Search tool"}
		obj2 := TestObject{ID: "2", Name: "Calculate", Description: "Calculator"}

		err := index.Add(context.Background(), obj1, obj2)
		if err != nil {
			t.Fatalf("Add() error = %v", err)
		}

		// Test retrieval
		results, err := index.Retrieve(context.Background(), "search")
		if err != nil {
			t.Fatalf("Retrieve() error = %v", err)
		}

		if len(results) == 0 {
			t.Error("expected at least 1 result")
		}

		// Test get by ID
		obj, err := index.Get("1")
		if err != nil {
			t.Fatalf("Get() error = %v", err)
		}

		if obj.Name != "Search" {
			t.Errorf("expected name 'Search', got %s", obj.Name)
		}

		// Test all
		all := index.All()
		if len(all) != 2 {
			t.Errorf("expected 2 objects, got %d", len(all))
		}
	})
}

func TestInterfaceCompliance(t *testing.T) {
	// Verify interface compliance at compile time
	var _ ObjectNodeMapping = (*BaseObjectNodeMapping)(nil)
	var _ ObjectNodeMapping = (*SimpleObjectNodeMapping)(nil)
	var _ ObjectNodeMapping = (*ToolNodeMapping)(nil)
	var _ ObjectRetriever = (*ToolRetriever)(nil)

	t.Run("interfaces implemented", func(t *testing.T) {
		// This test just verifies the compile-time checks above
	})
}
