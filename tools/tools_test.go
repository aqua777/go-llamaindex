package tools

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/aqua777/go-llamaindex/rag/synthesizer"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestToolMetadata tests the ToolMetadata struct.
func TestToolMetadata(t *testing.T) {
	t.Run("NewToolMetadata", func(t *testing.T) {
		meta := NewToolMetadata("test_tool", "A test tool")

		assert.Equal(t, "test_tool", meta.Name)
		assert.Equal(t, "A test tool", meta.Description)
		assert.NotNil(t, meta.Parameters)
	})

	t.Run("GetName", func(t *testing.T) {
		meta := NewToolMetadata("my_tool", "Description")
		assert.Equal(t, "my_tool", meta.GetName())
	})

	t.Run("GetParametersDict", func(t *testing.T) {
		meta := NewToolMetadata("tool", "desc")
		params := meta.GetParametersDict()

		assert.Equal(t, "object", params["type"])
		props := params["properties"].(map[string]interface{})
		assert.Contains(t, props, "input")
	})

	t.Run("GetParametersJSON", func(t *testing.T) {
		meta := NewToolMetadata("tool", "desc")
		jsonStr, err := meta.GetParametersJSON()

		require.NoError(t, err)
		assert.Contains(t, jsonStr, "input")
		assert.Contains(t, jsonStr, "string")
	})

	t.Run("ToOpenAITool", func(t *testing.T) {
		meta := NewToolMetadata("search", "Search the web")
		openAITool := meta.ToOpenAITool()

		assert.Equal(t, "function", openAITool["type"])
		fn := openAITool["function"].(map[string]interface{})
		assert.Equal(t, "search", fn["name"])
		assert.Equal(t, "Search the web", fn["description"])
		assert.NotNil(t, fn["parameters"])
	})

	t.Run("ToOpenAIFunction", func(t *testing.T) {
		meta := NewToolMetadata("calc", "Calculate")
		openAIFn := meta.ToOpenAIFunction()

		assert.Equal(t, "calc", openAIFn["name"])
		assert.Equal(t, "Calculate", openAIFn["description"])
	})
}

// TestToolOutput tests the ToolOutput struct.
func TestToolOutput(t *testing.T) {
	t.Run("NewToolOutput", func(t *testing.T) {
		output := NewToolOutput("test_tool", "Hello, World!")

		assert.Equal(t, "test_tool", output.ToolName)
		assert.Equal(t, "Hello, World!", output.Content)
		assert.False(t, output.IsError)
	})

	t.Run("NewToolOutputWithInput", func(t *testing.T) {
		rawInput := map[string]interface{}{"query": "test"}
		output := NewToolOutputWithInput("tool", "result", rawInput, "raw_result")

		assert.Equal(t, "tool", output.ToolName)
		assert.Equal(t, "result", output.Content)
		assert.Equal(t, rawInput, output.RawInput)
		assert.Equal(t, "raw_result", output.RawOutput)
	})

	t.Run("NewErrorToolOutput", func(t *testing.T) {
		err := errors.New("something went wrong")
		output := NewErrorToolOutput("error_tool", err)

		assert.Equal(t, "error_tool", output.ToolName)
		assert.Equal(t, "something went wrong", output.Content)
		assert.True(t, output.IsError)
		assert.Equal(t, err, output.Error)
	})

	t.Run("String", func(t *testing.T) {
		output := NewToolOutput("tool", "content here")
		assert.Equal(t, "content here", output.String())
	})
}

// TestToolSpec tests the ToolSpec struct.
func TestToolSpec(t *testing.T) {
	t.Run("ToTool", func(t *testing.T) {
		spec := &ToolSpec{
			Name:        "add",
			Description: "Add two numbers",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"a": map[string]interface{}{"type": "number"},
					"b": map[string]interface{}{"type": "number"},
				},
				"required": []string{"a", "b"},
			},
			Handler: func(ctx context.Context, input map[string]interface{}) (interface{}, error) {
				a := input["a"].(float64)
				b := input["b"].(float64)
				return a + b, nil
			},
		}

		tool := spec.ToTool()
		assert.Equal(t, "add", tool.Metadata().Name)
		assert.Equal(t, "Add two numbers", tool.Metadata().Description)

		// Test calling the tool
		ctx := context.Background()
		output, err := tool.Call(ctx, map[string]interface{}{"a": 2.0, "b": 3.0})
		require.NoError(t, err)
		assert.Equal(t, "5", output.Content)
	})

	t.Run("ToTool with string input", func(t *testing.T) {
		spec := &ToolSpec{
			Name:        "echo",
			Description: "Echo the input",
			Handler: func(ctx context.Context, input map[string]interface{}) (interface{}, error) {
				return input["input"], nil
			},
		}

		tool := spec.ToTool()
		ctx := context.Background()
		output, err := tool.Call(ctx, "hello")
		require.NoError(t, err)
		assert.Equal(t, "hello", output.Content)
	})
}

// TestFunctionTool tests the FunctionTool.
func TestFunctionTool(t *testing.T) {
	t.Run("NewFunctionTool with simple function", func(t *testing.T) {
		fn := func(input string) (string, error) {
			return "Hello, " + input, nil
		}

		tool, err := NewFunctionTool(fn,
			WithFunctionToolName("greet"),
			WithFunctionToolDescription("Greet someone"),
		)
		require.NoError(t, err)

		assert.Equal(t, "greet", tool.Metadata().Name)
		assert.Equal(t, "Greet someone", tool.Metadata().Description)

		ctx := context.Background()
		output, err := tool.Call(ctx, "World")
		require.NoError(t, err)
		assert.Equal(t, "Hello, World", output.Content)
	})

	t.Run("NewFunctionTool with context", func(t *testing.T) {
		fn := func(ctx context.Context, query string) (string, error) {
			return "Result for: " + query, nil
		}

		tool, err := NewFunctionTool(fn,
			WithFunctionToolName("search"),
			WithFunctionToolDescription("Search for something"),
		)
		require.NoError(t, err)

		ctx := context.Background()
		output, err := tool.Call(ctx, "test query")
		require.NoError(t, err)
		assert.Equal(t, "Result for: test query", output.Content)
	})

	t.Run("NewFunctionTool with map input", func(t *testing.T) {
		fn := func(a int, b int) (int, error) {
			return a + b, nil
		}

		tool, err := NewFunctionTool(fn,
			WithFunctionToolName("add"),
			WithFunctionToolDescription("Add two numbers"),
		)
		require.NoError(t, err)

		ctx := context.Background()
		output, err := tool.Call(ctx, map[string]interface{}{"arg0": 5, "arg1": 3})
		require.NoError(t, err)
		assert.Equal(t, "8", output.Content)
	})

	t.Run("NewFunctionTool with error", func(t *testing.T) {
		fn := func(input string) (string, error) {
			return "", errors.New("intentional error")
		}

		tool, err := NewFunctionTool(fn,
			WithFunctionToolName("failing"),
			WithFunctionToolDescription("A failing tool"),
		)
		require.NoError(t, err)

		ctx := context.Background()
		output, err := tool.Call(ctx, "test")
		assert.Error(t, err)
		assert.True(t, output.IsError)
		assert.Contains(t, output.Content, "intentional error")
	})

	t.Run("NewFunctionToolFromDefaults", func(t *testing.T) {
		fn := func(x float64) (float64, error) {
			return x * 2, nil
		}

		tool, err := NewFunctionToolFromDefaults(fn, "double", "Double a number")
		require.NoError(t, err)

		assert.Equal(t, "double", tool.Metadata().Name)
		assert.Equal(t, "Double a number", tool.Metadata().Description)
	})

	t.Run("Invalid function type", func(t *testing.T) {
		_, err := NewFunctionTool("not a function")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "must be a function")
	})
}

// MockQueryEngine is a mock query engine for testing.
type MockQueryEngine struct {
	response string
	err      error
}

func (m *MockQueryEngine) Query(ctx context.Context, query string) (*QueryResponse, error) {
	if m.err != nil {
		return nil, m.err
	}
	return &QueryResponse{Response: m.response}, nil
}

// QueryResponse represents a query response.
type QueryResponse struct {
	Response string
}

// TestQueryEngineTool tests the QueryEngineTool.
func TestQueryEngineTool(t *testing.T) {
	t.Run("NewQueryEngineTool", func(t *testing.T) {
		mockQE := &mockQueryEngineImpl{response: "test response"}
		tool := NewQueryEngineTool(mockQE)

		assert.Equal(t, DefaultQueryEngineToolName, tool.Metadata().Name)
		assert.Equal(t, DefaultQueryEngineToolDescription, tool.Metadata().Description)
	})

	t.Run("NewQueryEngineToolFromDefaults", func(t *testing.T) {
		mockQE := &mockQueryEngineImpl{response: "test response"}
		tool := NewQueryEngineToolFromDefaults(mockQE, "custom_qe", "Custom query engine")

		assert.Equal(t, "custom_qe", tool.Metadata().Name)
		assert.Equal(t, "Custom query engine", tool.Metadata().Description)
	})

	t.Run("Call with string input", func(t *testing.T) {
		mockQE := &mockQueryEngineImpl{response: "The answer is 42"}
		tool := NewQueryEngineTool(mockQE)

		ctx := context.Background()
		output, err := tool.Call(ctx, "What is the answer?")
		require.NoError(t, err)
		assert.Equal(t, "The answer is 42", output.Content)
		assert.Equal(t, "What is the answer?", output.RawInput["input"])
	})

	t.Run("Call with map input", func(t *testing.T) {
		mockQE := &mockQueryEngineImpl{response: "Response"}
		tool := NewQueryEngineTool(mockQE)

		ctx := context.Background()
		output, err := tool.Call(ctx, map[string]interface{}{"input": "test query"})
		require.NoError(t, err)
		assert.Equal(t, "Response", output.Content)
	})

	t.Run("Call with error", func(t *testing.T) {
		mockQE := &mockQueryEngineImpl{err: errors.New("query failed")}
		tool := NewQueryEngineTool(mockQE)

		ctx := context.Background()
		output, err := tool.Call(ctx, "test")
		assert.Error(t, err)
		assert.True(t, output.IsError)
	})
}

// mockQueryEngineImpl implements queryengine.QueryEngine for testing.
type mockQueryEngineImpl struct {
	response string
	err      error
}

func (m *mockQueryEngineImpl) Query(ctx context.Context, query string) (*synthesizer.Response, error) {
	if m.err != nil {
		return nil, m.err
	}
	return &synthesizer.Response{Response: m.response}, nil
}

// TestRetrieverTool tests the RetrieverTool.
func TestRetrieverTool(t *testing.T) {
	t.Run("NewRetrieverTool", func(t *testing.T) {
		mockRet := &mockRetrieverImpl{}
		tool := NewRetrieverTool(mockRet)

		assert.Equal(t, DefaultRetrieverToolName, tool.Metadata().Name)
		assert.Equal(t, DefaultRetrieverToolDescription, tool.Metadata().Description)
	})

	t.Run("NewRetrieverToolFromDefaults", func(t *testing.T) {
		mockRet := &mockRetrieverImpl{}
		tool := NewRetrieverToolFromDefaults(mockRet, "custom_ret", "Custom retriever")

		assert.Equal(t, "custom_ret", tool.Metadata().Name)
		assert.Equal(t, "Custom retriever", tool.Metadata().Description)
	})

	t.Run("Call with string input", func(t *testing.T) {
		mockRet := &mockRetrieverImpl{
			nodes: []schema.NodeWithScore{
				{Node: schema.Node{Text: "Document 1"}, Score: 0.9},
				{Node: schema.Node{Text: "Document 2"}, Score: 0.8},
			},
		}
		tool := NewRetrieverTool(mockRet)

		ctx := context.Background()
		output, err := tool.Call(ctx, "search query")
		require.NoError(t, err)
		assert.Contains(t, output.Content, "Document 1")
		assert.Contains(t, output.Content, "Document 2")
	})

	t.Run("Call with map input", func(t *testing.T) {
		mockRet := &mockRetrieverImpl{
			nodes: []schema.NodeWithScore{
				{Node: schema.Node{Text: "Result"}, Score: 1.0},
			},
		}
		tool := NewRetrieverTool(mockRet)

		ctx := context.Background()
		output, err := tool.Call(ctx, map[string]interface{}{"input": "test"})
		require.NoError(t, err)
		assert.Contains(t, output.Content, "Result")
	})

	t.Run("Call with error", func(t *testing.T) {
		mockRet := &mockRetrieverImpl{err: errors.New("retrieval failed")}
		tool := NewRetrieverTool(mockRet)

		ctx := context.Background()
		output, err := tool.Call(ctx, "test")
		assert.Error(t, err)
		assert.True(t, output.IsError)
	})
}

// mockRetrieverImpl implements retriever.Retriever for testing.
type mockRetrieverImpl struct {
	nodes []schema.NodeWithScore
	err   error
}

func (m *mockRetrieverImpl) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.nodes, nil
}

// TestTypeToJSONSchema tests the typeToJSONSchema function.
func TestTypeToJSONSchema(t *testing.T) {
	t.Run("String type", func(t *testing.T) {
		schema := typeToJSONSchema(reflect.TypeOf(""))
		assert.Equal(t, "string", schema["type"])
	})

	t.Run("Int type", func(t *testing.T) {
		schema := typeToJSONSchema(reflect.TypeOf(0))
		assert.Equal(t, "integer", schema["type"])
	})

	t.Run("Float type", func(t *testing.T) {
		schema := typeToJSONSchema(reflect.TypeOf(0.0))
		assert.Equal(t, "number", schema["type"])
	})

	t.Run("Bool type", func(t *testing.T) {
		schema := typeToJSONSchema(reflect.TypeOf(false))
		assert.Equal(t, "boolean", schema["type"])
	})
}

// TestToolInterface tests that all tools implement the Tool interface.
func TestToolInterface(t *testing.T) {
	t.Run("FunctionTool implements Tool", func(t *testing.T) {
		fn := func(s string) (string, error) { return s, nil }
		tool, err := NewFunctionTool(fn)
		require.NoError(t, err)
		var _ Tool = tool
	})

	t.Run("QueryEngineTool implements Tool", func(t *testing.T) {
		mockQE := &mockQueryEngineImpl{}
		tool := NewQueryEngineTool(mockQE)
		var _ Tool = tool
	})

	t.Run("RetrieverTool implements Tool", func(t *testing.T) {
		mockRet := &mockRetrieverImpl{}
		tool := NewRetrieverTool(mockRet)
		var _ Tool = tool
	})

	t.Run("ToolSpec.ToTool implements Tool", func(t *testing.T) {
		spec := &ToolSpec{
			Name:        "test",
			Description: "test",
			Handler: func(ctx context.Context, input map[string]interface{}) (interface{}, error) {
				return nil, nil
			},
		}
		var _ Tool = spec.ToTool()
	})
}
