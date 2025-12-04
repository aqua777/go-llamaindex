# Tool Examples

This directory contains examples demonstrating tool abstractions in `go-llamaindex`.

## Examples

### 1. Function Tool (`function_tool/`)

Demonstrates wrapping Go functions as tools for LLM agents.

**Features:**
- Wrap any Go function as a tool
- Automatic parameter schema generation
- Context support for cancellation/timeout
- Error handling with IsError flag
- Struct-based inputs and outputs
- ToolSpec for declarative creation

**Run:**
```bash
cd function_tool && go run main.go
```

### 2. Query Engine Tool (`query_engine_tool/`)

Demonstrates wrapping query engines as tools for LLM agents.

**Features:**
- Wrap QueryEngine as a tool
- Configurable name and description
- ReturnDirect option
- Multiple knowledge base tools
- OpenAI tool format conversion

**Run:**
```bash
cd query_engine_tool && go run main.go
```

## Key Concepts

### Tool Interface

```go
type Tool interface {
    Metadata() *ToolMetadata
    Call(ctx context.Context, input interface{}) (*ToolOutput, error)
}
```

### ToolMetadata

```go
type ToolMetadata struct {
    Name         string
    Description  string
    Parameters   map[string]interface{}  // JSON Schema
    ReturnDirect bool
}

// Convert to OpenAI format
openAITool := metadata.ToOpenAITool()
```

### FunctionTool

```go
// From function with options
tool, err := tools.NewFunctionTool(myFunc,
    tools.WithFunctionToolName("my_tool"),
    tools.WithFunctionToolDescription("Does something useful"),
)

// From function with explicit config
tool, err := tools.NewFunctionToolFromDefaults(
    myFunc,
    "my_tool",
    "Does something useful",
)

// Call the tool
output, err := tool.Call(ctx, map[string]interface{}{
    "arg0": "value1",
    "arg1": 42,
})
```

### QueryEngineTool

```go
// Create from query engine
tool := tools.NewQueryEngineTool(queryEngine,
    tools.WithQueryEngineToolName("my_kb"),
    tools.WithQueryEngineToolDescription("Query my knowledge base"),
    tools.WithQueryEngineToolReturnDirect(true),
)

// Or with defaults
tool := tools.NewQueryEngineToolFromDefaults(
    queryEngine,
    "my_kb",
    "Query my knowledge base",
)

// Call the tool
output, err := tool.Call(ctx, "What is X?")
// or
output, err := tool.Call(ctx, map[string]interface{}{"input": "What is X?"})
```

### ToolSpec (Declarative)

```go
spec := &tools.ToolSpec{
    Name:        "calculate",
    Description: "Perform calculation",
    Parameters: map[string]interface{}{
        "type": "object",
        "properties": map[string]interface{}{
            "expression": map[string]interface{}{
                "type": "string",
                "description": "Math expression",
            },
        },
        "required": []string{"expression"},
    },
    Handler: func(ctx context.Context, input map[string]interface{}) (interface{}, error) {
        expr := input["expression"].(string)
        // ... evaluate expression
        return result, nil
    },
}

tool := spec.ToTool()
```

### ToolOutput

```go
type ToolOutput struct {
    Content   string                 // Text content
    ToolName  string                 // Name of the tool
    RawInput  map[string]interface{} // Original input
    RawOutput interface{}            // Raw result
    IsError   bool                   // Error flag
    Error     error                  // Error if IsError
}
```

### Function Signatures

FunctionTool supports various function signatures:

```go
// Simple function
func add(a, b float64) float64

// With error return
func divide(a, b float64) (float64, error)

// With context
func fetch(ctx context.Context, url string) (string, error)

// With struct input
func search(params SearchParams) ([]Result, error)
```

### Automatic Schema Generation

Go types are automatically converted to JSON Schema:

| Go Type | JSON Schema Type |
|---------|------------------|
| `string` | `string` |
| `int`, `int64`, etc. | `integer` |
| `float64`, `float32` | `number` |
| `bool` | `boolean` |
| `[]T` | `array` |
| `map[string]T` | `object` |
| `struct` | `object` with properties |

## Environment Variables

Query engine tool examples may require:
- `OPENAI_API_KEY` - For real query engine implementations
