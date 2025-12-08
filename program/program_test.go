package program

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/prompts"
)

// MockLLM implements llm.LLM for testing.
type MockLLM struct {
	CompleteResponse string
	ChatResponse     string
	Error            error
}

func (m *MockLLM) Complete(ctx context.Context, prompt string) (string, error) {
	if m.Error != nil {
		return "", m.Error
	}
	return m.CompleteResponse, nil
}

func (m *MockLLM) Chat(ctx context.Context, messages []llm.ChatMessage) (string, error) {
	if m.Error != nil {
		return "", m.Error
	}
	return m.ChatResponse, nil
}

func (m *MockLLM) Stream(ctx context.Context, prompt string) (<-chan string, error) {
	ch := make(chan string, 1)
	ch <- m.CompleteResponse
	close(ch)
	return ch, nil
}

// MockToolLLM implements llm.LLMWithToolCalling for testing.
type MockToolLLM struct {
	MockLLM
	ToolCallResponse llm.CompletionResponse
}

func (m *MockToolLLM) ChatWithTools(ctx context.Context, messages []llm.ChatMessage, tools []*llm.ToolMetadata, opts *llm.ChatCompletionOptions) (llm.CompletionResponse, error) {
	if m.Error != nil {
		return llm.CompletionResponse{}, m.Error
	}
	return m.ToolCallResponse, nil
}

func (m *MockToolLLM) SupportsToolCalling() bool {
	return true
}

// MockStructuredLLM implements llm.LLMWithStructuredOutput for testing.
type MockStructuredLLM struct {
	MockLLM
	StructuredResponse string
}

func (m *MockStructuredLLM) ChatWithFormat(ctx context.Context, messages []llm.ChatMessage, format *llm.ResponseFormat) (string, error) {
	if m.Error != nil {
		return "", m.Error
	}
	return m.StructuredResponse, nil
}

func (m *MockStructuredLLM) SupportsStructuredOutput() bool {
	return true
}

// Test types for parsing
type TestPerson struct {
	Name string `json:"name"`
	Age  int    `json:"age"`
}

type TestOutput struct {
	Result  string   `json:"result"`
	Items   []string `json:"items,omitempty"`
	Success bool     `json:"success"`
}

func TestJSONOutputParser(t *testing.T) {
	t.Run("parse simple JSON", func(t *testing.T) {
		parser := NewJSONOutputParser()
		output := `{"name": "John", "age": 30}`

		result, err := parser.Parse(output)
		if err != nil {
			t.Fatalf("Parse() error = %v", err)
		}

		m, ok := result.(map[string]interface{})
		if !ok {
			t.Fatalf("expected map, got %T", result)
		}

		if m["name"] != "John" {
			t.Errorf("expected name 'John', got %v", m["name"])
		}
	})

	t.Run("parse JSON with surrounding text", func(t *testing.T) {
		parser := NewJSONOutputParser()
		output := `Here is the result: {"value": 42} That's all.`

		result, err := parser.Parse(output)
		if err != nil {
			t.Fatalf("Parse() error = %v", err)
		}

		m, ok := result.(map[string]interface{})
		if !ok {
			t.Fatalf("expected map, got %T", result)
		}

		if m["value"] != float64(42) {
			t.Errorf("expected value 42, got %v", m["value"])
		}
	})

	t.Run("parse JSON array", func(t *testing.T) {
		parser := NewJSONOutputParser()
		output := `[1, 2, 3]`

		result, err := parser.Parse(output)
		if err != nil {
			t.Fatalf("Parse() error = %v", err)
		}

		arr, ok := result.([]interface{})
		if !ok {
			t.Fatalf("expected array, got %T", result)
		}

		if len(arr) != 3 {
			t.Errorf("expected 3 items, got %d", len(arr))
		}
	})

	t.Run("parse with target type", func(t *testing.T) {
		parser := NewJSONOutputParserWithType(TestPerson{})
		output := `{"name": "Alice", "age": 25}`

		result, err := parser.Parse(output)
		if err != nil {
			t.Fatalf("Parse() error = %v", err)
		}

		person, ok := result.(TestPerson)
		if !ok {
			t.Fatalf("expected TestPerson, got %T", result)
		}

		if person.Name != "Alice" {
			t.Errorf("expected name 'Alice', got %s", person.Name)
		}
		if person.Age != 25 {
			t.Errorf("expected age 25, got %d", person.Age)
		}
	})

	t.Run("format instructions", func(t *testing.T) {
		parser := NewJSONOutputParser()
		instructions := parser.GetFormatInstructions()

		if instructions == "" {
			t.Error("expected non-empty format instructions")
		}
	})

	t.Run("format instructions with schema", func(t *testing.T) {
		schema := map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"name": map[string]interface{}{"type": "string"},
			},
		}
		parser := NewJSONOutputParserWithSchema(schema)
		instructions := parser.GetFormatInstructions()

		if instructions == "" {
			t.Error("expected non-empty format instructions")
		}
	})
}

func TestPydanticOutputParser(t *testing.T) {
	t.Run("parse into struct", func(t *testing.T) {
		parser := NewPydanticOutputParser(TestPerson{})
		output := `{"name": "Bob", "age": 35}`

		result, err := parser.Parse(output)
		if err != nil {
			t.Fatalf("Parse() error = %v", err)
		}

		person, ok := result.(TestPerson)
		if !ok {
			t.Fatalf("expected TestPerson, got %T", result)
		}

		if person.Name != "Bob" {
			t.Errorf("expected name 'Bob', got %s", person.Name)
		}
	})

	t.Run("format instructions include type name", func(t *testing.T) {
		parser := NewPydanticOutputParser(TestPerson{})
		instructions := parser.GetFormatInstructions()

		if instructions == "" {
			t.Error("expected non-empty format instructions")
		}
	})
}

func TestProgramOutput(t *testing.T) {
	t.Run("create output", func(t *testing.T) {
		output := NewProgramOutput("raw text", map[string]interface{}{"key": "value"})

		if output.RawOutput != "raw text" {
			t.Errorf("expected raw output 'raw text', got %s", output.RawOutput)
		}
		if output.String() != "raw text" {
			t.Errorf("expected String() to return raw output")
		}
	})

	t.Run("get parsed as type", func(t *testing.T) {
		parsed := map[string]interface{}{"name": "Test", "age": float64(30)}
		output := NewProgramOutput("", parsed)

		var person TestPerson
		err := output.GetParsedAs(&person)
		if err != nil {
			t.Fatalf("GetParsedAs() error = %v", err)
		}

		if person.Name != "Test" {
			t.Errorf("expected name 'Test', got %s", person.Name)
		}
	})
}

func TestLLMProgram(t *testing.T) {
	t.Run("call with completion", func(t *testing.T) {
		mockLLM := &MockLLM{
			CompleteResponse: `{"result": "success"}`,
		}

		program := NewLLMProgram(mockLLM)
		output, err := program.Call(context.Background(), map[string]interface{}{
			"input": "test query",
		})

		if err != nil {
			t.Fatalf("Call() error = %v", err)
		}

		if output.RawOutput != `{"result": "success"}` {
			t.Errorf("unexpected raw output: %s", output.RawOutput)
		}
	})

	t.Run("call with prompt template", func(t *testing.T) {
		mockLLM := &MockLLM{
			CompleteResponse: `{"answer": "42"}`,
		}

		prompt := prompts.NewPromptTemplate("Question: {question}", prompts.PromptTypeCustom)
		program := NewLLMProgram(mockLLM).WithPrompt(prompt)

		output, err := program.Call(context.Background(), map[string]interface{}{
			"question": "What is the meaning of life?",
		})

		if err != nil {
			t.Fatalf("Call() error = %v", err)
		}

		if output.RawOutput == "" {
			t.Error("expected non-empty output")
		}
	})

	t.Run("call with output parser", func(t *testing.T) {
		mockLLM := &MockLLM{
			CompleteResponse: `{"name": "Test", "age": 25}`,
		}

		parser := NewPydanticOutputParser(TestPerson{})
		program := NewLLMProgram(mockLLM).WithOutputParser(parser)

		output, err := program.Call(context.Background(), map[string]interface{}{
			"input": "Generate a person",
		})

		if err != nil {
			t.Fatalf("Call() error = %v", err)
		}

		if output.ParsedOutput == nil {
			t.Error("expected parsed output")
		}

		person, ok := output.ParsedOutput.(TestPerson)
		if !ok {
			t.Fatalf("expected TestPerson, got %T", output.ParsedOutput)
		}

		if person.Name != "Test" {
			t.Errorf("expected name 'Test', got %s", person.Name)
		}
	})

	t.Run("call with structured output LLM", func(t *testing.T) {
		mockLLM := &MockStructuredLLM{
			StructuredResponse: `{"result": "structured"}`,
		}

		program := NewLLMProgram(mockLLM)
		output, err := program.Call(context.Background(), map[string]interface{}{
			"input": "test",
		})

		if err != nil {
			t.Fatalf("Call() error = %v", err)
		}

		if output.RawOutput != `{"result": "structured"}` {
			t.Errorf("unexpected output: %s", output.RawOutput)
		}
	})

	t.Run("error without LLM", func(t *testing.T) {
		program := NewLLMProgram(nil)
		_, err := program.Call(context.Background(), map[string]interface{}{})

		if err == nil {
			t.Error("expected error without LLM")
		}
	})
}

func TestFunctionProgram(t *testing.T) {
	t.Run("create with options", func(t *testing.T) {
		mockLLM := &MockToolLLM{}
		program := NewFunctionProgram(mockLLM,
			WithFunctionName("test_func"),
			WithFunctionDescription("A test function"),
			WithFunctionParameters(map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"input": map[string]interface{}{"type": "string"},
				},
			}),
		)

		if program.FunctionName != "test_func" {
			t.Errorf("expected function name 'test_func', got %s", program.FunctionName)
		}
		if program.FunctionDescription != "A test function" {
			t.Errorf("expected description 'A test function', got %s", program.FunctionDescription)
		}
	})

	t.Run("create from schema", func(t *testing.T) {
		mockLLM := &MockToolLLM{}
		schema := map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"name": map[string]interface{}{"type": "string"},
			},
		}

		program := NewFunctionProgramFromSchema(mockLLM, "get_name", "Get a name", schema)

		if program.FunctionName != "get_name" {
			t.Errorf("expected function name 'get_name', got %s", program.FunctionName)
		}
	})

	t.Run("create from type", func(t *testing.T) {
		mockLLM := &MockToolLLM{}
		program := NewFunctionProgramFromType(mockLLM, "get_person", "Get person info", TestPerson{})

		if program.FunctionName != "get_person" {
			t.Errorf("expected function name 'get_person', got %s", program.FunctionName)
		}
		if program.Parameters == nil {
			t.Error("expected parameters to be set")
		}
	})

	t.Run("call with tool response", func(t *testing.T) {
		toolCall := llm.NewToolCall("call_1", "output", `{"name": "John", "age": 30}`)
		mockLLM := &MockToolLLM{
			ToolCallResponse: llm.CompletionResponse{
				Text: "",
				Message: &llm.ChatMessage{
					Role: llm.MessageRoleAssistant,
					Blocks: []llm.ContentBlock{
						llm.NewToolCallBlock(toolCall),
					},
				},
			},
		}

		program := NewFunctionProgram(mockLLM)
		output, err := program.Call(context.Background(), map[string]interface{}{
			"input": "Get person info",
		})

		if err != nil {
			t.Fatalf("Call() error = %v", err)
		}

		if output.ParsedOutput == nil {
			t.Error("expected parsed output")
		}

		parsed, ok := output.ParsedOutput.(map[string]interface{})
		if !ok {
			t.Fatalf("expected map, got %T", output.ParsedOutput)
		}

		if parsed["name"] != "John" {
			t.Errorf("expected name 'John', got %v", parsed["name"])
		}
	})

	t.Run("error without tool calling support", func(t *testing.T) {
		mockLLM := &MockLLM{} // Does not implement LLMWithToolCalling
		program := NewFunctionProgram(mockLLM)

		_, err := program.Call(context.Background(), map[string]interface{}{})
		if err == nil {
			t.Error("expected error without tool calling support")
		}
	})
}

func TestStructToSchema(t *testing.T) {
	t.Run("simple struct", func(t *testing.T) {
		type Simple struct {
			Name string `json:"name"`
			Age  int    `json:"age"`
		}

		parser := NewPydanticOutputParser(Simple{})
		schema := structToSchema(parser.TargetType)

		if schema["type"] != "object" {
			t.Errorf("expected type 'object', got %v", schema["type"])
		}

		props, ok := schema["properties"].(map[string]interface{})
		if !ok {
			t.Fatal("expected properties map")
		}

		if _, exists := props["name"]; !exists {
			t.Error("expected 'name' property")
		}
		if _, exists := props["age"]; !exists {
			t.Error("expected 'age' property")
		}
	})

	t.Run("nested struct", func(t *testing.T) {
		type Address struct {
			City string `json:"city"`
		}
		type Person struct {
			Name    string  `json:"name"`
			Address Address `json:"address"`
		}

		parser := NewPydanticOutputParser(Person{})
		schema := structToSchema(parser.TargetType)

		props := schema["properties"].(map[string]interface{})
		addrSchema, ok := props["address"].(map[string]interface{})
		if !ok {
			t.Fatal("expected address schema")
		}

		if addrSchema["type"] != "object" {
			t.Errorf("expected nested type 'object', got %v", addrSchema["type"])
		}
	})

	t.Run("slice field", func(t *testing.T) {
		type WithSlice struct {
			Items []string `json:"items"`
		}

		parser := NewPydanticOutputParser(WithSlice{})
		schema := structToSchema(parser.TargetType)

		props := schema["properties"].(map[string]interface{})
		itemsSchema, ok := props["items"].(map[string]interface{})
		if !ok {
			t.Fatal("expected items schema")
		}

		if itemsSchema["type"] != "array" {
			t.Errorf("expected type 'array', got %v", itemsSchema["type"])
		}
	})
}

func TestExtractJSON(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "simple object",
			input:    `{"key": "value"}`,
			expected: `{"key": "value"}`,
		},
		{
			name:     "object with surrounding text",
			input:    `Here is the JSON: {"key": "value"} That's it.`,
			expected: `{"key": "value"}`,
		},
		{
			name:     "nested object",
			input:    `{"outer": {"inner": "value"}}`,
			expected: `{"outer": {"inner": "value"}}`,
		},
		{
			name:     "array",
			input:    `[1, 2, 3]`,
			expected: `[1, 2, 3]`,
		},
		{
			name:     "array with surrounding text",
			input:    `The array is: [1, 2, 3] done`,
			expected: `[1, 2, 3]`,
		},
		{
			name:     "no JSON",
			input:    `Just plain text`,
			expected: "",
		},
		{
			name:     "string with braces",
			input:    `{"text": "hello {world}"}`,
			expected: `{"text": "hello {world}"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractJSON(tt.input)
			if result != tt.expected {
				t.Errorf("extractJSON(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

func TestMultiOutputProgram(t *testing.T) {
	t.Run("create and call", func(t *testing.T) {
		mockLLM := &MockLLM{
			CompleteResponse: `[{"name": "A"}, {"name": "B"}]`,
		}

		parser := NewJSONOutputParser()
		program := NewMultiOutputProgram(mockLLM, 2, parser)

		output, err := program.Call(context.Background(), map[string]interface{}{
			"input": "Generate names",
		})

		if err != nil {
			t.Fatalf("Call() error = %v", err)
		}

		if output.Metadata["num_outputs"] != 2 {
			t.Errorf("expected num_outputs 2, got %v", output.Metadata["num_outputs"])
		}
	})
}

func TestInterfaceCompliance(t *testing.T) {
	// Verify interface compliance at compile time
	var _ Program = (*LLMProgram)(nil)
	var _ Program = (*FunctionProgram)(nil)
	var _ Program = (*MultiOutputProgram)(nil)
	var _ ProgramWithPrompt = (*LLMProgram)(nil)
	var _ ProgramWithPrompt = (*FunctionProgram)(nil)
	var _ OutputParser = (*JSONOutputParser)(nil)
	var _ OutputParser = (*PydanticOutputParser)(nil)

	t.Run("interfaces implemented", func(t *testing.T) {
		// This test just verifies the compile-time checks above
	})
}

// Helper to verify JSON is valid
func isValidJSON(s string) bool {
	var js json.RawMessage
	return json.Unmarshal([]byte(s), &js) == nil
}
