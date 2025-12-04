# Program Examples

This directory contains examples demonstrating structured output programs in `go-llamaindex`.

## Examples

### 1. Function Program (`function_program/`)

Demonstrates structured output using LLM function calling.

**Features:**
- Schema-based function definitions
- Go type to schema conversion
- Custom prompt templates
- Typed output parsing
- Nested schema support

**Run:**
```bash
cd function_program && go run main.go
```

### 2. LLM Program (`llm_program/`)

Demonstrates structured output using text generation with parsing.

**Features:**
- JSON output parsing
- Pydantic-style type parsing
- Automatic format instructions
- Multi-output generation
- Custom output parsers

**Run:**
```bash
cd llm_program && go run main.go
```

## Key Concepts

### Program Interface

```go
type Program interface {
    Call(ctx context.Context, args map[string]interface{}) (*ProgramOutput, error)
}
```

### FunctionProgram

Uses LLM function calling for guaranteed structured output.

```go
// From JSON schema
prog := program.NewFunctionProgramFromSchema(
    llm,
    "extract_info",
    "Extract information from text",
    schema,
)

// From Go type
prog := program.NewFunctionProgramFromType(
    llm,
    "extract_person",
    "Extract person information",
    Person{},
)

// Call the program
output, err := prog.Call(ctx, map[string]interface{}{
    "input": "John is 30 years old.",
})
```

### LLMProgram

Uses text generation with output parsing.

```go
// With JSON parser
prog := program.NewLLMProgramWithParser(llm, program.NewJSONOutputParser())

// For specific type
prog := program.NewLLMProgramForType(llm, Recipe{})

// With custom prompt
prog.WithPrompt(prompts.NewPromptTemplate(
    "Generate a recipe for: {request}",
    prompts.PromptTypeCustom,
))

output, err := prog.Call(ctx, map[string]interface{}{
    "request": "chocolate cake",
})
```

### Output Parsers

| Parser | Description |
|--------|-------------|
| `JSONOutputParser` | Generic JSON parsing |
| `JSONOutputParserWithSchema` | Schema-validated JSON |
| `PydanticOutputParser` | Go struct-based parsing |

```go
// Generic JSON
parser := program.NewJSONOutputParser()

// With schema
parser := program.NewJSONOutputParserWithSchema(schema)

// From Go type
parser := program.NewPydanticOutputParser(MyStruct{})

// Get format instructions
instructions := parser.GetFormatInstructions()
```

### ProgramOutput

```go
output, err := prog.Call(ctx, args)

// Raw string output
raw := output.RawOutput

// Parsed output (interface{})
parsed := output.ParsedOutput

// Type-safe parsing
var result MyStruct
err := output.GetParsedAs(&result)

// Metadata
meta := output.Metadata
```

### Go Struct to Schema

```go
type Person struct {
    Name string `json:"name" description:"Person's name"`
    Age  int    `json:"age" description:"Age in years"`
}

// Automatically generates JSON schema:
// {
//   "type": "object",
//   "properties": {
//     "name": {"type": "string", "description": "Person's name"},
//     "age": {"type": "integer", "description": "Age in years"}
//   },
//   "required": ["name", "age"]
// }
```

### MultiOutputProgram

```go
prog := program.NewMultiOutputProgram(llm, 3, parser)
prog.Prompt = prompts.NewPromptTemplate(
    "Generate {num} ideas for: {topic}",
    prompts.PromptTypeCustom,
)

output, err := prog.Call(ctx, map[string]interface{}{
    "topic": "learning Go",
})
// Returns array of outputs
```

## Comparison

| Feature | FunctionProgram | LLMProgram |
|---------|-----------------|------------|
| Requires function calling | Yes | No |
| Output reliability | Higher | Depends on parsing |
| LLM compatibility | Limited | Universal |
| Schema enforcement | Native | Via instructions |

## Environment Variables

All examples require:
- `OPENAI_API_KEY` - OpenAI API key for LLM operations
