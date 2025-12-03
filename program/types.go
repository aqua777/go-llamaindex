// Package program provides structured LLM output programs for go-llamaindex.
package program

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/prompts"
)

// Program is the interface for structured LLM output programs.
// Programs take a prompt and return structured output.
type Program interface {
	// Call executes the program with the given arguments.
	Call(ctx context.Context, args map[string]interface{}) (*ProgramOutput, error)
}

// ProgramWithPrompt extends Program with prompt access.
type ProgramWithPrompt interface {
	Program
	// GetPrompt returns the program's prompt template.
	GetPrompt() *prompts.PromptTemplate
	// SetPrompt sets the program's prompt template.
	SetPrompt(prompt *prompts.PromptTemplate)
}

// ProgramOutput represents the output of a program execution.
type ProgramOutput struct {
	// RawOutput is the raw string output from the LLM.
	RawOutput string `json:"raw_output"`
	// ParsedOutput is the parsed structured output.
	ParsedOutput interface{} `json:"parsed_output,omitempty"`
	// Metadata contains additional information about the execution.
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// NewProgramOutput creates a new ProgramOutput.
func NewProgramOutput(rawOutput string, parsedOutput interface{}) *ProgramOutput {
	return &ProgramOutput{
		RawOutput:    rawOutput,
		ParsedOutput: parsedOutput,
		Metadata:     make(map[string]interface{}),
	}
}

// String returns the raw output as a string.
func (o *ProgramOutput) String() string {
	return o.RawOutput
}

// GetParsedAs attempts to unmarshal the parsed output into the given type.
func (o *ProgramOutput) GetParsedAs(target interface{}) error {
	if o.ParsedOutput == nil {
		return fmt.Errorf("no parsed output available")
	}

	// If already the right type, just assign
	targetVal := reflect.ValueOf(target)
	if targetVal.Kind() != reflect.Ptr {
		return fmt.Errorf("target must be a pointer")
	}

	parsedVal := reflect.ValueOf(o.ParsedOutput)
	if parsedVal.Type().AssignableTo(targetVal.Elem().Type()) {
		targetVal.Elem().Set(parsedVal)
		return nil
	}

	// Try JSON marshal/unmarshal
	data, err := json.Marshal(o.ParsedOutput)
	if err != nil {
		return fmt.Errorf("failed to marshal parsed output: %w", err)
	}

	if err := json.Unmarshal(data, target); err != nil {
		return fmt.Errorf("failed to unmarshal to target type: %w", err)
	}

	return nil
}

// OutputParser is the interface for parsing LLM output into structured data.
type OutputParser interface {
	// Parse parses the raw output into structured data.
	Parse(output string) (interface{}, error)
	// GetFormatInstructions returns instructions for the LLM on output format.
	GetFormatInstructions() string
}

// JSONOutputParser parses JSON output from LLM.
type JSONOutputParser struct {
	// TargetType is the type to unmarshal into (optional).
	TargetType reflect.Type
	// Schema is the JSON schema for the expected output (optional).
	Schema map[string]interface{}
}

// NewJSONOutputParser creates a new JSONOutputParser.
func NewJSONOutputParser() *JSONOutputParser {
	return &JSONOutputParser{}
}

// NewJSONOutputParserWithType creates a parser for a specific type.
func NewJSONOutputParserWithType(targetType interface{}) *JSONOutputParser {
	return &JSONOutputParser{
		TargetType: reflect.TypeOf(targetType),
	}
}

// NewJSONOutputParserWithSchema creates a parser with a JSON schema.
func NewJSONOutputParserWithSchema(schema map[string]interface{}) *JSONOutputParser {
	return &JSONOutputParser{
		Schema: schema,
	}
}

// Parse parses JSON output.
func (p *JSONOutputParser) Parse(output string) (interface{}, error) {
	// Try to extract JSON from the output
	jsonStr := extractJSON(output)
	if jsonStr == "" {
		return nil, fmt.Errorf("no JSON found in output")
	}

	if p.TargetType != nil {
		// Create a new instance of the target type
		target := reflect.New(p.TargetType).Interface()
		if err := json.Unmarshal([]byte(jsonStr), target); err != nil {
			return nil, fmt.Errorf("failed to parse JSON: %w", err)
		}
		return reflect.ValueOf(target).Elem().Interface(), nil
	}

	// Parse as generic map/slice
	var result interface{}
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}

	return result, nil
}

// GetFormatInstructions returns JSON format instructions.
func (p *JSONOutputParser) GetFormatInstructions() string {
	if p.Schema != nil {
		schemaJSON, _ := json.MarshalIndent(p.Schema, "", "  ")
		return fmt.Sprintf("Please respond with a JSON object matching this schema:\n```json\n%s\n```", string(schemaJSON))
	}
	return "Please respond with a valid JSON object."
}

// extractJSON extracts JSON from a string that may contain other text.
func extractJSON(s string) string {
	// Try to find JSON object
	start := -1
	braceCount := 0
	inString := false
	escaped := false

	for i, c := range s {
		if escaped {
			escaped = false
			continue
		}
		if c == '\\' && inString {
			escaped = true
			continue
		}
		if c == '"' {
			inString = !inString
			continue
		}
		if inString {
			continue
		}

		if c == '{' {
			if start == -1 {
				start = i
			}
			braceCount++
		} else if c == '}' {
			braceCount--
			if braceCount == 0 && start != -1 {
				return s[start : i+1]
			}
		}
	}

	// Try to find JSON array
	start = -1
	bracketCount := 0
	inString = false
	escaped = false

	for i, c := range s {
		if escaped {
			escaped = false
			continue
		}
		if c == '\\' && inString {
			escaped = true
			continue
		}
		if c == '"' {
			inString = !inString
			continue
		}
		if inString {
			continue
		}

		if c == '[' {
			if start == -1 {
				start = i
			}
			bracketCount++
		} else if c == ']' {
			bracketCount--
			if bracketCount == 0 && start != -1 {
				return s[start : i+1]
			}
		}
	}

	return ""
}

// PydanticOutputParser parses output into a struct type (Go equivalent of Pydantic).
type PydanticOutputParser struct {
	// TargetType is the struct type to parse into.
	TargetType reflect.Type
	// TypeName is the name of the type for instructions.
	TypeName string
}

// NewPydanticOutputParser creates a parser for a struct type.
func NewPydanticOutputParser(target interface{}) *PydanticOutputParser {
	t := reflect.TypeOf(target)
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	return &PydanticOutputParser{
		TargetType: t,
		TypeName:   t.Name(),
	}
}

// Parse parses output into the target struct.
func (p *PydanticOutputParser) Parse(output string) (interface{}, error) {
	jsonStr := extractJSON(output)
	if jsonStr == "" {
		return nil, fmt.Errorf("no JSON found in output")
	}

	target := reflect.New(p.TargetType).Interface()
	if err := json.Unmarshal([]byte(jsonStr), target); err != nil {
		return nil, fmt.Errorf("failed to parse into %s: %w", p.TypeName, err)
	}

	return reflect.ValueOf(target).Elem().Interface(), nil
}

// GetFormatInstructions returns format instructions based on struct fields.
func (p *PydanticOutputParser) GetFormatInstructions() string {
	schema := structToSchema(p.TargetType)
	schemaJSON, _ := json.MarshalIndent(schema, "", "  ")
	return fmt.Sprintf("Please respond with a JSON object for type '%s' matching this schema:\n```json\n%s\n```", p.TypeName, string(schemaJSON))
}

// structToSchema converts a struct type to a JSON schema.
func structToSchema(t reflect.Type) map[string]interface{} {
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	schema := map[string]interface{}{
		"type":       "object",
		"properties": map[string]interface{}{},
	}

	properties := schema["properties"].(map[string]interface{})
	required := []string{}

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		if !field.IsExported() {
			continue
		}

		jsonTag := field.Tag.Get("json")
		if jsonTag == "-" {
			continue
		}

		fieldName := field.Name
		if jsonTag != "" {
			// Parse json tag to get field name
			for j, c := range jsonTag {
				if c == ',' {
					jsonTag = jsonTag[:j]
					break
				}
			}
			if jsonTag != "" {
				fieldName = jsonTag
			}
		}

		fieldSchema := typeToSchema(field.Type)

		// Add description from struct tag if available
		if desc := field.Tag.Get("description"); desc != "" {
			fieldSchema["description"] = desc
		}

		properties[fieldName] = fieldSchema

		// Check if required
		if !hasOmitempty(field.Tag.Get("json")) {
			required = append(required, fieldName)
		}
	}

	if len(required) > 0 {
		schema["required"] = required
	}

	return schema
}

// typeToSchema converts a Go type to JSON schema type.
func typeToSchema(t reflect.Type) map[string]interface{} {
	switch t.Kind() {
	case reflect.String:
		return map[string]interface{}{"type": "string"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]interface{}{"type": "integer"}
	case reflect.Float32, reflect.Float64:
		return map[string]interface{}{"type": "number"}
	case reflect.Bool:
		return map[string]interface{}{"type": "boolean"}
	case reflect.Slice, reflect.Array:
		return map[string]interface{}{
			"type":  "array",
			"items": typeToSchema(t.Elem()),
		}
	case reflect.Map:
		return map[string]interface{}{
			"type":                 "object",
			"additionalProperties": typeToSchema(t.Elem()),
		}
	case reflect.Struct:
		return structToSchema(t)
	case reflect.Ptr:
		return typeToSchema(t.Elem())
	default:
		return map[string]interface{}{"type": "string"}
	}
}

// hasOmitempty checks if a json tag has omitempty.
func hasOmitempty(tag string) bool {
	for _, part := range splitTag(tag) {
		if part == "omitempty" {
			return true
		}
	}
	return false
}

// splitTag splits a struct tag by comma.
func splitTag(tag string) []string {
	var parts []string
	current := ""
	for _, c := range tag {
		if c == ',' {
			parts = append(parts, current)
			current = ""
		} else {
			current += string(c)
		}
	}
	if current != "" {
		parts = append(parts, current)
	}
	return parts
}

// BaseProgram provides common functionality for programs.
type BaseProgram struct {
	LLM          llm.LLM
	Prompt       *prompts.PromptTemplate
	OutputParser OutputParser
	Verbose      bool
}

// BaseProgramOption configures a BaseProgram.
type BaseProgramOption func(*BaseProgram)

// WithProgramLLM sets the LLM for the program.
func WithProgramLLM(l llm.LLM) BaseProgramOption {
	return func(p *BaseProgram) {
		p.LLM = l
	}
}

// WithProgramPrompt sets the prompt template.
func WithProgramPrompt(prompt *prompts.PromptTemplate) BaseProgramOption {
	return func(p *BaseProgram) {
		p.Prompt = prompt
	}
}

// WithProgramOutputParser sets the output parser.
func WithProgramOutputParser(parser OutputParser) BaseProgramOption {
	return func(p *BaseProgram) {
		p.OutputParser = parser
	}
}

// WithProgramVerbose enables verbose output.
func WithProgramVerbose(verbose bool) BaseProgramOption {
	return func(p *BaseProgram) {
		p.Verbose = verbose
	}
}

// NewBaseProgram creates a new BaseProgram.
func NewBaseProgram(opts ...BaseProgramOption) *BaseProgram {
	p := &BaseProgram{
		OutputParser: NewJSONOutputParser(),
	}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

// GetPrompt returns the prompt template.
func (p *BaseProgram) GetPrompt() *prompts.PromptTemplate {
	return p.Prompt
}

// SetPrompt sets the prompt template.
func (p *BaseProgram) SetPrompt(prompt *prompts.PromptTemplate) {
	p.Prompt = prompt
}

// Ensure interfaces are implemented.
var _ OutputParser = (*JSONOutputParser)(nil)
var _ OutputParser = (*PydanticOutputParser)(nil)
