package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
)

// FunctionTool wraps a Go function as a tool.
type FunctionTool struct {
	*BaseTool
	fn           interface{}
	fnValue      reflect.Value
	fnType       reflect.Type
	paramNames   []string
	returnDirect bool
}

// FunctionToolOption configures a FunctionTool.
type FunctionToolOption func(*FunctionTool)

// WithFunctionToolName sets the tool name.
func WithFunctionToolName(name string) FunctionToolOption {
	return func(ft *FunctionTool) {
		ft.metadata.Name = name
	}
}

// WithFunctionToolDescription sets the tool description.
func WithFunctionToolDescription(description string) FunctionToolOption {
	return func(ft *FunctionTool) {
		ft.metadata.Description = description
	}
}

// WithFunctionToolReturnDirect sets whether to return the output directly.
func WithFunctionToolReturnDirect(returnDirect bool) FunctionToolOption {
	return func(ft *FunctionTool) {
		ft.returnDirect = returnDirect
		ft.metadata.ReturnDirect = returnDirect
	}
}

// WithFunctionToolParameters sets custom parameters schema.
func WithFunctionToolParameters(params map[string]interface{}) FunctionToolOption {
	return func(ft *FunctionTool) {
		ft.metadata.Parameters = params
	}
}

// NewFunctionTool creates a new FunctionTool from a function.
// The function should have the signature:
//
//	func(ctx context.Context, args...) (result, error)
//
// or
//
//	func(args...) (result, error)
func NewFunctionTool(fn interface{}, opts ...FunctionToolOption) (*FunctionTool, error) {
	fnValue := reflect.ValueOf(fn)
	fnType := fnValue.Type()

	if fnType.Kind() != reflect.Func {
		return nil, fmt.Errorf("fn must be a function, got %s", fnType.Kind())
	}

	// Get function name from runtime
	fnName := getFunctionName(fn)

	// Generate parameters schema from function signature
	params, paramNames := generateParametersSchema(fnType)

	ft := &FunctionTool{
		BaseTool: NewBaseTool(&ToolMetadata{
			Name:        fnName,
			Description: fmt.Sprintf("Function: %s", fnName),
			Parameters:  params,
		}),
		fn:         fn,
		fnValue:    fnValue,
		fnType:     fnType,
		paramNames: paramNames,
	}

	for _, opt := range opts {
		opt(ft)
	}

	return ft, nil
}

// NewFunctionToolFromDefaults creates a FunctionTool with explicit name and description.
func NewFunctionToolFromDefaults(
	fn interface{},
	name string,
	description string,
	opts ...FunctionToolOption,
) (*FunctionTool, error) {
	ft, err := NewFunctionTool(fn, opts...)
	if err != nil {
		return nil, err
	}

	ft.metadata.Name = name
	ft.metadata.Description = description

	return ft, nil
}

// Call executes the function with the given input.
func (ft *FunctionTool) Call(ctx context.Context, input interface{}) (*ToolOutput, error) {
	// Convert input to the expected arguments
	args, rawInput, err := ft.prepareArgs(ctx, input)
	if err != nil {
		return NewErrorToolOutput(ft.metadata.Name, err), err
	}

	// Call the function
	results := ft.fnValue.Call(args)

	// Process results
	return ft.processResults(results, rawInput)
}

// prepareArgs converts the input to function arguments.
func (ft *FunctionTool) prepareArgs(ctx context.Context, input interface{}) ([]reflect.Value, map[string]interface{}, error) {
	var args []reflect.Value
	rawInput := make(map[string]interface{})

	// Check if first parameter is context.Context
	startIdx := 0
	if ft.fnType.NumIn() > 0 && ft.fnType.In(0) == reflect.TypeOf((*context.Context)(nil)).Elem() {
		args = append(args, reflect.ValueOf(ctx))
		startIdx = 1
	}

	// Convert input to map
	var inputMap map[string]interface{}
	switch v := input.(type) {
	case map[string]interface{}:
		inputMap = v
	case string:
		// If single string input and function expects one string param
		if ft.fnType.NumIn()-startIdx == 1 && ft.fnType.In(startIdx).Kind() == reflect.String {
			args = append(args, reflect.ValueOf(v))
			rawInput["input"] = v
			return args, rawInput, nil
		}
		inputMap = map[string]interface{}{"input": v}
	default:
		// Try to marshal and unmarshal to convert to map
		data, err := json.Marshal(input)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to convert input: %w", err)
		}
		if err := json.Unmarshal(data, &inputMap); err != nil {
			return nil, nil, fmt.Errorf("failed to convert input: %w", err)
		}
	}

	rawInput = inputMap

	// Map input to function parameters
	for i := startIdx; i < ft.fnType.NumIn(); i++ {
		paramType := ft.fnType.In(i)
		paramName := ft.paramNames[i-startIdx]

		value, ok := inputMap[paramName]
		if !ok {
			// Try "input" as fallback for single parameter
			if len(ft.paramNames) == 1 {
				value, ok = inputMap["input"]
			}
			if !ok {
				// Use zero value
				args = append(args, reflect.Zero(paramType))
				continue
			}
		}

		// Convert value to the expected type
		convertedValue, err := convertValue(value, paramType)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to convert parameter %s: %w", paramName, err)
		}
		args = append(args, convertedValue)
	}

	return args, rawInput, nil
}

// processResults processes the function results.
func (ft *FunctionTool) processResults(results []reflect.Value, rawInput map[string]interface{}) (*ToolOutput, error) {
	var result interface{}
	var err error

	// Check for error in last return value
	if len(results) > 0 {
		lastResult := results[len(results)-1]
		if lastResult.Type().Implements(reflect.TypeOf((*error)(nil)).Elem()) {
			if !lastResult.IsNil() {
				err = lastResult.Interface().(error)
			}
			results = results[:len(results)-1]
		}
	}

	// Get the result value
	if len(results) > 0 {
		result = results[0].Interface()
	}

	if err != nil {
		return NewErrorToolOutput(ft.metadata.Name, err), err
	}

	content := fmt.Sprintf("%v", result)
	return NewToolOutputWithInput(ft.metadata.Name, content, rawInput, result), nil
}

// generateParametersSchema generates a JSON Schema from function signature.
func generateParametersSchema(fnType reflect.Type) (map[string]interface{}, []string) {
	properties := make(map[string]interface{})
	required := []string{}
	paramNames := []string{}

	startIdx := 0
	// Skip context.Context parameter
	if fnType.NumIn() > 0 && fnType.In(0) == reflect.TypeOf((*context.Context)(nil)).Elem() {
		startIdx = 1
	}

	for i := startIdx; i < fnType.NumIn(); i++ {
		paramType := fnType.In(i)
		paramName := fmt.Sprintf("arg%d", i-startIdx)
		paramNames = append(paramNames, paramName)

		properties[paramName] = typeToJSONSchema(paramType)
		required = append(required, paramName)
	}

	// If no parameters, use default input schema
	if len(properties) == 0 {
		return DefaultParameters(), []string{"input"}
	}

	return map[string]interface{}{
		"type":       "object",
		"properties": properties,
		"required":   required,
	}, paramNames
}

// typeToJSONSchema converts a Go type to JSON Schema.
func typeToJSONSchema(t reflect.Type) map[string]interface{} {
	switch t.Kind() {
	case reflect.String:
		return map[string]interface{}{"type": "string"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return map[string]interface{}{"type": "integer"}
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]interface{}{"type": "integer"}
	case reflect.Float32, reflect.Float64:
		return map[string]interface{}{"type": "number"}
	case reflect.Bool:
		return map[string]interface{}{"type": "boolean"}
	case reflect.Slice, reflect.Array:
		return map[string]interface{}{
			"type":  "array",
			"items": typeToJSONSchema(t.Elem()),
		}
	case reflect.Map:
		return map[string]interface{}{
			"type": "object",
		}
	case reflect.Struct:
		return structToJSONSchema(t)
	case reflect.Ptr:
		return typeToJSONSchema(t.Elem())
	default:
		return map[string]interface{}{"type": "string"}
	}
}

// structToJSONSchema converts a struct type to JSON Schema.
func structToJSONSchema(t reflect.Type) map[string]interface{} {
	properties := make(map[string]interface{})
	required := []string{}

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		if !field.IsExported() {
			continue
		}

		// Get JSON tag name
		jsonTag := field.Tag.Get("json")
		fieldName := field.Name
		if jsonTag != "" {
			parts := strings.Split(jsonTag, ",")
			if parts[0] != "" && parts[0] != "-" {
				fieldName = parts[0]
			}
		}

		properties[fieldName] = typeToJSONSchema(field.Type)

		// Check if field is required (not a pointer and no omitempty)
		if field.Type.Kind() != reflect.Ptr && !strings.Contains(jsonTag, "omitempty") {
			required = append(required, fieldName)
		}
	}

	return map[string]interface{}{
		"type":       "object",
		"properties": properties,
		"required":   required,
	}
}

// convertValue converts a value to the target type.
func convertValue(value interface{}, targetType reflect.Type) (reflect.Value, error) {
	if value == nil {
		return reflect.Zero(targetType), nil
	}

	valueType := reflect.TypeOf(value)
	if valueType == targetType {
		return reflect.ValueOf(value), nil
	}

	// Handle common conversions
	switch targetType.Kind() {
	case reflect.String:
		return reflect.ValueOf(fmt.Sprintf("%v", value)), nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		switch v := value.(type) {
		case float64:
			return reflect.ValueOf(int64(v)).Convert(targetType), nil
		case int:
			return reflect.ValueOf(int64(v)).Convert(targetType), nil
		case string:
			var i int64
			fmt.Sscanf(v, "%d", &i)
			return reflect.ValueOf(i).Convert(targetType), nil
		}
	case reflect.Float32, reflect.Float64:
		switch v := value.(type) {
		case float64:
			return reflect.ValueOf(v).Convert(targetType), nil
		case int:
			return reflect.ValueOf(float64(v)).Convert(targetType), nil
		case string:
			var f float64
			fmt.Sscanf(v, "%f", &f)
			return reflect.ValueOf(f).Convert(targetType), nil
		}
	case reflect.Bool:
		switch v := value.(type) {
		case bool:
			return reflect.ValueOf(v), nil
		case string:
			return reflect.ValueOf(v == "true" || v == "1"), nil
		}
	case reflect.Struct, reflect.Ptr:
		// Try JSON marshal/unmarshal
		data, err := json.Marshal(value)
		if err != nil {
			return reflect.Value{}, err
		}
		newValue := reflect.New(targetType)
		if err := json.Unmarshal(data, newValue.Interface()); err != nil {
			return reflect.Value{}, err
		}
		if targetType.Kind() == reflect.Ptr {
			return newValue, nil
		}
		return newValue.Elem(), nil
	}

	// Try direct conversion
	if valueType.ConvertibleTo(targetType) {
		return reflect.ValueOf(value).Convert(targetType), nil
	}

	return reflect.Value{}, fmt.Errorf("cannot convert %T to %s", value, targetType)
}

// getFunctionName extracts the function name from a function value.
func getFunctionName(fn interface{}) string {
	fnValue := reflect.ValueOf(fn)
	fnPtr := fnValue.Pointer()
	// Use a simple default name based on pointer
	return fmt.Sprintf("func_%x", fnPtr)
}

// Ensure FunctionTool implements Tool.
var _ Tool = (*FunctionTool)(nil)
