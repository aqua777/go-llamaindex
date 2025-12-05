package outputparser

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestStructuredOutput tests the StructuredOutput struct.
func TestStructuredOutput(t *testing.T) {
	t.Run("Create StructuredOutput", func(t *testing.T) {
		output := &StructuredOutput{
			RawOutput:    "raw text",
			ParsedOutput: map[string]string{"key": "value"},
		}
		assert.Equal(t, "raw text", output.RawOutput)
		assert.NotNil(t, output.ParsedOutput)
	})
}

// TestOutputParserError tests the OutputParserError.
func TestOutputParserError(t *testing.T) {
	t.Run("Create OutputParserError", func(t *testing.T) {
		err := NewOutputParserError("parse failed", "invalid output")
		assert.Contains(t, err.Error(), "parse failed")
		assert.Contains(t, err.Error(), "invalid output")
	})
}

// TestBaseOutputParser tests the BaseOutputParser.
func TestBaseOutputParser(t *testing.T) {
	t.Run("NewBaseOutputParser", func(t *testing.T) {
		parser := NewBaseOutputParser()
		assert.NotNil(t, parser)
		assert.Equal(t, "BaseOutputParser", parser.Name())
	})

	t.Run("WithParserName", func(t *testing.T) {
		parser := NewBaseOutputParser(WithParserName("custom"))
		assert.Equal(t, "custom", parser.Name())
	})

	t.Run("Parse returns raw output", func(t *testing.T) {
		parser := NewBaseOutputParser()
		output, err := parser.Parse("test output")
		require.NoError(t, err)
		assert.Equal(t, "test output", output.RawOutput)
		assert.Equal(t, "test output", output.ParsedOutput)
	})

	t.Run("Format returns template unchanged", func(t *testing.T) {
		parser := NewBaseOutputParser()
		result := parser.Format("template")
		assert.Equal(t, "template", result)
	})
}

// TestJSONOutputParser tests the JSONOutputParser.
func TestJSONOutputParser(t *testing.T) {
	t.Run("NewJSONOutputParser", func(t *testing.T) {
		parser := NewJSONOutputParser()
		assert.NotNil(t, parser)
		assert.Equal(t, "JSONOutputParser", parser.Name())
	})

	t.Run("Parse JSON object", func(t *testing.T) {
		parser := NewJSONOutputParser()
		output, err := parser.Parse(`{"key": "value"}`)
		require.NoError(t, err)
		assert.NotNil(t, output.ParsedOutput)
	})

	t.Run("Parse JSON array", func(t *testing.T) {
		parser := NewJSONOutputParser()
		output, err := parser.Parse(`[1, 2, 3]`)
		require.NoError(t, err)
		assert.NotNil(t, output.ParsedOutput)
	})

	t.Run("Parse JSON from code block", func(t *testing.T) {
		parser := NewJSONOutputParser()
		output, err := parser.Parse("```json\n{\"key\": \"value\"}\n```")
		require.NoError(t, err)
		assert.NotNil(t, output.ParsedOutput)
	})

	t.Run("Parse JSON with surrounding text", func(t *testing.T) {
		parser := NewJSONOutputParser()
		output, err := parser.Parse(`Here is the result: {"key": "value"} That's all.`)
		require.NoError(t, err)
		assert.NotNil(t, output.ParsedOutput)
	})

	t.Run("Parse invalid JSON returns error", func(t *testing.T) {
		parser := NewJSONOutputParser()
		_, err := parser.Parse("not json")
		assert.Error(t, err)
	})

	t.Run("Format adds instructions", func(t *testing.T) {
		parser := NewJSONOutputParser()
		result := parser.Format("template")
		assert.Contains(t, result, "template")
		assert.Contains(t, result, "JSON")
	})
}

// TestListOutputParser tests the ListOutputParser.
func TestListOutputParser(t *testing.T) {
	t.Run("NewListOutputParser", func(t *testing.T) {
		parser := NewListOutputParser()
		assert.NotNil(t, parser)
		assert.Equal(t, "ListOutputParser", parser.Name())
	})

	t.Run("Parse newline-separated list", func(t *testing.T) {
		parser := NewListOutputParser()
		output, err := parser.Parse("item1\nitem2\nitem3")
		require.NoError(t, err)

		items, ok := output.ParsedOutput.([]string)
		require.True(t, ok)
		assert.Len(t, items, 3)
		assert.Equal(t, "item1", items[0])
	})

	t.Run("Parse with custom separator", func(t *testing.T) {
		parser := NewListOutputParser(WithListSeparator(","))
		output, err := parser.Parse("a,b,c")
		require.NoError(t, err)

		items, ok := output.ParsedOutput.([]string)
		require.True(t, ok)
		assert.Len(t, items, 3)
	})

	t.Run("Filters empty items", func(t *testing.T) {
		parser := NewListOutputParser()
		output, err := parser.Parse("item1\n\nitem2\n  \nitem3")
		require.NoError(t, err)

		items, ok := output.ParsedOutput.([]string)
		require.True(t, ok)
		assert.Len(t, items, 3)
	})
}

// TestBooleanOutputParser tests the BooleanOutputParser.
func TestBooleanOutputParser(t *testing.T) {
	t.Run("NewBooleanOutputParser", func(t *testing.T) {
		parser := NewBooleanOutputParser()
		assert.NotNil(t, parser)
		assert.Equal(t, "BooleanOutputParser", parser.Name())
	})

	t.Run("Parse true values", func(t *testing.T) {
		parser := NewBooleanOutputParser()

		trueValues := []string{"yes", "YES", "true", "True", "1", "y", "Y"}
		for _, v := range trueValues {
			output, err := parser.Parse(v)
			require.NoError(t, err, "failed for value: %s", v)
			assert.True(t, output.ParsedOutput.(bool), "expected true for: %s", v)
		}
	})

	t.Run("Parse false values", func(t *testing.T) {
		parser := NewBooleanOutputParser()

		falseValues := []string{"no", "NO", "false", "False", "0", "n", "N"}
		for _, v := range falseValues {
			output, err := parser.Parse(v)
			require.NoError(t, err, "failed for value: %s", v)
			assert.False(t, output.ParsedOutput.(bool), "expected false for: %s", v)
		}
	})

	t.Run("Parse with surrounding text", func(t *testing.T) {
		parser := NewBooleanOutputParser()
		output, err := parser.Parse("The answer is yes.")
		require.NoError(t, err)
		assert.True(t, output.ParsedOutput.(bool))
	})

	t.Run("Parse ambiguous returns error", func(t *testing.T) {
		parser := NewBooleanOutputParser()
		_, err := parser.Parse("perhaps")
		assert.Error(t, err)
	})

	t.Run("Custom true/false values", func(t *testing.T) {
		parser := NewBooleanOutputParser(
			WithTrueValues([]string{"correct", "right"}),
			WithFalseValues([]string{"incorrect", "wrong"}),
		)

		output, err := parser.Parse("correct")
		require.NoError(t, err)
		assert.True(t, output.ParsedOutput.(bool))

		output, err = parser.Parse("wrong")
		require.NoError(t, err)
		assert.False(t, output.ParsedOutput.(bool))
	})
}

// TestExtractJSON tests the extractJSON function.
func TestExtractJSON(t *testing.T) {
	t.Run("Extract from code block", func(t *testing.T) {
		text := "```json\n{\"key\": \"value\"}\n```"
		result := extractJSON(text)
		assert.Equal(t, "{\"key\": \"value\"}", result)
	})

	t.Run("Extract object", func(t *testing.T) {
		text := "Here is the result: {\"key\": \"value\"}"
		result := extractJSON(text)
		assert.Equal(t, "{\"key\": \"value\"}", result)
	})

	t.Run("Extract array", func(t *testing.T) {
		text := "The list is: [1, 2, 3]"
		result := extractJSON(text)
		assert.Equal(t, "[1, 2, 3]", result)
	})

	t.Run("No JSON returns empty", func(t *testing.T) {
		text := "No JSON here"
		result := extractJSON(text)
		assert.Empty(t, result)
	})
}

// TestInterfaceCompliance tests that all parsers implement OutputParser.
func TestInterfaceCompliance(t *testing.T) {
	var _ OutputParser = (*BaseOutputParser)(nil)
	var _ OutputParser = (*JSONOutputParser)(nil)
	var _ OutputParser = (*ListOutputParser)(nil)
	var _ OutputParser = (*BooleanOutputParser)(nil)
}
