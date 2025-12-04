package bedrock

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestLLM tests the AWS Bedrock LLM implementation.
func TestLLM(t *testing.T) {
	t.Run("New with defaults", func(t *testing.T) {
		llm := New()
		assert.NotNil(t, llm)
		assert.Equal(t, DefaultModel, llm.model)
		assert.Equal(t, DefaultMaxTokens, llm.maxTokens)
	})

	t.Run("New with options", func(t *testing.T) {
		llm := New(
			WithModel(Claude3Haiku),
			WithMaxTokens(2048),
			WithTemperature(0.5),
			WithTopP(0.9),
			WithRegion("us-west-2"),
		)

		assert.Equal(t, Claude3Haiku, llm.model)
		assert.Equal(t, 2048, llm.maxTokens)
		assert.Equal(t, float32(0.5), llm.temperature)
		assert.Equal(t, float32(0.9), llm.topP)
		assert.Equal(t, "us-west-2", llm.region)
	})

	t.Run("Metadata returns correct values", func(t *testing.T) {
		tests := []struct {
			model         string
			contextWindow int
			toolCalling   bool
			multiModal    bool
		}{
			{Claude35SonnetV2, 200000, true, true},
			{Claude3Haiku, 200000, true, true},
			{NovaProV1, 300000, true, true},
			{NovaMicroV1, 128000, true, false},
			{Llama33_70BInstruct, 128000, true, false},
			{Mistral7BInstruct, 32000, false, false},
			{TitanTextExpressV1, 8192, false, false},
		}

		for _, tt := range tests {
			llm := New(WithModel(tt.model))
			meta := llm.Metadata()
			assert.Equal(t, tt.model, meta.ModelName)
			assert.Equal(t, tt.contextWindow, meta.ContextWindow, "model: %s", tt.model)
			assert.Equal(t, tt.toolCalling, meta.IsFunctionCalling, "model: %s", tt.model)
			assert.Equal(t, tt.multiModal, meta.IsMultiModal, "model: %s", tt.model)
		}
	})

	t.Run("SupportsToolCalling depends on model", func(t *testing.T) {
		llm := New(WithModel(Claude35SonnetV2))
		assert.True(t, llm.SupportsToolCalling())

		llm = New(WithModel(Mistral7BInstruct))
		assert.False(t, llm.SupportsToolCalling())
	})

	t.Run("SupportsStructuredOutput depends on tool calling", func(t *testing.T) {
		llm := New(WithModel(Claude35SonnetV2))
		assert.True(t, llm.SupportsStructuredOutput())

		llm = New(WithModel(Mistral7BInstruct))
		assert.False(t, llm.SupportsStructuredOutput())
	})

	t.Run("IsFunctionCallingModel helper", func(t *testing.T) {
		assert.True(t, IsFunctionCallingModel(Claude35SonnetV2))
		assert.True(t, IsFunctionCallingModel(NovaProV1))
		assert.True(t, IsFunctionCallingModel(Llama33_70BInstruct))
		assert.False(t, IsFunctionCallingModel(Mistral7BInstruct))
		assert.False(t, IsFunctionCallingModel(TitanTextExpressV1))
	})

	t.Run("ModelContextSize helper", func(t *testing.T) {
		assert.Equal(t, 200000, ModelContextSize(Claude35SonnetV2))
		assert.Equal(t, 300000, ModelContextSize(NovaProV1))
		assert.Equal(t, 128000, ModelContextSize(Llama31_70BInstruct))
		assert.Equal(t, 32000, ModelContextSize(Mistral7BInstruct))
		assert.Equal(t, 128000, ModelContextSize("unknown-model")) // default
	})

	t.Run("Region-prefixed models work correctly", func(t *testing.T) {
		// Test that us. prefixed models are handled correctly
		assert.True(t, IsFunctionCallingModel("us.anthropic.claude-3-5-sonnet-20241022-v2:0"))
		assert.Equal(t, 200000, ModelContextSize("us.anthropic.claude-3-5-sonnet-20241022-v2:0"))

		// Test eu. prefix
		assert.True(t, IsFunctionCallingModel("eu.anthropic.claude-3-5-sonnet-20241022-v2:0"))

		// Test apac. prefix
		assert.True(t, IsFunctionCallingModel("apac.anthropic.claude-3-5-sonnet-20241022-v2:0"))
	})
}
