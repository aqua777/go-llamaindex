package bedrock

import (
	"testing"

	"github.com/aqua777/go-llamaindex/embedding"
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

// TestEmbedding tests the AWS Bedrock Embedding implementation.
func TestEmbedding(t *testing.T) {
	t.Run("NewEmbedding with defaults", func(t *testing.T) {
		emb := NewEmbedding()
		assert.NotNil(t, emb)
		assert.Equal(t, DefaultEmbeddingModel, emb.model)
		assert.Equal(t, 1024, emb.dimensions)
		assert.True(t, emb.normalize)
	})

	t.Run("NewEmbedding with options", func(t *testing.T) {
		emb := NewEmbedding(
			WithEmbeddingModel(TitanEmbedTextV1),
			WithEmbeddingRegion("us-west-2"),
			WithEmbeddingDimensions(512),
			WithEmbeddingNormalize(false),
		)

		assert.Equal(t, TitanEmbedTextV1, emb.model)
		assert.Equal(t, "us-west-2", emb.region)
		assert.Equal(t, 512, emb.dimensions)
		assert.False(t, emb.normalize)
	})

	t.Run("Info returns correct values for Titan models", func(t *testing.T) {
		tests := []struct {
			model      string
			dimensions int
			maxTokens  int
		}{
			{TitanEmbedTextV1, 1536, 8192},
			{TitanEmbedTextV2, 1024, 8192},
			{TitanEmbedG1Text02, 1536, 8192},
		}

		for _, tt := range tests {
			emb := NewEmbedding(WithEmbeddingModel(tt.model))
			info := emb.Info()
			assert.Equal(t, tt.model, info.ModelName)
			assert.Equal(t, tt.dimensions, info.Dimensions, "model: %s", tt.model)
			assert.Equal(t, tt.maxTokens, info.MaxTokens, "model: %s", tt.model)
		}
	})

	t.Run("Info returns correct values for Cohere models", func(t *testing.T) {
		tests := []struct {
			model      string
			dimensions int
			maxTokens  int
		}{
			{CohereEmbedEnglishV3, 1024, 512},
			{CohereEmbedMultilingualV3, 1024, 512},
			{CohereEmbedV4, 1024, 512},
		}

		for _, tt := range tests {
			emb := NewEmbedding(WithEmbeddingModel(tt.model))
			info := emb.Info()
			assert.Equal(t, tt.model, info.ModelName)
			assert.Equal(t, tt.dimensions, info.Dimensions, "model: %s", tt.model)
			assert.Equal(t, tt.maxTokens, info.MaxTokens, "model: %s", tt.model)
		}
	})

	t.Run("Titan V2 respects custom dimensions", func(t *testing.T) {
		emb := NewEmbedding(
			WithEmbeddingModel(TitanEmbedTextV2),
			WithEmbeddingDimensions(256),
		)
		info := emb.Info()
		assert.Equal(t, 256, info.Dimensions)
	})

	t.Run("getProvider extracts provider correctly", func(t *testing.T) {
		tests := []struct {
			model    string
			provider string
		}{
			{TitanEmbedTextV1, "amazon"},
			{TitanEmbedTextV2, "amazon"},
			{CohereEmbedEnglishV3, "cohere"},
			{"us.amazon.titan-embed-text-v2:0", "amazon"},
			{"eu.cohere.embed-english-v3", "cohere"},
		}

		for _, tt := range tests {
			emb := NewEmbedding(WithEmbeddingModel(tt.model))
			assert.Equal(t, tt.provider, emb.getProvider(), "model: %s", tt.model)
		}
	})

	t.Run("Embedding model constants are correct", func(t *testing.T) {
		assert.Equal(t, "amazon.titan-embed-text-v1", TitanEmbedTextV1)
		assert.Equal(t, "amazon.titan-embed-text-v2:0", TitanEmbedTextV2)
		assert.Equal(t, "amazon.titan-embed-g1-text-02", TitanEmbedG1Text02)
		assert.Equal(t, "cohere.embed-english-v3", CohereEmbedEnglishV3)
		assert.Equal(t, "cohere.embed-multilingual-v3", CohereEmbedMultilingualV3)
		assert.Equal(t, "cohere.embed-v4:0", CohereEmbedV4)
	})

	t.Run("Embedding implements all interfaces", func(t *testing.T) {
		var _ embedding.EmbeddingModel = (*Embedding)(nil)
		var _ embedding.EmbeddingModelWithInfo = (*Embedding)(nil)
		var _ embedding.EmbeddingModelWithBatch = (*Embedding)(nil)
	})
}
