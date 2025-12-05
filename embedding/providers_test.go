package embedding

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestOllamaEmbedding tests the Ollama embedding implementation.
func TestOllamaEmbedding(t *testing.T) {
	t.Run("NewOllamaEmbedding with defaults", func(t *testing.T) {
		e := NewOllamaEmbedding()
		assert.NotNil(t, e)
		assert.Equal(t, OllamaNomicEmbedText, e.model)
		assert.Equal(t, OllamaDefaultURL, e.baseURL)
	})

	t.Run("NewOllamaEmbedding with options", func(t *testing.T) {
		e := NewOllamaEmbedding(
			WithOllamaEmbeddingModel(OllamaMxbaiEmbedLarge),
			WithOllamaEmbeddingBaseURL("http://custom:11434"),
		)
		assert.Equal(t, OllamaMxbaiEmbedLarge, e.model)
		assert.Equal(t, "http://custom:11434", e.baseURL)
	})

	t.Run("Info returns correct values", func(t *testing.T) {
		tests := []struct {
			model      string
			dimensions int
		}{
			{OllamaMxbaiEmbedLarge, 1024},
			{OllamaAllMiniLM, 384},
			{OllamaNomicEmbedText, 768},
			{OllamaSnowflakeArctic, 1024},
			{OllamaBgeSmall, 384},
			{OllamaBgeLarge, 1024},
		}

		for _, tt := range tests {
			e := NewOllamaEmbedding(WithOllamaEmbeddingModel(tt.model))
			info := e.Info()
			assert.Equal(t, tt.model, info.ModelName)
			assert.Equal(t, tt.dimensions, info.Dimensions)
		}
	})

	t.Run("SupportsMultiModal returns false", func(t *testing.T) {
		e := NewOllamaEmbedding()
		assert.False(t, e.SupportsMultiModal())
	})

	t.Run("GetTextEmbedding with mock server", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, "POST", r.Method)
			assert.Equal(t, "/api/embeddings", r.URL.Path)

			var req ollamaEmbeddingRequest
			json.NewDecoder(r.Body).Decode(&req)
			assert.Equal(t, "nomic-embed-text", req.Model)
			assert.Equal(t, "test text", req.Prompt)

			resp := ollamaEmbeddingResponse{
				Embedding: []float64{0.1, 0.2, 0.3, 0.4, 0.5},
			}
			json.NewEncoder(w).Encode(resp)
		}))
		defer server.Close()

		e := NewOllamaEmbedding(WithOllamaEmbeddingBaseURL(server.URL))

		embedding, err := e.GetTextEmbedding(context.Background(), "test text")
		require.NoError(t, err)
		assert.Equal(t, []float64{0.1, 0.2, 0.3, 0.4, 0.5}, embedding)
	})

	t.Run("GetQueryEmbedding with mock server", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			resp := ollamaEmbeddingResponse{
				Embedding: []float64{0.5, 0.4, 0.3, 0.2, 0.1},
			}
			json.NewEncoder(w).Encode(resp)
		}))
		defer server.Close()

		e := NewOllamaEmbedding(WithOllamaEmbeddingBaseURL(server.URL))

		embedding, err := e.GetQueryEmbedding(context.Background(), "test query")
		require.NoError(t, err)
		assert.Len(t, embedding, 5)
	})

	t.Run("GetTextEmbeddingsBatch with mock server", func(t *testing.T) {
		callCount := 0
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			callCount++
			resp := ollamaEmbeddingResponse{
				Embedding: []float64{float64(callCount), float64(callCount) * 0.1},
			}
			json.NewEncoder(w).Encode(resp)
		}))
		defer server.Close()

		e := NewOllamaEmbedding(WithOllamaEmbeddingBaseURL(server.URL))

		texts := []string{"text1", "text2", "text3"}
		embeddings, err := e.GetTextEmbeddingsBatch(context.Background(), texts, nil)
		require.NoError(t, err)
		assert.Len(t, embeddings, 3)
		assert.Equal(t, 3, callCount)
	})
}

// TestCohereEmbedding tests the Cohere embedding implementation.
func TestCohereEmbedding(t *testing.T) {
	t.Run("NewCohereEmbedding with defaults", func(t *testing.T) {
		e := NewCohereEmbedding()
		assert.NotNil(t, e)
		assert.Equal(t, CohereEmbedEnglishV3, e.model)
		assert.Equal(t, CohereAPIURL, e.baseURL)
		assert.Equal(t, CohereInputTypeSearchDocument, e.inputType)
	})

	t.Run("NewCohereEmbedding with options", func(t *testing.T) {
		e := NewCohereEmbedding(
			WithCohereEmbeddingAPIKey("test-key"),
			WithCohereEmbeddingModel(CohereEmbedMultilingualV3),
			WithCohereEmbeddingInputType(CohereInputTypeSearchQuery),
			WithCohereEmbeddingTruncate("START"),
			WithCohereEmbeddingBaseURL("https://custom.cohere.ai"),
		)
		assert.Equal(t, "test-key", e.apiKey)
		assert.Equal(t, CohereEmbedMultilingualV3, e.model)
		assert.Equal(t, CohereInputTypeSearchQuery, e.inputType)
		assert.Equal(t, "START", e.truncate)
		assert.Equal(t, "https://custom.cohere.ai", e.baseURL)
	})

	t.Run("Info returns correct values", func(t *testing.T) {
		tests := []struct {
			model      string
			dimensions int
		}{
			{CohereEmbedEnglishV3, 1024},
			{CohereEmbedMultilingualV3, 1024},
			{CohereEmbedEnglishLightV3, 384},
			{CohereEmbedMultilingualLightV3, 384},
		}

		for _, tt := range tests {
			e := NewCohereEmbedding(WithCohereEmbeddingModel(tt.model))
			info := e.Info()
			assert.Equal(t, tt.model, info.ModelName)
			assert.Equal(t, tt.dimensions, info.Dimensions)
		}
	})

	t.Run("SupportsMultiModal returns false", func(t *testing.T) {
		e := NewCohereEmbedding()
		assert.False(t, e.SupportsMultiModal())
	})

	t.Run("GetTextEmbedding with mock server", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, "POST", r.Method)
			assert.Equal(t, "/embed", r.URL.Path)
			assert.Contains(t, r.Header.Get("Authorization"), "Bearer")

			var req cohereEmbedRequest
			json.NewDecoder(r.Body).Decode(&req)
			assert.Equal(t, "search_document", req.InputType)

			resp := cohereEmbedResponse{
				ID:         "embed_123",
				Embeddings: [][]float64{{0.1, 0.2, 0.3}},
			}
			json.NewEncoder(w).Encode(resp)
		}))
		defer server.Close()

		e := NewCohereEmbedding(
			WithCohereEmbeddingAPIKey("test-key"),
			WithCohereEmbeddingBaseURL(server.URL),
		)

		embedding, err := e.GetTextEmbedding(context.Background(), "test text")
		require.NoError(t, err)
		assert.Equal(t, []float64{0.1, 0.2, 0.3}, embedding)
	})

	t.Run("GetQueryEmbedding uses search_query input type", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var req cohereEmbedRequest
			json.NewDecoder(r.Body).Decode(&req)
			assert.Equal(t, "search_query", req.InputType)

			resp := cohereEmbedResponse{
				Embeddings: [][]float64{{0.5, 0.4, 0.3}},
			}
			json.NewEncoder(w).Encode(resp)
		}))
		defer server.Close()

		e := NewCohereEmbedding(
			WithCohereEmbeddingAPIKey("test-key"),
			WithCohereEmbeddingBaseURL(server.URL),
		)

		embedding, err := e.GetQueryEmbedding(context.Background(), "test query")
		require.NoError(t, err)
		assert.Len(t, embedding, 3)
	})

	t.Run("GetTextEmbeddingsBatch with mock server", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var req cohereEmbedRequest
			json.NewDecoder(r.Body).Decode(&req)

			embeddings := make([][]float64, len(req.Texts))
			for i := range req.Texts {
				embeddings[i] = []float64{float64(i), float64(i) * 0.1}
			}

			resp := cohereEmbedResponse{
				Embeddings: embeddings,
			}
			json.NewEncoder(w).Encode(resp)
		}))
		defer server.Close()

		e := NewCohereEmbedding(
			WithCohereEmbeddingAPIKey("test-key"),
			WithCohereEmbeddingBaseURL(server.URL),
		)

		texts := []string{"text1", "text2", "text3"}
		embeddings, err := e.GetTextEmbeddingsBatch(context.Background(), texts, nil)
		require.NoError(t, err)
		assert.Len(t, embeddings, 3)
	})
}

// TestHuggingFaceEmbedding tests the HuggingFace embedding implementation.
func TestHuggingFaceEmbedding(t *testing.T) {
	t.Run("NewHuggingFaceEmbedding with defaults", func(t *testing.T) {
		e := NewHuggingFaceEmbedding()
		assert.NotNil(t, e)
		assert.Equal(t, HFSentenceTransformersMiniLM, e.model)
		assert.Equal(t, HuggingFaceInferenceAPIURL, e.baseURL)
		assert.False(t, e.useTEI)
	})

	t.Run("NewHuggingFaceEmbedding with options", func(t *testing.T) {
		e := NewHuggingFaceEmbedding(
			WithHuggingFaceAPIKey("test-key"),
			WithHuggingFaceModel(HFBGELarge),
			WithHuggingFaceBaseURL("http://custom:8080"),
			WithHuggingFaceTEI(true),
			WithHuggingFaceQueryPrefix("query: "),
			WithHuggingFaceDocPrefix("passage: "),
		)
		assert.Equal(t, "test-key", e.apiKey)
		assert.Equal(t, HFBGELarge, e.model)
		assert.Equal(t, "http://custom:8080", e.baseURL)
		assert.True(t, e.useTEI)
		assert.Equal(t, "query: ", e.queryPrefix)
		assert.Equal(t, "passage: ", e.docPrefix)
	})

	t.Run("E5 models get default prefixes", func(t *testing.T) {
		e := NewHuggingFaceEmbedding(WithHuggingFaceModel(HFE5Large))
		assert.Equal(t, "query: ", e.queryPrefix)
		assert.Equal(t, "passage: ", e.docPrefix)
	})

	t.Run("Info returns correct values", func(t *testing.T) {
		tests := []struct {
			model      string
			dimensions int
		}{
			{HFSentenceTransformersMiniLM, 384},
			{HFSentenceTransformersMpnet, 768},
			{HFBGEM3, 1024},
			{HFBGESmall, 384},
			{HFBGEBase, 768},
			{HFBGELarge, 1024},
			{HFE5Small, 384},
			{HFE5Base, 768},
			{HFE5Large, 1024},
			{HFGTESmall, 384},
			{HFGTEBase, 768},
			{HFGTELarge, 1024},
		}

		for _, tt := range tests {
			e := NewHuggingFaceEmbedding(WithHuggingFaceModel(tt.model))
			info := e.Info()
			assert.Equal(t, tt.model, info.ModelName)
			assert.Equal(t, tt.dimensions, info.Dimensions)
		}
	})

	t.Run("SupportsMultiModal returns false", func(t *testing.T) {
		e := NewHuggingFaceEmbedding()
		assert.False(t, e.SupportsMultiModal())
	})

	t.Run("GetTextEmbedding with Inference API mock", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, "POST", r.Method)
			assert.Contains(t, r.URL.Path, "/pipeline/feature-extraction/")

			// Return simple embedding
			embedding := []float64{0.1, 0.2, 0.3, 0.4}
			json.NewEncoder(w).Encode(embedding)
		}))
		defer server.Close()

		e := NewHuggingFaceEmbedding(
			WithHuggingFaceBaseURL(server.URL),
		)

		embedding, err := e.GetTextEmbedding(context.Background(), "test text")
		require.NoError(t, err)
		assert.Equal(t, []float64{0.1, 0.2, 0.3, 0.4}, embedding)
	})

	t.Run("GetTextEmbedding with TEI mock", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, "POST", r.Method)
			assert.Equal(t, "/embed", r.URL.Path)

			var req teiEmbedRequest
			json.NewDecoder(r.Body).Decode(&req)

			embeddings := make([][]float64, len(req.Inputs))
			for i := range req.Inputs {
				embeddings[i] = []float64{float64(i), float64(i) * 0.1}
			}
			json.NewEncoder(w).Encode(embeddings)
		}))
		defer server.Close()

		e := NewHuggingFaceEmbedding(
			WithHuggingFaceBaseURL(server.URL),
			WithHuggingFaceTEI(true),
		)

		embedding, err := e.GetTextEmbedding(context.Background(), "test text")
		require.NoError(t, err)
		assert.Len(t, embedding, 2)
	})

	t.Run("GetQueryEmbedding adds prefix", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var req hfInferenceRequest
			json.NewDecoder(r.Body).Decode(&req)
			// Check that prefix was added
			assert.Equal(t, "query: test query", req.Inputs)

			json.NewEncoder(w).Encode([]float64{0.1, 0.2})
		}))
		defer server.Close()

		e := NewHuggingFaceEmbedding(
			WithHuggingFaceBaseURL(server.URL),
			WithHuggingFaceQueryPrefix("query: "),
		)

		_, err := e.GetQueryEmbedding(context.Background(), "test query")
		require.NoError(t, err)
	})
}

// TestAzureOpenAIEmbedding tests the Azure OpenAI embedding implementation.
func TestAzureOpenAIEmbedding(t *testing.T) {
	t.Run("NewAzureOpenAIEmbedding with defaults", func(t *testing.T) {
		e := NewAzureOpenAIEmbedding()
		assert.NotNil(t, e)
	})

	t.Run("NewAzureOpenAIEmbedding with options", func(t *testing.T) {
		e := NewAzureOpenAIEmbedding(
			WithAzureEmbeddingDeployment("text-embedding-ada-002"),
		)
		assert.Equal(t, "text-embedding-ada-002", e.deployment)
	})

	t.Run("NewAzureOpenAIEmbeddingWithConfig", func(t *testing.T) {
		e := NewAzureOpenAIEmbeddingWithConfig(
			"https://myresource.openai.azure.com",
			"my-api-key",
			"my-embedding-deployment",
		)
		assert.NotNil(t, e)
		assert.Equal(t, "my-embedding-deployment", e.deployment)
	})

	t.Run("Info returns correct values", func(t *testing.T) {
		e := NewAzureOpenAIEmbedding(WithAzureEmbeddingDeployment("text-embedding-ada-002"))
		info := e.Info()
		assert.Equal(t, "text-embedding-ada-002", info.ModelName)
		assert.Equal(t, 1536, info.Dimensions)
	})

	t.Run("SupportsMultiModal returns false", func(t *testing.T) {
		e := NewAzureOpenAIEmbedding()
		assert.False(t, e.SupportsMultiModal())
	})
}

// TestMeanPool tests the mean pooling function.
func TestMeanPool(t *testing.T) {
	t.Run("Mean pools token embeddings", func(t *testing.T) {
		tokens := [][]float64{
			{1.0, 2.0, 3.0},
			{2.0, 4.0, 6.0},
			{3.0, 6.0, 9.0},
		}
		result := meanPool(tokens)
		assert.Equal(t, []float64{2.0, 4.0, 6.0}, result)
	})

	t.Run("Handles single token", func(t *testing.T) {
		tokens := [][]float64{{1.0, 2.0, 3.0}}
		result := meanPool(tokens)
		assert.Equal(t, []float64{1.0, 2.0, 3.0}, result)
	})

	t.Run("Handles empty input", func(t *testing.T) {
		result := meanPool([][]float64{})
		assert.Nil(t, result)
	})
}

// TestEmbeddingInterfaceCompliance verifies all providers implement required interfaces.
func TestEmbeddingInterfaceCompliance(t *testing.T) {
	t.Run("OllamaEmbedding implements all interfaces", func(t *testing.T) {
		var _ EmbeddingModel = (*OllamaEmbedding)(nil)
		var _ EmbeddingModelWithInfo = (*OllamaEmbedding)(nil)
		var _ EmbeddingModelWithBatch = (*OllamaEmbedding)(nil)
		var _ MultiModalEmbeddingModel = (*OllamaEmbedding)(nil)
		var _ FullEmbeddingModel = (*OllamaEmbedding)(nil)
	})

	t.Run("CohereEmbedding implements all interfaces", func(t *testing.T) {
		var _ EmbeddingModel = (*CohereEmbedding)(nil)
		var _ EmbeddingModelWithInfo = (*CohereEmbedding)(nil)
		var _ EmbeddingModelWithBatch = (*CohereEmbedding)(nil)
		var _ MultiModalEmbeddingModel = (*CohereEmbedding)(nil)
		var _ FullEmbeddingModel = (*CohereEmbedding)(nil)
	})

	t.Run("HuggingFaceEmbedding implements all interfaces", func(t *testing.T) {
		var _ EmbeddingModel = (*HuggingFaceEmbedding)(nil)
		var _ EmbeddingModelWithInfo = (*HuggingFaceEmbedding)(nil)
		var _ EmbeddingModelWithBatch = (*HuggingFaceEmbedding)(nil)
		var _ MultiModalEmbeddingModel = (*HuggingFaceEmbedding)(nil)
		var _ FullEmbeddingModel = (*HuggingFaceEmbedding)(nil)
	})

	t.Run("AzureOpenAIEmbedding implements all interfaces", func(t *testing.T) {
		var _ EmbeddingModel = (*AzureOpenAIEmbedding)(nil)
		var _ EmbeddingModelWithInfo = (*AzureOpenAIEmbedding)(nil)
		var _ EmbeddingModelWithBatch = (*AzureOpenAIEmbedding)(nil)
		var _ MultiModalEmbeddingModel = (*AzureOpenAIEmbedding)(nil)
		var _ FullEmbeddingModel = (*AzureOpenAIEmbedding)(nil)
	})
}

// TestEmbeddingModelConstants verifies model constants are defined correctly.
func TestEmbeddingModelConstants(t *testing.T) {
	t.Run("Ollama embedding models", func(t *testing.T) {
		assert.Equal(t, "mxbai-embed-large", OllamaMxbaiEmbedLarge)
		assert.Equal(t, "all-minilm", OllamaAllMiniLM)
		assert.Equal(t, "nomic-embed-text", OllamaNomicEmbedText)
	})

	t.Run("Cohere embedding models", func(t *testing.T) {
		assert.Equal(t, "embed-english-v3.0", CohereEmbedEnglishV3)
		assert.Equal(t, "embed-multilingual-v3.0", CohereEmbedMultilingualV3)
		assert.Equal(t, "embed-english-light-v3.0", CohereEmbedEnglishLightV3)
	})

	t.Run("HuggingFace embedding models", func(t *testing.T) {
		assert.Equal(t, "sentence-transformers/all-MiniLM-L6-v2", HFSentenceTransformersMiniLM)
		assert.Equal(t, "BAAI/bge-large-en-v1.5", HFBGELarge)
		assert.Equal(t, "intfloat/e5-large-v2", HFE5Large)
	})
}
