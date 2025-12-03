package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestAnthropicLLM tests the Anthropic LLM implementation.
func TestAnthropicLLM(t *testing.T) {
	t.Run("NewAnthropicLLM with defaults", func(t *testing.T) {
		llm := NewAnthropicLLM()
		assert.NotNil(t, llm)
		assert.Equal(t, Claude35Sonnet, llm.model)
		assert.Equal(t, AnthropicAPIURL, llm.baseURL)
		assert.Equal(t, 4096, llm.maxTokens)
	})

	t.Run("NewAnthropicLLM with options", func(t *testing.T) {
		llm := NewAnthropicLLM(
			WithAnthropicAPIKey("test-key"),
			WithAnthropicModel(Claude3Opus),
			WithAnthropicMaxTokens(8192),
			WithAnthropicBaseURL("https://custom.api.com"),
		)
		assert.Equal(t, "test-key", llm.apiKey)
		assert.Equal(t, Claude3Opus, llm.model)
		assert.Equal(t, 8192, llm.maxTokens)
		assert.Equal(t, "https://custom.api.com", llm.baseURL)
	})

	t.Run("Metadata returns correct values", func(t *testing.T) {
		tests := []struct {
			model         string
			contextWindow int
		}{
			{Claude3Opus, 200000},
			{Claude35Sonnet, 200000},
			{Claude3Haiku, 200000},
		}

		for _, tt := range tests {
			llm := NewAnthropicLLM(WithAnthropicModel(tt.model))
			meta := llm.Metadata()
			assert.Equal(t, tt.model, meta.ModelName)
			assert.Equal(t, tt.contextWindow, meta.ContextWindow)
			assert.True(t, meta.IsFunctionCalling)
			assert.True(t, meta.IsMultiModal)
		}
	})

	t.Run("SupportsToolCalling returns true", func(t *testing.T) {
		llm := NewAnthropicLLM()
		assert.True(t, llm.SupportsToolCalling())
	})

	t.Run("SupportsStructuredOutput returns true", func(t *testing.T) {
		llm := NewAnthropicLLM()
		assert.True(t, llm.SupportsStructuredOutput())
	})

	t.Run("Complete with mock server", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, "POST", r.Method)
			assert.Equal(t, "/messages", r.URL.Path)
			assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
			assert.Equal(t, "test-api-key", r.Header.Get("x-api-key"))
			assert.Equal(t, AnthropicAPIVersion, r.Header.Get("anthropic-version"))

			resp := anthropicResponse{
				ID:   "msg_123",
				Type: "message",
				Role: "assistant",
				Content: []anthropicContent{
					{Type: "text", Text: "Hello, I'm Claude!"},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}))
		defer server.Close()

		llm := NewAnthropicLLM(
			WithAnthropicAPIKey("test-api-key"),
			WithAnthropicBaseURL(server.URL),
		)

		result, err := llm.Complete(context.Background(), "Hello")
		require.NoError(t, err)
		assert.Equal(t, "Hello, I'm Claude!", result)
	})

	t.Run("Chat with mock server", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var req anthropicRequest
			json.NewDecoder(r.Body).Decode(&req)

			// Verify system message is extracted
			assert.Equal(t, "You are helpful", req.System)
			assert.Len(t, req.Messages, 1)
			assert.Equal(t, "user", req.Messages[0].Role)

			resp := anthropicResponse{
				Content: []anthropicContent{
					{Type: "text", Text: "I can help you!"},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}))
		defer server.Close()

		llm := NewAnthropicLLM(
			WithAnthropicAPIKey("test-key"),
			WithAnthropicBaseURL(server.URL),
		)

		messages := []ChatMessage{
			NewSystemMessage("You are helpful"),
			NewUserMessage("Help me"),
		}

		result, err := llm.Chat(context.Background(), messages)
		require.NoError(t, err)
		assert.Equal(t, "I can help you!", result)
	})

	t.Run("convertMessages handles all roles", func(t *testing.T) {
		llm := NewAnthropicLLM()
		messages := []ChatMessage{
			NewSystemMessage("System prompt"),
			NewUserMessage("User message"),
			NewAssistantMessage("Assistant response"),
		}

		anthropicMsgs, systemPrompt := llm.convertMessages(messages)
		assert.Equal(t, "System prompt", systemPrompt)
		assert.Len(t, anthropicMsgs, 2)
		assert.Equal(t, "user", anthropicMsgs[0].Role)
		assert.Equal(t, "assistant", anthropicMsgs[1].Role)
	})
}

// TestOllamaLLM tests the Ollama LLM implementation.
func TestOllamaLLM(t *testing.T) {
	t.Run("NewOllamaLLM with defaults", func(t *testing.T) {
		llm := NewOllamaLLM()
		assert.NotNil(t, llm)
		assert.Equal(t, OllamaLlama31, llm.model)
		assert.Equal(t, OllamaDefaultURL, llm.baseURL)
	})

	t.Run("NewOllamaLLM with options", func(t *testing.T) {
		temp := float32(0.7)
		topP := float32(0.9)
		topK := 40
		numPredict := 2048
		numCtx := 4096
		seed := 42

		llm := NewOllamaLLM(
			WithOllamaModel(OllamaMistral),
			WithOllamaBaseURL("http://custom:11434"),
			WithOllamaTemperature(temp),
			WithOllamaTopP(topP),
			WithOllamaTopK(topK),
			WithOllamaNumPredict(numPredict),
			WithOllamaNumCtx(numCtx),
			WithOllamaSeed(seed),
			WithOllamaStop([]string{"END"}),
		)

		assert.Equal(t, OllamaMistral, llm.model)
		assert.Equal(t, "http://custom:11434", llm.baseURL)
		assert.Equal(t, &temp, llm.temperature)
		assert.Equal(t, &topP, llm.topP)
		assert.Equal(t, &topK, llm.topK)
		assert.Equal(t, &numPredict, llm.numPredict)
		assert.Equal(t, &numCtx, llm.numCtx)
		assert.Equal(t, &seed, llm.seed)
		assert.Equal(t, []string{"END"}, llm.stop)
	})

	t.Run("Metadata returns correct values", func(t *testing.T) {
		tests := []struct {
			model         string
			contextWindow int
			toolCalling   bool
		}{
			{OllamaLlama31, 128000, true},
			{OllamaMistral, 32768, true},
			{OllamaLlama2, 4096, false},
			{OllamaCodeLlama, 16384, false},
		}

		for _, tt := range tests {
			llm := NewOllamaLLM(WithOllamaModel(tt.model))
			meta := llm.Metadata()
			assert.Equal(t, tt.model, meta.ModelName)
			assert.Equal(t, tt.contextWindow, meta.ContextWindow)
			assert.Equal(t, tt.toolCalling, meta.IsFunctionCalling)
		}
	})

	t.Run("SupportsToolCalling depends on model", func(t *testing.T) {
		llm := NewOllamaLLM(WithOllamaModel(OllamaLlama31))
		assert.True(t, llm.SupportsToolCalling())

		llm = NewOllamaLLM(WithOllamaModel(OllamaLlama2))
		assert.False(t, llm.SupportsToolCalling())
	})

	t.Run("buildOptions includes all set options", func(t *testing.T) {
		temp := float32(0.5)
		topP := float32(0.8)
		topK := 50
		numPredict := 1024

		llm := NewOllamaLLM(
			WithOllamaTemperature(temp),
			WithOllamaTopP(topP),
			WithOllamaTopK(topK),
			WithOllamaNumPredict(numPredict),
		)

		opts := llm.buildOptions()
		assert.Equal(t, temp, opts["temperature"])
		assert.Equal(t, topP, opts["top_p"])
		assert.Equal(t, topK, opts["top_k"])
		assert.Equal(t, numPredict, opts["num_predict"])
	})

	t.Run("Complete with mock server", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, "POST", r.Method)
			assert.Equal(t, "/api/generate", r.URL.Path)

			resp := ollamaGenerateResponse{
				Model:    "llama3.1",
				Response: "Hello from Ollama!",
				Done:     true,
			}
			json.NewEncoder(w).Encode(resp)
		}))
		defer server.Close()

		llm := NewOllamaLLM(WithOllamaBaseURL(server.URL))

		result, err := llm.Complete(context.Background(), "Hello")
		require.NoError(t, err)
		assert.Equal(t, "Hello from Ollama!", result)
	})

	t.Run("Chat with mock server", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, "/api/chat", r.URL.Path)

			resp := ollamaChatResponse{
				Model: "llama3.1",
				Message: ollamaMessage{
					Role:    "assistant",
					Content: "Chat response from Ollama!",
				},
				Done: true,
			}
			json.NewEncoder(w).Encode(resp)
		}))
		defer server.Close()

		llm := NewOllamaLLM(WithOllamaBaseURL(server.URL))

		messages := []ChatMessage{NewUserMessage("Hello")}
		result, err := llm.Chat(context.Background(), messages)
		require.NoError(t, err)
		assert.Equal(t, "Chat response from Ollama!", result)
	})
}

// TestCohereLLM tests the Cohere LLM implementation.
func TestCohereLLM(t *testing.T) {
	t.Run("NewCohereLLM with defaults", func(t *testing.T) {
		llm := NewCohereLLM()
		assert.NotNil(t, llm)
		assert.Equal(t, CohereCommandRPlus, llm.model)
		assert.Equal(t, CohereAPIURL, llm.baseURL)
		assert.Equal(t, 4096, llm.maxTokens)
	})

	t.Run("NewCohereLLM with options", func(t *testing.T) {
		temp := float32(0.7)
		llm := NewCohereLLM(
			WithCohereAPIKey("test-key"),
			WithCohereModel(CohereCommandR),
			WithCohereMaxTokens(2048),
			WithCohereTemperature(temp),
			WithCohereBaseURL("https://custom.cohere.ai"),
		)

		assert.Equal(t, "test-key", llm.apiKey)
		assert.Equal(t, CohereCommandR, llm.model)
		assert.Equal(t, 2048, llm.maxTokens)
		assert.Equal(t, &temp, llm.temperature)
		assert.Equal(t, "https://custom.cohere.ai", llm.baseURL)
	})

	t.Run("Metadata returns correct values", func(t *testing.T) {
		tests := []struct {
			model         string
			contextWindow int
			toolCalling   bool
		}{
			{CohereCommandRPlus, 128000, true},
			{CohereCommandR, 128000, true},
			{CohereCommand, 4096, false},
		}

		for _, tt := range tests {
			llm := NewCohereLLM(WithCohereModel(tt.model))
			meta := llm.Metadata()
			assert.Equal(t, tt.model, meta.ModelName)
			assert.Equal(t, tt.contextWindow, meta.ContextWindow)
			assert.Equal(t, tt.toolCalling, meta.IsFunctionCalling)
		}
	})

	t.Run("SupportsToolCalling depends on model", func(t *testing.T) {
		llm := NewCohereLLM(WithCohereModel(CohereCommandRPlus))
		assert.True(t, llm.SupportsToolCalling())

		llm = NewCohereLLM(WithCohereModel(CohereCommand))
		assert.False(t, llm.SupportsToolCalling())
	})

	t.Run("Complete with mock server", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, "POST", r.Method)
			assert.Equal(t, "/generate", r.URL.Path)
			assert.Contains(t, r.Header.Get("Authorization"), "Bearer")

			resp := cohereGenerateResponse{
				ID: "gen_123",
				Generations: []struct {
					ID   string `json:"id"`
					Text string `json:"text"`
				}{
					{ID: "0", Text: "Hello from Cohere!"},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}))
		defer server.Close()

		llm := NewCohereLLM(
			WithCohereAPIKey("test-key"),
			WithCohereBaseURL(server.URL),
		)

		result, err := llm.Complete(context.Background(), "Hello")
		require.NoError(t, err)
		assert.Equal(t, "Hello from Cohere!", result)
	})

	t.Run("Chat with mock server", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, "/chat", r.URL.Path)

			var req cohereChatRequest
			json.NewDecoder(r.Body).Decode(&req)
			assert.Equal(t, "System prompt", req.Preamble)

			resp := cohereChatResponse{
				ResponseID: "resp_123",
				Text:       "Chat response from Cohere!",
			}
			json.NewEncoder(w).Encode(resp)
		}))
		defer server.Close()

		llm := NewCohereLLM(
			WithCohereAPIKey("test-key"),
			WithCohereBaseURL(server.URL),
		)

		messages := []ChatMessage{
			NewSystemMessage("System prompt"),
			NewUserMessage("Hello"),
		}
		result, err := llm.Chat(context.Background(), messages)
		require.NoError(t, err)
		assert.Equal(t, "Chat response from Cohere!", result)
	})

	t.Run("convertMessages extracts preamble", func(t *testing.T) {
		llm := NewCohereLLM()
		messages := []ChatMessage{
			NewSystemMessage("System prompt"),
			NewUserMessage("First message"),
			NewAssistantMessage("Response"),
			NewUserMessage("Second message"),
		}

		history, current, preamble := llm.convertMessages(messages)
		assert.Equal(t, "System prompt", preamble)
		assert.Equal(t, "Second message", current)
		assert.Len(t, history, 2)
		assert.Equal(t, "USER", history[0].Role)
		assert.Equal(t, "CHATBOT", history[1].Role)
	})
}

// TestAzureOpenAILLM tests the Azure OpenAI LLM implementation.
func TestAzureOpenAILLM(t *testing.T) {
	t.Run("NewAzureOpenAILLM with defaults", func(t *testing.T) {
		llm := NewAzureOpenAILLM()
		assert.NotNil(t, llm)
		assert.Equal(t, "2024-02-15-preview", llm.apiVersion)
	})

	t.Run("NewAzureOpenAILLM with options", func(t *testing.T) {
		llm := NewAzureOpenAILLM(
			WithAzureDeployment("gpt-4-deployment"),
			WithAzureAPIVersion("2024-01-01"),
		)

		assert.Equal(t, "gpt-4-deployment", llm.model)
		assert.Equal(t, "2024-01-01", llm.apiVersion)
	})

	t.Run("NewAzureOpenAILLMWithConfig", func(t *testing.T) {
		llm := NewAzureOpenAILLMWithConfig(
			"https://myresource.openai.azure.com",
			"my-api-key",
			"gpt-4",
			"2024-02-01",
		)

		assert.NotNil(t, llm)
		assert.Equal(t, "gpt-4", llm.model)
		assert.Equal(t, "2024-02-01", llm.apiVersion)
	})

	t.Run("Metadata returns correct values", func(t *testing.T) {
		llm := NewAzureOpenAILLM(WithAzureDeployment("gpt-4"))
		meta := llm.Metadata()
		assert.Equal(t, "gpt-4", meta.ModelName)
		assert.True(t, meta.IsFunctionCalling)
		assert.True(t, meta.IsMultiModal)
	})

	t.Run("SupportsToolCalling returns true", func(t *testing.T) {
		llm := NewAzureOpenAILLM()
		assert.True(t, llm.SupportsToolCalling())
	})

	t.Run("SupportsStructuredOutput returns true", func(t *testing.T) {
		llm := NewAzureOpenAILLM()
		assert.True(t, llm.SupportsStructuredOutput())
	})
}

// TestInterfaceCompliance verifies all providers implement required interfaces.
func TestLLMInterfaceCompliance(t *testing.T) {
	t.Run("AnthropicLLM implements all interfaces", func(t *testing.T) {
		var _ LLM = (*AnthropicLLM)(nil)
		var _ LLMWithMetadata = (*AnthropicLLM)(nil)
		var _ LLMWithToolCalling = (*AnthropicLLM)(nil)
		var _ LLMWithStructuredOutput = (*AnthropicLLM)(nil)
		var _ FullLLM = (*AnthropicLLM)(nil)
	})

	t.Run("OllamaLLM implements all interfaces", func(t *testing.T) {
		var _ LLM = (*OllamaLLM)(nil)
		var _ LLMWithMetadata = (*OllamaLLM)(nil)
		var _ LLMWithToolCalling = (*OllamaLLM)(nil)
		var _ LLMWithStructuredOutput = (*OllamaLLM)(nil)
		var _ FullLLM = (*OllamaLLM)(nil)
	})

	t.Run("CohereLLM implements all interfaces", func(t *testing.T) {
		var _ LLM = (*CohereLLM)(nil)
		var _ LLMWithMetadata = (*CohereLLM)(nil)
		var _ LLMWithToolCalling = (*CohereLLM)(nil)
		var _ LLMWithStructuredOutput = (*CohereLLM)(nil)
		var _ FullLLM = (*CohereLLM)(nil)
	})

	t.Run("AzureOpenAILLM implements all interfaces", func(t *testing.T) {
		var _ LLM = (*AzureOpenAILLM)(nil)
		var _ LLMWithMetadata = (*AzureOpenAILLM)(nil)
		var _ LLMWithToolCalling = (*AzureOpenAILLM)(nil)
		var _ LLMWithStructuredOutput = (*AzureOpenAILLM)(nil)
		var _ FullLLM = (*AzureOpenAILLM)(nil)
	})
}

// TestModelConstants verifies model constants are defined correctly.
func TestModelConstants(t *testing.T) {
	t.Run("Anthropic models", func(t *testing.T) {
		assert.Equal(t, "claude-3-opus-20240229", Claude3Opus)
		assert.Equal(t, "claude-3-sonnet-20240229", Claude3Sonnet)
		assert.Equal(t, "claude-3-haiku-20240307", Claude3Haiku)
		assert.Equal(t, "claude-3-5-sonnet-20241022", Claude35Sonnet)
		assert.Equal(t, "claude-3-5-haiku-20241022", Claude35Haiku)
	})

	t.Run("Ollama models", func(t *testing.T) {
		assert.Equal(t, "llama2", OllamaLlama2)
		assert.Equal(t, "llama3", OllamaLlama3)
		assert.Equal(t, "llama3.1", OllamaLlama31)
		assert.Equal(t, "mistral", OllamaMistral)
		assert.Equal(t, "codellama", OllamaCodeLlama)
	})

	t.Run("Cohere models", func(t *testing.T) {
		assert.Equal(t, "command", CohereCommand)
		assert.Equal(t, "command-light", CohereCommandLight)
		assert.Equal(t, "command-r", CohereCommandR)
		assert.Equal(t, "command-r-plus", CohereCommandRPlus)
	})
}
