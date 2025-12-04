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

// TestBedrockLLM tests the AWS Bedrock LLM implementation.
func TestBedrockLLM(t *testing.T) {
	t.Run("NewBedrockLLM with defaults", func(t *testing.T) {
		llm := NewBedrockLLM()
		assert.NotNil(t, llm)
		assert.Equal(t, DefaultBedrockModel, llm.model)
		assert.Equal(t, DefaultBedrockMaxTokens, llm.maxTokens)
	})

	t.Run("NewBedrockLLM with options", func(t *testing.T) {
		llm := NewBedrockLLM(
			WithBedrockModel(BedrockClaude3Haiku),
			WithBedrockMaxTokens(2048),
			WithBedrockTemperature(0.5),
			WithBedrockTopP(0.9),
			WithBedrockRegion("us-west-2"),
		)

		assert.Equal(t, BedrockClaude3Haiku, llm.model)
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
			{BedrockClaude35SonnetV2, 200000, true, true},
			{BedrockClaude3Haiku, 200000, true, true},
			{BedrockNovaProV1, 300000, true, true},
			{BedrockNovaMicroV1, 128000, true, false},
			{BedrockLlama33_70BInstruct, 128000, true, false},
			{BedrockMistral7BInstruct, 32000, false, false},
			{BedrockTitanTextExpressV1, 8192, false, false},
		}

		for _, tt := range tests {
			llm := NewBedrockLLM(WithBedrockModel(tt.model))
			meta := llm.Metadata()
			assert.Equal(t, tt.model, meta.ModelName)
			assert.Equal(t, tt.contextWindow, meta.ContextWindow, "model: %s", tt.model)
			assert.Equal(t, tt.toolCalling, meta.IsFunctionCalling, "model: %s", tt.model)
			assert.Equal(t, tt.multiModal, meta.IsMultiModal, "model: %s", tt.model)
		}
	})

	t.Run("SupportsToolCalling depends on model", func(t *testing.T) {
		llm := NewBedrockLLM(WithBedrockModel(BedrockClaude35SonnetV2))
		assert.True(t, llm.SupportsToolCalling())

		llm = NewBedrockLLM(WithBedrockModel(BedrockMistral7BInstruct))
		assert.False(t, llm.SupportsToolCalling())
	})

	t.Run("SupportsStructuredOutput depends on tool calling", func(t *testing.T) {
		llm := NewBedrockLLM(WithBedrockModel(BedrockClaude35SonnetV2))
		assert.True(t, llm.SupportsStructuredOutput())

		llm = NewBedrockLLM(WithBedrockModel(BedrockMistral7BInstruct))
		assert.False(t, llm.SupportsStructuredOutput())
	})

	t.Run("IsBedrockFunctionCallingModel helper", func(t *testing.T) {
		assert.True(t, IsBedrockFunctionCallingModel(BedrockClaude35SonnetV2))
		assert.True(t, IsBedrockFunctionCallingModel(BedrockNovaProV1))
		assert.True(t, IsBedrockFunctionCallingModel(BedrockLlama33_70BInstruct))
		assert.False(t, IsBedrockFunctionCallingModel(BedrockMistral7BInstruct))
		assert.False(t, IsBedrockFunctionCallingModel(BedrockTitanTextExpressV1))
	})

	t.Run("BedrockModelContextSize helper", func(t *testing.T) {
		assert.Equal(t, 200000, BedrockModelContextSize(BedrockClaude35SonnetV2))
		assert.Equal(t, 300000, BedrockModelContextSize(BedrockNovaProV1))
		assert.Equal(t, 128000, BedrockModelContextSize(BedrockLlama31_70BInstruct))
		assert.Equal(t, 32000, BedrockModelContextSize(BedrockMistral7BInstruct))
		assert.Equal(t, 128000, BedrockModelContextSize("unknown-model")) // default
	})

	t.Run("Region-prefixed models work correctly", func(t *testing.T) {
		// Test that us. prefixed models are handled correctly
		assert.True(t, IsBedrockFunctionCallingModel("us.anthropic.claude-3-5-sonnet-20241022-v2:0"))
		assert.Equal(t, 200000, BedrockModelContextSize("us.anthropic.claude-3-5-sonnet-20241022-v2:0"))

		// Test eu. prefix
		assert.True(t, IsBedrockFunctionCallingModel("eu.anthropic.claude-3-5-sonnet-20241022-v2:0"))

		// Test apac. prefix
		assert.True(t, IsBedrockFunctionCallingModel("apac.anthropic.claude-3-5-sonnet-20241022-v2:0"))
	})
}

// TestDeepSeekLLM tests the DeepSeek LLM implementation.
func TestDeepSeekLLM(t *testing.T) {
	t.Run("NewDeepSeekLLM with defaults", func(t *testing.T) {
		llm := NewDeepSeekLLM()
		assert.NotNil(t, llm)
		assert.Equal(t, DefaultDeepSeekModel, llm.model)
	})

	t.Run("NewDeepSeekLLM with options", func(t *testing.T) {
		llm := NewDeepSeekLLM(
			WithDeepSeekAPIKey("test-key"),
			WithDeepSeekModel(DeepSeekCoder),
		)

		assert.Equal(t, DeepSeekCoder, llm.model)
	})

	t.Run("Metadata returns correct values", func(t *testing.T) {
		tests := []struct {
			model         string
			contextWindow int
			toolCalling   bool
		}{
			{DeepSeekChat, 64000, true},
			{DeepSeekCoder, 128000, true},
			{DeepSeekReasoner, 64000, false},
		}

		for _, tt := range tests {
			llm := NewDeepSeekLLM(WithDeepSeekModel(tt.model))
			meta := llm.Metadata()
			assert.Equal(t, tt.model, meta.ModelName)
			assert.Equal(t, tt.contextWindow, meta.ContextWindow)
			assert.Equal(t, tt.toolCalling, meta.IsFunctionCalling, "model: %s", tt.model)
		}
	})

	t.Run("SupportsToolCalling depends on model", func(t *testing.T) {
		llm := NewDeepSeekLLM(WithDeepSeekModel(DeepSeekChat))
		assert.True(t, llm.SupportsToolCalling())

		llm = NewDeepSeekLLM(WithDeepSeekModel(DeepSeekReasoner))
		assert.False(t, llm.SupportsToolCalling())
	})

	t.Run("SupportsStructuredOutput returns true", func(t *testing.T) {
		llm := NewDeepSeekLLM()
		assert.True(t, llm.SupportsStructuredOutput())
	})

	t.Run("IsDeepSeekFunctionCallingModel helper", func(t *testing.T) {
		assert.True(t, IsDeepSeekFunctionCallingModel(DeepSeekChat))
		assert.True(t, IsDeepSeekFunctionCallingModel(DeepSeekCoder))
		assert.False(t, IsDeepSeekFunctionCallingModel(DeepSeekReasoner))
		assert.False(t, IsDeepSeekFunctionCallingModel("unknown-model"))
	})

	t.Run("DeepSeekModelContextSize helper", func(t *testing.T) {
		assert.Equal(t, 64000, DeepSeekModelContextSize(DeepSeekChat))
		assert.Equal(t, 128000, DeepSeekModelContextSize(DeepSeekCoder))
		assert.Equal(t, 64000, DeepSeekModelContextSize(DeepSeekReasoner))
		assert.Equal(t, 64000, DeepSeekModelContextSize("unknown-model")) // default
	})
}

// TestGroqLLM tests the Groq LLM implementation.
func TestGroqLLM(t *testing.T) {
	t.Run("NewGroqLLM with defaults", func(t *testing.T) {
		llm := NewGroqLLM()
		assert.NotNil(t, llm)
		assert.Equal(t, DefaultGroqModel, llm.model)
	})

	t.Run("NewGroqLLM with options", func(t *testing.T) {
		llm := NewGroqLLM(
			WithGroqAPIKey("test-key"),
			WithGroqModel(GroqLlama31_8B),
		)

		assert.Equal(t, GroqLlama31_8B, llm.model)
	})

	t.Run("Metadata returns correct values", func(t *testing.T) {
		tests := []struct {
			model         string
			contextWindow int
			toolCalling   bool
		}{
			{GroqLlama31_8B, 128000, true},
			{GroqLlama33_70B, 128000, true},
			{GroqMixtral8x7B, 32768, false},
			{GroqGemma2_9B, 8192, true},
			{GroqDeepSeekR1_70B, 128000, true},
			{GroqQwenQWQ32B, 32768, true},
		}

		for _, tt := range tests {
			llm := NewGroqLLM(WithGroqModel(tt.model))
			meta := llm.Metadata()
			assert.Equal(t, tt.model, meta.ModelName)
			assert.Equal(t, tt.contextWindow, meta.ContextWindow)
			assert.Equal(t, tt.toolCalling, meta.IsFunctionCalling, "model: %s", tt.model)
		}
	})

	t.Run("SupportsToolCalling depends on model", func(t *testing.T) {
		llm := NewGroqLLM(WithGroqModel(GroqLlama33_70B))
		assert.True(t, llm.SupportsToolCalling())

		llm = NewGroqLLM(WithGroqModel(GroqMixtral8x7B))
		assert.False(t, llm.SupportsToolCalling())
	})

	t.Run("SupportsStructuredOutput returns true", func(t *testing.T) {
		llm := NewGroqLLM()
		assert.True(t, llm.SupportsStructuredOutput())
	})

	t.Run("IsGroqFunctionCallingModel helper", func(t *testing.T) {
		assert.True(t, IsGroqFunctionCallingModel(GroqLlama33_70B))
		assert.True(t, IsGroqFunctionCallingModel(GroqLlama31_8B))
		assert.True(t, IsGroqFunctionCallingModel(GroqGemma2_9B))
		assert.False(t, IsGroqFunctionCallingModel(GroqMixtral8x7B))
		assert.False(t, IsGroqFunctionCallingModel("unknown-model"))
	})

	t.Run("GroqModelContextSize helper", func(t *testing.T) {
		assert.Equal(t, 128000, GroqModelContextSize(GroqLlama33_70B))
		assert.Equal(t, 32768, GroqModelContextSize(GroqMixtral8x7B))
		assert.Equal(t, 8192, GroqModelContextSize(GroqGemma2_9B))
		assert.Equal(t, 8192, GroqModelContextSize("unknown-model")) // default
	})
}

// TestMistralLLM tests the Mistral AI LLM implementation.
func TestMistralLLM(t *testing.T) {
	t.Run("NewMistralLLM with defaults", func(t *testing.T) {
		llm := NewMistralLLM()
		assert.NotNil(t, llm)
		assert.Equal(t, DefaultMistralModel, llm.model)
		assert.Equal(t, MistralAPIURL, llm.baseURL)
		assert.Equal(t, DefaultMistralMaxTokens, llm.maxTokens)
		assert.Equal(t, float32(0.1), llm.temperature)
		assert.Equal(t, float32(1.0), llm.topP)
		assert.False(t, llm.safeMode)
	})

	t.Run("NewMistralLLM with options", func(t *testing.T) {
		seed := 42
		llm := NewMistralLLM(
			WithMistralAPIKey("test-key"),
			WithMistralModel(MistralSmallLatest),
			WithMistralMaxTokens(2048),
			WithMistralTemperature(0.7),
			WithMistralTopP(0.9),
			WithMistralSafeMode(true),
			WithMistralRandomSeed(seed),
			WithMistralBaseURL("https://custom.mistral.ai"),
		)

		assert.Equal(t, "test-key", llm.apiKey)
		assert.Equal(t, MistralSmallLatest, llm.model)
		assert.Equal(t, 2048, llm.maxTokens)
		assert.Equal(t, float32(0.7), llm.temperature)
		assert.Equal(t, float32(0.9), llm.topP)
		assert.True(t, llm.safeMode)
		assert.Equal(t, &seed, llm.randomSeed)
		assert.Equal(t, "https://custom.mistral.ai", llm.baseURL)
	})

	t.Run("Metadata returns correct values", func(t *testing.T) {
		tests := []struct {
			model         string
			contextWindow int
			toolCalling   bool
			multiModal    bool
		}{
			{MistralSmallLatest, 32000, true, false},
			{MistralLargeLatest, 131000, true, false},
			{CodestralLatest, 256000, true, false},
			{PixtralLargeLatest, 131000, true, true},
			{Ministral3BLatest, 131000, true, false},
			{Ministral8BLatest, 131000, true, false},
			{MistralTiny, 32000, false, false},
		}

		for _, tt := range tests {
			llm := NewMistralLLM(WithMistralModel(tt.model))
			meta := llm.Metadata()
			assert.Equal(t, tt.model, meta.ModelName)
			assert.Equal(t, tt.contextWindow, meta.ContextWindow)
			assert.Equal(t, tt.toolCalling, meta.IsFunctionCalling, "model: %s", tt.model)
			assert.Equal(t, tt.multiModal, meta.IsMultiModal, "model: %s", tt.model)
		}
	})

	t.Run("SupportsToolCalling depends on model", func(t *testing.T) {
		llm := NewMistralLLM(WithMistralModel(MistralLargeLatest))
		assert.True(t, llm.SupportsToolCalling())

		llm = NewMistralLLM(WithMistralModel(MistralTiny))
		assert.False(t, llm.SupportsToolCalling())
	})

	t.Run("SupportsStructuredOutput returns true", func(t *testing.T) {
		llm := NewMistralLLM()
		assert.True(t, llm.SupportsStructuredOutput())
	})

	t.Run("Complete with mock server", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, "POST", r.Method)
			assert.Equal(t, "/chat/completions", r.URL.Path)
			assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
			assert.Equal(t, "Bearer test-api-key", r.Header.Get("Authorization"))

			resp := mistralResponse{
				ID:     "chat_123",
				Object: "chat.completion",
				Model:  "mistral-large-latest",
				Choices: []struct {
					Index        int            `json:"index"`
					Message      mistralMessage `json:"message"`
					FinishReason string         `json:"finish_reason"`
				}{
					{
						Index:        0,
						Message:      mistralMessage{Role: "assistant", Content: "Hello from Mistral!"},
						FinishReason: "stop",
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}))
		defer server.Close()

		llm := NewMistralLLM(
			WithMistralAPIKey("test-api-key"),
			WithMistralBaseURL(server.URL),
		)

		result, err := llm.Complete(context.Background(), "Hello")
		require.NoError(t, err)
		assert.Equal(t, "Hello from Mistral!", result)
	})

	t.Run("Chat with mock server", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var req mistralRequest
			json.NewDecoder(r.Body).Decode(&req)

			// Verify messages are converted correctly
			assert.Len(t, req.Messages, 2)
			assert.Equal(t, "system", req.Messages[0].Role)
			assert.Equal(t, "user", req.Messages[1].Role)

			resp := mistralResponse{
				Choices: []struct {
					Index        int            `json:"index"`
					Message      mistralMessage `json:"message"`
					FinishReason string         `json:"finish_reason"`
				}{
					{
						Index:        0,
						Message:      mistralMessage{Role: "assistant", Content: "I can help you!"},
						FinishReason: "stop",
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}))
		defer server.Close()

		llm := NewMistralLLM(
			WithMistralAPIKey("test-key"),
			WithMistralBaseURL(server.URL),
		)

		messages := []ChatMessage{
			NewSystemMessage("You are helpful"),
			NewUserMessage("Help me"),
		}

		result, err := llm.Chat(context.Background(), messages)
		require.NoError(t, err)
		assert.Equal(t, "I can help you!", result)
	})

	t.Run("ChatWithTools with mock server", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var req mistralRequest
			json.NewDecoder(r.Body).Decode(&req)

			// Verify tools are included
			assert.Len(t, req.Tools, 1)
			assert.Equal(t, "function", req.Tools[0].Type)
			assert.Equal(t, "get_weather", req.Tools[0].Function.Name)

			resp := mistralResponse{
				Choices: []struct {
					Index        int            `json:"index"`
					Message      mistralMessage `json:"message"`
					FinishReason string         `json:"finish_reason"`
				}{
					{
						Index: 0,
						Message: mistralMessage{
							Role:    "assistant",
							Content: "",
							ToolCalls: []mistralTool{
								{
									ID:   "call_123",
									Type: "function",
									Function: mistralFunctionCall{
										Name:      "get_weather",
										Arguments: `{"location": "Paris"}`,
									},
								},
							},
						},
						FinishReason: "tool_calls",
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}))
		defer server.Close()

		llm := NewMistralLLM(
			WithMistralAPIKey("test-key"),
			WithMistralBaseURL(server.URL),
		)

		tools := []*ToolMetadata{
			{
				Name:        "get_weather",
				Description: "Get the weather for a location",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"location": map[string]interface{}{
							"type":        "string",
							"description": "The city name",
						},
					},
					"required": []string{"location"},
				},
			},
		}

		messages := []ChatMessage{
			NewUserMessage("What's the weather in Paris?"),
		}

		resp, err := llm.ChatWithTools(context.Background(), messages, tools, nil)
		require.NoError(t, err)
		assert.NotNil(t, resp.Message)
		assert.True(t, resp.Message.HasToolCalls())

		toolCalls := resp.Message.GetToolCalls()
		assert.Len(t, toolCalls, 1)
		assert.Equal(t, "get_weather", toolCalls[0].Name)
		assert.Equal(t, "call_123", toolCalls[0].ID)
	})

	t.Run("convertMessages handles all roles", func(t *testing.T) {
		llm := NewMistralLLM()
		messages := []ChatMessage{
			NewSystemMessage("System prompt"),
			NewUserMessage("User message"),
			NewAssistantMessage("Assistant response"),
			NewToolMessage("tool_123", "Tool result"),
		}

		mistralMsgs := llm.convertMessages(messages)
		assert.Len(t, mistralMsgs, 4)
		assert.Equal(t, "system", mistralMsgs[0].Role)
		assert.Equal(t, "user", mistralMsgs[1].Role)
		assert.Equal(t, "assistant", mistralMsgs[2].Role)
		assert.Equal(t, "tool", mistralMsgs[3].Role)
		assert.Equal(t, "tool_123", mistralMsgs[3].ToolCallID)
	})

	t.Run("IsMistralFunctionCallingModel helper", func(t *testing.T) {
		assert.True(t, IsMistralFunctionCallingModel(MistralLargeLatest))
		assert.True(t, IsMistralFunctionCallingModel(MistralSmallLatest))
		assert.True(t, IsMistralFunctionCallingModel(CodestralLatest))
		assert.False(t, IsMistralFunctionCallingModel(MistralTiny))
		assert.False(t, IsMistralFunctionCallingModel("unknown-model"))
	})

	t.Run("MistralModelContextSize helper", func(t *testing.T) {
		assert.Equal(t, 131000, MistralModelContextSize(MistralLargeLatest))
		assert.Equal(t, 256000, MistralModelContextSize(CodestralLatest))
		assert.Equal(t, 32000, MistralModelContextSize(MistralTiny))
		assert.Equal(t, 32000, MistralModelContextSize("unknown-model")) // default
	})

	t.Run("API error handling", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusUnauthorized)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"error": map[string]string{
					"message": "Invalid API key",
					"type":    "authentication_error",
				},
			})
		}))
		defer server.Close()

		llm := NewMistralLLM(
			WithMistralAPIKey("invalid-key"),
			WithMistralBaseURL(server.URL),
		)

		_, err := llm.Complete(context.Background(), "Hello")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "Invalid API key")
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

	t.Run("MistralLLM implements all interfaces", func(t *testing.T) {
		var _ LLM = (*MistralLLM)(nil)
		var _ LLMWithMetadata = (*MistralLLM)(nil)
		var _ LLMWithToolCalling = (*MistralLLM)(nil)
		var _ LLMWithStructuredOutput = (*MistralLLM)(nil)
		var _ FullLLM = (*MistralLLM)(nil)
	})

	t.Run("GroqLLM implements all interfaces", func(t *testing.T) {
		var _ LLM = (*GroqLLM)(nil)
		var _ LLMWithMetadata = (*GroqLLM)(nil)
		var _ LLMWithToolCalling = (*GroqLLM)(nil)
		var _ LLMWithStructuredOutput = (*GroqLLM)(nil)
		var _ FullLLM = (*GroqLLM)(nil)
	})

	t.Run("DeepSeekLLM implements all interfaces", func(t *testing.T) {
		var _ LLM = (*DeepSeekLLM)(nil)
		var _ LLMWithMetadata = (*DeepSeekLLM)(nil)
		var _ LLMWithToolCalling = (*DeepSeekLLM)(nil)
		var _ LLMWithStructuredOutput = (*DeepSeekLLM)(nil)
		var _ FullLLM = (*DeepSeekLLM)(nil)
	})

	t.Run("BedrockLLM implements all interfaces", func(t *testing.T) {
		var _ LLM = (*BedrockLLM)(nil)
		var _ LLMWithMetadata = (*BedrockLLM)(nil)
		var _ LLMWithToolCalling = (*BedrockLLM)(nil)
		var _ LLMWithStructuredOutput = (*BedrockLLM)(nil)
		var _ FullLLM = (*BedrockLLM)(nil)
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

	t.Run("Mistral models", func(t *testing.T) {
		assert.Equal(t, "mistral-tiny", MistralTiny)
		assert.Equal(t, "mistral-small", MistralSmall)
		assert.Equal(t, "mistral-medium", MistralMedium)
		assert.Equal(t, "mistral-small-latest", MistralSmallLatest)
		assert.Equal(t, "mistral-large-latest", MistralLargeLatest)
		assert.Equal(t, "codestral-latest", CodestralLatest)
		assert.Equal(t, "pixtral-large-latest", PixtralLargeLatest)
		assert.Equal(t, "ministral-3b-latest", Ministral3BLatest)
		assert.Equal(t, "ministral-8b-latest", Ministral8BLatest)
		assert.Equal(t, "open-mistral-nemo", OpenMistralNemo)
	})

	t.Run("Groq models", func(t *testing.T) {
		assert.Equal(t, "llama-3.1-8b-instant", GroqLlama31_8B)
		assert.Equal(t, "llama-3.3-70b-versatile", GroqLlama33_70B)
		assert.Equal(t, "mixtral-8x7b-32768", GroqMixtral8x7B)
		assert.Equal(t, "gemma2-9b-it", GroqGemma2_9B)
		assert.Equal(t, "deepseek-r1-distill-llama-70b", GroqDeepSeekR1_70B)
		assert.Equal(t, "qwen-qwq-32b", GroqQwenQWQ32B)
	})

	t.Run("DeepSeek models", func(t *testing.T) {
		assert.Equal(t, "deepseek-chat", DeepSeekChat)
		assert.Equal(t, "deepseek-reasoner", DeepSeekReasoner)
		assert.Equal(t, "deepseek-coder", DeepSeekCoder)
	})

	t.Run("Bedrock models", func(t *testing.T) {
		// Claude models
		assert.Equal(t, "anthropic.claude-3-5-sonnet-20241022-v2:0", BedrockClaude35SonnetV2)
		assert.Equal(t, "anthropic.claude-3-haiku-20240307-v1:0", BedrockClaude3Haiku)
		assert.Equal(t, "anthropic.claude-3-opus-20240229-v1:0", BedrockClaude3Opus)
		// Amazon Nova models
		assert.Equal(t, "amazon.nova-pro-v1:0", BedrockNovaProV1)
		assert.Equal(t, "amazon.nova-lite-v1:0", BedrockNovaLiteV1)
		// Meta Llama models
		assert.Equal(t, "meta.llama3-3-70b-instruct-v1:0", BedrockLlama33_70BInstruct)
		// Mistral models
		assert.Equal(t, "mistral.mistral-7b-instruct-v0:2", BedrockMistral7BInstruct)
	})
}
