package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
)

const (
	// OllamaDefaultURL is the default Ollama API endpoint.
	OllamaDefaultURL = "http://localhost:11434"
)

// Common Ollama embedding model names.
const (
	OllamaMxbaiEmbedLarge = "mxbai-embed-large"
	OllamaAllMiniLM       = "all-minilm"
	OllamaNomicEmbedText  = "nomic-embed-text"
	OllamaSnowflakeArctic = "snowflake-arctic-embed"
	OllamaBgeSmall        = "bge-small"
	OllamaBgeLarge        = "bge-large"
)

// OllamaEmbedding implements the EmbeddingModel interface for Ollama.
type OllamaEmbedding struct {
	baseURL    string
	model      string
	httpClient *http.Client
	logger     *slog.Logger
}

// OllamaEmbeddingOption configures an OllamaEmbedding.
type OllamaEmbeddingOption func(*OllamaEmbedding)

// WithOllamaEmbeddingBaseURL sets the base URL.
func WithOllamaEmbeddingBaseURL(baseURL string) OllamaEmbeddingOption {
	return func(o *OllamaEmbedding) {
		o.baseURL = baseURL
	}
}

// WithOllamaEmbeddingModel sets the model.
func WithOllamaEmbeddingModel(model string) OllamaEmbeddingOption {
	return func(o *OllamaEmbedding) {
		o.model = model
	}
}

// WithOllamaEmbeddingHTTPClient sets a custom HTTP client.
func WithOllamaEmbeddingHTTPClient(client *http.Client) OllamaEmbeddingOption {
	return func(o *OllamaEmbedding) {
		o.httpClient = client
	}
}

// NewOllamaEmbedding creates a new Ollama embedding client.
func NewOllamaEmbedding(opts ...OllamaEmbeddingOption) *OllamaEmbedding {
	baseURL := os.Getenv("OLLAMA_HOST")
	if baseURL == "" {
		baseURL = OllamaDefaultURL
	}

	o := &OllamaEmbedding{
		baseURL:    baseURL,
		model:      OllamaNomicEmbedText,
		httpClient: http.DefaultClient,
		logger:     slog.New(slog.NewJSONHandler(os.Stdout, nil)),
	}

	for _, opt := range opts {
		opt(o)
	}

	return o
}

// ollamaEmbeddingRequest represents a request to the Ollama embedding API.
type ollamaEmbeddingRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

// ollamaEmbeddingResponse represents a response from the Ollama embedding API.
type ollamaEmbeddingResponse struct {
	Embedding []float64 `json:"embedding"`
}

// GetTextEmbedding generates an embedding for a given text.
func (o *OllamaEmbedding) GetTextEmbedding(ctx context.Context, text string) ([]float64, error) {
	return o.getEmbedding(ctx, text)
}

// GetQueryEmbedding generates an embedding for a given query.
func (o *OllamaEmbedding) GetQueryEmbedding(ctx context.Context, query string) ([]float64, error) {
	return o.getEmbedding(ctx, query)
}

// getEmbedding performs the actual embedding request.
func (o *OllamaEmbedding) getEmbedding(ctx context.Context, text string) ([]float64, error) {
	reqBody := ollamaEmbeddingRequest{
		Model:  o.model,
		Prompt: text,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", o.baseURL+"/api/embeddings", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := o.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama API error (%d): %s", resp.StatusCode, string(respBody))
	}

	var result ollamaEmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result.Embedding, nil
}

// Info returns information about the model's capabilities.
func (o *OllamaEmbedding) Info() EmbeddingInfo {
	return getOllamaEmbeddingInfo(o.model)
}

// GetTextEmbeddingsBatch generates embeddings for multiple texts.
func (o *OllamaEmbedding) GetTextEmbeddingsBatch(ctx context.Context, texts []string, callback ProgressCallback) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	o.logger.Info("GetTextEmbeddingsBatch called", "model", o.model, "count", len(texts))

	results := make([][]float64, len(texts))
	for i, text := range texts {
		embedding, err := o.getEmbedding(ctx, text)
		if err != nil {
			return nil, fmt.Errorf("failed to get embedding for text %d: %w", i, err)
		}
		results[i] = embedding

		if callback != nil {
			callback(i+1, len(texts))
		}
	}

	return results, nil
}

// SupportsMultiModal returns false as Ollama embedding models don't support images.
func (o *OllamaEmbedding) SupportsMultiModal() bool {
	return false
}

// GetImageEmbedding is not supported by Ollama embedding models.
func (o *OllamaEmbedding) GetImageEmbedding(ctx context.Context, image ImageType) ([]float64, error) {
	return nil, fmt.Errorf("image embedding not supported by Ollama model %s", o.model)
}

// getOllamaEmbeddingInfo returns embedding info for Ollama models.
func getOllamaEmbeddingInfo(model string) EmbeddingInfo {
	switch model {
	case OllamaMxbaiEmbedLarge:
		return EmbeddingInfo{
			ModelName:  model,
			Dimensions: 1024,
			MaxTokens:  512,
		}
	case OllamaAllMiniLM:
		return EmbeddingInfo{
			ModelName:  model,
			Dimensions: 384,
			MaxTokens:  256,
		}
	case OllamaNomicEmbedText:
		return EmbeddingInfo{
			ModelName:  model,
			Dimensions: 768,
			MaxTokens:  8192,
		}
	case OllamaSnowflakeArctic:
		return EmbeddingInfo{
			ModelName:  model,
			Dimensions: 1024,
			MaxTokens:  512,
		}
	case OllamaBgeSmall:
		return EmbeddingInfo{
			ModelName:  model,
			Dimensions: 384,
			MaxTokens:  512,
		}
	case OllamaBgeLarge:
		return EmbeddingInfo{
			ModelName:  model,
			Dimensions: 1024,
			MaxTokens:  512,
		}
	default:
		return DefaultEmbeddingInfo(model)
	}
}

// Ensure OllamaEmbedding implements the interfaces.
var _ EmbeddingModel = (*OllamaEmbedding)(nil)
var _ EmbeddingModelWithInfo = (*OllamaEmbedding)(nil)
var _ EmbeddingModelWithBatch = (*OllamaEmbedding)(nil)
var _ MultiModalEmbeddingModel = (*OllamaEmbedding)(nil)
var _ FullEmbeddingModel = (*OllamaEmbedding)(nil)
