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
	// HuggingFaceInferenceAPIURL is the default HuggingFace Inference API endpoint.
	HuggingFaceInferenceAPIURL = "https://api-inference.huggingface.co"
	// HuggingFaceTEIURL is the default Text Embeddings Inference endpoint.
	HuggingFaceTEIURL = "http://localhost:8080"
)

// Common HuggingFace embedding model names.
const (
	HFSentenceTransformersMiniLM = "sentence-transformers/all-MiniLM-L6-v2"
	HFSentenceTransformersMpnet  = "sentence-transformers/all-mpnet-base-v2"
	HFBGEM3                      = "BAAI/bge-m3"
	HFBGESmall                   = "BAAI/bge-small-en-v1.5"
	HFBGEBase                    = "BAAI/bge-base-en-v1.5"
	HFBGELarge                   = "BAAI/bge-large-en-v1.5"
	HFE5Small                    = "intfloat/e5-small-v2"
	HFE5Base                     = "intfloat/e5-base-v2"
	HFE5Large                    = "intfloat/e5-large-v2"
	HFGTESmall                   = "thenlper/gte-small"
	HFGTEBase                    = "thenlper/gte-base"
	HFGTELarge                   = "thenlper/gte-large"
)

// HuggingFaceEmbedding implements the EmbeddingModel interface for HuggingFace models.
// It supports both the HuggingFace Inference API and Text Embeddings Inference (TEI).
type HuggingFaceEmbedding struct {
	apiKey      string
	baseURL     string
	model       string
	useTEI      bool // Use Text Embeddings Inference format
	httpClient  *http.Client
	logger      *slog.Logger
	queryPrefix string // Prefix for query embeddings (e.g., "query: " for E5)
	docPrefix   string // Prefix for document embeddings (e.g., "passage: " for E5)
}

// HuggingFaceEmbeddingOption configures a HuggingFaceEmbedding.
type HuggingFaceEmbeddingOption func(*HuggingFaceEmbedding)

// WithHuggingFaceAPIKey sets the API key.
func WithHuggingFaceAPIKey(apiKey string) HuggingFaceEmbeddingOption {
	return func(h *HuggingFaceEmbedding) {
		h.apiKey = apiKey
	}
}

// WithHuggingFaceBaseURL sets the base URL.
func WithHuggingFaceBaseURL(baseURL string) HuggingFaceEmbeddingOption {
	return func(h *HuggingFaceEmbedding) {
		h.baseURL = baseURL
	}
}

// WithHuggingFaceModel sets the model.
func WithHuggingFaceModel(model string) HuggingFaceEmbeddingOption {
	return func(h *HuggingFaceEmbedding) {
		h.model = model
	}
}

// WithHuggingFaceTEI enables Text Embeddings Inference mode.
func WithHuggingFaceTEI(useTEI bool) HuggingFaceEmbeddingOption {
	return func(h *HuggingFaceEmbedding) {
		h.useTEI = useTEI
	}
}

// WithHuggingFaceHTTPClient sets a custom HTTP client.
func WithHuggingFaceHTTPClient(client *http.Client) HuggingFaceEmbeddingOption {
	return func(h *HuggingFaceEmbedding) {
		h.httpClient = client
	}
}

// WithHuggingFaceQueryPrefix sets the query prefix.
func WithHuggingFaceQueryPrefix(prefix string) HuggingFaceEmbeddingOption {
	return func(h *HuggingFaceEmbedding) {
		h.queryPrefix = prefix
	}
}

// WithHuggingFaceDocPrefix sets the document prefix.
func WithHuggingFaceDocPrefix(prefix string) HuggingFaceEmbeddingOption {
	return func(h *HuggingFaceEmbedding) {
		h.docPrefix = prefix
	}
}

// NewHuggingFaceEmbedding creates a new HuggingFace embedding client.
func NewHuggingFaceEmbedding(opts ...HuggingFaceEmbeddingOption) *HuggingFaceEmbedding {
	h := &HuggingFaceEmbedding{
		apiKey:     os.Getenv("HUGGINGFACE_API_KEY"),
		baseURL:    HuggingFaceInferenceAPIURL,
		model:      HFSentenceTransformersMiniLM,
		useTEI:     false,
		httpClient: http.DefaultClient,
		logger:     slog.New(slog.NewJSONHandler(os.Stdout, nil)),
	}

	for _, opt := range opts {
		opt(h)
	}

	// Set default prefixes for E5 models
	if h.queryPrefix == "" && h.docPrefix == "" {
		if isE5Model(h.model) {
			h.queryPrefix = "query: "
			h.docPrefix = "passage: "
		}
	}

	return h
}

// isE5Model checks if the model is an E5 model that requires prefixes.
func isE5Model(model string) bool {
	switch model {
	case HFE5Small, HFE5Base, HFE5Large:
		return true
	default:
		return false
	}
}

// hfInferenceRequest represents a request to the HuggingFace Inference API.
type hfInferenceRequest struct {
	Inputs  interface{} `json:"inputs"`
	Options struct {
		WaitForModel bool `json:"wait_for_model"`
	} `json:"options,omitempty"`
}

// teiEmbedRequest represents a request to Text Embeddings Inference.
type teiEmbedRequest struct {
	Inputs   []string `json:"inputs"`
	Truncate bool     `json:"truncate,omitempty"`
}

// GetTextEmbedding generates an embedding for a given text.
func (h *HuggingFaceEmbedding) GetTextEmbedding(ctx context.Context, text string) ([]float64, error) {
	// Add document prefix if configured
	if h.docPrefix != "" {
		text = h.docPrefix + text
	}
	return h.getEmbedding(ctx, text)
}

// GetQueryEmbedding generates an embedding for a given query.
func (h *HuggingFaceEmbedding) GetQueryEmbedding(ctx context.Context, query string) ([]float64, error) {
	// Add query prefix if configured
	if h.queryPrefix != "" {
		query = h.queryPrefix + query
	}
	return h.getEmbedding(ctx, query)
}

// getEmbedding performs the actual embedding request.
func (h *HuggingFaceEmbedding) getEmbedding(ctx context.Context, text string) ([]float64, error) {
	if h.useTEI {
		embeddings, err := h.getTEIEmbeddings(ctx, []string{text})
		if err != nil {
			return nil, err
		}
		if len(embeddings) == 0 {
			return nil, fmt.Errorf("no embeddings returned")
		}
		return embeddings[0], nil
	}

	return h.getInferenceAPIEmbedding(ctx, text)
}

// getInferenceAPIEmbedding gets embedding from HuggingFace Inference API.
func (h *HuggingFaceEmbedding) getInferenceAPIEmbedding(ctx context.Context, text string) ([]float64, error) {
	reqBody := hfInferenceRequest{
		Inputs: text,
	}
	reqBody.Options.WaitForModel = true

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/pipeline/feature-extraction/%s", h.baseURL, h.model)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if h.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+h.apiKey)
	}

	resp, err := h.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("huggingface API error (%d): %s", resp.StatusCode, string(respBody))
	}

	// Response can be nested arrays for sentence-transformers models
	// Try to parse as []float64 first, then as [][]float64
	var embedding []float64
	if err := json.Unmarshal(respBody, &embedding); err == nil {
		return embedding, nil
	}

	// Try nested format (common for sentence-transformers)
	var nestedEmbedding [][]float64
	if err := json.Unmarshal(respBody, &nestedEmbedding); err == nil && len(nestedEmbedding) > 0 {
		// For sentence-transformers, we typically want the mean of token embeddings
		// But the API usually returns the pooled output directly
		return nestedEmbedding[0], nil
	}

	// Try triple nested (token-level embeddings)
	var tokenEmbeddings [][][]float64
	if err := json.Unmarshal(respBody, &tokenEmbeddings); err == nil && len(tokenEmbeddings) > 0 && len(tokenEmbeddings[0]) > 0 {
		// Mean pooling over tokens
		return meanPool(tokenEmbeddings[0]), nil
	}

	return nil, fmt.Errorf("failed to parse embedding response: %s", string(respBody))
}

// getTEIEmbeddings gets embeddings from Text Embeddings Inference.
func (h *HuggingFaceEmbedding) getTEIEmbeddings(ctx context.Context, texts []string) ([][]float64, error) {
	reqBody := teiEmbedRequest{
		Inputs:   texts,
		Truncate: true,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", h.baseURL+"/embed", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := h.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("TEI API error (%d): %s", resp.StatusCode, string(respBody))
	}

	var embeddings [][]float64
	if err := json.Unmarshal(respBody, &embeddings); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return embeddings, nil
}

// meanPool computes mean pooling over token embeddings.
func meanPool(tokenEmbeddings [][]float64) []float64 {
	if len(tokenEmbeddings) == 0 {
		return nil
	}

	dims := len(tokenEmbeddings[0])
	result := make([]float64, dims)

	for _, token := range tokenEmbeddings {
		for i, v := range token {
			result[i] += v
		}
	}

	numTokens := float64(len(tokenEmbeddings))
	for i := range result {
		result[i] /= numTokens
	}

	return result
}

// Info returns information about the model's capabilities.
func (h *HuggingFaceEmbedding) Info() EmbeddingInfo {
	return getHuggingFaceEmbeddingInfo(h.model)
}

// GetTextEmbeddingsBatch generates embeddings for multiple texts.
func (h *HuggingFaceEmbedding) GetTextEmbeddingsBatch(ctx context.Context, texts []string, callback ProgressCallback) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	h.logger.Info("GetTextEmbeddingsBatch called", "model", h.model, "count", len(texts))

	// Add document prefix if configured
	prefixedTexts := texts
	if h.docPrefix != "" {
		prefixedTexts = make([]string, len(texts))
		for i, text := range texts {
			prefixedTexts[i] = h.docPrefix + text
		}
	}

	if h.useTEI {
		// TEI supports batch natively
		const batchSize = 32
		results := make([][]float64, 0, len(prefixedTexts))
		processed := 0

		for i := 0; i < len(prefixedTexts); i += batchSize {
			end := i + batchSize
			if end > len(prefixedTexts) {
				end = len(prefixedTexts)
			}
			batch := prefixedTexts[i:end]

			embeddings, err := h.getTEIEmbeddings(ctx, batch)
			if err != nil {
				return nil, fmt.Errorf("failed to get embeddings for batch starting at %d: %w", i, err)
			}

			results = append(results, embeddings...)
			processed += len(batch)

			if callback != nil {
				callback(processed, len(texts))
			}
		}

		return results, nil
	}

	// For Inference API, process one at a time
	results := make([][]float64, len(prefixedTexts))
	for i, text := range prefixedTexts {
		embedding, err := h.getInferenceAPIEmbedding(ctx, text)
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

// SupportsMultiModal returns false as HuggingFace text embedding models don't support images.
func (h *HuggingFaceEmbedding) SupportsMultiModal() bool {
	return false
}

// GetImageEmbedding is not supported by HuggingFace text embedding models.
func (h *HuggingFaceEmbedding) GetImageEmbedding(ctx context.Context, image ImageType) ([]float64, error) {
	return nil, fmt.Errorf("image embedding not supported by HuggingFace model %s", h.model)
}

// getHuggingFaceEmbeddingInfo returns embedding info for HuggingFace models.
func getHuggingFaceEmbeddingInfo(model string) EmbeddingInfo {
	switch model {
	case HFSentenceTransformersMiniLM:
		return EmbeddingInfo{
			ModelName:  model,
			Dimensions: 384,
			MaxTokens:  256,
		}
	case HFSentenceTransformersMpnet:
		return EmbeddingInfo{
			ModelName:  model,
			Dimensions: 768,
			MaxTokens:  384,
		}
	case HFBGEM3:
		return EmbeddingInfo{
			ModelName:  model,
			Dimensions: 1024,
			MaxTokens:  8192,
		}
	case HFBGESmall:
		return EmbeddingInfo{
			ModelName:  model,
			Dimensions: 384,
			MaxTokens:  512,
		}
	case HFBGEBase:
		return EmbeddingInfo{
			ModelName:  model,
			Dimensions: 768,
			MaxTokens:  512,
		}
	case HFBGELarge:
		return EmbeddingInfo{
			ModelName:  model,
			Dimensions: 1024,
			MaxTokens:  512,
		}
	case HFE5Small:
		return EmbeddingInfo{
			ModelName:  model,
			Dimensions: 384,
			MaxTokens:  512,
		}
	case HFE5Base:
		return EmbeddingInfo{
			ModelName:  model,
			Dimensions: 768,
			MaxTokens:  512,
		}
	case HFE5Large:
		return EmbeddingInfo{
			ModelName:  model,
			Dimensions: 1024,
			MaxTokens:  512,
		}
	case HFGTESmall:
		return EmbeddingInfo{
			ModelName:  model,
			Dimensions: 384,
			MaxTokens:  512,
		}
	case HFGTEBase:
		return EmbeddingInfo{
			ModelName:  model,
			Dimensions: 768,
			MaxTokens:  512,
		}
	case HFGTELarge:
		return EmbeddingInfo{
			ModelName:  model,
			Dimensions: 1024,
			MaxTokens:  512,
		}
	default:
		return DefaultEmbeddingInfo(model)
	}
}

// Ensure HuggingFaceEmbedding implements the interfaces.
var _ EmbeddingModel = (*HuggingFaceEmbedding)(nil)
var _ EmbeddingModelWithInfo = (*HuggingFaceEmbedding)(nil)
var _ EmbeddingModelWithBatch = (*HuggingFaceEmbedding)(nil)
var _ MultiModalEmbeddingModel = (*HuggingFaceEmbedding)(nil)
var _ FullEmbeddingModel = (*HuggingFaceEmbedding)(nil)
