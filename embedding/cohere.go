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
	// CohereAPIURL is the default Cohere API endpoint.
	CohereAPIURL = "https://api.cohere.ai/v1"
)

// Cohere embedding model constants.
const (
	CohereEmbedEnglishV3           = "embed-english-v3.0"
	CohereEmbedMultilingualV3      = "embed-multilingual-v3.0"
	CohereEmbedEnglishLightV3      = "embed-english-light-v3.0"
	CohereEmbedMultilingualLightV3 = "embed-multilingual-light-v3.0"
)

// CohereInputType specifies the type of input for embedding.
type CohereInputType string

const (
	// CohereInputTypeSearchDocument for documents to be searched.
	CohereInputTypeSearchDocument CohereInputType = "search_document"
	// CohereInputTypeSearchQuery for search queries.
	CohereInputTypeSearchQuery CohereInputType = "search_query"
	// CohereInputTypeClassification for classification inputs.
	CohereInputTypeClassification CohereInputType = "classification"
	// CohereInputTypeClustering for clustering inputs.
	CohereInputTypeClustering CohereInputType = "clustering"
)

// CohereEmbedding implements the EmbeddingModel interface for Cohere.
type CohereEmbedding struct {
	apiKey     string
	baseURL    string
	model      string
	inputType  CohereInputType
	truncate   string // "NONE", "START", "END"
	httpClient *http.Client
	logger     *slog.Logger
}

// CohereEmbeddingOption configures a CohereEmbedding.
type CohereEmbeddingOption func(*CohereEmbedding)

// WithCohereEmbeddingAPIKey sets the API key.
func WithCohereEmbeddingAPIKey(apiKey string) CohereEmbeddingOption {
	return func(c *CohereEmbedding) {
		c.apiKey = apiKey
	}
}

// WithCohereEmbeddingBaseURL sets the base URL.
func WithCohereEmbeddingBaseURL(baseURL string) CohereEmbeddingOption {
	return func(c *CohereEmbedding) {
		c.baseURL = baseURL
	}
}

// WithCohereEmbeddingModel sets the model.
func WithCohereEmbeddingModel(model string) CohereEmbeddingOption {
	return func(c *CohereEmbedding) {
		c.model = model
	}
}

// WithCohereEmbeddingInputType sets the input type.
func WithCohereEmbeddingInputType(inputType CohereInputType) CohereEmbeddingOption {
	return func(c *CohereEmbedding) {
		c.inputType = inputType
	}
}

// WithCohereEmbeddingTruncate sets the truncation mode.
func WithCohereEmbeddingTruncate(truncate string) CohereEmbeddingOption {
	return func(c *CohereEmbedding) {
		c.truncate = truncate
	}
}

// WithCohereEmbeddingHTTPClient sets a custom HTTP client.
func WithCohereEmbeddingHTTPClient(client *http.Client) CohereEmbeddingOption {
	return func(c *CohereEmbedding) {
		c.httpClient = client
	}
}

// NewCohereEmbedding creates a new Cohere embedding client.
func NewCohereEmbedding(opts ...CohereEmbeddingOption) *CohereEmbedding {
	c := &CohereEmbedding{
		apiKey:     os.Getenv("COHERE_API_KEY"),
		baseURL:    CohereAPIURL,
		model:      CohereEmbedEnglishV3,
		inputType:  CohereInputTypeSearchDocument,
		truncate:   "END",
		httpClient: http.DefaultClient,
		logger:     slog.New(slog.NewJSONHandler(os.Stdout, nil)),
	}

	for _, opt := range opts {
		opt(c)
	}

	return c
}

// cohereEmbedRequest represents a request to the Cohere embed API.
type cohereEmbedRequest struct {
	Model     string   `json:"model"`
	Texts     []string `json:"texts"`
	InputType string   `json:"input_type"`
	Truncate  string   `json:"truncate,omitempty"`
}

// cohereEmbedResponse represents a response from the Cohere embed API.
type cohereEmbedResponse struct {
	ID         string      `json:"id"`
	Embeddings [][]float64 `json:"embeddings"`
	Texts      []string    `json:"texts"`
	Meta       struct {
		APIVersion struct {
			Version string `json:"version"`
		} `json:"api_version"`
		BilledUnits struct {
			InputTokens int `json:"input_tokens"`
		} `json:"billed_units"`
	} `json:"meta"`
}

// GetTextEmbedding generates an embedding for a given text.
func (c *CohereEmbedding) GetTextEmbedding(ctx context.Context, text string) ([]float64, error) {
	embeddings, err := c.getEmbeddings(ctx, []string{text}, CohereInputTypeSearchDocument)
	if err != nil {
		return nil, err
	}
	if len(embeddings) == 0 {
		return nil, fmt.Errorf("cohere returned no embeddings")
	}
	return embeddings[0], nil
}

// GetQueryEmbedding generates an embedding for a given query.
func (c *CohereEmbedding) GetQueryEmbedding(ctx context.Context, query string) ([]float64, error) {
	embeddings, err := c.getEmbeddings(ctx, []string{query}, CohereInputTypeSearchQuery)
	if err != nil {
		return nil, err
	}
	if len(embeddings) == 0 {
		return nil, fmt.Errorf("cohere returned no embeddings")
	}
	return embeddings[0], nil
}

// getEmbeddings performs the actual embedding request.
func (c *CohereEmbedding) getEmbeddings(ctx context.Context, texts []string, inputType CohereInputType) ([][]float64, error) {
	reqBody := cohereEmbedRequest{
		Model:     c.model,
		Texts:     texts,
		InputType: string(inputType),
		Truncate:  c.truncate,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/embed", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("cohere API error (%d): %s", resp.StatusCode, string(respBody))
	}

	var result cohereEmbedResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return result.Embeddings, nil
}

// Info returns information about the model's capabilities.
func (c *CohereEmbedding) Info() EmbeddingInfo {
	return getCohereEmbeddingInfo(c.model)
}

// GetTextEmbeddingsBatch generates embeddings for multiple texts.
func (c *CohereEmbedding) GetTextEmbeddingsBatch(ctx context.Context, texts []string, callback ProgressCallback) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	c.logger.Info("GetTextEmbeddingsBatch called", "model", c.model, "count", len(texts))

	// Cohere supports batch embedding natively (up to 96 texts per request)
	const batchSize = 96
	results := make([][]float64, 0, len(texts))
	processed := 0

	for i := 0; i < len(texts); i += batchSize {
		end := i + batchSize
		if end > len(texts) {
			end = len(texts)
		}
		batch := texts[i:end]

		embeddings, err := c.getEmbeddings(ctx, batch, CohereInputTypeSearchDocument)
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

// SupportsMultiModal returns false as Cohere embedding models don't support images.
func (c *CohereEmbedding) SupportsMultiModal() bool {
	return false
}

// GetImageEmbedding is not supported by Cohere embedding models.
func (c *CohereEmbedding) GetImageEmbedding(ctx context.Context, image ImageType) ([]float64, error) {
	return nil, fmt.Errorf("image embedding not supported by Cohere model %s", c.model)
}

// getCohereEmbeddingInfo returns embedding info for Cohere models.
func getCohereEmbeddingInfo(model string) EmbeddingInfo {
	switch model {
	case CohereEmbedEnglishV3, CohereEmbedMultilingualV3:
		return EmbeddingInfo{
			ModelName:  model,
			Dimensions: 1024,
			MaxTokens:  512,
		}
	case CohereEmbedEnglishLightV3, CohereEmbedMultilingualLightV3:
		return EmbeddingInfo{
			ModelName:  model,
			Dimensions: 384,
			MaxTokens:  512,
		}
	default:
		return DefaultEmbeddingInfo(model)
	}
}

// Ensure CohereEmbedding implements the interfaces.
var _ EmbeddingModel = (*CohereEmbedding)(nil)
var _ EmbeddingModelWithInfo = (*CohereEmbedding)(nil)
var _ EmbeddingModelWithBatch = (*CohereEmbedding)(nil)
var _ MultiModalEmbeddingModel = (*CohereEmbedding)(nil)
var _ FullEmbeddingModel = (*CohereEmbedding)(nil)
