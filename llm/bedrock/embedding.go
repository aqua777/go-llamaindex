package bedrock

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

// Embedding model constants - Amazon Titan.
const (
	TitanEmbedTextV1   = "amazon.titan-embed-text-v1"
	TitanEmbedTextV2   = "amazon.titan-embed-text-v2:0"
	TitanEmbedG1Text02 = "amazon.titan-embed-g1-text-02"
)

// Embedding model constants - Cohere.
const (
	CohereEmbedEnglishV3      = "cohere.embed-english-v3"
	CohereEmbedMultilingualV3 = "cohere.embed-multilingual-v3"
	CohereEmbedV4             = "cohere.embed-v4:0"
)

// DefaultEmbeddingModel is the default embedding model.
const DefaultEmbeddingModel = TitanEmbedTextV2

// embeddingModelInfo maps model names to their embedding info.
var embeddingModelInfo = map[string]embedding.EmbeddingInfo{
	TitanEmbedTextV1: {
		ModelName:  TitanEmbedTextV1,
		Dimensions: 1536,
		MaxTokens:  8192,
	},
	TitanEmbedTextV2: {
		ModelName:  TitanEmbedTextV2,
		Dimensions: 1024, // Default, can be configured to 256, 512, or 1024
		MaxTokens:  8192,
	},
	TitanEmbedG1Text02: {
		ModelName:  TitanEmbedG1Text02,
		Dimensions: 1536,
		MaxTokens:  8192,
	},
	CohereEmbedEnglishV3: {
		ModelName:  CohereEmbedEnglishV3,
		Dimensions: 1024,
		MaxTokens:  512,
	},
	CohereEmbedMultilingualV3: {
		ModelName:  CohereEmbedMultilingualV3,
		Dimensions: 1024,
		MaxTokens:  512,
	},
	CohereEmbedV4: {
		ModelName:  CohereEmbedV4,
		Dimensions: 1024,
		MaxTokens:  512,
	},
}

// Embedding implements the EmbeddingModel interface for AWS Bedrock.
type Embedding struct {
	client     *bedrockruntime.Client
	model      string
	region     string
	dimensions int  // For Titan V2, can be 256, 512, or 1024
	normalize  bool // For Titan V2, whether to normalize embeddings
	logger     *slog.Logger
}

// EmbeddingOption configures an Embedding.
type EmbeddingOption func(*Embedding)

// WithEmbeddingModel sets the embedding model.
func WithEmbeddingModel(model string) EmbeddingOption {
	return func(e *Embedding) {
		e.model = model
	}
}

// WithEmbeddingRegion sets the AWS region.
func WithEmbeddingRegion(region string) EmbeddingOption {
	return func(e *Embedding) {
		e.region = region
	}
}

// WithEmbeddingDimensions sets the embedding dimensions (Titan V2 only).
func WithEmbeddingDimensions(dimensions int) EmbeddingOption {
	return func(e *Embedding) {
		e.dimensions = dimensions
	}
}

// WithEmbeddingNormalize sets whether to normalize embeddings (Titan V2 only).
func WithEmbeddingNormalize(normalize bool) EmbeddingOption {
	return func(e *Embedding) {
		e.normalize = normalize
	}
}

// WithEmbeddingCredentials sets explicit AWS credentials.
func WithEmbeddingCredentials(accessKeyID, secretAccessKey, sessionToken string) EmbeddingOption {
	return func(e *Embedding) {
		cfg, err := config.LoadDefaultConfig(context.Background(),
			config.WithRegion(e.region),
			config.WithCredentialsProvider(credentials.NewStaticCredentialsProvider(
				accessKeyID,
				secretAccessKey,
				sessionToken,
			)),
		)
		if err == nil {
			e.client = bedrockruntime.NewFromConfig(cfg)
		}
	}
}

// WithEmbeddingClient sets a custom Bedrock client (for testing).
func WithEmbeddingClient(client *bedrockruntime.Client) EmbeddingOption {
	return func(e *Embedding) {
		e.client = client
	}
}

// NewEmbedding creates a new AWS Bedrock Embedding client.
func NewEmbedding(opts ...EmbeddingOption) *Embedding {
	region := os.Getenv("AWS_REGION")
	if region == "" {
		region = os.Getenv("AWS_DEFAULT_REGION")
	}
	if region == "" {
		region = "us-east-1"
	}

	e := &Embedding{
		model:      DefaultEmbeddingModel,
		region:     region,
		dimensions: 1024, // Default for Titan V2
		normalize:  true,
		logger:     slog.New(slog.NewJSONHandler(os.Stdout, nil)),
	}

	// Apply options first to get region
	for _, opt := range opts {
		opt(e)
	}

	// Initialize client if not already set
	if e.client == nil {
		cfg, err := config.LoadDefaultConfig(context.Background(),
			config.WithRegion(e.region),
		)
		if err == nil {
			e.client = bedrockruntime.NewFromConfig(cfg)
		}
	}

	return e
}

// GetTextEmbedding generates an embedding for a given text.
func (e *Embedding) GetTextEmbedding(ctx context.Context, text string) ([]float64, error) {
	return e.getEmbedding(ctx, text, "text")
}

// GetQueryEmbedding generates an embedding for a given query.
func (e *Embedding) GetQueryEmbedding(ctx context.Context, query string) ([]float64, error) {
	return e.getEmbedding(ctx, query, "query")
}

// Info returns information about the model's capabilities.
func (e *Embedding) Info() embedding.EmbeddingInfo {
	if info, ok := embeddingModelInfo[e.model]; ok {
		// For Titan V2, use configured dimensions
		if e.model == TitanEmbedTextV2 {
			info.Dimensions = e.dimensions
		}
		return info
	}
	return embedding.DefaultEmbeddingInfo(e.model)
}

// GetTextEmbeddingsBatch generates embeddings for multiple texts.
func (e *Embedding) GetTextEmbeddingsBatch(ctx context.Context, texts []string, callback embedding.ProgressCallback) ([][]float64, error) {
	provider := e.getProvider()

	// Cohere supports batch embedding natively
	if provider == "cohere" {
		return e.getCohereBatchEmbeddings(ctx, texts, "text", callback)
	}

	// For Amazon Titan, process one at a time
	results := make([][]float64, len(texts))
	for i, text := range texts {
		emb, err := e.GetTextEmbedding(ctx, text)
		if err != nil {
			return nil, fmt.Errorf("failed to get embedding for text %d: %w", i, err)
		}
		results[i] = emb
		if callback != nil {
			callback(i+1, len(texts))
		}
	}
	return results, nil
}

// getEmbedding generates an embedding for a given text.
func (e *Embedding) getEmbedding(ctx context.Context, text string, inputType string) ([]float64, error) {
	e.logger.Info("getEmbedding called", "model", e.model, "text_len", len(text), "input_type", inputType)

	provider := e.getProvider()
	requestBody, err := e.buildRequestBody(provider, text, inputType)
	if err != nil {
		return nil, err
	}

	resp, err := e.client.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(e.model),
		Body:        requestBody,
		Accept:      aws.String("application/json"),
		ContentType: aws.String("application/json"),
	})
	if err != nil {
		e.logger.Error("InvokeModel failed", "error", err)
		return nil, fmt.Errorf("bedrock invoke model failed: %w", err)
	}

	return e.parseResponse(provider, resp.Body)
}

// getCohereBatchEmbeddings gets embeddings for multiple texts using Cohere's batch API.
func (e *Embedding) getCohereBatchEmbeddings(ctx context.Context, texts []string, inputType string, callback embedding.ProgressCallback) ([][]float64, error) {
	e.logger.Info("getCohereBatchEmbeddings called", "model", e.model, "text_count", len(texts))

	// Truncate texts to 2048 chars (Cohere limit)
	truncatedTexts := make([]string, len(texts))
	for i, text := range texts {
		if len(text) > 2048 {
			truncatedTexts[i] = text[:2048]
		} else {
			truncatedTexts[i] = text
		}
	}

	// Build Cohere batch request
	cohereInputType := "search_document"
	if inputType == "query" {
		cohereInputType = "search_query"
	}

	request := map[string]interface{}{
		"texts":      truncatedTexts,
		"input_type": cohereInputType,
	}

	requestBody, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := e.client.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(e.model),
		Body:        requestBody,
		Accept:      aws.String("application/json"),
		ContentType: aws.String("application/json"),
	})
	if err != nil {
		e.logger.Error("InvokeModel failed", "error", err)
		return nil, fmt.Errorf("bedrock invoke model failed: %w", err)
	}

	embeddings, err := e.parseCohereResponse(resp.Body, true)
	if err != nil {
		return nil, err
	}

	if callback != nil {
		callback(len(texts), len(texts))
	}

	return embeddings, nil
}

// getProvider extracts the provider name from the model.
func (e *Embedding) getProvider() string {
	parts := strings.Split(e.model, ".")
	if len(parts) == 2 {
		return parts[0]
	}
	if len(parts) == 3 {
		// Region-prefixed model: us.amazon.titan-embed-text-v2:0
		return parts[1]
	}
	return "amazon" // default
}

// buildRequestBody builds the request body based on provider.
func (e *Embedding) buildRequestBody(provider, text, inputType string) ([]byte, error) {
	switch provider {
	case "amazon":
		request := map[string]interface{}{
			"inputText": text,
		}

		// Titan V2 supports additional parameters
		if e.model == TitanEmbedTextV2 {
			request["dimensions"] = e.dimensions
			request["normalize"] = e.normalize
		}

		return json.Marshal(request)

	case "cohere":
		// Truncate to 2048 chars
		if len(text) > 2048 {
			text = text[:2048]
		}

		cohereInputType := "search_document"
		if inputType == "query" {
			cohereInputType = "search_query"
		}

		request := map[string]interface{}{
			"texts":      []string{text},
			"input_type": cohereInputType,
		}

		return json.Marshal(request)

	default:
		return nil, fmt.Errorf("unsupported provider: %s", provider)
	}
}

// parseResponse parses the response based on provider.
func (e *Embedding) parseResponse(provider string, body []byte) ([]float64, error) {
	switch provider {
	case "amazon":
		return e.parseAmazonResponse(body)
	case "cohere":
		embeddings, err := e.parseCohereResponse(body, false)
		if err != nil {
			return nil, err
		}
		if len(embeddings) == 0 {
			return nil, fmt.Errorf("no embeddings in response")
		}
		return embeddings[0], nil
	default:
		return nil, fmt.Errorf("unsupported provider: %s", provider)
	}
}

// parseAmazonResponse parses Amazon Titan embedding response.
func (e *Embedding) parseAmazonResponse(body []byte) ([]float64, error) {
	var response struct {
		Embedding []float64 `json:"embedding"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return response.Embedding, nil
}

// parseCohereResponse parses Cohere embedding response.
// Handles both v3 and v4 response formats.
func (e *Embedding) parseCohereResponse(body []byte, isBatch bool) ([][]float64, error) {
	var response map[string]interface{}
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Try to find embeddings in various response formats
	embeddings := e.extractCohereEmbeddings(response)
	if embeddings == nil {
		return nil, fmt.Errorf("could not find embeddings in response")
	}

	return embeddings, nil
}

// extractCohereEmbeddings extracts embeddings from Cohere response.
// Handles v3 format: {"embeddings": [[...]]}
// Handles v4 format: {"embeddings": {"float": [[...]]}}
func (e *Embedding) extractCohereEmbeddings(response map[string]interface{}) [][]float64 {
	embeddings, ok := response["embeddings"]
	if !ok {
		return nil
	}

	// Check if it's a nested dict (v4 format)
	if embMap, ok := embeddings.(map[string]interface{}); ok {
		if floatEmb, ok := embMap["float"]; ok {
			return e.convertToFloat64Slice(floatEmb)
		}
		if nestedEmb, ok := embMap["embeddings"]; ok {
			if nestedMap, ok := nestedEmb.(map[string]interface{}); ok {
				if floatEmb, ok := nestedMap["float"]; ok {
					return e.convertToFloat64Slice(floatEmb)
				}
			}
			return e.convertToFloat64Slice(nestedEmb)
		}
	}

	// Direct array format (v3)
	return e.convertToFloat64Slice(embeddings)
}

// convertToFloat64Slice converts interface{} to [][]float64.
func (e *Embedding) convertToFloat64Slice(data interface{}) [][]float64 {
	arr, ok := data.([]interface{})
	if !ok {
		return nil
	}

	result := make([][]float64, len(arr))
	for i, item := range arr {
		embArr, ok := item.([]interface{})
		if !ok {
			return nil
		}
		result[i] = make([]float64, len(embArr))
		for j, val := range embArr {
			if f, ok := val.(float64); ok {
				result[i][j] = f
			}
		}
	}
	return result
}

// Ensure Embedding implements the interfaces.
var _ embedding.EmbeddingModel = (*Embedding)(nil)
var _ embedding.EmbeddingModelWithInfo = (*Embedding)(nil)
var _ embedding.EmbeddingModelWithBatch = (*Embedding)(nil)
