package embedding

import (
	"context"
	"fmt"
	"log/slog"
	"os"

	openai "github.com/sashabaranov/go-openai"
)

// AzureOpenAIEmbedding implements the EmbeddingModel interface for Azure OpenAI.
type AzureOpenAIEmbedding struct {
	client     *openai.Client
	deployment string // Azure deployment name
	logger     *slog.Logger
}

// AzureOpenAIEmbeddingOption configures an AzureOpenAIEmbedding.
type AzureOpenAIEmbeddingOption func(*AzureOpenAIEmbedding)

// WithAzureEmbeddingDeployment sets the deployment name.
func WithAzureEmbeddingDeployment(deployment string) AzureOpenAIEmbeddingOption {
	return func(a *AzureOpenAIEmbedding) {
		a.deployment = deployment
	}
}

// NewAzureOpenAIEmbedding creates a new Azure OpenAI embedding client.
// It requires the Azure endpoint and API key, which can be provided via
// environment variables AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY.
func NewAzureOpenAIEmbedding(opts ...AzureOpenAIEmbeddingOption) *AzureOpenAIEmbedding {
	endpoint := os.Getenv("AZURE_OPENAI_ENDPOINT")
	apiKey := os.Getenv("AZURE_OPENAI_API_KEY")
	deployment := os.Getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

	a := &AzureOpenAIEmbedding{
		deployment: deployment,
		logger:     slog.New(slog.NewJSONHandler(os.Stdout, nil)),
	}

	for _, opt := range opts {
		opt(a)
	}

	// Create Azure OpenAI config
	config := openai.DefaultAzureConfig(apiKey, endpoint)
	a.client = openai.NewClientWithConfig(config)

	return a
}

// NewAzureOpenAIEmbeddingWithConfig creates a new Azure OpenAI embedding client with explicit configuration.
func NewAzureOpenAIEmbeddingWithConfig(endpoint, apiKey, deployment string) *AzureOpenAIEmbedding {
	config := openai.DefaultAzureConfig(apiKey, endpoint)

	return &AzureOpenAIEmbedding{
		client:     openai.NewClientWithConfig(config),
		deployment: deployment,
		logger:     slog.New(slog.NewJSONHandler(os.Stdout, nil)),
	}
}

// GetTextEmbedding generates an embedding for a given text.
func (a *AzureOpenAIEmbedding) GetTextEmbedding(ctx context.Context, text string) ([]float64, error) {
	return a.getEmbedding(ctx, text)
}

// GetQueryEmbedding generates an embedding for a given query.
func (a *AzureOpenAIEmbedding) GetQueryEmbedding(ctx context.Context, query string) ([]float64, error) {
	return a.getEmbedding(ctx, query)
}

// getEmbedding performs the actual embedding request.
func (a *AzureOpenAIEmbedding) getEmbedding(ctx context.Context, text string) ([]float64, error) {
	resp, err := a.client.CreateEmbeddings(
		ctx,
		openai.EmbeddingRequest{
			Input: []string{text},
			Model: openai.EmbeddingModel(a.deployment),
		},
	)

	if err != nil {
		a.logger.Error("GetEmbedding failed", "error", err)
		return nil, fmt.Errorf("azure openai embedding failed: %w", err)
	}

	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("azure openai returned no embeddings")
	}

	// Convert float32 to float64
	embedding32 := resp.Data[0].Embedding
	embedding64 := make([]float64, len(embedding32))
	for i, v := range embedding32 {
		embedding64[i] = float64(v)
	}

	return embedding64, nil
}

// Info returns information about the model's capabilities.
func (a *AzureOpenAIEmbedding) Info() EmbeddingInfo {
	// Azure deployments can use various models, return generic info
	return EmbeddingInfo{
		ModelName:  a.deployment,
		Dimensions: 1536, // Common for text-embedding-ada-002 and text-embedding-3-small
		MaxTokens:  8191,
	}
}

// GetTextEmbeddingsBatch generates embeddings for multiple texts.
func (a *AzureOpenAIEmbedding) GetTextEmbeddingsBatch(ctx context.Context, texts []string, callback ProgressCallback) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	a.logger.Info("GetTextEmbeddingsBatch called", "deployment", a.deployment, "count", len(texts))

	// Process in chunks of 2048 (Azure OpenAI's limit)
	const batchSize = 2048
	results := make([][]float64, len(texts))
	processed := 0

	for i := 0; i < len(texts); i += batchSize {
		end := i + batchSize
		if end > len(texts) {
			end = len(texts)
		}
		batch := texts[i:end]

		resp, err := a.client.CreateEmbeddings(
			ctx,
			openai.EmbeddingRequest{
				Input: batch,
				Model: openai.EmbeddingModel(a.deployment),
			},
		)

		if err != nil {
			a.logger.Error("GetTextEmbeddingsBatch failed", "error", err)
			return nil, fmt.Errorf("azure openai batch embedding failed: %w", err)
		}

		// Process results
		for j, data := range resp.Data {
			embedding64 := make([]float64, len(data.Embedding))
			for k, v := range data.Embedding {
				embedding64[k] = float64(v)
			}
			results[i+j] = embedding64
		}

		processed += len(batch)
		if callback != nil {
			callback(processed, len(texts))
		}
	}

	return results, nil
}

// SupportsMultiModal returns false as Azure OpenAI embeddings don't support images.
func (a *AzureOpenAIEmbedding) SupportsMultiModal() bool {
	return false
}

// GetImageEmbedding is not supported by Azure OpenAI embedding models.
func (a *AzureOpenAIEmbedding) GetImageEmbedding(ctx context.Context, image ImageType) ([]float64, error) {
	return nil, fmt.Errorf("image embedding not supported by Azure OpenAI deployment %s", a.deployment)
}

// Ensure AzureOpenAIEmbedding implements the interfaces.
var _ EmbeddingModel = (*AzureOpenAIEmbedding)(nil)
var _ EmbeddingModelWithInfo = (*AzureOpenAIEmbedding)(nil)
var _ EmbeddingModelWithBatch = (*AzureOpenAIEmbedding)(nil)
var _ MultiModalEmbeddingModel = (*AzureOpenAIEmbedding)(nil)
var _ FullEmbeddingModel = (*AzureOpenAIEmbedding)(nil)
