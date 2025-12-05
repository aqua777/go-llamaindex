package embedding

import (
	"context"
	"fmt"
	"log/slog"
	"os"

	openai "github.com/sashabaranov/go-openai"
)

type OpenAIEmbedding struct {
	client *openai.Client
	model  openai.EmbeddingModel
	logger *slog.Logger
}

func NewOpenAIEmbedding(apiKey string, modelName string) *OpenAIEmbedding {
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}

	var model openai.EmbeddingModel
	if modelName == "" {
		model = openai.SmallEmbedding3
	} else {
		model = openai.EmbeddingModel(modelName)
	}

	client := openai.NewClient(apiKey)
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))

	return &OpenAIEmbedding{
		client: client,
		model:  model,
		logger: logger,
	}
}

func NewOpenAIEmbeddingWithClient(client *openai.Client, modelName string) *OpenAIEmbedding {
	var model openai.EmbeddingModel
	if modelName == "" {
		model = openai.SmallEmbedding3
	} else {
		model = openai.EmbeddingModel(modelName)
	}

	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))

	return &OpenAIEmbedding{
		client: client,
		model:  model,
		logger: logger,
	}
}

func (o *OpenAIEmbedding) GetTextEmbedding(ctx context.Context, text string) ([]float64, error) {
	return o.getEmbedding(ctx, text, "text")
}

func (o *OpenAIEmbedding) GetQueryEmbedding(ctx context.Context, query string) ([]float64, error) {
	return o.getEmbedding(ctx, query, "query")
}

func (o *OpenAIEmbedding) getEmbedding(ctx context.Context, input string, typeLabel string) ([]float64, error) {
	// o.logger.Info("GetEmbedding called", "type", typeLabel, "model", o.model)

	resp, err := o.client.CreateEmbeddings(
		ctx,
		openai.EmbeddingRequest{
			Input: []string{input},
			Model: o.model,
		},
	)

	if err != nil {
		o.logger.Error("GetEmbedding failed", "type", typeLabel, "error", err)
		return nil, fmt.Errorf("openai embedding failed: %w", err)
	}

	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("openai returned no embeddings")
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
func (o *OpenAIEmbedding) Info() EmbeddingInfo {
	return getModelInfo(string(o.model))
}

// GetTextEmbeddingsBatch generates embeddings for multiple texts.
func (o *OpenAIEmbedding) GetTextEmbeddingsBatch(ctx context.Context, texts []string, callback ProgressCallback) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	o.logger.Info("GetTextEmbeddingsBatch called", "model", o.model, "count", len(texts))

	// OpenAI supports batch embedding natively
	// Process in chunks of 2048 (OpenAI's limit)
	const batchSize = 2048
	results := make([][]float64, len(texts))
	processed := 0

	for i := 0; i < len(texts); i += batchSize {
		end := i + batchSize
		if end > len(texts) {
			end = len(texts)
		}
		batch := texts[i:end]

		resp, err := o.client.CreateEmbeddings(
			ctx,
			openai.EmbeddingRequest{
				Input: batch,
				Model: o.model,
			},
		)

		if err != nil {
			o.logger.Error("GetTextEmbeddingsBatch failed", "error", err)
			return nil, fmt.Errorf("openai batch embedding failed: %w", err)
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

// SupportsMultiModal returns false as standard OpenAI embeddings don't support images.
func (o *OpenAIEmbedding) SupportsMultiModal() bool {
	return false
}

// GetImageEmbedding is not supported by standard OpenAI embedding models.
func (o *OpenAIEmbedding) GetImageEmbedding(ctx context.Context, image ImageType) ([]float64, error) {
	return nil, fmt.Errorf("image embedding not supported by model %s", o.model)
}

// getModelInfo returns embedding info for known models.
func getModelInfo(model string) EmbeddingInfo {
	switch model {
	case "text-embedding-3-small":
		return OpenAISmallEmbedding3Info()
	case "text-embedding-3-large":
		return OpenAILargeEmbedding3Info()
	case "text-embedding-ada-002":
		return OpenAIAdaEmbeddingInfo()
	case "mxbai-embed-large":
		return MxbaiEmbedLargeInfo()
	case "all-minilm", "all-minilm:22m", "all-minilm:33m":
		return AllMiniLMInfo()
	case "nomic-embed-text":
		return NomicEmbedTextInfo()
	default:
		return DefaultEmbeddingInfo(model)
	}
}
