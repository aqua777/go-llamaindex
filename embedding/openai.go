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

