package rag

import (
	"context"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/schema"
)

// SimpleSynthesizer generates a response by stuffing retrieved context into a prompt.
type SimpleSynthesizer struct {
	llm llm.LLM
}

// NewSimpleSynthesizer creates a new SimpleSynthesizer.
func NewSimpleSynthesizer(llm llm.LLM) *SimpleSynthesizer {
	return &SimpleSynthesizer{
		llm: llm,
	}
}

func (s *SimpleSynthesizer) Synthesize(ctx context.Context, query schema.QueryBundle, nodes []schema.NodeWithScore) (schema.EngineResponse, error) {
	contextStr := s.formatContext(nodes)
	prompt := s.createPrompt(contextStr, query.QueryString)

	responseStr, err := s.llm.Complete(ctx, prompt)
	if err != nil {
		return schema.EngineResponse{}, fmt.Errorf("llm completion failed: %w", err)
	}

	return schema.EngineResponse{
		Response:    responseStr,
		SourceNodes: nodes,
	}, nil
}

func (s *SimpleSynthesizer) SynthesizeStream(ctx context.Context, query schema.QueryBundle, nodes []schema.NodeWithScore) (schema.StreamingEngineResponse, error) {
	contextStr := s.formatContext(nodes)
	prompt := s.createPrompt(contextStr, query.QueryString)

	tokenStream, err := s.llm.Stream(ctx, prompt)
	if err != nil {
		return schema.StreamingEngineResponse{}, fmt.Errorf("llm stream failed: %w", err)
	}

	return schema.StreamingEngineResponse{
		ResponseStream: tokenStream,
		SourceNodes:    nodes,
	}, nil
}

func (s *SimpleSynthesizer) formatContext(nodes []schema.NodeWithScore) string {
	var sb strings.Builder
	for _, n := range nodes {
		sb.WriteString(n.Node.Text)
		sb.WriteString("\n")
	}
	return sb.String()
}

func (s *SimpleSynthesizer) createPrompt(context, query string) string {
	return fmt.Sprintf("Context information is below.\n---------------------\n%s\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: %s\nAnswer:", context, query)
}
