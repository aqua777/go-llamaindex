package synthesizer

import (
	"context"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/schema"
)

// CompactAndRefineSynthesizer compacts text chunks before refining.
// This reduces the number of LLM calls by combining chunks that fit
// within the context window.
type CompactAndRefineSynthesizer struct {
	*RefineSynthesizer
	// MaxChunkSize is the maximum size for compacted chunks.
	MaxChunkSize int
	// ChunkSeparator is the separator between chunks when compacting.
	ChunkSeparator string
}

// CompactAndRefineSynthesizerOption is a functional option.
type CompactAndRefineSynthesizerOption func(*CompactAndRefineSynthesizer)

// WithMaxChunkSize sets the maximum chunk size for compaction.
func WithMaxChunkSize(size int) CompactAndRefineSynthesizerOption {
	return func(cs *CompactAndRefineSynthesizer) {
		cs.MaxChunkSize = size
	}
}

// WithChunkSeparator sets the separator for compacted chunks.
func WithChunkSeparator(sep string) CompactAndRefineSynthesizerOption {
	return func(cs *CompactAndRefineSynthesizer) {
		cs.ChunkSeparator = sep
	}
}

// NewCompactAndRefineSynthesizer creates a new CompactAndRefineSynthesizer.
func NewCompactAndRefineSynthesizer(llmModel llm.LLM, opts ...CompactAndRefineSynthesizerOption) *CompactAndRefineSynthesizer {
	cs := &CompactAndRefineSynthesizer{
		RefineSynthesizer: NewRefineSynthesizer(llmModel),
		MaxChunkSize:      4096, // Default context window estimate
		ChunkSeparator:    "\n\n",
	}

	for _, opt := range opts {
		opt(cs)
	}

	return cs
}

// Synthesize generates a response from the query and source nodes.
func (cs *CompactAndRefineSynthesizer) Synthesize(ctx context.Context, query string, nodes []schema.NodeWithScore) (*Response, error) {
	if len(nodes) == 0 {
		return NewResponse("Empty Response", nil), nil
	}

	textChunks := GetTextChunksFromNodes(nodes, schema.MetadataModeLLM)
	responseStr, err := cs.GetResponse(ctx, query, textChunks)
	if err != nil {
		return nil, err
	}

	return cs.PrepareResponseOutput(responseStr, nodes), nil
}

// GetResponse generates a response from query and text chunks.
func (cs *CompactAndRefineSynthesizer) GetResponse(ctx context.Context, query string, textChunks []string) (string, error) {
	// Compact chunks first
	compactedChunks := CompactTextChunks(textChunks, cs.MaxChunkSize, cs.ChunkSeparator)

	// Use refine logic on compacted chunks
	return cs.RefineSynthesizer.GetResponse(ctx, query, compactedChunks)
}

// Ensure CompactAndRefineSynthesizer implements Synthesizer.
var _ Synthesizer = (*CompactAndRefineSynthesizer)(nil)
