package synthesizer

import (
	"context"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/prompts"
	"github.com/aqua777/go-llamaindex/schema"
)

// AccumulateSynthesizer generates a response for each chunk and concatenates them.
type AccumulateSynthesizer struct {
	*BaseSynthesizer
	// TextQATemplate is the prompt template for QA.
	TextQATemplate prompts.BasePromptTemplate
	// Separator is the separator between accumulated responses.
	Separator string
}

// AccumulateSynthesizerOption is a functional option.
type AccumulateSynthesizerOption func(*AccumulateSynthesizer)

// WithAccumulateTextQATemplate sets the QA template.
func WithAccumulateTextQATemplate(template prompts.BasePromptTemplate) AccumulateSynthesizerOption {
	return func(as *AccumulateSynthesizer) {
		as.TextQATemplate = template
	}
}

// WithAccumulateSeparator sets the separator between responses.
func WithAccumulateSeparator(sep string) AccumulateSynthesizerOption {
	return func(as *AccumulateSynthesizer) {
		as.Separator = sep
	}
}

// NewAccumulateSynthesizer creates a new AccumulateSynthesizer.
func NewAccumulateSynthesizer(llmModel llm.LLM, opts ...AccumulateSynthesizerOption) *AccumulateSynthesizer {
	as := &AccumulateSynthesizer{
		BaseSynthesizer: NewBaseSynthesizer(llmModel),
		TextQATemplate:  prompts.DefaultTextQAPrompt,
		Separator:       "\n\n",
	}

	for _, opt := range opts {
		opt(as)
	}

	// Register prompt
	as.SetPrompt("text_qa_template", as.TextQATemplate)

	return as
}

// Synthesize generates a response from the query and source nodes.
func (as *AccumulateSynthesizer) Synthesize(ctx context.Context, query string, nodes []schema.NodeWithScore) (*Response, error) {
	if len(nodes) == 0 {
		return NewResponse("Empty Response", nil), nil
	}

	textChunks := GetTextChunksFromNodes(nodes, schema.MetadataModeLLM)
	responseStr, err := as.GetResponse(ctx, query, textChunks)
	if err != nil {
		return nil, err
	}

	return as.PrepareResponseOutput(responseStr, nodes), nil
}

// GetResponse generates a response for each chunk and concatenates them.
func (as *AccumulateSynthesizer) GetResponse(ctx context.Context, query string, textChunks []string) (string, error) {
	if len(textChunks) == 0 {
		return "Empty Response", nil
	}

	responses := make([]string, 0, len(textChunks))

	for _, chunk := range textChunks {
		prompt := as.TextQATemplate.Format(map[string]string{
			"query_str":   query,
			"context_str": chunk,
		})

		response, err := as.LLM.Complete(ctx, prompt)
		if err != nil {
			return "", err
		}

		responses = append(responses, response)
	}

	return strings.Join(responses, as.Separator), nil
}

// Ensure AccumulateSynthesizer implements Synthesizer.
var _ Synthesizer = (*AccumulateSynthesizer)(nil)

// CompactAccumulateSynthesizer compacts chunks before accumulating.
type CompactAccumulateSynthesizer struct {
	*AccumulateSynthesizer
	// MaxChunkSize is the maximum size for compacted chunks.
	MaxChunkSize int
	// ChunkSeparator is the separator between chunks when compacting.
	ChunkSeparator string
}

// CompactAccumulateSynthesizerOption is a functional option.
type CompactAccumulateSynthesizerOption func(*CompactAccumulateSynthesizer)

// WithCompactAccumulateMaxChunkSize sets the maximum chunk size.
func WithCompactAccumulateMaxChunkSize(size int) CompactAccumulateSynthesizerOption {
	return func(cas *CompactAccumulateSynthesizer) {
		cas.MaxChunkSize = size
	}
}

// NewCompactAccumulateSynthesizer creates a new CompactAccumulateSynthesizer.
func NewCompactAccumulateSynthesizer(llmModel llm.LLM, opts ...CompactAccumulateSynthesizerOption) *CompactAccumulateSynthesizer {
	cas := &CompactAccumulateSynthesizer{
		AccumulateSynthesizer: NewAccumulateSynthesizer(llmModel),
		MaxChunkSize:          4096,
		ChunkSeparator:        "\n\n",
	}

	for _, opt := range opts {
		opt(cas)
	}

	return cas
}

// GetResponse compacts chunks then accumulates responses.
func (cas *CompactAccumulateSynthesizer) GetResponse(ctx context.Context, query string, textChunks []string) (string, error) {
	// Compact chunks first
	compactedChunks := CompactTextChunks(textChunks, cas.MaxChunkSize, cas.ChunkSeparator)

	// Use accumulate logic on compacted chunks
	return cas.AccumulateSynthesizer.GetResponse(ctx, query, compactedChunks)
}

// Ensure CompactAccumulateSynthesizer implements Synthesizer.
var _ Synthesizer = (*CompactAccumulateSynthesizer)(nil)
