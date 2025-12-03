package synthesizer

import (
	"context"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/prompts"
	"github.com/aqua777/go-llamaindex/schema"
)

// TreeSummarizeSynthesizer recursively summarizes text chunks in a tree structure.
// It builds a tree from leaves to root, summarizing at each level.
type TreeSummarizeSynthesizer struct {
	*BaseSynthesizer
	// SummaryTemplate is the prompt template for summarization.
	SummaryTemplate prompts.BasePromptTemplate
	// MaxChunkSize is the maximum size for repacked chunks.
	MaxChunkSize int
}

// TreeSummarizeSynthesizerOption is a functional option.
type TreeSummarizeSynthesizerOption func(*TreeSummarizeSynthesizer)

// WithSummaryTemplate sets the summary template.
func WithSummaryTemplate(template prompts.BasePromptTemplate) TreeSummarizeSynthesizerOption {
	return func(ts *TreeSummarizeSynthesizer) {
		ts.SummaryTemplate = template
	}
}

// WithTreeMaxChunkSize sets the maximum chunk size.
func WithTreeMaxChunkSize(size int) TreeSummarizeSynthesizerOption {
	return func(ts *TreeSummarizeSynthesizer) {
		ts.MaxChunkSize = size
	}
}

// NewTreeSummarizeSynthesizer creates a new TreeSummarizeSynthesizer.
func NewTreeSummarizeSynthesizer(llmModel llm.LLM, opts ...TreeSummarizeSynthesizerOption) *TreeSummarizeSynthesizer {
	ts := &TreeSummarizeSynthesizer{
		BaseSynthesizer: NewBaseSynthesizer(llmModel),
		SummaryTemplate: prompts.DefaultTreeSummarizePrompt,
		MaxChunkSize:    4096,
	}

	for _, opt := range opts {
		opt(ts)
	}

	// Register prompt
	ts.SetPrompt("summary_template", ts.SummaryTemplate)

	return ts
}

// Synthesize generates a response from the query and source nodes.
func (ts *TreeSummarizeSynthesizer) Synthesize(ctx context.Context, query string, nodes []schema.NodeWithScore) (*Response, error) {
	if len(nodes) == 0 {
		return NewResponse("Empty Response", nil), nil
	}

	textChunks := GetTextChunksFromNodes(nodes, schema.MetadataModeLLM)
	responseStr, err := ts.GetResponse(ctx, query, textChunks)
	if err != nil {
		return nil, err
	}

	return ts.PrepareResponseOutput(responseStr, nodes), nil
}

// GetResponse generates a response using tree summarization.
func (ts *TreeSummarizeSynthesizer) GetResponse(ctx context.Context, query string, textChunks []string) (string, error) {
	if len(textChunks) == 0 {
		return "Empty Response", nil
	}

	// Repack chunks to better utilize context window
	repackedChunks := CompactTextChunks(textChunks, ts.MaxChunkSize, "\n\n")

	if ts.Verbose {
		// Could add logging here
	}

	// Base case: single chunk, generate final response
	if len(repackedChunks) == 1 {
		return ts.summarizeChunk(ctx, query, repackedChunks[0])
	}

	// Recursive case: summarize each chunk, then recursively summarize summaries
	summaries := make([]string, len(repackedChunks))
	for i, chunk := range repackedChunks {
		summary, err := ts.summarizeChunk(ctx, query, chunk)
		if err != nil {
			return "", err
		}
		summaries[i] = summary
	}

	// Recursively summarize the summaries
	return ts.GetResponse(ctx, query, summaries)
}

// summarizeChunk summarizes a single chunk.
func (ts *TreeSummarizeSynthesizer) summarizeChunk(ctx context.Context, query, chunk string) (string, error) {
	prompt := ts.SummaryTemplate.Format(map[string]string{
		"query_str":   query,
		"context_str": chunk,
	})

	return ts.LLM.Complete(ctx, prompt)
}

// Ensure TreeSummarizeSynthesizer implements Synthesizer.
var _ Synthesizer = (*TreeSummarizeSynthesizer)(nil)
