package synthesizer

import (
	"context"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/prompts"
	"github.com/aqua777/go-llamaindex/schema"
)

// SimpleSynthesizer merges all text chunks and makes a single LLM call.
type SimpleSynthesizer struct {
	*BaseSynthesizer
	// TextQATemplate is the prompt template for QA.
	TextQATemplate prompts.BasePromptTemplate
}

// SimpleSynthesizerOption is a functional option for SimpleSynthesizer.
type SimpleSynthesizerOption func(*SimpleSynthesizer)

// WithTextQATemplate sets the QA template.
func WithTextQATemplate(template prompts.BasePromptTemplate) SimpleSynthesizerOption {
	return func(ss *SimpleSynthesizer) {
		ss.TextQATemplate = template
	}
}

// NewSimpleSynthesizer creates a new SimpleSynthesizer.
func NewSimpleSynthesizer(llmModel llm.LLM, opts ...SimpleSynthesizerOption) *SimpleSynthesizer {
	ss := &SimpleSynthesizer{
		BaseSynthesizer: NewBaseSynthesizer(llmModel),
		TextQATemplate:  prompts.DefaultTextQAPrompt,
	}

	for _, opt := range opts {
		opt(ss)
	}

	// Register prompt
	ss.SetPrompt("text_qa_template", ss.TextQATemplate)

	return ss
}

// Synthesize generates a response from the query and source nodes.
func (ss *SimpleSynthesizer) Synthesize(ctx context.Context, query string, nodes []schema.NodeWithScore) (*Response, error) {
	if len(nodes) == 0 {
		return NewResponse("Empty Response", nil), nil
	}

	textChunks := GetTextChunksFromNodes(nodes, schema.MetadataModeLLM)
	responseStr, err := ss.GetResponse(ctx, query, textChunks)
	if err != nil {
		return nil, err
	}

	return ss.PrepareResponseOutput(responseStr, nodes), nil
}

// GetResponse generates a response from query and text chunks.
func (ss *SimpleSynthesizer) GetResponse(ctx context.Context, query string, textChunks []string) (string, error) {
	// Merge all chunks into one context
	contextStr := strings.Join(textChunks, "\n\n")

	// Format prompt
	prompt := ss.TextQATemplate.Format(map[string]string{
		"query_str":   query,
		"context_str": contextStr,
	})

	// Get LLM response
	response, err := ss.LLM.Complete(ctx, prompt)
	if err != nil {
		return "", err
	}

	return response, nil
}

// Ensure SimpleSynthesizer implements Synthesizer.
var _ Synthesizer = (*SimpleSynthesizer)(nil)
