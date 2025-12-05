package synthesizer

import (
	"context"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/prompts"
	"github.com/aqua777/go-llamaindex/schema"
)

// RefineSynthesizer iteratively refines the response across text chunks.
type RefineSynthesizer struct {
	*BaseSynthesizer
	// TextQATemplate is the prompt template for initial QA.
	TextQATemplate prompts.BasePromptTemplate
	// RefineTemplate is the prompt template for refining answers.
	RefineTemplate prompts.BasePromptTemplate
}

// RefineSynthesizerOption is a functional option for RefineSynthesizer.
type RefineSynthesizerOption func(*RefineSynthesizer)

// WithRefineTextQATemplate sets the QA template.
func WithRefineTextQATemplate(template prompts.BasePromptTemplate) RefineSynthesizerOption {
	return func(rs *RefineSynthesizer) {
		rs.TextQATemplate = template
	}
}

// WithRefineTemplate sets the refine template.
func WithRefineTemplate(template prompts.BasePromptTemplate) RefineSynthesizerOption {
	return func(rs *RefineSynthesizer) {
		rs.RefineTemplate = template
	}
}

// NewRefineSynthesizer creates a new RefineSynthesizer.
func NewRefineSynthesizer(llmModel llm.LLM, opts ...RefineSynthesizerOption) *RefineSynthesizer {
	rs := &RefineSynthesizer{
		BaseSynthesizer: NewBaseSynthesizer(llmModel),
		TextQATemplate:  prompts.DefaultTextQAPrompt,
		RefineTemplate:  prompts.DefaultRefinePrompt,
	}

	for _, opt := range opts {
		opt(rs)
	}

	// Register prompts
	rs.SetPrompt("text_qa_template", rs.TextQATemplate)
	rs.SetPrompt("refine_template", rs.RefineTemplate)

	return rs
}

// Synthesize generates a response from the query and source nodes.
func (rs *RefineSynthesizer) Synthesize(ctx context.Context, query string, nodes []schema.NodeWithScore) (*Response, error) {
	if len(nodes) == 0 {
		return NewResponse("Empty Response", nil), nil
	}

	textChunks := GetTextChunksFromNodes(nodes, schema.MetadataModeLLM)
	responseStr, err := rs.GetResponse(ctx, query, textChunks)
	if err != nil {
		return nil, err
	}

	return rs.PrepareResponseOutput(responseStr, nodes), nil
}

// GetResponse generates a response from query and text chunks using refinement.
func (rs *RefineSynthesizer) GetResponse(ctx context.Context, query string, textChunks []string) (string, error) {
	if len(textChunks) == 0 {
		return "Empty Response", nil
	}

	var response string
	var err error

	for i, textChunk := range textChunks {
		if i == 0 {
			// First chunk: generate initial response
			response, err = rs.giveResponseSingle(ctx, query, textChunk)
			if err != nil {
				return "", err
			}
		} else {
			// Subsequent chunks: refine the response
			response, err = rs.refineResponseSingle(ctx, response, query, textChunk)
			if err != nil {
				return "", err
			}
		}
	}

	if response == "" {
		response = "Empty Response"
	}

	return response, nil
}

// giveResponseSingle generates initial response from a single chunk.
func (rs *RefineSynthesizer) giveResponseSingle(ctx context.Context, query, textChunk string) (string, error) {
	prompt := rs.TextQATemplate.Format(map[string]string{
		"query_str":   query,
		"context_str": textChunk,
	})

	return rs.LLM.Complete(ctx, prompt)
}

// refineResponseSingle refines an existing response with new context.
func (rs *RefineSynthesizer) refineResponseSingle(ctx context.Context, existingAnswer, query, textChunk string) (string, error) {
	prompt := rs.RefineTemplate.Format(map[string]string{
		"query_str":       query,
		"existing_answer": existingAnswer,
		"context_msg":     textChunk,
	})

	return rs.LLM.Complete(ctx, prompt)
}

// Ensure RefineSynthesizer implements Synthesizer.
var _ Synthesizer = (*RefineSynthesizer)(nil)
