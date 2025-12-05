package extractors

import (
	"context"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/schema"
)

// DefaultSummaryExtractTemplate is the default template for summary extraction.
const DefaultSummaryExtractTemplate = `Here is the content of the section:
{context_str}

Summarize the key topics and entities of the section.

Summary: `

// SummaryType specifies which summaries to extract.
type SummaryType string

const (
	// SummaryTypeSelf extracts summary for the current node.
	SummaryTypeSelf SummaryType = "self"
	// SummaryTypePrev extracts summary from the previous node.
	SummaryTypePrev SummaryType = "prev"
	// SummaryTypeNext extracts summary from the next node.
	SummaryTypeNext SummaryType = "next"
)

// SummaryExtractor extracts summaries from nodes.
// It can extract summaries for the current node, as well as adjacent nodes
// (previous and next) to provide context.
type SummaryExtractor struct {
	*LLMExtractor
	summaries      []SummaryType // which summaries to extract
	promptTemplate string        // template for summary extraction
}

// SummaryExtractorOption configures a SummaryExtractor.
type SummaryExtractorOption func(*SummaryExtractor)

// WithSummaryTypes sets which summaries to extract.
func WithSummaryTypes(types ...SummaryType) SummaryExtractorOption {
	return func(e *SummaryExtractor) {
		e.summaries = types
	}
}

// WithSummaryPromptTemplate sets the summary extraction template.
func WithSummaryPromptTemplate(template string) SummaryExtractorOption {
	return func(e *SummaryExtractor) {
		e.promptTemplate = template
	}
}

// WithSummaryLLM sets the LLM for summary extraction.
func WithSummaryLLM(l llm.LLM) SummaryExtractorOption {
	return func(e *SummaryExtractor) {
		e.llm = l
	}
}

// NewSummaryExtractor creates a new SummaryExtractor.
func NewSummaryExtractor(opts ...SummaryExtractorOption) *SummaryExtractor {
	e := &SummaryExtractor{
		LLMExtractor: NewLLMExtractor(
			[]BaseExtractorOption{
				WithExtractorName("SummaryExtractor"),
				WithTextNodeOnly(true),
			},
		),
		summaries:      []SummaryType{SummaryTypeSelf},
		promptTemplate: DefaultSummaryExtractTemplate,
	}

	for _, opt := range opts {
		opt(e)
	}

	return e
}

// hasSummaryType checks if a summary type is in the list.
func (e *SummaryExtractor) hasSummaryType(t SummaryType) bool {
	for _, s := range e.summaries {
		if s == t {
			return true
		}
	}
	return false
}

// Extract extracts summaries from nodes.
// Returns metadata with "section_summary", "prev_section_summary", and/or
// "next_section_summary" fields depending on configuration.
func (e *SummaryExtractor) Extract(ctx context.Context, nodes []*schema.Node) ([]ExtractedMetadata, error) {
	if e.llm == nil {
		return nil, fmt.Errorf("LLM must be provided for SummaryExtractor")
	}

	if len(nodes) == 0 {
		return []ExtractedMetadata{}, nil
	}

	// Generate summaries for all nodes
	summaries, err := e.generateSummaries(ctx, nodes)
	if err != nil {
		return nil, err
	}

	// Build result with appropriate summary fields
	result := make([]ExtractedMetadata, len(nodes))
	for i := range nodes {
		result[i] = make(ExtractedMetadata)

		// Add self summary
		if e.hasSummaryType(SummaryTypeSelf) && summaries[i] != "" {
			result[i]["section_summary"] = summaries[i]
		}

		// Add previous section summary
		if e.hasSummaryType(SummaryTypePrev) && i > 0 && summaries[i-1] != "" {
			result[i]["prev_section_summary"] = summaries[i-1]
		}

		// Add next section summary
		if e.hasSummaryType(SummaryTypeNext) && i < len(nodes)-1 && summaries[i+1] != "" {
			result[i]["next_section_summary"] = summaries[i+1]
		}
	}

	return result, nil
}

// generateSummaries generates summaries for all nodes.
func (e *SummaryExtractor) generateSummaries(ctx context.Context, nodes []*schema.Node) ([]string, error) {
	return runConcurrent(ctx, nodes, e.numWorkers, func(ctx context.Context, node *schema.Node, _ int) (string, error) {
		return e.generateNodeSummary(ctx, node)
	})
}

// generateNodeSummary generates a summary for a single node.
func (e *SummaryExtractor) generateNodeSummary(ctx context.Context, node *schema.Node) (string, error) {
	content := e.GetNodeContent(node)
	prompt := formatPrompt(e.promptTemplate, map[string]string{
		"context_str": content,
	})

	summary, err := e.llm.Complete(ctx, prompt)
	if err != nil {
		return "", fmt.Errorf("failed to generate summary: %w", err)
	}

	return strings.TrimSpace(summary), nil
}

// ProcessNodes extracts summaries and updates nodes.
func (e *SummaryExtractor) ProcessNodes(ctx context.Context, nodes []*schema.Node) ([]*schema.Node, error) {
	var newNodes []*schema.Node
	if e.inPlace {
		newNodes = nodes
	} else {
		newNodes = make([]*schema.Node, len(nodes))
		for i, node := range nodes {
			nodeCopy := *node
			newNodes[i] = &nodeCopy
		}
	}

	metadataList, err := e.Extract(ctx, newNodes)
	if err != nil {
		return nil, err
	}

	for i, node := range newNodes {
		if node.Metadata == nil {
			node.Metadata = make(map[string]interface{})
		}
		for k, v := range metadataList[i] {
			node.Metadata[k] = v
		}
	}

	return newNodes, nil
}

// Ensure SummaryExtractor implements MetadataExtractor.
var _ MetadataExtractor = (*SummaryExtractor)(nil)
