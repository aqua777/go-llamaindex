package extractors

import (
	"context"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/schema"
)

// DefaultKeywordExtractTemplate is the default template for keyword extraction.
const DefaultKeywordExtractTemplate = `{context_str}. Give {keywords} unique keywords for this document. Format as comma separated. Keywords: `

// KeywordsExtractor extracts keywords from nodes.
// It uses an LLM to identify unique keywords that characterize each node's content.
type KeywordsExtractor struct {
	*LLMExtractor
	keywords       int    // number of keywords to extract
	promptTemplate string // template for keyword extraction
}

// KeywordsExtractorOption configures a KeywordsExtractor.
type KeywordsExtractorOption func(*KeywordsExtractor)

// WithKeywordsCount sets the number of keywords to extract.
func WithKeywordsCount(n int) KeywordsExtractorOption {
	return func(e *KeywordsExtractor) {
		if n > 0 {
			e.keywords = n
		}
	}
}

// WithKeywordsPromptTemplate sets the keyword extraction template.
func WithKeywordsPromptTemplate(template string) KeywordsExtractorOption {
	return func(e *KeywordsExtractor) {
		e.promptTemplate = template
	}
}

// WithKeywordsLLM sets the LLM for keyword extraction.
func WithKeywordsLLM(l llm.LLM) KeywordsExtractorOption {
	return func(e *KeywordsExtractor) {
		e.llm = l
	}
}

// NewKeywordsExtractor creates a new KeywordsExtractor.
func NewKeywordsExtractor(opts ...KeywordsExtractorOption) *KeywordsExtractor {
	e := &KeywordsExtractor{
		LLMExtractor: NewLLMExtractor(
			[]BaseExtractorOption{
				WithExtractorName("KeywordsExtractor"),
				WithTextNodeOnly(true),
			},
		),
		keywords:       5,
		promptTemplate: DefaultKeywordExtractTemplate,
	}

	for _, opt := range opts {
		opt(e)
	}

	return e
}

// Extract extracts keywords from nodes.
// Returns metadata with "excerpt_keywords" field for each node.
func (e *KeywordsExtractor) Extract(ctx context.Context, nodes []*schema.Node) ([]ExtractedMetadata, error) {
	if e.llm == nil {
		return nil, fmt.Errorf("LLM must be provided for KeywordsExtractor")
	}

	if len(nodes) == 0 {
		return []ExtractedMetadata{}, nil
	}

	return runConcurrent(ctx, nodes, e.numWorkers, func(ctx context.Context, node *schema.Node, _ int) (ExtractedMetadata, error) {
		return e.extractKeywordsFromNode(ctx, node)
	})
}

// extractKeywordsFromNode extracts keywords from a single node.
func (e *KeywordsExtractor) extractKeywordsFromNode(ctx context.Context, node *schema.Node) (ExtractedMetadata, error) {
	content := e.GetNodeContent(node)
	prompt := formatPrompt(e.promptTemplate, map[string]string{
		"context_str": content,
		"keywords":    fmt.Sprintf("%d", e.keywords),
	})

	keywords, err := e.llm.Complete(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("failed to extract keywords: %w", err)
	}

	return ExtractedMetadata{
		"excerpt_keywords": strings.TrimSpace(keywords),
	}, nil
}

// ParseKeywords parses a comma-separated keyword string into a slice.
func ParseKeywords(keywordsStr string) []string {
	parts := strings.Split(keywordsStr, ",")
	keywords := make([]string, 0, len(parts))
	for _, part := range parts {
		keyword := strings.TrimSpace(part)
		if keyword != "" {
			keywords = append(keywords, keyword)
		}
	}
	return keywords
}

// ProcessNodes extracts keywords and updates nodes.
func (e *KeywordsExtractor) ProcessNodes(ctx context.Context, nodes []*schema.Node) ([]*schema.Node, error) {
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

// Ensure KeywordsExtractor implements MetadataExtractor.
var _ MetadataExtractor = (*KeywordsExtractor)(nil)
