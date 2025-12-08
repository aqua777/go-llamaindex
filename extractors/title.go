package extractors

import (
	"context"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/schema"
)

// DefaultTitleNodeTemplate is the default template for extracting title candidates from nodes.
const DefaultTitleNodeTemplate = `Context: {context_str}. Give a title that summarizes all of the unique entities, titles or themes found in the context. Title: `

// DefaultTitleCombineTemplate is the default template for combining title candidates.
const DefaultTitleCombineTemplate = `{context_str}. Based on the above candidate titles and content, what is the comprehensive title for this document? Title: `

// TitleExtractor extracts document titles from nodes.
// It processes multiple nodes to generate title candidates and combines them
// into a comprehensive document title.
type TitleExtractor struct {
	*LLMExtractor
	nodes           int    // number of nodes to use for title extraction
	nodeTemplate    string // template for node-level title extraction
	combineTemplate string // template for combining titles
}

// TitleExtractorOption configures a TitleExtractor.
type TitleExtractorOption func(*TitleExtractor)

// WithTitleNodes sets the number of nodes to use for title extraction.
func WithTitleNodes(n int) TitleExtractorOption {
	return func(e *TitleExtractor) {
		if n > 0 {
			e.nodes = n
		}
	}
}

// WithTitleNodeTemplate sets the node-level title extraction template.
func WithTitleNodeTemplate(template string) TitleExtractorOption {
	return func(e *TitleExtractor) {
		e.nodeTemplate = template
	}
}

// WithTitleCombineTemplate sets the title combination template.
func WithTitleCombineTemplate(template string) TitleExtractorOption {
	return func(e *TitleExtractor) {
		e.combineTemplate = template
	}
}

// WithTitleLLM sets the LLM for title extraction.
func WithTitleLLM(l llm.LLM) TitleExtractorOption {
	return func(e *TitleExtractor) {
		e.llm = l
	}
}

// NewTitleExtractor creates a new TitleExtractor.
func NewTitleExtractor(opts ...TitleExtractorOption) *TitleExtractor {
	e := &TitleExtractor{
		LLMExtractor: NewLLMExtractor(
			[]BaseExtractorOption{
				WithExtractorName("TitleExtractor"),
				WithTextNodeOnly(false), // can work with mixed node types
			},
		),
		nodes:           5,
		nodeTemplate:    DefaultTitleNodeTemplate,
		combineTemplate: DefaultTitleCombineTemplate,
	}

	for _, opt := range opts {
		opt(e)
	}

	return e
}

// Extract extracts document titles from nodes.
// Returns metadata with "document_title" field for each node.
func (e *TitleExtractor) Extract(ctx context.Context, nodes []*schema.Node) ([]ExtractedMetadata, error) {
	if e.llm == nil {
		return nil, fmt.Errorf("LLM must be provided for TitleExtractor")
	}

	if len(nodes) == 0 {
		return []ExtractedMetadata{}, nil
	}

	// Group nodes by document ID
	nodesByDocID := e.separateNodesByRefID(nodes)

	// Extract titles for each document
	titlesByDocID, err := e.extractTitles(ctx, nodesByDocID)
	if err != nil {
		return nil, err
	}

	// Build result with document_title for each node
	result := make([]ExtractedMetadata, len(nodes))
	for i, node := range nodes {
		docID := e.getRefDocID(node)
		title := titlesByDocID[docID]
		result[i] = ExtractedMetadata{
			"document_title": title,
		}
	}

	return result, nil
}

// separateNodesByRefID groups nodes by their reference document ID.
func (e *TitleExtractor) separateNodesByRefID(nodes []*schema.Node) map[string][]*schema.Node {
	separated := make(map[string][]*schema.Node)

	for _, node := range nodes {
		key := e.getRefDocID(node)
		if len(separated[key]) < e.nodes {
			separated[key] = append(separated[key], node)
		}
	}

	return separated
}

// getRefDocID gets the reference document ID from a node.
func (e *TitleExtractor) getRefDocID(node *schema.Node) string {
	// Check for ref_doc_id in metadata
	if node.Metadata != nil {
		if refDocID, ok := node.Metadata["ref_doc_id"].(string); ok && refDocID != "" {
			return refDocID
		}
	}
	// Fall back to node ID
	return node.ID
}

// extractTitles extracts titles for each document group.
func (e *TitleExtractor) extractTitles(ctx context.Context, nodesByDocID map[string][]*schema.Node) (map[string]string, error) {
	titles := make(map[string]string)

	for docID, docNodes := range nodesByDocID {
		// Get title candidates from each node
		candidates, err := e.getTitleCandidates(ctx, docNodes)
		if err != nil {
			return nil, err
		}

		// Combine candidates into final title
		combinedCandidates := strings.Join(candidates, ", ")
		prompt := formatPrompt(e.combineTemplate, map[string]string{
			"context_str": combinedCandidates,
		})

		title, err := e.llm.Complete(ctx, prompt)
		if err != nil {
			return nil, fmt.Errorf("failed to combine titles: %w", err)
		}

		titles[docID] = strings.TrimSpace(title)
	}

	return titles, nil
}

// getTitleCandidates extracts title candidates from nodes.
func (e *TitleExtractor) getTitleCandidates(ctx context.Context, nodes []*schema.Node) ([]string, error) {
	candidates := make([]string, 0, len(nodes))

	for _, node := range nodes {
		content := e.GetNodeContent(node)
		prompt := formatPrompt(e.nodeTemplate, map[string]string{
			"context_str": content,
		})

		candidate, err := e.llm.Complete(ctx, prompt)
		if err != nil {
			return nil, fmt.Errorf("failed to extract title candidate: %w", err)
		}

		candidates = append(candidates, strings.TrimSpace(candidate))
	}

	return candidates, nil
}

// ProcessNodes extracts titles and updates nodes.
func (e *TitleExtractor) ProcessNodes(ctx context.Context, nodes []*schema.Node) ([]*schema.Node, error) {
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

// Ensure TitleExtractor implements MetadataExtractor.
var _ MetadataExtractor = (*TitleExtractor)(nil)
