package nodeparser

import (
	"regexp"
	"strings"

	"github.com/aqua777/go-llamaindex/schema"
)

// MetadataKeyUnstructuredElement is the metadata key for the unstructured block kind (e.g. paragraph).
const MetadataKeyUnstructuredElement = "unstructured_element"

var unstructuredParagraphBoundary = regexp.MustCompile(`\n\s*\n`)

// UnstructuredElementNodeParser parses plain unstructured text into element nodes by paragraph
// boundaries (blocks separated by blank lines). This mirrors coarse Unstructured.io-style
// elements without requiring an external API.
type UnstructuredElementNodeParser struct {
	*BaseNodeParser
}

var _ NodeParser = (*UnstructuredElementNodeParser)(nil)

// NewUnstructuredElementNodeParser creates a new UnstructuredElementNodeParser.
//
// Returns:
//
//	A configured UnstructuredElementNodeParser.
func NewUnstructuredElementNodeParser() *UnstructuredElementNodeParser {
	return &UnstructuredElementNodeParser{
		BaseNodeParser: NewBaseNodeParser(),
	}
}

// WithIncludeMetadata sets whether to include parent metadata in child nodes.
func (p *UnstructuredElementNodeParser) WithIncludeMetadata(include bool) NodeParserWithOptions {
	p.BaseNodeParser.WithIncludeMetadata(include)
	return p
}

// WithIncludePrevNextRel sets whether to establish PREVIOUS/NEXT relationships.
func (p *UnstructuredElementNodeParser) WithIncludePrevNextRel(include bool) NodeParserWithOptions {
	p.BaseNodeParser.WithIncludePrevNextRel(include)
	return p
}

// GetNodesFromDocuments parses documents into nodes.
func (p *UnstructuredElementNodeParser) GetNodesFromDocuments(documents []schema.Document) []*schema.Node {
	var allNodes []*schema.Node
	for _, doc := range documents {
		p.appendUnstructuredForSource(&allNodes, doc.ID, doc.Text, nil, &doc, "source_doc_id", doc.ID)
	}
	return allNodes
}

// ParseNodes parses nodes into smaller nodes.
func (p *UnstructuredElementNodeParser) ParseNodes(nodes []*schema.Node) []*schema.Node {
	var allNodes []*schema.Node
	for _, node := range nodes {
		p.appendUnstructuredForSource(&allNodes, node.ID, node.Text, node, nil, "source_node_id", node.ID)
	}
	return allNodes
}

func (p *UnstructuredElementNodeParser) appendUnstructuredForSource(
	allNodes *[]*schema.Node,
	id string,
	text string,
	parentNode *schema.Node,
	parentDoc *schema.Document,
	sourceMetaKey string,
	sourceMetaVal string,
) {
	p.EmitStart(id)
	parts := unstructuredParagraphParts(text)
	appendNodesFromParsedTextParts(p.BaseNodeParser, allNodes, id, parts, parentNode, parentDoc, sourceMetaKey, sourceMetaVal)
}

// unstructuredParagraphParts splits text into paragraph-level chunks (blank-line boundaries).
func unstructuredParagraphParts(text string) []textPart {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	raw := unstructuredParagraphBoundary.Split(text, -1)
	var parts []textPart
	for _, block := range raw {
		b := strings.TrimSpace(block)
		if b == "" {
			continue
		}
		parts = append(parts, textPart{
			Text: b,
			Meta: map[string]interface{}{MetadataKeyUnstructuredElement: "paragraph"},
		})
	}
	return parts
}
