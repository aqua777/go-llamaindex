package nodeparser

import (
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/schema"
	"github.com/yuin/goldmark"
	"github.com/yuin/goldmark/ast"
	"github.com/yuin/goldmark/extension"
	extast "github.com/yuin/goldmark/extension/ast"
	"github.com/yuin/goldmark/text"
)

// MetadataKeyMarkdownElement is the metadata key for the markdown block kind (e.g. heading, table).
const MetadataKeyMarkdownElement = "markdown_element"

// MarkdownElementNodeParser parses markdown documents into element nodes (e.g., text, tables).
type MarkdownElementNodeParser struct {
	*BaseNodeParser
}

var _ NodeParser = (*MarkdownElementNodeParser)(nil)

// NewMarkdownElementNodeParser creates a new MarkdownElementNodeParser.
//
// Returns:
//
//	A configured MarkdownElementNodeParser.
func NewMarkdownElementNodeParser() *MarkdownElementNodeParser {
	return &MarkdownElementNodeParser{
		BaseNodeParser: NewBaseNodeParser(),
	}
}

// WithIncludeMetadata sets whether to include parent metadata in child nodes.
func (p *MarkdownElementNodeParser) WithIncludeMetadata(include bool) NodeParserWithOptions {
	p.BaseNodeParser.WithIncludeMetadata(include)
	return p
}

// WithIncludePrevNextRel sets whether to establish PREVIOUS/NEXT relationships.
func (p *MarkdownElementNodeParser) WithIncludePrevNextRel(include bool) NodeParserWithOptions {
	p.BaseNodeParser.WithIncludePrevNextRel(include)
	return p
}

// GetNodesFromDocuments parses documents into nodes.
func (p *MarkdownElementNodeParser) GetNodesFromDocuments(documents []schema.Document) []*schema.Node {
	var allNodes []*schema.Node
	for _, doc := range documents {
		p.appendMarkdownForSource(&allNodes, doc.ID, doc.Text, nil, &doc, "source_doc_id", doc.ID)
	}
	return allNodes
}

// ParseNodes parses nodes into smaller nodes.
func (p *MarkdownElementNodeParser) ParseNodes(nodes []*schema.Node) []*schema.Node {
	var allNodes []*schema.Node
	for _, node := range nodes {
		p.appendMarkdownForSource(&allNodes, node.ID, node.Text, node, nil, "source_node_id", node.ID)
	}
	return allNodes
}

func (p *MarkdownElementNodeParser) appendMarkdownForSource(
	allNodes *[]*schema.Node,
	id string,
	text string,
	parentNode *schema.Node,
	parentDoc *schema.Document,
	sourceMetaKey string,
	sourceMetaVal string,
) {
	p.EmitStart(id)
	parts, err := markdownTopLevelTextParts([]byte(text))
	if err != nil {
		p.EmitError(id, err)
		return
	}
	appendNodesFromParsedTextParts(p.BaseNodeParser, allNodes, id, parts, parentNode, parentDoc, sourceMetaKey, sourceMetaVal)
}

func newMarkdownGoldmark() goldmark.Markdown {
	return goldmark.New(
		goldmark.WithExtensions(extension.GFM),
	)
}

func parseMarkdownDocument(md goldmark.Markdown, source []byte) (*ast.Document, error) {
	reader := text.NewReader(source)
	root := md.Parser().Parse(reader)
	doc, ok := root.(*ast.Document)
	if !ok {
		return nil, fmt.Errorf("markdown parse root: expected *ast.Document, got %T", root)
	}
	return doc, nil
}

// blockKindForMarkdownNode returns a stable label for a top-level markdown block.
func blockKindForMarkdownNode(n ast.Node) string {
	switch n.(type) {
	case *ast.Heading:
		return "heading"
	case *ast.Paragraph:
		return "paragraph"
	case *ast.FencedCodeBlock:
		return "code_block"
	case *ast.Blockquote:
		return "blockquote"
	case *ast.List:
		return "list"
	case *ast.HTMLBlock:
		return "html_block"
	case *ast.ThematicBreak:
		return "thematic_break"
	case *extast.Table:
		return "table"
	default:
		return "block"
	}
}

// blockSourceText returns the markdown source segment for a block node.
func blockSourceText(source []byte, n ast.Node) string {
	if n == nil {
		return ""
	}
	if tbl, ok := n.(*extast.Table); ok {
		return tableBlockText(source, tbl)
	}
	segs := n.Lines()
	if segs == nil || segs.Len() == 0 {
		return ""
	}
	return strings.TrimSpace(string(segs.Value(source)))
}

func collectInlineTextFromNode(source []byte, n ast.Node) string {
	if n == nil {
		return ""
	}
	var b strings.Builder
	_ = ast.Walk(n, func(node ast.Node, entering bool) (ast.WalkStatus, error) {
		if !entering {
			return ast.WalkContinue, nil
		}
		if tn, ok := node.(*ast.Text); ok {
			b.Write(tn.Segment.Value(source))
		}
		return ast.WalkContinue, nil
	})
	return strings.TrimSpace(b.String())
}

func tableBlockText(source []byte, t *extast.Table) string {
	if t == nil {
		return ""
	}
	var rows []string
	for sec := t.FirstChild(); sec != nil; sec = sec.NextSibling() {
		switch s := sec.(type) {
		case *extast.TableHeader:
			rows = append(rows, tableCellsRowText(source, s))
		case *extast.TableRow:
			rows = append(rows, tableCellsRowText(source, s))
		}
	}
	return strings.Join(rows, "\n")
}

// tableCellsRowText collects cell text from a TableHeader or TableRow (direct TableCell children).
func tableCellsRowText(source []byte, n ast.Node) string {
	var cells []string
	for c := n.FirstChild(); c != nil; c = c.NextSibling() {
		tc, ok := c.(*extast.TableCell)
		if !ok {
			continue
		}
		cells = append(cells, collectInlineTextFromNode(source, tc))
	}
	if len(cells) == 0 {
		return ""
	}
	return strings.Join(cells, " | ")
}

func markdownTopLevelTextParts(source []byte) ([]textPart, error) {
	md := newMarkdownGoldmark()
	doc, err := parseMarkdownDocument(md, source)
	if err != nil {
		return nil, err
	}
	var parts []textPart
	for c := doc.FirstChild(); c != nil; c = c.NextSibling() {
		kind := blockKindForMarkdownNode(c)
		txt := blockSourceText(source, c)
		if txt == "" {
			continue
		}
		parts = append(parts, textPart{
			Text: txt,
			Meta: map[string]interface{}{MetadataKeyMarkdownElement: kind},
		})
	}
	return parts, nil
}
