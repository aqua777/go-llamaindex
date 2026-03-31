package nodeparser

import (
	"strings"

	"github.com/aqua777/go-llamaindex/schema"
	"golang.org/x/net/html"
)

// Metadata key for the source HTML element tag name on extracted nodes.
const MetadataKeyHTMLTag = "html_tag"

// htmlTagFragment is one extracted block of text and its element tag name.
type htmlTagFragment struct {
	Tag  string
	Text string
}

// HTMLNodeParser parses HTML documents into nodes based on HTML tags.
type HTMLNodeParser struct {
	*BaseNodeParser
	tagsToExtract []string
}

var _ NodeParser = (*HTMLNodeParser)(nil)

// NewHTMLNodeParser creates a new HTMLNodeParser with default tags (p, h1–h6, li, b, i, u, section).
func NewHTMLNodeParser() *HTMLNodeParser {
	return NewHTMLNodeParserWithTags(cloneStringSlice(defaultHTMLExtractTags()))
}

// NewHTMLNodeParserWithTags creates a new HTMLNodeParser extracting specific tags.
func NewHTMLNodeParserWithTags(tags []string) *HTMLNodeParser {
	cp := cloneStringSlice(tags)
	return &HTMLNodeParser{
		BaseNodeParser: NewBaseNodeParser(),
		tagsToExtract:  cp,
	}
}

// WithIncludeMetadata sets whether to include parent metadata in child nodes.
func (p *HTMLNodeParser) WithIncludeMetadata(include bool) NodeParserWithOptions {
	p.BaseNodeParser.WithIncludeMetadata(include)
	return p
}

// WithIncludePrevNextRel sets whether to establish PREVIOUS/NEXT relationships.
func (p *HTMLNodeParser) WithIncludePrevNextRel(include bool) NodeParserWithOptions {
	p.BaseNodeParser.WithIncludePrevNextRel(include)
	return p
}

// GetNodesFromDocuments parses documents into nodes.
func (p *HTMLNodeParser) GetNodesFromDocuments(documents []schema.Document) []*schema.Node {
	var allNodes []*schema.Node
	tagSet := htmlTagSetFromList(p.tagsToExtract)

	for _, doc := range documents {
		p.appendParsedHTMLForSource(&allNodes, tagSet, doc.ID, doc.Text, nil, &doc, "source_doc_id", doc.ID)
	}

	return allNodes
}

// ParseNodes parses existing nodes into smaller nodes.
func (p *HTMLNodeParser) ParseNodes(nodes []*schema.Node) []*schema.Node {
	var allNodes []*schema.Node
	tagSet := htmlTagSetFromList(p.tagsToExtract)

	for _, node := range nodes {
		p.appendParsedHTMLForSource(&allNodes, tagSet, node.ID, node.Text, node, nil, "source_node_id", node.ID)
	}

	return allNodes
}

func (p *HTMLNodeParser) appendParsedHTMLForSource(
	allNodes *[]*schema.Node,
	tagSet map[string]struct{},
	id string,
	text string,
	parentNode *schema.Node,
	parentDoc *schema.Document,
	sourceMetaKey string,
	sourceMetaVal string,
) {
	p.EmitStart(id)

	frags, err := htmlFragmentsFromString(text, tagSet)
	if err != nil {
		p.EmitError(id, err)
		return
	}

	nodes := buildNodesFromHTMLFragments(p.BaseNodeParser, frags, parentNode, parentDoc)
	for _, n := range nodes {
		n.Metadata[sourceMetaKey] = sourceMetaVal
	}

	*allNodes = append(*allNodes, nodes...)
	p.EmitComplete(id, len(nodes))
}

func defaultHTMLExtractTags() []string {
	return []string{
		"p", "h1", "h2", "h3", "h4", "h5", "h6",
		"li", "b", "i", "u", "section",
	}
}

func cloneStringSlice(s []string) []string {
	if len(s) == 0 {
		return nil
	}
	out := make([]string, len(s))
	copy(out, s)
	return out
}

func normalizeHTMLTagName(s string) string {
	return strings.ToLower(strings.TrimSpace(s))
}

func htmlTagSetFromList(tags []string) map[string]struct{} {
	set := make(map[string]struct{})
	for _, t := range tags {
		n := normalizeHTMLTagName(t)
		if n == "" {
			continue
		}
		set[n] = struct{}{}
	}
	return set
}

func htmlTextContentSkippingUnsafe(n *html.Node) string {
	if n == nil {
		return ""
	}
	var b strings.Builder
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		walkHTMLTextContent(c, &b)
	}
	return strings.TrimSpace(b.String())
}

func walkHTMLTextContent(cur *html.Node, b *strings.Builder) {
	if cur == nil {
		return
	}
	switch cur.Type {
	case html.TextNode:
		b.WriteString(cur.Data)
	case html.ElementNode:
		switch strings.ToLower(cur.Data) {
		case "script", "style", "noscript":
			return
		}
	}
	for c := cur.FirstChild; c != nil; c = c.NextSibling {
		walkHTMLTextContent(c, b)
	}
}

func walkCollectHTMLFragments(n *html.Node, tagSet map[string]struct{}, out *[]htmlTagFragment) {
	if n == nil {
		return
	}
	if n.Type == html.ElementNode {
		tag := strings.ToLower(n.Data)
		if _, ok := tagSet[tag]; ok {
			text := htmlTextContentSkippingUnsafe(n)
			if text != "" {
				*out = append(*out, htmlTagFragment{Tag: tag, Text: text})
			}
		}
	}
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		walkCollectHTMLFragments(c, tagSet, out)
	}
}

func htmlFragmentsFromString(htmlStr string, tagSet map[string]struct{}) ([]htmlTagFragment, error) {
	if len(tagSet) == 0 {
		return nil, nil
	}
	doc, err := html.Parse(strings.NewReader(htmlStr))
	if err != nil {
		return nil, err
	}
	var frags []htmlTagFragment
	walkCollectHTMLFragments(doc, tagSet, &frags)
	return frags, nil
}

func buildNodesFromHTMLFragments(
	base *BaseNodeParser,
	fragments []htmlTagFragment,
	parentNode *schema.Node,
	parentDoc *schema.Document,
) []*schema.Node {
	if len(fragments) == 0 {
		return nil
	}
	splits := make([]string, len(fragments))
	for i := range fragments {
		splits[i] = fragments[i].Text
	}
	nodes := make([]*schema.Node, len(splits))
	for i, text := range splits {
		node := schema.NewNode()
		node.ID = base.GenerateID()
		node.Text = text
		node.Type = schema.ObjectTypeText
		node.Metadata[MetadataKeyHTMLTag] = fragments[i].Tag
		node.Metadata["chunk_index"] = i
		node.Metadata["chunk_count"] = len(splits)
		node.Hash = node.GenerateHash()
		nodes[i] = node
	}
	return base.PostProcessNodes(nodes, parentNode, parentDoc)
}
