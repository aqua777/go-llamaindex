package nodeparser

import (
	"errors"

	"github.com/aqua777/go-llamaindex/schema"
)

// LangchainTextSplitter represents a text splitter compatible with Langchain-style
// SplitText(text string) ([]string, error) (e.g. langchaingo or a user adapter).
type LangchainTextSplitter interface {
	SplitText(text string) ([]string, error)
}

// LangchainNodeParser bridges a Langchain text splitter to a LlamaIndex node parser.
type LangchainNodeParser struct {
	*BaseNodeParser
	Splitter LangchainTextSplitter
}

var _ NodeParser = (*LangchainNodeParser)(nil)

// NewLangchainNodeParser creates a new LangchainNodeParser with the given Langchain text splitter.
//
// Args:
//
//	splitter: The Langchain text splitter to use.
//
// Returns:
//
//	A configured LangchainNodeParser.
func NewLangchainNodeParser(splitter LangchainTextSplitter) *LangchainNodeParser {
	return &LangchainNodeParser{
		BaseNodeParser: NewBaseNodeParser(),
		Splitter:       splitter,
	}
}

// WithIncludeMetadata sets whether to include parent metadata in child nodes.
func (p *LangchainNodeParser) WithIncludeMetadata(include bool) NodeParserWithOptions {
	p.BaseNodeParser.WithIncludeMetadata(include)
	return p
}

// WithIncludePrevNextRel sets whether to establish PREVIOUS/NEXT relationships.
func (p *LangchainNodeParser) WithIncludePrevNextRel(include bool) NodeParserWithOptions {
	p.BaseNodeParser.WithIncludePrevNextRel(include)
	return p
}

// GetNodesFromDocuments parses documents into nodes.
func (p *LangchainNodeParser) GetNodesFromDocuments(documents []schema.Document) []*schema.Node {
	var allNodes []*schema.Node
	for _, doc := range documents {
		appendLangchainNodesForSource(
			p.BaseNodeParser, p.Splitter, &allNodes,
			doc.ID, doc.Text, nil, &doc, "source_doc_id", doc.ID,
		)
	}
	return allNodes
}

// ParseNodes parses nodes into smaller nodes.
func (p *LangchainNodeParser) ParseNodes(nodes []*schema.Node) []*schema.Node {
	var allNodes []*schema.Node
	for _, node := range nodes {
		appendLangchainNodesForSource(
			p.BaseNodeParser, p.Splitter, &allNodes,
			node.ID, node.Text, node, nil, "source_node_id", node.ID,
		)
	}
	return allNodes
}

// appendLangchainNodesForSource splits text with the Langchain splitter, builds nodes, and appends them.
func appendLangchainNodesForSource(
	base *BaseNodeParser,
	splitter LangchainTextSplitter,
	allNodes *[]*schema.Node,
	emitID string,
	text string,
	parentNode *schema.Node,
	parentDoc *schema.Document,
	sourceMetaKey, sourceMetaVal string,
) {
	base.EmitStart(emitID)
	splits, err := splitWithLangchainSplitter(splitter, text)
	if err != nil {
		base.EmitError(emitID, err)
		return
	}
	nodes := base.BuildNodesFromSplits(splits, parentNode, parentDoc)
	applySourceNodeMetadata(nodes, sourceMetaKey, sourceMetaVal)
	*allNodes = append(*allNodes, nodes...)
	base.EmitComplete(emitID, len(nodes))
}

// splitWithLangchainSplitter runs SplitText on the splitter (module-level for tests).
func splitWithLangchainSplitter(splitter LangchainTextSplitter, text string) ([]string, error) {
	if splitter == nil {
		return nil, errors.New("langchain splitter is nil")
	}
	return splitter.SplitText(text)
}
