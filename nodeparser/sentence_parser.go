package nodeparser

import (
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/textsplitter"
)

// SentenceNodeParser parses documents into nodes using sentence-based splitting.
// It wraps the SentenceSplitter and adds node relationship management.
type SentenceNodeParser struct {
	*BaseNodeParser
	splitter *textsplitter.SentenceSplitter
}

// NewSentenceNodeParser creates a new SentenceNodeParser with default settings.
func NewSentenceNodeParser() *SentenceNodeParser {
	return &SentenceNodeParser{
		BaseNodeParser: NewBaseNodeParser(),
		splitter:       textsplitter.NewSentenceSplitter(0, 0, nil, nil),
	}
}

// NewSentenceNodeParserWithConfig creates a new SentenceNodeParser with custom settings.
func NewSentenceNodeParserWithConfig(chunkSize, chunkOverlap int) *SentenceNodeParser {
	return &SentenceNodeParser{
		BaseNodeParser: NewBaseNodeParser(),
		splitter:       textsplitter.NewSentenceSplitter(chunkSize, chunkOverlap, nil, nil),
	}
}

// NewSentenceNodeParserWithSplitter creates a new SentenceNodeParser with a custom splitter.
func NewSentenceNodeParserWithSplitter(splitter *textsplitter.SentenceSplitter) *SentenceNodeParser {
	return &SentenceNodeParser{
		BaseNodeParser: NewBaseNodeParser(),
		splitter:       splitter,
	}
}

// WithIncludeMetadata sets whether to include parent metadata in child nodes.
func (p *SentenceNodeParser) WithIncludeMetadata(include bool) NodeParserWithOptions {
	p.BaseNodeParser.WithIncludeMetadata(include)
	return p
}

// WithIncludePrevNextRel sets whether to establish PREVIOUS/NEXT relationships.
func (p *SentenceNodeParser) WithIncludePrevNextRel(include bool) NodeParserWithOptions {
	p.BaseNodeParser.WithIncludePrevNextRel(include)
	return p
}

// GetNodesFromDocuments parses documents into nodes.
func (p *SentenceNodeParser) GetNodesFromDocuments(documents []schema.Document) []*schema.Node {
	var allNodes []*schema.Node

	for _, doc := range documents {
		p.EmitStart(doc.ID)

		// Split the document text
		splits := p.splitter.SplitText(doc.Text)

		// Build nodes from splits
		nodes := p.BuildNodesFromSplits(splits, nil, &doc)

		// Add document-specific metadata
		for _, node := range nodes {
			node.Metadata["source_doc_id"] = doc.ID
		}

		allNodes = append(allNodes, nodes...)

		p.EmitComplete(doc.ID, len(nodes))
	}

	return allNodes
}

// ParseNodes parses nodes into smaller nodes.
func (p *SentenceNodeParser) ParseNodes(nodes []*schema.Node) []*schema.Node {
	var allNodes []*schema.Node

	for _, node := range nodes {
		p.EmitStart(node.ID)

		// Split the node text
		splits := p.splitter.SplitText(node.Text)

		// Build child nodes from splits
		childNodes := p.BuildNodesFromSplits(splits, node, nil)

		// Add parent node metadata
		for _, childNode := range childNodes {
			childNode.Metadata["source_node_id"] = node.ID
		}

		allNodes = append(allNodes, childNodes...)

		p.EmitComplete(node.ID, len(childNodes))
	}

	return allNodes
}

// SimpleNodeParser is a basic node parser that creates one node per document.
// Useful for documents that don't need splitting.
type SimpleNodeParser struct {
	*BaseNodeParser
}

// NewSimpleNodeParser creates a new SimpleNodeParser.
func NewSimpleNodeParser() *SimpleNodeParser {
	return &SimpleNodeParser{
		BaseNodeParser: NewBaseNodeParser(),
	}
}

// WithIncludeMetadata sets whether to include parent metadata in child nodes.
func (p *SimpleNodeParser) WithIncludeMetadata(include bool) NodeParserWithOptions {
	p.BaseNodeParser.WithIncludeMetadata(include)
	return p
}

// WithIncludePrevNextRel sets whether to establish PREVIOUS/NEXT relationships.
func (p *SimpleNodeParser) WithIncludePrevNextRel(include bool) NodeParserWithOptions {
	p.BaseNodeParser.WithIncludePrevNextRel(include)
	return p
}

// GetNodesFromDocuments creates one node per document.
func (p *SimpleNodeParser) GetNodesFromDocuments(documents []schema.Document) []*schema.Node {
	nodes := make([]*schema.Node, len(documents))

	for i, doc := range documents {
		node := schema.NewNode()
		node.ID = p.GenerateID()
		node.Text = doc.Text
		node.Type = schema.ObjectTypeText
		node.Hash = node.GenerateHash()

		// Copy document metadata
		if p.options.IncludeMetadata && doc.Metadata != nil {
			for k, v := range doc.Metadata {
				node.Metadata[k] = v
			}
		}
		node.Metadata["source_doc_id"] = doc.ID

		// Set SOURCE relationship
		node.Relationships.SetSource(schema.RelatedNodeInfo{
			NodeID:   doc.ID,
			NodeType: schema.ObjectTypeDocument,
			Metadata: doc.Metadata,
		})

		nodes[i] = node
	}

	// Set PREVIOUS/NEXT relationships
	if p.options.IncludePrevNextRel {
		for i := range nodes {
			if i > 0 {
				nodes[i].Relationships.SetPrevious(nodes[i-1].AsRelatedNodeInfo())
			}
			if i < len(nodes)-1 {
				nodes[i].Relationships.SetNext(nodes[i+1].AsRelatedNodeInfo())
			}
		}
	}

	return nodes
}

// ParseNodes returns nodes unchanged (no splitting).
func (p *SimpleNodeParser) ParseNodes(nodes []*schema.Node) []*schema.Node {
	return nodes
}
