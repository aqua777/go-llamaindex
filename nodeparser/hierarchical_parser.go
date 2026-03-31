package nodeparser

import (
	"sort"

	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/textsplitter"
)

// MetadataKeyHierarchyLevel is the metadata key for the 0-based hierarchy depth.
const MetadataKeyHierarchyLevel = "hierarchy_level"

// HierarchicalNodeParser creates a hierarchy of nodes using multiple chunk sizes.
type HierarchicalNodeParser struct {
	*BaseNodeParser
	chunkSizes []int
}

var _ NodeParser = (*HierarchicalNodeParser)(nil)

// defaultHierarchicalChunkSizes returns the default chunk size ladder (largest first).
func defaultHierarchicalChunkSizes() []int {
	return []int{2048, 512, 128}
}

// normalizeHierarchicalChunkSizes returns a strictly descending positive slice.
// Empty input yields defaults; values are sorted descending and deduplicated.
func normalizeHierarchicalChunkSizes(sizes []int) []int {
	if len(sizes) == 0 {
		return cloneIntSlice(defaultHierarchicalChunkSizes())
	}
	cp := make([]int, 0, len(sizes))
	for _, s := range sizes {
		if s > 0 {
			cp = append(cp, s)
		}
	}
	if len(cp) == 0 {
		return cloneIntSlice(defaultHierarchicalChunkSizes())
	}
	sort.Slice(cp, func(i, j int) bool { return cp[i] > cp[j] })
	out := make([]int, 0, len(cp))
	var prev int
	for i, v := range cp {
		if i == 0 || v != prev {
			out = append(out, v)
		}
		prev = v
	}
	return out
}

func cloneIntSlice(s []int) []int {
	if s == nil {
		return nil
	}
	c := make([]int, len(s))
	copy(c, s)
	return c
}

// splitTextForHierarchy splits text using a SentenceSplitter at the given chunk size.
func splitTextForHierarchy(text string, chunkSize int) []string {
	if text == "" {
		return nil
	}
	s := textsplitter.NewSentenceSplitter(chunkSize, 0, nil, nil)
	parts := s.SplitText(text)
	return filterNonEmptyTextParts(parts)
}

func filterNonEmptyTextParts(parts []string) []string {
	var out []string
	for _, p := range parts {
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

// establishParentChildLinks sets PARENT on each child and CHILD on the parent.
func establishParentChildLinks(parent *schema.Node, children []*schema.Node) {
	if parent == nil || len(children) == 0 {
		return
	}
	infos := make([]schema.RelatedNodeInfo, len(children))
	for i, ch := range children {
		ch.Relationships.SetParent(parent.AsRelatedNodeInfo())
		infos[i] = ch.AsRelatedNodeInfo()
	}
	parent.Relationships.SetChildren(infos)
}

// hierarchicalNodesFromText builds all hierarchy levels and returns a flat list of nodes.
func hierarchicalNodesFromText(
	base *BaseNodeParser,
	text string,
	sizes []int,
	parentNode *schema.Node,
	parentDoc *schema.Document,
	levelOffset int,
	sourceMetaKey string,
	sourceMetaValue string,
) []*schema.Node {
	sizes = normalizeHierarchicalChunkSizes(sizes)
	if len(sizes) == 0 || text == "" {
		return nil
	}

	splits := splitTextForHierarchy(text, sizes[0])
	if len(splits) == 0 {
		return nil
	}

	nodes := base.BuildNodesFromSplits(splits, parentNode, parentDoc)
	for _, n := range nodes {
		n.Metadata[MetadataKeyHierarchyLevel] = levelOffset
		if sourceMetaKey != "" {
			n.Metadata[sourceMetaKey] = sourceMetaValue
		}
	}
	if parentNode != nil && len(nodes) > 0 {
		establishParentChildLinks(parentNode, nodes)
	}

	all := append([]*schema.Node(nil), nodes...)
	if len(sizes) == 1 {
		return all
	}

	rest := sizes[1:]
	nextLevel := levelOffset + 1
	for _, parent := range nodes {
		children := hierarchicalNodesFromText(
			base, parent.Text, rest, parent, nil, nextLevel,
			sourceMetaKey, sourceMetaValue,
		)
		if len(children) == 0 {
			continue
		}
		all = append(all, children...)
	}
	return all
}

// NewHierarchicalNodeParser creates a new HierarchicalNodeParser with default chunk sizes (2048, 512, 128).
func NewHierarchicalNodeParser() *HierarchicalNodeParser {
	return NewHierarchicalNodeParserWithSizes(nil)
}

// NewHierarchicalNodeParserWithSizes creates a new HierarchicalNodeParser with specific chunk sizes (descending).
func NewHierarchicalNodeParserWithSizes(chunkSizes []int) *HierarchicalNodeParser {
	sizes := normalizeHierarchicalChunkSizes(chunkSizes)
	return &HierarchicalNodeParser{
		BaseNodeParser: NewBaseNodeParser(),
		chunkSizes:     sizes,
	}
}

// WithIncludeMetadata sets whether to include parent metadata in child nodes.
func (p *HierarchicalNodeParser) WithIncludeMetadata(include bool) NodeParserWithOptions {
	p.BaseNodeParser.WithIncludeMetadata(include)
	return p
}

// WithIncludePrevNextRel sets whether to establish PREVIOUS/NEXT relationships.
func (p *HierarchicalNodeParser) WithIncludePrevNextRel(include bool) NodeParserWithOptions {
	p.BaseNodeParser.WithIncludePrevNextRel(include)
	return p
}

// GetNodesFromDocuments parses documents into a hierarchy of nodes.
func (p *HierarchicalNodeParser) GetNodesFromDocuments(documents []schema.Document) []*schema.Node {
	var allNodes []*schema.Node
	for _, doc := range documents {
		p.EmitStart(doc.ID)
		nodes := hierarchicalNodesFromText(
			p.BaseNodeParser, doc.Text, p.chunkSizes,
			nil, &doc, 0, "source_doc_id", doc.ID,
		)
		allNodes = append(allNodes, nodes...)
		p.EmitComplete(doc.ID, len(nodes))
	}
	return allNodes
}

// ParseNodes parses existing nodes into a hierarchy.
func (p *HierarchicalNodeParser) ParseNodes(nodes []*schema.Node) []*schema.Node {
	var allNodes []*schema.Node
	for _, node := range nodes {
		p.EmitStart(node.ID)
		parsed := hierarchicalNodesFromText(
			p.BaseNodeParser, node.Text, p.chunkSizes,
			node, nil, 0, "source_node_id", node.ID,
		)
		allNodes = append(allNodes, parsed...)
		p.EmitComplete(node.ID, len(parsed))
	}
	return allNodes
}
