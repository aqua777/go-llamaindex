package nodeparser

import (
	"strings"
	"testing"

	"github.com/aqua777/go-llamaindex/schema"
	"github.com/stretchr/testify/suite"
)

type HierarchicalNodeParserTestSuite struct {
	suite.Suite
}

func TestHierarchicalNodeParserTestSuite(t *testing.T) {
	suite.Run(t, new(HierarchicalNodeParserTestSuite))
}

func (s *HierarchicalNodeParserTestSuite) TestNormalizeHierarchicalChunkSizes_DefaultsWhenEmpty() {
	out := normalizeHierarchicalChunkSizes(nil)
	s.Equal([]int{2048, 512, 128}, out)

	out2 := normalizeHierarchicalChunkSizes([]int{})
	s.Equal([]int{2048, 512, 128}, out2)
}

func (s *HierarchicalNodeParserTestSuite) TestNormalizeHierarchicalChunkSizes_SortsDescendingDedup() {
	out := normalizeHierarchicalChunkSizes([]int{10, 100, 50, 100})
	s.Equal([]int{100, 50, 10}, out)
}

func (s *HierarchicalNodeParserTestSuite) TestSplitTextForHierarchy_Empty() {
	s.Nil(splitTextForHierarchy("", 100))
}

func (s *HierarchicalNodeParserTestSuite) TestEstablishParentChildLinks_NilParent() {
	ch := schema.NewNode()
	ch.ID = "c"
	establishParentChildLinks(nil, []*schema.Node{ch})
	s.Nil(ch.Relationships.GetParent())
}

func (s *HierarchicalNodeParserTestSuite) TestGetNodesFromDocuments_DefaultSizes() {
	parser := NewHierarchicalNodeParser()
	long := strings.Repeat("alpha beta gamma. ", 400)
	docs := []schema.Document{{ID: "doc-1", Text: long}}
	nodes := parser.GetNodesFromDocuments(docs)
	s.NotEmpty(nodes)
	for _, n := range nodes {
		_, ok := n.Metadata[MetadataKeyHierarchyLevel].(int)
		s.True(ok)
		s.Equal("doc-1", n.Metadata["source_doc_id"])
	}
}

func (s *HierarchicalNodeParserTestSuite) TestGetNodesFromDocuments_CustomSizes_ParentChild() {
	sizes := []int{80, 40, 20}
	parser := NewHierarchicalNodeParserWithSizes(sizes)
	text := strings.Repeat("sentence one two three. ", 50)
	docs := []schema.Document{{ID: "h1", Text: text}}
	nodes := parser.GetNodesFromDocuments(docs)

	var withChildren []*schema.Node
	for _, n := range nodes {
		if len(n.Relationships.GetChildren()) > 0 {
			withChildren = append(withChildren, n)
		}
	}
	s.NotEmpty(withChildren, "expected at least one parent with children")

	parent := withChildren[0]
	children := parent.Relationships.GetChildren()
	s.NotEmpty(children)
	for _, info := range children {
		s.NotEmpty(info.NodeID)
	}

	childID := children[0].NodeID
	var childNode *schema.Node
	for _, n := range nodes {
		if n.ID == childID {
			childNode = n
			break
		}
	}
	s.Require().NotNil(childNode)
	pi := childNode.Relationships.GetParent()
	s.Require().NotNil(pi)
	s.Equal(parent.ID, pi.NodeID)
}

func (s *HierarchicalNodeParserTestSuite) TestParseNodes_SourceAndHierarchy() {
	base := schema.NewNode()
	base.ID = "root-node"
	base.Text = strings.Repeat("chunkable text here. ", 60)
	base.Type = schema.ObjectTypeText

	parser := NewHierarchicalNodeParserWithSizes([]int{100, 50, 25})
	out := parser.ParseNodes([]*schema.Node{base})
	s.NotEmpty(out)
	s.Contains(out[0].Metadata, "source_node_id")
	s.Equal("root-node", out[0].Metadata["source_node_id"])
}

func (s *HierarchicalNodeParserTestSuite) TestFilterNonEmptyTextParts() {
	s.Equal([]string{"a"}, filterNonEmptyTextParts([]string{"", "a", ""}))
	s.Empty(filterNonEmptyTextParts([]string{"", ""}))
}
