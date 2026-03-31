package nodeparser

import (
	"strings"
	"testing"

	"github.com/aqua777/go-llamaindex/schema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBlockKindForMarkdownNode_HeadingAndParagraph(t *testing.T) {
	src := []byte("# H\n\nP text.\n")
	md := newMarkdownGoldmark()
	doc, err := parseMarkdownDocument(md, src)
	require.NoError(t, err)
	var kinds []string
	for c := doc.FirstChild(); c != nil; c = c.NextSibling() {
		kinds = append(kinds, blockKindForMarkdownNode(c))
	}
	assert.Equal(t, []string{"heading", "paragraph"}, kinds)
}

func TestBlockSourceText_PreservesTable(t *testing.T) {
	src := []byte("| a | b |\n|---|---|\n| 1 | 2 |\n")
	md := newMarkdownGoldmark()
	doc, err := parseMarkdownDocument(md, src)
	require.NoError(t, err)
	require.NotNil(t, doc.FirstChild())
	tab := doc.FirstChild()
	assert.Equal(t, "table", blockKindForMarkdownNode(tab))
	got := strings.TrimSpace(blockSourceText(src, tab))
	assert.Contains(t, got, "|")
	assert.Contains(t, got, "a")
}

func TestMarkdownTopLevelTextParts_HeadersAndParagraphs(t *testing.T) {
	src := []byte("# Title\n\nFirst para.\n\n## Sub\n\nSecond.\n")
	parts, err := markdownTopLevelTextParts(src)
	require.NoError(t, err)
	require.Len(t, parts, 4)
	assert.Equal(t, "heading", parts[0].Meta[MetadataKeyMarkdownElement])
	assert.Contains(t, parts[0].Text, "Title")
	assert.Equal(t, "paragraph", parts[1].Meta[MetadataKeyMarkdownElement])
	assert.Equal(t, "First para.", strings.TrimSpace(parts[1].Text))
	assert.Equal(t, "heading", parts[2].Meta[MetadataKeyMarkdownElement])
	assert.Equal(t, "paragraph", parts[3].Meta[MetadataKeyMarkdownElement])
}

func TestMarkdownTopLevelTextParts_Table(t *testing.T) {
	src := []byte("Intro line.\n\n| ColA | ColB |\n|------|------|\n| x    | y    |\n")
	parts, err := markdownTopLevelTextParts(src)
	require.NoError(t, err)
	var foundTable bool
	for _, p := range parts {
		if p.Meta[MetadataKeyMarkdownElement] == "table" {
			foundTable = true
			assert.Contains(t, p.Text, "ColA")
			assert.Contains(t, p.Text, "|")
		}
	}
	assert.True(t, foundTable, "expected a table block")
}

func TestMarkdownElementNodeParser_Relationships(t *testing.T) {
	p := NewMarkdownElementNodeParser()
	docs := []schema.Document{
		{ID: "doc-rel", Text: "# A\n\nB.\n\nC.\n"},
	}
	nodes := p.GetNodesFromDocuments(docs)
	require.Len(t, nodes, 3)
	src := nodes[0].Relationships.GetSource()
	require.NotNil(t, src)
	assert.Equal(t, "doc-rel", src.NodeID)
	assert.Equal(t, schema.ObjectTypeDocument, src.NodeType)

	prev1 := nodes[1].Relationships.GetPrevious()
	require.NotNil(t, prev1)
	assert.Equal(t, nodes[0].ID, prev1.NodeID)
	next0 := nodes[0].Relationships.GetNext()
	require.NotNil(t, next0)
	assert.Equal(t, nodes[1].ID, next0.NodeID)
}

func TestMarkdownElementNodeParser_ParseNodes(t *testing.T) {
	p := NewMarkdownElementNodeParser()
	parent := schema.NewTextNode("# In node\n\nBody.")
	parent.ID = "parent-md"
	out := p.ParseNodes([]*schema.Node{parent})
	require.Len(t, out, 2)
	assert.Equal(t, "parent-md", out[0].Metadata["source_node_id"])
	assert.Equal(t, "heading", out[0].Metadata[MetadataKeyMarkdownElement])
}
