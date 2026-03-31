package nodeparser

import (
	"strings"
	"testing"

	"github.com/aqua777/go-llamaindex/schema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestUnstructuredParagraphParts_Basic(t *testing.T) {
	text := "First block.\n\nSecond block.\n\nThird."
	parts := unstructuredParagraphParts(text)
	require.Len(t, parts, 3)
	assert.Equal(t, "paragraph", parts[0].Meta[MetadataKeyUnstructuredElement])
	assert.Equal(t, "First block.", parts[0].Text)
	assert.Equal(t, "Second block.", parts[1].Text)
	assert.Equal(t, "Third.", parts[2].Text)
}

func TestUnstructuredParagraphParts_SingleParagraph(t *testing.T) {
	parts := unstructuredParagraphParts("Only one paragraph without blank lines.")
	require.Len(t, parts, 1)
	assert.Equal(t, "Only one paragraph without blank lines.", parts[0].Text)
}

func TestUnstructuredParagraphParts_NormalizesCRLF(t *testing.T) {
	parts := unstructuredParagraphParts("A\r\n\r\nB")
	require.Len(t, parts, 2)
	assert.Equal(t, "A", parts[0].Text)
	assert.Equal(t, "B", parts[1].Text)
}

func TestUnstructuredParagraphParts_Empty(t *testing.T) {
	assert.Empty(t, unstructuredParagraphParts(""))
	assert.Empty(t, unstructuredParagraphParts("   \n\n  "))
}

func TestUnstructuredElementNodeParser_Relationships(t *testing.T) {
	p := NewUnstructuredElementNodeParser()
	docs := []schema.Document{
		{ID: "doc-u", Text: "Alpha.\n\nBeta.\n\nGamma.\n"},
	}
	nodes := p.GetNodesFromDocuments(docs)
	require.Len(t, nodes, 3)
	src := nodes[0].Relationships.GetSource()
	require.NotNil(t, src)
	assert.Equal(t, "doc-u", src.NodeID)
	assert.Equal(t, schema.ObjectTypeDocument, src.NodeType)

	prev := nodes[1].Relationships.GetPrevious()
	require.NotNil(t, prev)
	assert.Equal(t, nodes[0].ID, prev.NodeID)
	next := nodes[0].Relationships.GetNext()
	require.NotNil(t, next)
	assert.Equal(t, nodes[1].ID, next.NodeID)
}

func TestUnstructuredElementNodeParser_ParseNodes(t *testing.T) {
	p := NewUnstructuredElementNodeParser()
	parent := schema.NewTextNode("Line one.\n\nLine two.")
	parent.ID = "parent-u"
	out := p.ParseNodes([]*schema.Node{parent})
	require.Len(t, out, 2)
	assert.Equal(t, "parent-u", out[0].Metadata["source_node_id"])
	assert.Equal(t, "paragraph", out[0].Metadata[MetadataKeyUnstructuredElement])
	assert.True(t, strings.Contains(out[0].Text, "Line one"))
}
