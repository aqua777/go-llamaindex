package nodeparser

import (
	"testing"

	"github.com/aqua777/go-llamaindex/schema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestJSONPathAppendKey(t *testing.T) {
	assert.Equal(t, "a", jsonPathAppendKey("", "a"))
	assert.Equal(t, "a.b", jsonPathAppendKey("a", "b"))
	assert.Equal(t, "[0].k", jsonPathAppendKey("[0]", "k"))
}

func TestJSONPathAppendIndex(t *testing.T) {
	assert.Equal(t, "[0]", jsonPathAppendIndex("", 0))
	assert.Equal(t, "x[1]", jsonPathAppendIndex("x", 1))
}

func TestJSONLeavesFromString_FlatObject(t *testing.T) {
	parts, err := jsonLeavesFromString(`{"name":"alice","count":3}`)
	require.NoError(t, err)
	require.Len(t, parts, 2)
	byPath := mapPartsByPath(parts)
	assert.Equal(t, "3", byPath["count"].Text)
	assert.Equal(t, "alice", byPath["name"].Text)
}

func TestJSONLeavesFromString_NestedObject(t *testing.T) {
	parts, err := jsonLeavesFromString(`{"outer":{"inner":"val"}}`)
	require.NoError(t, err)
	require.Len(t, parts, 1)
	assert.Equal(t, "val", parts[0].Text)
	assert.Equal(t, "outer.inner", parts[0].Meta[MetadataKeyJSONPath])
}

func TestJSONLeavesFromString_Array(t *testing.T) {
	parts, err := jsonLeavesFromString(`[10,20,30]`)
	require.NoError(t, err)
	require.Len(t, parts, 3)
	assert.Equal(t, "10", parts[0].Text)
	assert.Equal(t, "[0]", parts[0].Meta[MetadataKeyJSONPath])
	assert.Equal(t, "20", parts[1].Text)
	assert.Equal(t, "[1]", parts[1].Meta[MetadataKeyJSONPath])
	assert.Equal(t, "30", parts[2].Text)
	assert.Equal(t, "[2]", parts[2].Meta[MetadataKeyJSONPath])
}

func TestJSONLeavesFromString_EmptyContainersYieldNoLeaves(t *testing.T) {
	p1, err := jsonLeavesFromString(`{}`)
	require.NoError(t, err)
	assert.Empty(t, p1)
	p2, err := jsonLeavesFromString(`[]`)
	require.NoError(t, err)
	assert.Empty(t, p2)
}

func TestJSONLeavesFromString_InvalidJSON(t *testing.T) {
	_, err := jsonLeavesFromString(`{`)
	require.Error(t, err)
}

func TestJSONNodeParser_GetNodesFromDocuments(t *testing.T) {
	p := NewJSONNodeParser()
	docs := []schema.Document{
		{ID: "doc1", Text: `{"title":"hi","n":2}`},
	}
	nodes := p.GetNodesFromDocuments(docs)
	require.Len(t, nodes, 2)
	byPath := nodesByJSONPath(nodes)
	assert.Equal(t, "hi", byPath["title"].Text)
	assert.Equal(t, "2", byPath["n"].Text)
	assert.Equal(t, "doc1", byPath["title"].Metadata["source_doc_id"])
}

func TestJSONNodeParser_ParseNodes(t *testing.T) {
	p := NewJSONNodeParser()
	parent := schema.NewTextNode(`{"x":true}`)
	parent.ID = "nid-1"
	nodes := p.ParseNodes([]*schema.Node{parent})
	require.Len(t, nodes, 1)
	assert.Equal(t, "true", nodes[0].Text)
	assert.Equal(t, "x", nodes[0].Metadata[MetadataKeyJSONPath])
	assert.Equal(t, "nid-1", nodes[0].Metadata["source_node_id"])
}

func mapPartsByPath(parts []textPart) map[string]textPart {
	m := make(map[string]textPart, len(parts))
	for _, p := range parts {
		path, _ := p.Meta[MetadataKeyJSONPath].(string)
		m[path] = p
	}
	return m
}

func nodesByJSONPath(nodes []*schema.Node) map[string]*schema.Node {
	m := make(map[string]*schema.Node, len(nodes))
	for _, n := range nodes {
		path, _ := n.Metadata[MetadataKeyJSONPath].(string)
		m[path] = n
	}
	return m
}
