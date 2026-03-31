package nodeparser

import (
	"errors"
	"strings"
	"testing"

	"github.com/aqua777/go-llamaindex/schema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockLangchainSplitter splits on a delimiter for deterministic tests.
type mockLangchainSplitter struct {
	sep   string
	err   error
	parts []string
}

func (m *mockLangchainSplitter) SplitText(text string) ([]string, error) {
	if m.err != nil {
		return nil, m.err
	}
	if m.parts != nil {
		return m.parts, nil
	}
	if m.sep == "" {
		return []string{text}, nil
	}
	return strings.Split(text, m.sep), nil
}

func TestSplitWithLangchainSplitter_NilSplitter(t *testing.T) {
	out, err := splitWithLangchainSplitter(nil, "hello")
	require.Error(t, err)
	assert.Nil(t, out)
}

func TestSplitWithLangchainSplitter_OK(t *testing.T) {
	s := &mockLangchainSplitter{sep: "|"}
	out, err := splitWithLangchainSplitter(s, "a|b|c")
	require.NoError(t, err)
	assert.Equal(t, []string{"a", "b", "c"}, out)
}

func TestLangchainNodeParser_GetNodesFromDocuments_MockSplitter(t *testing.T) {
	s := &mockLangchainSplitter{sep: "||"}
	p := NewLangchainNodeParser(s)
	docs := []schema.Document{
		{ID: "doc-lc", Text: "chunk one||chunk two||chunk three"},
	}
	nodes := p.GetNodesFromDocuments(docs)
	require.Len(t, nodes, 3)
	assert.Equal(t, "doc-lc", nodes[0].Metadata["source_doc_id"])
	assert.Equal(t, "chunk one", nodes[0].Text)
	assert.Equal(t, "chunk two", nodes[1].Text)
	assert.Equal(t, "chunk three", nodes[2].Text)
	for i, n := range nodes {
		assert.Equal(t, i, n.Metadata["chunk_index"])
		assert.Equal(t, 3, n.Metadata["chunk_count"])
	}
}

func TestLangchainNodeParser_Relationships(t *testing.T) {
	s := &mockLangchainSplitter{sep: ";"}
	p := NewLangchainNodeParser(s)
	docs := []schema.Document{{ID: "rel-doc", Text: "A;B;C"}}
	nodes := p.GetNodesFromDocuments(docs)
	require.Len(t, nodes, 3)
	src := nodes[0].Relationships.GetSource()
	require.NotNil(t, src)
	assert.Equal(t, "rel-doc", src.NodeID)
	assert.Equal(t, schema.ObjectTypeDocument, src.NodeType)
	prev := nodes[1].Relationships.GetPrevious()
	require.NotNil(t, prev)
	assert.Equal(t, nodes[0].ID, prev.NodeID)
	next := nodes[0].Relationships.GetNext()
	require.NotNil(t, next)
	assert.Equal(t, nodes[1].ID, next.NodeID)
}

func TestLangchainNodeParser_ParseNodes(t *testing.T) {
	s := &mockLangchainSplitter{sep: ","}
	p := NewLangchainNodeParser(s)
	parent := schema.NewTextNode("x,y,z")
	parent.ID = "parent-lc"
	out := p.ParseNodes([]*schema.Node{parent})
	require.Len(t, out, 3)
	assert.Equal(t, "parent-lc", out[0].Metadata["source_node_id"])
	assert.Equal(t, "x", strings.TrimSpace(out[0].Text))
}

func TestLangchainNodeParser_SplitError_EmitsAndSkips(t *testing.T) {
	s := &mockLangchainSplitter{err: errors.New("split failed")}
	var events []NodeParserEvent
	p := NewLangchainNodeParser(s)
	p.WithCallback(func(e NodeParserEvent) { events = append(events, e) })
	nodes := p.GetNodesFromDocuments([]schema.Document{{ID: "e1", Text: "x"}})
	assert.Empty(t, nodes)
	var gotStart, gotErr bool
	for _, e := range events {
		if e.Type == EventTypeStart {
			gotStart = true
		}
		if e.Type == EventTypeError {
			gotErr = true
		}
	}
	assert.True(t, gotStart)
	assert.True(t, gotErr)
}
