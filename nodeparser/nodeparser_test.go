package nodeparser

import (
	"testing"

	"github.com/aqua777/go-llamaindex/schema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDefaultNodeParserOptions(t *testing.T) {
	opts := DefaultNodeParserOptions()
	assert.True(t, opts.IncludeMetadata)
	assert.True(t, opts.IncludePrevNextRel)
	assert.Nil(t, opts.IDFunc)
}

func TestBaseNodeParser(t *testing.T) {
	parser := NewBaseNodeParser()

	// Test default options
	opts := parser.Options()
	assert.True(t, opts.IncludeMetadata)
	assert.True(t, opts.IncludePrevNextRel)

	// Test fluent configuration
	parser.WithIncludeMetadata(false).WithIncludePrevNextRel(false)
	opts = parser.Options()
	assert.False(t, opts.IncludeMetadata)
	assert.False(t, opts.IncludePrevNextRel)
}

func TestBaseNodeParserGenerateID(t *testing.T) {
	parser := NewBaseNodeParser()

	// Default UUID generation
	id1 := parser.GenerateID()
	id2 := parser.GenerateID()
	assert.NotEmpty(t, id1)
	assert.NotEmpty(t, id2)
	assert.NotEqual(t, id1, id2)

	// Custom ID function
	counter := 0
	parser.WithIDFunc(func() string {
		counter++
		return "custom-" + string(rune('0'+counter))
	})
	assert.Equal(t, "custom-1", parser.GenerateID())
	assert.Equal(t, "custom-2", parser.GenerateID())
}

func TestBaseNodeParserBuildNodesFromSplits(t *testing.T) {
	parser := NewBaseNodeParser()

	doc := &schema.Document{
		ID:   "doc-1",
		Text: "First chunk. Second chunk. Third chunk.",
		Metadata: map[string]interface{}{
			"filename": "test.txt",
		},
	}

	splits := []string{"First chunk.", "Second chunk.", "Third chunk."}
	nodes := parser.BuildNodesFromSplits(splits, nil, doc)

	require.Len(t, nodes, 3)

	// Check first node
	assert.Equal(t, "First chunk.", nodes[0].Text)
	assert.Equal(t, 0, nodes[0].Metadata["chunk_index"])
	assert.Equal(t, 3, nodes[0].Metadata["chunk_count"])
	assert.Equal(t, "test.txt", nodes[0].Metadata["filename"])

	// Check character indices
	assert.NotNil(t, nodes[0].StartCharIdx)
	assert.NotNil(t, nodes[0].EndCharIdx)
	assert.Equal(t, 0, *nodes[0].StartCharIdx)
	assert.Equal(t, 12, *nodes[0].EndCharIdx) // len("First chunk.")

	// Check SOURCE relationship
	source := nodes[0].Relationships.GetSource()
	require.NotNil(t, source)
	assert.Equal(t, "doc-1", source.NodeID)
	assert.Equal(t, schema.ObjectTypeDocument, source.NodeType)

	// Check PREVIOUS/NEXT relationships
	assert.Nil(t, nodes[0].Relationships.GetPrevious())
	assert.NotNil(t, nodes[0].Relationships.GetNext())
	assert.Equal(t, nodes[1].ID, nodes[0].Relationships.GetNext().NodeID)

	assert.NotNil(t, nodes[1].Relationships.GetPrevious())
	assert.NotNil(t, nodes[1].Relationships.GetNext())

	assert.NotNil(t, nodes[2].Relationships.GetPrevious())
	assert.Nil(t, nodes[2].Relationships.GetNext())
}

func TestBaseNodeParserWithoutRelationships(t *testing.T) {
	parser := NewBaseNodeParser().WithIncludePrevNextRel(false)

	doc := &schema.Document{
		ID:   "doc-1",
		Text: "Test",
	}

	splits := []string{"First", "Second", "Third"}
	nodes := parser.BuildNodesFromSplits(splits, nil, doc)

	require.Len(t, nodes, 3)

	// Should have SOURCE but no PREVIOUS/NEXT
	for _, node := range nodes {
		assert.NotNil(t, node.Relationships.GetSource())
		assert.Nil(t, node.Relationships.GetPrevious())
		assert.Nil(t, node.Relationships.GetNext())
	}
}

func TestBaseNodeParserWithoutMetadata(t *testing.T) {
	parser := NewBaseNodeParser().WithIncludeMetadata(false)

	doc := &schema.Document{
		ID:   "doc-1",
		Text: "Test",
		Metadata: map[string]interface{}{
			"filename": "test.txt",
			"author":   "John",
		},
	}

	splits := []string{"First chunk"}
	nodes := parser.BuildNodesFromSplits(splits, nil, doc)

	require.Len(t, nodes, 1)

	// Should have chunk metadata but not document metadata
	assert.Equal(t, 0, nodes[0].Metadata["chunk_index"])
	assert.Nil(t, nodes[0].Metadata["filename"])
	assert.Nil(t, nodes[0].Metadata["author"])
}

func TestSentenceNodeParser(t *testing.T) {
	parser := NewSentenceNodeParser()

	docs := []schema.Document{
		{
			ID:   "doc-1",
			Text: "This is the first sentence. This is the second sentence. This is the third sentence.",
			Metadata: map[string]interface{}{
				"source": "test",
			},
		},
	}

	nodes := parser.GetNodesFromDocuments(docs)

	// Should have at least one node
	require.NotEmpty(t, nodes)

	// All nodes should have source_doc_id
	for _, node := range nodes {
		assert.Equal(t, "doc-1", node.Metadata["source_doc_id"])
		assert.Equal(t, "test", node.Metadata["source"])
	}

	// Check relationships
	if len(nodes) > 1 {
		// First node has no previous
		assert.Nil(t, nodes[0].Relationships.GetPrevious())
		assert.NotNil(t, nodes[0].Relationships.GetNext())

		// Last node has no next
		lastIdx := len(nodes) - 1
		assert.NotNil(t, nodes[lastIdx].Relationships.GetPrevious())
		assert.Nil(t, nodes[lastIdx].Relationships.GetNext())
	}
}

func TestSentenceNodeParserWithConfig(t *testing.T) {
	// Small chunk size to force splitting
	parser := NewSentenceNodeParserWithConfig(20, 5)

	docs := []schema.Document{
		{
			ID:   "doc-1",
			Text: "This is sentence one. This is sentence two. This is sentence three. This is sentence four. This is sentence five.",
		},
	}

	nodes := parser.GetNodesFromDocuments(docs)

	// With small chunk size, should create multiple nodes
	require.GreaterOrEqual(t, len(nodes), 1)

	// Verify all nodes have proper metadata
	for _, node := range nodes {
		assert.Equal(t, "doc-1", node.Metadata["source_doc_id"])
	}
}

func TestSentenceNodeParserParseNodes(t *testing.T) {
	parser := NewSentenceNodeParserWithConfig(30, 5)

	// Create a parent node
	parentNode := schema.NewTextNode("First part. Second part. Third part.")
	parentNode.ID = "parent-1"
	parentNode.Metadata["custom"] = "value"

	nodes := parser.ParseNodes([]*schema.Node{parentNode})

	require.NotEmpty(t, nodes)

	// All nodes should reference the parent
	for _, node := range nodes {
		assert.Equal(t, "parent-1", node.Metadata["source_node_id"])
		assert.Equal(t, "value", node.Metadata["custom"])

		// Check SOURCE relationship points to parent
		source := node.Relationships.GetSource()
		require.NotNil(t, source)
		assert.Equal(t, "parent-1", source.NodeID)
	}
}

func TestSimpleNodeParser(t *testing.T) {
	parser := NewSimpleNodeParser()

	docs := []schema.Document{
		{
			ID:   "doc-1",
			Text: "First document content.",
			Metadata: map[string]interface{}{
				"filename": "doc1.txt",
			},
		},
		{
			ID:   "doc-2",
			Text: "Second document content.",
			Metadata: map[string]interface{}{
				"filename": "doc2.txt",
			},
		},
	}

	nodes := parser.GetNodesFromDocuments(docs)

	require.Len(t, nodes, 2)

	// Check first node
	assert.Equal(t, "First document content.", nodes[0].Text)
	assert.Equal(t, "doc-1", nodes[0].Metadata["source_doc_id"])
	assert.Equal(t, "doc1.txt", nodes[0].Metadata["filename"])

	// Check second node
	assert.Equal(t, "Second document content.", nodes[1].Text)
	assert.Equal(t, "doc-2", nodes[1].Metadata["source_doc_id"])
	assert.Equal(t, "doc2.txt", nodes[1].Metadata["filename"])

	// Check relationships
	assert.Nil(t, nodes[0].Relationships.GetPrevious())
	assert.NotNil(t, nodes[0].Relationships.GetNext())
	assert.Equal(t, nodes[1].ID, nodes[0].Relationships.GetNext().NodeID)

	assert.NotNil(t, nodes[1].Relationships.GetPrevious())
	assert.Equal(t, nodes[0].ID, nodes[1].Relationships.GetPrevious().NodeID)
	assert.Nil(t, nodes[1].Relationships.GetNext())
}

func TestSimpleNodeParserWithoutRelationships(t *testing.T) {
	parser := NewSimpleNodeParser()
	parser.WithIncludePrevNextRel(false)

	docs := []schema.Document{
		{ID: "doc-1", Text: "First"},
		{ID: "doc-2", Text: "Second"},
	}

	nodes := parser.GetNodesFromDocuments(docs)

	require.Len(t, nodes, 2)

	// Should have SOURCE but no PREVIOUS/NEXT
	for _, node := range nodes {
		assert.NotNil(t, node.Relationships.GetSource())
		assert.Nil(t, node.Relationships.GetPrevious())
		assert.Nil(t, node.Relationships.GetNext())
	}
}

func TestNodeParserCallback(t *testing.T) {
	var events []NodeParserEvent

	parser := NewSentenceNodeParser()
	parser.WithCallback(func(event NodeParserEvent) {
		events = append(events, event)
	})

	docs := []schema.Document{
		{ID: "doc-1", Text: "Test content."},
	}

	parser.GetNodesFromDocuments(docs)

	// Should have start and complete events
	require.GreaterOrEqual(t, len(events), 2)

	// First event should be start
	assert.Equal(t, EventTypeStart, events[0].Type)
	assert.Equal(t, "doc-1", events[0].DocumentID)

	// Last event should be complete
	lastEvent := events[len(events)-1]
	assert.Equal(t, EventTypeComplete, lastEvent.Type)
	assert.Equal(t, "doc-1", lastEvent.DocumentID)
}

func TestNodeParserInterface(t *testing.T) {
	// Verify implementations satisfy the interface
	var _ NodeParser = &SentenceNodeParser{}
	var _ NodeParser = &SimpleNodeParser{}
	var _ NodeParserWithOptions = &SentenceNodeParser{}
	var _ NodeParserWithOptions = &SimpleNodeParser{}
}

func TestNodeParserEventTypes(t *testing.T) {
	assert.Equal(t, NodeParserEventType("start"), EventTypeStart)
	assert.Equal(t, NodeParserEventType("progress"), EventTypeProgress)
	assert.Equal(t, NodeParserEventType("complete"), EventTypeComplete)
	assert.Equal(t, NodeParserEventType("error"), EventTypeError)
}

func TestEmptyDocuments(t *testing.T) {
	parser := NewSentenceNodeParser()

	// Empty slice
	nodes := parser.GetNodesFromDocuments([]schema.Document{})
	assert.Empty(t, nodes)

	// Document with empty text
	docs := []schema.Document{
		{ID: "doc-1", Text: ""},
	}
	nodes = parser.GetNodesFromDocuments(docs)
	// Should handle gracefully (may return empty or single empty node)
	assert.NotNil(t, nodes)
}

func TestCharacterIndicesAccuracy(t *testing.T) {
	parser := NewBaseNodeParser()

	splits := []string{"Hello", " World", "!"}
	nodes := parser.BuildNodesFromSplits(splits, nil, nil)

	require.Len(t, nodes, 3)

	// First: 0-5
	assert.Equal(t, 0, *nodes[0].StartCharIdx)
	assert.Equal(t, 5, *nodes[0].EndCharIdx)

	// Second: 5-11
	assert.Equal(t, 5, *nodes[1].StartCharIdx)
	assert.Equal(t, 11, *nodes[1].EndCharIdx)

	// Third: 11-12
	assert.Equal(t, 11, *nodes[2].StartCharIdx)
	assert.Equal(t, 12, *nodes[2].EndCharIdx)
}

func TestMetadataMergeDoesNotOverwrite(t *testing.T) {
	parser := NewBaseNodeParser()

	doc := &schema.Document{
		ID:   "doc-1",
		Text: "Test",
		Metadata: map[string]interface{}{
			"shared_key": "doc_value",
			"doc_only":   "doc_value",
		},
	}

	splits := []string{"Test"}
	nodes := parser.BuildNodesFromSplits(splits, nil, doc)

	require.Len(t, nodes, 1)

	// chunk_index is set first, then doc metadata is merged
	// chunk_index should not be overwritten
	assert.Equal(t, 0, nodes[0].Metadata["chunk_index"])

	// doc metadata should be present
	assert.Equal(t, "doc_value", nodes[0].Metadata["doc_only"])
	assert.Equal(t, "doc_value", nodes[0].Metadata["shared_key"])
}
