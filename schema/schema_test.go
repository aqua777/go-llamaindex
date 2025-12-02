package schema

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewNode(t *testing.T) {
	node := NewNode()
	assert.NotEmpty(t, node.ID)
	assert.Equal(t, ObjectTypeText, node.Type)
	assert.NotNil(t, node.Metadata)
	assert.NotNil(t, node.Relationships)
	assert.Equal(t, DefaultMetadataTemplate, node.MetadataTemplate)
	assert.Equal(t, DefaultMetadataSeparator, node.MetadataSeparator)
}

func TestNewTextNode(t *testing.T) {
	text := "Hello, world!"
	node := NewTextNode(text)
	assert.Equal(t, text, node.Text)
	assert.NotEmpty(t, node.Hash)
	assert.Equal(t, "TextNode", node.ClassName())
}

func TestNodeGetContent(t *testing.T) {
	node := NewTextNode("Test content")
	node.Metadata = map[string]interface{}{
		"author": "John",
		"title":  "Test",
	}

	// Test MetadataModeNone
	content := node.GetContent(MetadataModeNone)
	assert.Equal(t, "Test content", content)

	// Test MetadataModeAll
	content = node.GetContent(MetadataModeAll)
	assert.Contains(t, content, "Test content")
	assert.Contains(t, content, "author: John")
	assert.Contains(t, content, "title: Test")
}

func TestNodeMetadataExclusion(t *testing.T) {
	node := NewTextNode("Test content")
	node.Metadata = map[string]interface{}{
		"author":   "John",
		"title":    "Test",
		"internal": "secret",
	}
	node.ExcludedLLMMetadataKeys = []string{"internal"}
	node.ExcludedEmbedMetadataKeys = []string{"author"}

	// LLM mode should exclude "internal"
	llmStr := node.GetMetadataStr(MetadataModeLLM)
	assert.Contains(t, llmStr, "author")
	assert.Contains(t, llmStr, "title")
	assert.NotContains(t, llmStr, "internal")

	// Embed mode should exclude "author"
	embedStr := node.GetMetadataStr(MetadataModeEmbed)
	assert.NotContains(t, embedStr, "author")
	assert.Contains(t, embedStr, "title")
	assert.Contains(t, embedStr, "internal")
}

func TestNodeRelationships(t *testing.T) {
	parent := NewTextNode("Parent node")
	child1 := NewTextNode("Child 1")
	child2 := NewTextNode("Child 2")

	// Set parent relationship on child
	child1.GetRelationships().SetParent(parent.AsRelatedNodeInfo())

	// Set children on parent
	parent.GetRelationships().SetChildren([]RelatedNodeInfo{
		child1.AsRelatedNodeInfo(),
		child2.AsRelatedNodeInfo(),
	})

	// Verify relationships
	parentInfo := child1.GetRelationships().GetParent()
	require.NotNil(t, parentInfo)
	assert.Equal(t, parent.ID, parentInfo.NodeID)

	children := parent.GetRelationships().GetChildren()
	require.Len(t, children, 2)
	assert.Equal(t, child1.ID, children[0].NodeID)
	assert.Equal(t, child2.ID, children[1].NodeID)
}

func TestNodeHash(t *testing.T) {
	node1 := NewTextNode("Same content")
	node2 := NewTextNode("Same content")

	// Same content should produce same hash
	assert.Equal(t, node1.GetHash(), node2.GetHash())

	// Different content should produce different hash
	node3 := NewTextNode("Different content")
	assert.NotEqual(t, node1.GetHash(), node3.GetHash())
}

func TestNodeToJSON(t *testing.T) {
	node := NewTextNode("Test content")
	node.Metadata = map[string]interface{}{"key": "value"}

	jsonStr, err := node.ToJSON()
	require.NoError(t, err)

	var data map[string]interface{}
	err = json.Unmarshal([]byte(jsonStr), &data)
	require.NoError(t, err)

	assert.Equal(t, "TextNode", data["class_name"])
	assert.Equal(t, "Test content", data["text"])
	assert.Equal(t, "TEXT", data["type"])
}

func TestAsRelatedNodeInfo(t *testing.T) {
	node := NewTextNode("Test")
	node.Metadata = map[string]interface{}{"key": "value"}

	info := node.AsRelatedNodeInfo()
	assert.Equal(t, node.ID, info.NodeID)
	assert.Equal(t, node.Type, info.NodeType)
	assert.Equal(t, node.Metadata, info.Metadata)
	assert.NotEmpty(t, info.Hash)
}

func TestImageNode(t *testing.T) {
	node := NewImageNodeFromPath("/path/to/image.png", "image/png")
	assert.Equal(t, ObjectTypeImage, node.GetType())
	assert.Equal(t, "ImageNode", node.ClassName())
	assert.Equal(t, "/path/to/image.png", node.ImagePath)
	assert.Equal(t, "image/png", node.ImageMimeType)
	assert.True(t, node.HasImage())
	assert.NotEmpty(t, node.Hash)
}

func TestImageNodeFromURL(t *testing.T) {
	node := NewImageNodeFromURL("https://example.com/image.jpg", "image/jpeg")
	assert.Equal(t, "https://example.com/image.jpg", node.ImageURL)
	assert.Equal(t, "image/jpeg", node.ImageMimeType)
	assert.Equal(t, "https://example.com/image.jpg", node.GetImageSource())
}

func TestImageNodeFromBase64(t *testing.T) {
	node := NewImageNodeFromBase64("base64data", "image/png")
	assert.Equal(t, "base64data", node.Image)
	assert.Equal(t, "base64data", node.GetImageSource())
}

func TestIndexNode(t *testing.T) {
	node := NewIndexNode("index-123")
	assert.Equal(t, ObjectTypeIndex, node.GetType())
	assert.Equal(t, "IndexNode", node.ClassName())
	assert.Equal(t, "index-123", node.IndexID)
}

func TestIndexNodeFromTextNode(t *testing.T) {
	textNode := NewTextNode("Original text")
	textNode.Metadata = map[string]interface{}{"key": "value"}

	indexNode := NewIndexNodeFromTextNode(textNode, "index-456")
	assert.Equal(t, textNode.Text, indexNode.Text)
	assert.Equal(t, textNode.Metadata, indexNode.Metadata)
	assert.Equal(t, "index-456", indexNode.IndexID)
	assert.Equal(t, ObjectTypeIndex, indexNode.GetType())
}

func TestMediaResource(t *testing.T) {
	// Test from text
	mr := NewMediaResourceFromText("Hello")
	assert.Equal(t, "Hello", mr.Text)
	assert.Equal(t, "text/plain", mr.MimeType)
	assert.True(t, mr.HasText())
	assert.False(t, mr.IsEmpty())

	// Test from path
	mr = NewMediaResourceFromPath("/path/to/file.pdf", "application/pdf")
	assert.Equal(t, "/path/to/file.pdf", mr.Path)
	assert.True(t, mr.HasPath())

	// Test from URL
	mr = NewMediaResourceFromURL("https://example.com/file.pdf", "application/pdf")
	assert.Equal(t, "https://example.com/file.pdf", mr.URL)
	assert.True(t, mr.HasURL())

	// Test hash
	mr1 := NewMediaResourceFromText("Same text")
	mr2 := NewMediaResourceFromText("Same text")
	assert.Equal(t, mr1.Hash(), mr2.Hash())

	mr3 := NewMediaResourceFromText("Different text")
	assert.NotEqual(t, mr1.Hash(), mr3.Hash())
}

func TestMediaResourceBase64(t *testing.T) {
	data := []byte("binary data")
	mr := NewMediaResourceFromData(data, "application/octet-stream")

	b64 := mr.GetDataBase64()
	assert.NotEmpty(t, b64)

	mr2 := NewMediaResource()
	err := mr2.SetDataFromBase64(b64)
	require.NoError(t, err)
	assert.Equal(t, data, mr2.Data)
}

func TestMediaResourceEmbeddings(t *testing.T) {
	mr := NewMediaResource()
	embedding := []float64{0.1, 0.2, 0.3}

	mr.SetEmbedding(EmbeddingKindDense, embedding)
	assert.Equal(t, embedding, mr.GetDenseEmbedding())
	assert.Nil(t, mr.GetSparseEmbedding())

	sparseEmbedding := []float64{1.0, 0.0, 0.5}
	mr.SetEmbedding(EmbeddingKindSparse, sparseEmbedding)
	assert.Equal(t, sparseEmbedding, mr.GetSparseEmbedding())
}

func TestNodeRelationshipHelpers(t *testing.T) {
	rels := make(NodeRelationships)

	// Test source
	sourceInfo := RelatedNodeInfo{NodeID: "source-1"}
	rels.SetSource(sourceInfo)
	assert.Equal(t, "source-1", rels.GetSource().NodeID)

	// Test previous/next
	prevInfo := RelatedNodeInfo{NodeID: "prev-1"}
	nextInfo := RelatedNodeInfo{NodeID: "next-1"}
	rels.SetPrevious(prevInfo)
	rels.SetNext(nextInfo)
	assert.Equal(t, "prev-1", rels.GetPrevious().NodeID)
	assert.Equal(t, "next-1", rels.GetNext().NodeID)

	// Test parent
	parentInfo := RelatedNodeInfo{NodeID: "parent-1"}
	rels.SetParent(parentInfo)
	assert.Equal(t, "parent-1", rels.GetParent().NodeID)

	// Test children
	rels.AddChild(RelatedNodeInfo{NodeID: "child-1"})
	rels.AddChild(RelatedNodeInfo{NodeID: "child-2"})
	children := rels.GetChildren()
	assert.Len(t, children, 2)
	assert.Equal(t, "child-1", children[0].NodeID)
	assert.Equal(t, "child-2", children[1].NodeID)
}

func TestMetadataMode(t *testing.T) {
	assert.Equal(t, MetadataMode("all"), MetadataModeAll)
	assert.Equal(t, MetadataMode("embed"), MetadataModeEmbed)
	assert.Equal(t, MetadataMode("llm"), MetadataModeLLM)
	assert.Equal(t, MetadataMode("none"), MetadataModeNone)
}

func TestNodeRelationshipConstants(t *testing.T) {
	assert.Equal(t, NodeRelationship("SOURCE"), RelationshipSource)
	assert.Equal(t, NodeRelationship("PREVIOUS"), RelationshipPrevious)
	assert.Equal(t, NodeRelationship("NEXT"), RelationshipNext)
	assert.Equal(t, NodeRelationship("PARENT"), RelationshipParent)
	assert.Equal(t, NodeRelationship("CHILD"), RelationshipChild)
}

// ============================================================================
// Additional Node Tests
// ============================================================================

func TestNodeSetContent(t *testing.T) {
	node := NewTextNode("Original")
	originalHash := node.Hash

	node.SetContent("Updated content")
	assert.Equal(t, "Updated content", node.Text)
	assert.NotEqual(t, originalHash, node.Hash, "Hash should change when content changes")
}

func TestNodeGetSetID(t *testing.T) {
	node := NewNode()
	originalID := node.GetID()
	assert.NotEmpty(t, originalID)

	node.SetID("custom-id")
	assert.Equal(t, "custom-id", node.GetID())
}

func TestNodeGetSetEmbedding(t *testing.T) {
	node := NewNode()
	assert.Nil(t, node.GetEmbedding())

	embedding := []float64{0.1, 0.2, 0.3, 0.4}
	node.SetEmbedding(embedding)
	assert.Equal(t, embedding, node.GetEmbedding())
}

func TestNodeGetSetMetadata(t *testing.T) {
	node := NewNode()
	assert.NotNil(t, node.GetMetadata())
	assert.Empty(t, node.GetMetadata())

	metadata := map[string]interface{}{"key": "value", "num": 42}
	node.SetMetadata(metadata)
	assert.Equal(t, metadata, node.GetMetadata())
}

func TestNodeGetSetRelationships(t *testing.T) {
	node := NewNode()
	rels := node.GetRelationships()
	assert.NotNil(t, rels)

	newRels := make(NodeRelationships)
	newRels.SetSource(RelatedNodeInfo{NodeID: "source-123"})
	node.SetRelationships(newRels)

	assert.Equal(t, "source-123", node.GetRelationships().GetSource().NodeID)
}

func TestNodeGetText(t *testing.T) {
	node := NewTextNode("Plain text content")
	node.Metadata = map[string]interface{}{"key": "value"}

	// GetText should return content without metadata
	assert.Equal(t, "Plain text content", node.GetText())
}

func TestNodeGetNodeInfo(t *testing.T) {
	node := NewNode()
	info := node.GetNodeInfo()
	assert.Nil(t, info["start"])
	assert.Nil(t, info["end"])

	start, end := 10, 50
	node.StartCharIdx = &start
	node.EndCharIdx = &end

	info = node.GetNodeInfo()
	assert.Equal(t, &start, info["start"])
	assert.Equal(t, &end, info["end"])
}

func TestNodeToDict(t *testing.T) {
	node := NewTextNode("Test")
	node.Metadata = map[string]interface{}{"author": "Jane"}
	node.Embedding = []float64{0.1, 0.2}

	dict := node.ToDict()
	assert.Equal(t, "TextNode", dict["class_name"])
	assert.Equal(t, node.ID, dict["id"])
	assert.Equal(t, "Test", dict["text"])
	assert.Equal(t, "TEXT", dict["type"])
	assert.NotNil(t, dict["metadata"])
	assert.NotNil(t, dict["embedding"])
	assert.NotEmpty(t, dict["hash"])
}

func TestNodeToDictMinimal(t *testing.T) {
	node := NewNode()
	node.Text = "Minimal"

	dict := node.ToDict()
	// Should not include empty metadata/embedding
	_, hasMetadata := dict["metadata"]
	_, hasEmbedding := dict["embedding"]
	assert.False(t, hasMetadata)
	assert.False(t, hasEmbedding)
}

func TestNodeEmptyMetadataStr(t *testing.T) {
	node := NewNode()
	assert.Equal(t, "", node.GetMetadataStr(MetadataModeAll))
	assert.Equal(t, "", node.GetMetadataStr(MetadataModeNone))
}

func TestNodeMetadataStrWithCustomTemplate(t *testing.T) {
	node := NewTextNode("Content")
	node.Metadata = map[string]interface{}{"name": "Alice"}
	node.MetadataTemplate = "[{key}={value}]"
	node.MetadataSeparator = " | "

	str := node.GetMetadataStr(MetadataModeAll)
	assert.Equal(t, "[name=Alice]", str)

	node.Metadata["age"] = 30
	str = node.GetMetadataStr(MetadataModeAll)
	assert.Contains(t, str, "[age=30]")
	assert.Contains(t, str, "[name=Alice]")
	assert.Contains(t, str, " | ")
}

func TestNodeMetadataWithDifferentTypes(t *testing.T) {
	node := NewTextNode("Content")
	node.Metadata = map[string]interface{}{
		"string": "text",
		"number": 42,
		"float":  3.14,
		"bool":   true,
		"array":  []int{1, 2, 3},
	}

	str := node.GetMetadataStr(MetadataModeAll)
	assert.Contains(t, str, "string: text")
	assert.Contains(t, str, "number: 42")
	assert.Contains(t, str, "float: 3.14")
	assert.Contains(t, str, "bool: true")
	assert.Contains(t, str, "array: [1,2,3]")
}

func TestNodeGetContentWithCustomTemplate(t *testing.T) {
	node := NewTextNode("Main content")
	node.Metadata = map[string]interface{}{"title": "Doc"}
	node.TextTemplate = "--- {metadata_str} ---\n{content}"

	content := node.GetContent(MetadataModeAll)
	assert.Contains(t, content, "--- title: Doc ---")
	assert.Contains(t, content, "Main content")
}

func TestNodeGetContentEmptyMetadata(t *testing.T) {
	node := NewTextNode("Just text")
	// No metadata set

	content := node.GetContent(MetadataModeAll)
	assert.Equal(t, "Just text", content)
}

func TestNodeHashConsistency(t *testing.T) {
	// Hash should be deterministic
	node := NewTextNode("Consistent content")
	hash1 := node.GenerateHash()
	hash2 := node.GenerateHash()
	assert.Equal(t, hash1, hash2)
}

func TestNodeHashWithCharIndices(t *testing.T) {
	node1 := NewTextNode("Content")
	start1, end1 := 0, 10
	node1.StartCharIdx = &start1
	node1.EndCharIdx = &end1

	node2 := NewTextNode("Content")
	start2, end2 := 5, 15
	node2.StartCharIdx = &start2
	node2.EndCharIdx = &end2

	// Different char indices should produce different hashes
	assert.NotEqual(t, node1.GenerateHash(), node2.GenerateHash())
}

// ============================================================================
// Additional NodeRelationships Tests
// ============================================================================

func TestNodeRelationshipsGetNonExistent(t *testing.T) {
	rels := make(NodeRelationships)

	assert.Nil(t, rels.GetSource())
	assert.Nil(t, rels.GetPrevious())
	assert.Nil(t, rels.GetNext())
	assert.Nil(t, rels.GetParent())
	assert.Nil(t, rels.GetChildren())
}

func TestNodeRelationshipsOverwrite(t *testing.T) {
	rels := make(NodeRelationships)

	rels.SetSource(RelatedNodeInfo{NodeID: "first"})
	assert.Equal(t, "first", rels.GetSource().NodeID)

	rels.SetSource(RelatedNodeInfo{NodeID: "second"})
	assert.Equal(t, "second", rels.GetSource().NodeID)
}

func TestNodeRelationshipsSetChildrenOverwrite(t *testing.T) {
	rels := make(NodeRelationships)

	rels.AddChild(RelatedNodeInfo{NodeID: "child-1"})
	rels.AddChild(RelatedNodeInfo{NodeID: "child-2"})
	assert.Len(t, rels.GetChildren(), 2)

	// SetChildren should replace all
	rels.SetChildren([]RelatedNodeInfo{{NodeID: "new-child"}})
	assert.Len(t, rels.GetChildren(), 1)
	assert.Equal(t, "new-child", rels.GetChildren()[0].NodeID)
}

func TestRelatedNodeInfoFields(t *testing.T) {
	info := RelatedNodeInfo{
		NodeID:   "node-123",
		NodeType: ObjectTypeDocument,
		Metadata: map[string]interface{}{"key": "value"},
		Hash:     "abc123",
	}

	assert.Equal(t, "node-123", info.NodeID)
	assert.Equal(t, ObjectTypeDocument, info.NodeType)
	assert.Equal(t, "value", info.Metadata["key"])
	assert.Equal(t, "abc123", info.Hash)
}

// ============================================================================
// Additional ImageNode Tests
// ============================================================================

func TestNewImageNode(t *testing.T) {
	node := NewImageNode()
	assert.NotEmpty(t, node.ID)
	assert.Equal(t, ObjectTypeImage, node.Type)
	assert.NotNil(t, node.Metadata)
	assert.NotNil(t, node.Relationships)
}

func TestImageNodeHasImageFalse(t *testing.T) {
	node := NewImageNode()
	assert.False(t, node.HasImage())
}

func TestImageNodeGetImageSourcePriority(t *testing.T) {
	// Base64 takes priority
	node := NewImageNode()
	node.Image = "base64data"
	node.ImagePath = "/path"
	node.ImageURL = "https://url"
	assert.Equal(t, "base64data", node.GetImageSource())

	// Path is second priority
	node.Image = ""
	assert.Equal(t, "/path", node.GetImageSource())

	// URL is last
	node.ImagePath = ""
	assert.Equal(t, "https://url", node.GetImageSource())

	// Empty when nothing set
	node.ImageURL = ""
	assert.Equal(t, "", node.GetImageSource())
}

func TestImageNodeToDict(t *testing.T) {
	node := NewImageNodeFromPath("/img.png", "image/png")
	node.Text = "Image description"
	node.TextEmbedding = []float64{0.1, 0.2}

	dict := node.ToDict()
	assert.Equal(t, "ImageNode", dict["class_name"])
	assert.Equal(t, "IMAGE", dict["type"])
	assert.Equal(t, "/img.png", dict["image_path"])
	assert.Equal(t, "image/png", dict["image_mimetype"])
	assert.NotNil(t, dict["text_embedding"])
}

func TestImageNodeHashDifferentSources(t *testing.T) {
	node1 := NewImageNodeFromPath("/path1.png", "image/png")
	node2 := NewImageNodeFromPath("/path2.png", "image/png")
	assert.NotEqual(t, node1.GenerateHash(), node2.GenerateHash())

	node3 := NewImageNodeFromURL("https://a.com/img.png", "image/png")
	node4 := NewImageNodeFromURL("https://b.com/img.png", "image/png")
	assert.NotEqual(t, node3.GenerateHash(), node4.GenerateHash())
}

func TestImageNodeWithText(t *testing.T) {
	node := NewImageNodeFromPath("/img.png", "image/png")
	node.Text = "A beautiful sunset"

	// Text affects hash
	hash1 := node.GenerateHash()
	node.Text = "A different description"
	hash2 := node.GenerateHash()
	assert.NotEqual(t, hash1, hash2)
}

// ============================================================================
// Additional IndexNode Tests
// ============================================================================

func TestIndexNodeObject(t *testing.T) {
	node := NewIndexNode("idx-1")

	// Test setting and getting object
	obj := map[string]string{"key": "value"}
	node.SetObject(obj)
	assert.Equal(t, obj, node.GetObject())
}

func TestIndexNodeToDict(t *testing.T) {
	node := NewIndexNode("idx-123")
	node.Text = "Index description"

	dict := node.ToDict()
	assert.Equal(t, "IndexNode", dict["class_name"])
	assert.Equal(t, "INDEX", dict["type"])
	assert.Equal(t, "idx-123", dict["index_id"])
}

func TestIndexNodeFromTextNodePreservesFields(t *testing.T) {
	textNode := NewTextNode("Original")
	textNode.Metadata = map[string]interface{}{"key": "value"}
	textNode.Embedding = []float64{0.1, 0.2, 0.3}
	textNode.ExcludedLLMMetadataKeys = []string{"secret"}
	textNode.ExcludedEmbedMetadataKeys = []string{"internal"}
	start, end := 10, 20
	textNode.StartCharIdx = &start
	textNode.EndCharIdx = &end
	textNode.MimeType = "text/markdown"

	indexNode := NewIndexNodeFromTextNode(textNode, "idx-1")

	assert.Equal(t, textNode.ID, indexNode.ID)
	assert.Equal(t, textNode.Text, indexNode.Text)
	assert.Equal(t, textNode.Metadata, indexNode.Metadata)
	assert.Equal(t, textNode.Embedding, indexNode.Embedding)
	assert.Equal(t, textNode.ExcludedLLMMetadataKeys, indexNode.ExcludedLLMMetadataKeys)
	assert.Equal(t, textNode.ExcludedEmbedMetadataKeys, indexNode.ExcludedEmbedMetadataKeys)
	assert.Equal(t, textNode.StartCharIdx, indexNode.StartCharIdx)
	assert.Equal(t, textNode.EndCharIdx, indexNode.EndCharIdx)
	assert.Equal(t, textNode.MimeType, indexNode.MimeType)
	assert.Equal(t, ObjectTypeIndex, indexNode.Type)
}

// ============================================================================
// Additional MediaResource Tests
// ============================================================================

func TestMediaResourceEmpty(t *testing.T) {
	mr := NewMediaResource()
	assert.True(t, mr.IsEmpty())
	assert.Equal(t, "", mr.Hash())
}

func TestMediaResourceHasData(t *testing.T) {
	mr := NewMediaResource()
	assert.False(t, mr.HasData())

	mr.Data = []byte{}
	assert.False(t, mr.HasData())

	mr.Data = []byte("data")
	assert.True(t, mr.HasData())
}

func TestMediaResourceGetDataBase64Empty(t *testing.T) {
	mr := NewMediaResource()
	assert.Equal(t, "", mr.GetDataBase64())
}

func TestMediaResourceSetDataFromBase64Invalid(t *testing.T) {
	mr := NewMediaResource()
	err := mr.SetDataFromBase64("not-valid-base64!!!")
	assert.Error(t, err)
}

func TestMediaResourceGetParsedURL(t *testing.T) {
	mr := NewMediaResource()
	assert.Nil(t, mr.GetParsedURL())

	mr.URL = "https://example.com/path?query=1"
	parsed := mr.GetParsedURL()
	require.NotNil(t, parsed)
	assert.Equal(t, "https", parsed.Scheme)
	assert.Equal(t, "example.com", parsed.Host)
	assert.Equal(t, "/path", parsed.Path)
}

func TestMediaResourceGetParsedURLInvalid(t *testing.T) {
	mr := NewMediaResource()
	mr.URL = "://invalid"
	assert.Nil(t, mr.GetParsedURL())
}

func TestMediaResourceHashCombination(t *testing.T) {
	// Resource with multiple sources should have unique hash
	mr := NewMediaResource()
	mr.Text = "text"
	mr.Path = "/path"
	mr.URL = "https://url"
	mr.Data = []byte("data")

	hash := mr.Hash()
	assert.NotEmpty(t, hash)

	// Changing any source should change hash
	mr2 := NewMediaResource()
	mr2.Text = "text"
	mr2.Path = "/path"
	mr2.URL = "https://url"
	mr2.Data = []byte("different data")

	assert.NotEqual(t, hash, mr2.Hash())
}

func TestMediaResourceGetEmbeddingNil(t *testing.T) {
	mr := NewMediaResource()
	mr.Embeddings = nil
	assert.Nil(t, mr.GetEmbedding(EmbeddingKindDense))
}

// ============================================================================
// Component Tests
// ============================================================================

func TestBaseComponentImpl(t *testing.T) {
	comp := NewBaseComponentImpl("TestComponent")
	assert.Equal(t, "TestComponent", comp.ClassName())

	dict := comp.ToDict()
	assert.Equal(t, "TestComponent", dict["class_name"])

	jsonStr, err := comp.ToJSON()
	require.NoError(t, err)
	assert.Contains(t, jsonStr, "TestComponent")
}

func TestFromDict(t *testing.T) {
	data := map[string]interface{}{
		"class_name": "MyComponent",
	}
	comp := FromDict(data)
	assert.Equal(t, "MyComponent", comp.ClassName())
}

func TestFromJSON(t *testing.T) {
	jsonStr := `{"class_name": "JSONComponent"}`
	comp, err := FromJSON(jsonStr)
	require.NoError(t, err)
	assert.Equal(t, "JSONComponent", comp.ClassName())
}

func TestFromJSONInvalid(t *testing.T) {
	_, err := FromJSON("not valid json")
	assert.Error(t, err)
}

func TestTransformComponent(t *testing.T) {
	// Create a transform that uppercases text
	transform := NewTransformComponent("UppercaseTransform", func(nodes []*BaseNode) ([]*BaseNode, error) {
		// This is a no-op since we can't easily modify interface values
		return nodes, nil
	})

	assert.Equal(t, "UppercaseTransform", transform.ClassName())

	// Test with nil function
	nilTransform := NewTransformComponent("NilTransform", nil)
	result, err := nilTransform.Transform(nil)
	require.NoError(t, err)
	assert.Nil(t, result)
}

// ============================================================================
// NodeType Tests
// ============================================================================

func TestNodeTypes(t *testing.T) {
	assert.Equal(t, NodeType("TEXT"), ObjectTypeText)
	assert.Equal(t, NodeType("IMAGE"), ObjectTypeImage)
	assert.Equal(t, NodeType("INDEX"), ObjectTypeIndex)
	assert.Equal(t, NodeType("DOCUMENT"), ObjectTypeDocument)
	assert.Equal(t, NodeType("MULTIMODAL"), ObjectTypeMultimodal)
}

// ============================================================================
// JSON Serialization Tests
// ============================================================================

func TestNodeJSONRoundTrip(t *testing.T) {
	node := NewTextNode("Test content")
	node.Metadata = map[string]interface{}{"key": "value"}
	node.Embedding = []float64{0.1, 0.2, 0.3}

	// Serialize
	data, err := json.Marshal(node)
	require.NoError(t, err)

	// Deserialize
	var restored Node
	err = json.Unmarshal(data, &restored)
	require.NoError(t, err)

	assert.Equal(t, node.ID, restored.ID)
	assert.Equal(t, node.Text, restored.Text)
	assert.Equal(t, node.Type, restored.Type)
	assert.Equal(t, node.Embedding, restored.Embedding)
}

func TestImageNodeJSONRoundTrip(t *testing.T) {
	node := NewImageNodeFromURL("https://example.com/img.png", "image/png")
	node.Text = "Description"

	data, err := json.Marshal(node)
	require.NoError(t, err)

	var restored ImageNode
	err = json.Unmarshal(data, &restored)
	require.NoError(t, err)

	assert.Equal(t, node.ImageURL, restored.ImageURL)
	assert.Equal(t, node.ImageMimeType, restored.ImageMimeType)
	assert.Equal(t, node.Text, restored.Text)
}

func TestIndexNodeJSONRoundTrip(t *testing.T) {
	node := NewIndexNode("idx-123")
	node.Text = "Index description"

	data, err := json.Marshal(node)
	require.NoError(t, err)

	var restored IndexNode
	err = json.Unmarshal(data, &restored)
	require.NoError(t, err)

	assert.Equal(t, node.IndexID, restored.IndexID)
	assert.Equal(t, node.Text, restored.Text)
}

func TestMediaResourceJSONRoundTrip(t *testing.T) {
	mr := NewMediaResourceFromText("Hello world")
	mr.Embeddings[EmbeddingKindDense] = []float64{0.1, 0.2}

	data, err := json.Marshal(mr)
	require.NoError(t, err)

	var restored MediaResource
	err = json.Unmarshal(data, &restored)
	require.NoError(t, err)

	assert.Equal(t, mr.Text, restored.Text)
	assert.Equal(t, mr.MimeType, restored.MimeType)
}
