package schema

import (
	"github.com/google/uuid"
)

// IndexNode represents a node with a reference to another index or retriever.
// This can include other indices, query engines, retrievers, or other nodes.
type IndexNode struct {
	Node
	// IndexID is the ID of the referenced index/retriever.
	IndexID string `json:"index_id"`
	// Obj is an optional reference to the actual object (not serialized).
	Obj interface{} `json:"-"`
}

// NewIndexNode creates a new IndexNode with default values.
func NewIndexNode(indexID string) *IndexNode {
	return &IndexNode{
		Node: Node{
			ID:                uuid.New().String(),
			Type:              ObjectTypeIndex,
			Metadata:          make(map[string]interface{}),
			Relationships:     make(NodeRelationships),
			MetadataTemplate:  DefaultMetadataTemplate,
			MetadataSeparator: DefaultMetadataSeparator,
			TextTemplate:      DefaultTextNodeTemplate,
		},
		IndexID: indexID,
	}
}

// NewIndexNodeFromTextNode creates an IndexNode from an existing TextNode.
func NewIndexNodeFromTextNode(node *Node, indexID string) *IndexNode {
	return &IndexNode{
		Node: Node{
			ID:                        node.ID,
			Text:                      node.Text,
			Type:                      ObjectTypeIndex,
			Metadata:                  node.Metadata,
			Embedding:                 node.Embedding,
			Relationships:             node.Relationships,
			Hash:                      node.Hash,
			ExcludedEmbedMetadataKeys: node.ExcludedEmbedMetadataKeys,
			ExcludedLLMMetadataKeys:   node.ExcludedLLMMetadataKeys,
			MetadataTemplate:          node.MetadataTemplate,
			MetadataSeparator:         node.MetadataSeparator,
			TextTemplate:              node.TextTemplate,
			StartCharIdx:              node.StartCharIdx,
			EndCharIdx:                node.EndCharIdx,
			MimeType:                  node.MimeType,
		},
		IndexID: indexID,
	}
}

// ClassName returns the class name for serialization.
func (n *IndexNode) ClassName() string {
	return "IndexNode"
}

// GetType returns the node type.
func (n *IndexNode) GetType() NodeType {
	return ObjectTypeIndex
}

// ToDict converts the index node to a map representation.
func (n *IndexNode) ToDict() map[string]interface{} {
	result := n.Node.ToDict()
	result["class_name"] = n.ClassName()
	result["type"] = string(ObjectTypeIndex)
	result["index_id"] = n.IndexID
	return result
}

// SetObject sets the referenced object.
func (n *IndexNode) SetObject(obj interface{}) {
	n.Obj = obj
}

// GetObject returns the referenced object.
func (n *IndexNode) GetObject() interface{} {
	return n.Obj
}
