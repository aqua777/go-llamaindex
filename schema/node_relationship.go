package schema

// NodeRelationship represents the type of relationship between nodes.
type NodeRelationship string

const (
	// RelationshipSource indicates the node is the source document.
	RelationshipSource NodeRelationship = "SOURCE"
	// RelationshipPrevious indicates the node is the previous node in the document.
	RelationshipPrevious NodeRelationship = "PREVIOUS"
	// RelationshipNext indicates the node is the next node in the document.
	RelationshipNext NodeRelationship = "NEXT"
	// RelationshipParent indicates the node is the parent node in the document.
	RelationshipParent NodeRelationship = "PARENT"
	// RelationshipChild indicates the node is a child node in the document.
	RelationshipChild NodeRelationship = "CHILD"
)

// RelatedNodeInfo contains information about a related node.
type RelatedNodeInfo struct {
	NodeID   string                 `json:"node_id"`
	NodeType NodeType               `json:"node_type,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
	Hash     string                 `json:"hash,omitempty"`
}

// RelatedNodeType can be either a single RelatedNodeInfo or a slice of them.
// For SOURCE, PREVIOUS, NEXT, PARENT: single RelatedNodeInfo
// For CHILD: slice of RelatedNodeInfo
type RelatedNodeType interface {
	isRelatedNodeType()
}

// SingleRelatedNode wraps a single RelatedNodeInfo for relationship types
// that only allow one related node (SOURCE, PREVIOUS, NEXT, PARENT).
type SingleRelatedNode struct {
	Info RelatedNodeInfo
}

func (SingleRelatedNode) isRelatedNodeType() {}

// MultiRelatedNodes wraps multiple RelatedNodeInfo for relationship types
// that allow multiple related nodes (CHILD).
type MultiRelatedNodes struct {
	Infos []RelatedNodeInfo
}

func (MultiRelatedNodes) isRelatedNodeType() {}

// NodeRelationships maps relationship types to related node information.
type NodeRelationships map[NodeRelationship]RelatedNodeType

// SetSource sets the source relationship.
func (r NodeRelationships) SetSource(info RelatedNodeInfo) {
	r[RelationshipSource] = SingleRelatedNode{Info: info}
}

// GetSource returns the source node info if it exists.
func (r NodeRelationships) GetSource() *RelatedNodeInfo {
	if rel, ok := r[RelationshipSource]; ok {
		if single, ok := rel.(SingleRelatedNode); ok {
			return &single.Info
		}
	}
	return nil
}

// SetPrevious sets the previous relationship.
func (r NodeRelationships) SetPrevious(info RelatedNodeInfo) {
	r[RelationshipPrevious] = SingleRelatedNode{Info: info}
}

// GetPrevious returns the previous node info if it exists.
func (r NodeRelationships) GetPrevious() *RelatedNodeInfo {
	if rel, ok := r[RelationshipPrevious]; ok {
		if single, ok := rel.(SingleRelatedNode); ok {
			return &single.Info
		}
	}
	return nil
}

// SetNext sets the next relationship.
func (r NodeRelationships) SetNext(info RelatedNodeInfo) {
	r[RelationshipNext] = SingleRelatedNode{Info: info}
}

// GetNext returns the next node info if it exists.
func (r NodeRelationships) GetNext() *RelatedNodeInfo {
	if rel, ok := r[RelationshipNext]; ok {
		if single, ok := rel.(SingleRelatedNode); ok {
			return &single.Info
		}
	}
	return nil
}

// SetParent sets the parent relationship.
func (r NodeRelationships) SetParent(info RelatedNodeInfo) {
	r[RelationshipParent] = SingleRelatedNode{Info: info}
}

// GetParent returns the parent node info if it exists.
func (r NodeRelationships) GetParent() *RelatedNodeInfo {
	if rel, ok := r[RelationshipParent]; ok {
		if single, ok := rel.(SingleRelatedNode); ok {
			return &single.Info
		}
	}
	return nil
}

// SetChildren sets the child relationships.
func (r NodeRelationships) SetChildren(infos []RelatedNodeInfo) {
	r[RelationshipChild] = MultiRelatedNodes{Infos: infos}
}

// AddChild adds a child relationship.
func (r NodeRelationships) AddChild(info RelatedNodeInfo) {
	if rel, ok := r[RelationshipChild]; ok {
		if multi, ok := rel.(MultiRelatedNodes); ok {
			multi.Infos = append(multi.Infos, info)
			r[RelationshipChild] = multi
			return
		}
	}
	r[RelationshipChild] = MultiRelatedNodes{Infos: []RelatedNodeInfo{info}}
}

// GetChildren returns the child node infos if they exist.
func (r NodeRelationships) GetChildren() []RelatedNodeInfo {
	if rel, ok := r[RelationshipChild]; ok {
		if multi, ok := rel.(MultiRelatedNodes); ok {
			return multi.Infos
		}
	}
	return nil
}
