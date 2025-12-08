// Package objects provides object-to-node mapping for go-llamaindex.
// This enables storing and retrieving arbitrary objects (like tools) in an index.
package objects

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/aqua777/go-llamaindex/schema"
)

// ObjectNodeMapping is the interface for mapping objects to nodes.
// This allows arbitrary objects to be stored in and retrieved from an index.
type ObjectNodeMapping interface {
	// ToNode converts an object to a node for storage.
	ToNode(obj interface{}) (*schema.Node, error)
	// FromNode converts a node back to an object.
	FromNode(node *schema.Node) (interface{}, error)
	// GetObjects returns all objects in the mapping.
	GetObjects() []interface{}
	// AddObject adds an object to the mapping.
	AddObject(obj interface{}) error
	// GetObjectByID retrieves an object by its ID.
	GetObjectByID(id string) (interface{}, error)
	// GetNodes returns all nodes in the mapping.
	GetNodes() []*schema.Node
}

// ObjectWithID is an interface for objects that have an ID.
type ObjectWithID interface {
	GetID() string
}

// ObjectWithDescription is an interface for objects that have a description.
type ObjectWithDescription interface {
	GetDescription() string
}

// ObjectWithName is an interface for objects that have a name.
type ObjectWithName interface {
	GetName() string
}

// BaseObjectNodeMapping provides a base implementation of ObjectNodeMapping.
type BaseObjectNodeMapping struct {
	// objects stores the mapping from ID to object.
	objects map[string]interface{}
	// nodes stores the mapping from ID to node.
	nodes map[string]*schema.Node
	// ToNodeFunc is a custom function to convert objects to nodes.
	ToNodeFunc func(obj interface{}) (*schema.Node, error)
	// FromNodeFunc is a custom function to convert nodes to objects.
	FromNodeFunc func(node *schema.Node) (interface{}, error)
}

// BaseObjectNodeMappingOption configures a BaseObjectNodeMapping.
type BaseObjectNodeMappingOption func(*BaseObjectNodeMapping)

// WithToNodeFunc sets a custom ToNode function.
func WithToNodeFunc(fn func(obj interface{}) (*schema.Node, error)) BaseObjectNodeMappingOption {
	return func(m *BaseObjectNodeMapping) {
		m.ToNodeFunc = fn
	}
}

// WithFromNodeFunc sets a custom FromNode function.
func WithFromNodeFunc(fn func(node *schema.Node) (interface{}, error)) BaseObjectNodeMappingOption {
	return func(m *BaseObjectNodeMapping) {
		m.FromNodeFunc = fn
	}
}

// NewBaseObjectNodeMapping creates a new BaseObjectNodeMapping.
func NewBaseObjectNodeMapping(opts ...BaseObjectNodeMappingOption) *BaseObjectNodeMapping {
	m := &BaseObjectNodeMapping{
		objects: make(map[string]interface{}),
		nodes:   make(map[string]*schema.Node),
	}

	for _, opt := range opts {
		opt(m)
	}

	return m
}

// ToNode converts an object to a node.
func (m *BaseObjectNodeMapping) ToNode(obj interface{}) (*schema.Node, error) {
	if m.ToNodeFunc != nil {
		return m.ToNodeFunc(obj)
	}

	// Default implementation
	return defaultToNode(obj)
}

// FromNode converts a node back to an object.
func (m *BaseObjectNodeMapping) FromNode(node *schema.Node) (interface{}, error) {
	if m.FromNodeFunc != nil {
		return m.FromNodeFunc(node)
	}

	// Try to get from stored objects
	if obj, exists := m.objects[node.ID]; exists {
		return obj, nil
	}

	return nil, fmt.Errorf("object not found for node ID: %s", node.ID)
}

// GetObjects returns all objects in the mapping.
func (m *BaseObjectNodeMapping) GetObjects() []interface{} {
	objects := make([]interface{}, 0, len(m.objects))
	for _, obj := range m.objects {
		objects = append(objects, obj)
	}
	return objects
}

// AddObject adds an object to the mapping.
func (m *BaseObjectNodeMapping) AddObject(obj interface{}) error {
	node, err := m.ToNode(obj)
	if err != nil {
		return fmt.Errorf("failed to convert object to node: %w", err)
	}

	m.objects[node.ID] = obj
	m.nodes[node.ID] = node

	return nil
}

// GetObjectByID retrieves an object by its ID.
func (m *BaseObjectNodeMapping) GetObjectByID(id string) (interface{}, error) {
	if obj, exists := m.objects[id]; exists {
		return obj, nil
	}
	return nil, fmt.Errorf("object not found: %s", id)
}

// GetNodes returns all nodes in the mapping.
func (m *BaseObjectNodeMapping) GetNodes() []*schema.Node {
	nodes := make([]*schema.Node, 0, len(m.nodes))
	for _, node := range m.nodes {
		nodes = append(nodes, node)
	}
	return nodes
}

// defaultToNode is the default implementation for converting objects to nodes.
func defaultToNode(obj interface{}) (*schema.Node, error) {
	var id, text string
	metadata := make(map[string]interface{})

	// Try to get ID
	if withID, ok := obj.(ObjectWithID); ok {
		id = withID.GetID()
	}

	// Try to get description for text
	if withDesc, ok := obj.(ObjectWithDescription); ok {
		text = withDesc.GetDescription()
	}

	// Try to get name
	if withName, ok := obj.(ObjectWithName); ok {
		metadata["name"] = withName.GetName()
	}

	// If no text, try to serialize the object
	if text == "" {
		data, err := json.Marshal(obj)
		if err != nil {
			return nil, fmt.Errorf("failed to serialize object: %w", err)
		}
		text = string(data)
	}

	// Generate ID if not provided
	if id == "" {
		id = schema.NewNode().ID
	}

	node := schema.NewTextNode(text)
	node.ID = id
	node.Metadata = metadata
	node.Metadata["object_type"] = fmt.Sprintf("%T", obj)

	return node, nil
}

// SimpleObjectNodeMapping is a simple mapping that stores objects by name.
type SimpleObjectNodeMapping struct {
	*BaseObjectNodeMapping
}

// NewSimpleObjectNodeMapping creates a new SimpleObjectNodeMapping.
func NewSimpleObjectNodeMapping() *SimpleObjectNodeMapping {
	return &SimpleObjectNodeMapping{
		BaseObjectNodeMapping: NewBaseObjectNodeMapping(),
	}
}

// AddObjects adds multiple objects to the mapping.
func (m *SimpleObjectNodeMapping) AddObjects(objects ...interface{}) error {
	for _, obj := range objects {
		if err := m.AddObject(obj); err != nil {
			return err
		}
	}
	return nil
}

// ObjectRetriever retrieves objects from an index.
type ObjectRetriever interface {
	// RetrieveObjects retrieves objects matching the query.
	RetrieveObjects(ctx context.Context, query string) ([]interface{}, error)
}

// Ensure interfaces are implemented.
var _ ObjectNodeMapping = (*BaseObjectNodeMapping)(nil)
var _ ObjectNodeMapping = (*SimpleObjectNodeMapping)(nil)
