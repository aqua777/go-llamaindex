package schema

import (
	"encoding/json"
)

// BaseComponent defines the interface for serializable components.
// This provides a foundation for all LlamaIndex components that need
// to be serialized/deserialized.
type BaseComponent interface {
	// ClassName returns a unique identifier for the component type.
	// This is used for serialization to identify the concrete type.
	ClassName() string

	// ToDict converts the component to a map representation.
	ToDict() map[string]interface{}

	// ToJSON converts the component to a JSON string.
	ToJSON() (string, error)
}

// BaseComponentImpl provides a default implementation of BaseComponent.
type BaseComponentImpl struct {
	className string
}

// NewBaseComponentImpl creates a new BaseComponentImpl with the given class name.
func NewBaseComponentImpl(className string) BaseComponentImpl {
	return BaseComponentImpl{className: className}
}

// ClassName returns the class name.
func (b BaseComponentImpl) ClassName() string {
	return b.className
}

// ToDict converts to a map with class_name included.
func (b BaseComponentImpl) ToDict() map[string]interface{} {
	return map[string]interface{}{
		"class_name": b.className,
	}
}

// ToJSON converts to JSON string.
func (b BaseComponentImpl) ToJSON() (string, error) {
	data := b.ToDict()
	bytes, err := json.Marshal(data)
	if err != nil {
		return "", err
	}
	return string(bytes), nil
}

// FromDict creates a BaseComponentImpl from a map.
func FromDict(data map[string]interface{}) BaseComponentImpl {
	className, _ := data["class_name"].(string)
	return BaseComponentImpl{className: className}
}

// FromJSON creates a BaseComponentImpl from a JSON string.
func FromJSON(jsonStr string) (BaseComponentImpl, error) {
	var data map[string]interface{}
	if err := json.Unmarshal([]byte(jsonStr), &data); err != nil {
		return BaseComponentImpl{}, err
	}
	return FromDict(data), nil
}

// TransformComponent defines the interface for components that transform nodes.
// This is used for node parsers, extractors, and other transformation pipelines.
type TransformComponent interface {
	BaseComponent

	// Transform applies a transformation to a slice of nodes and returns the result.
	Transform(nodes []*BaseNode) ([]*BaseNode, error)
}

// TransformFunc is a function type that implements the Transform method.
type TransformFunc func(nodes []*BaseNode) ([]*BaseNode, error)

// TransformComponentImpl provides a base implementation of TransformComponent.
type TransformComponentImpl struct {
	BaseComponentImpl
	transformFn TransformFunc
}

// NewTransformComponent creates a new TransformComponentImpl.
func NewTransformComponent(className string, fn TransformFunc) *TransformComponentImpl {
	return &TransformComponentImpl{
		BaseComponentImpl: NewBaseComponentImpl(className),
		transformFn:       fn,
	}
}

// Transform applies the transformation function.
func (t *TransformComponentImpl) Transform(nodes []*BaseNode) ([]*BaseNode, error) {
	if t.transformFn == nil {
		return nodes, nil
	}
	return t.transformFn(nodes)
}
