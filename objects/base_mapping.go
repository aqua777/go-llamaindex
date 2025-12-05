package objects

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/schema"
)

// TypedObjectNodeMapping is a generic mapping for typed objects.
type TypedObjectNodeMapping[T any] struct {
	*BaseObjectNodeMapping
	// objects stores typed objects by ID.
	typedObjects map[string]T
	// textExtractor extracts text from an object for embedding.
	textExtractor func(T) string
	// idExtractor extracts ID from an object.
	idExtractor func(T) string
}

// TypedObjectNodeMappingOption configures a TypedObjectNodeMapping.
type TypedObjectNodeMappingOption[T any] func(*TypedObjectNodeMapping[T])

// WithTextExtractor sets the text extractor function.
func WithTextExtractor[T any](fn func(T) string) TypedObjectNodeMappingOption[T] {
	return func(m *TypedObjectNodeMapping[T]) {
		m.textExtractor = fn
	}
}

// WithIDExtractor sets the ID extractor function.
func WithIDExtractor[T any](fn func(T) string) TypedObjectNodeMappingOption[T] {
	return func(m *TypedObjectNodeMapping[T]) {
		m.idExtractor = fn
	}
}

// NewTypedObjectNodeMapping creates a new TypedObjectNodeMapping.
func NewTypedObjectNodeMapping[T any](opts ...TypedObjectNodeMappingOption[T]) *TypedObjectNodeMapping[T] {
	m := &TypedObjectNodeMapping[T]{
		BaseObjectNodeMapping: NewBaseObjectNodeMapping(),
		typedObjects:          make(map[string]T),
	}

	for _, opt := range opts {
		opt(m)
	}

	// Set up custom ToNode function
	m.ToNodeFunc = func(obj interface{}) (*schema.Node, error) {
		typed, ok := obj.(T)
		if !ok {
			return nil, fmt.Errorf("expected %T, got %T", *new(T), obj)
		}
		return m.typedToNode(typed)
	}

	return m
}

// typedToNode converts a typed object to a node.
func (m *TypedObjectNodeMapping[T]) typedToNode(obj T) (*schema.Node, error) {
	var id, text string

	// Get ID
	if m.idExtractor != nil {
		id = m.idExtractor(obj)
	} else {
		// Try to get ID from interface
		if withID, ok := any(obj).(ObjectWithID); ok {
			id = withID.GetID()
		} else {
			id = schema.NewNode().ID
		}
	}

	// Get text
	if m.textExtractor != nil {
		text = m.textExtractor(obj)
	} else {
		// Try to get description
		if withDesc, ok := any(obj).(ObjectWithDescription); ok {
			text = withDesc.GetDescription()
		} else {
			// Serialize to JSON
			data, err := json.Marshal(obj)
			if err != nil {
				return nil, fmt.Errorf("failed to serialize object: %w", err)
			}
			text = string(data)
		}
	}

	node := schema.NewTextNode(text)
	node.ID = id
	node.Metadata = map[string]interface{}{
		"object_type": reflect.TypeOf(obj).String(),
	}

	return node, nil
}

// AddTypedObject adds a typed object to the mapping.
func (m *TypedObjectNodeMapping[T]) AddTypedObject(obj T) error {
	node, err := m.typedToNode(obj)
	if err != nil {
		return err
	}

	m.typedObjects[node.ID] = obj
	m.objects[node.ID] = obj
	m.nodes[node.ID] = node

	return nil
}

// AddTypedObjects adds multiple typed objects.
func (m *TypedObjectNodeMapping[T]) AddTypedObjects(objects ...T) error {
	for _, obj := range objects {
		if err := m.AddTypedObject(obj); err != nil {
			return err
		}
	}
	return nil
}

// GetTypedObject retrieves a typed object by ID.
func (m *TypedObjectNodeMapping[T]) GetTypedObject(id string) (T, error) {
	obj, exists := m.typedObjects[id]
	if !exists {
		var zero T
		return zero, fmt.Errorf("object not found: %s", id)
	}
	return obj, nil
}

// GetTypedObjects returns all typed objects.
func (m *TypedObjectNodeMapping[T]) GetTypedObjects() []T {
	objects := make([]T, 0, len(m.typedObjects))
	for _, obj := range m.typedObjects {
		objects = append(objects, obj)
	}
	return objects
}

// TypedObjectRetriever retrieves typed objects based on query similarity.
type TypedObjectRetriever[T any] struct {
	// mapping is the typed object node mapping.
	mapping *TypedObjectNodeMapping[T]
	// embedModel is the embedding model for similarity search.
	embedModel embedding.EmbeddingModel
	// nodeEmbeddings stores embeddings for each node.
	nodeEmbeddings map[string][]float64
	// topK is the number of objects to retrieve.
	topK int
}

// TypedObjectRetrieverOption configures a TypedObjectRetriever.
type TypedObjectRetrieverOption[T any] func(*TypedObjectRetriever[T])

// WithTypedRetrieverTopK sets the number of objects to retrieve.
func WithTypedRetrieverTopK[T any](k int) TypedObjectRetrieverOption[T] {
	return func(r *TypedObjectRetriever[T]) {
		r.topK = k
	}
}

// NewTypedObjectRetriever creates a new TypedObjectRetriever.
func NewTypedObjectRetriever[T any](mapping *TypedObjectNodeMapping[T], embedModel embedding.EmbeddingModel, opts ...TypedObjectRetrieverOption[T]) *TypedObjectRetriever[T] {
	r := &TypedObjectRetriever[T]{
		mapping:        mapping,
		embedModel:     embedModel,
		nodeEmbeddings: make(map[string][]float64),
		topK:           3,
	}

	for _, opt := range opts {
		opt(r)
	}

	return r
}

// BuildIndex builds embeddings for all objects.
func (r *TypedObjectRetriever[T]) BuildIndex(ctx context.Context) error {
	nodes := r.mapping.GetNodes()

	for _, node := range nodes {
		text := node.GetContent(schema.MetadataModeNone)
		emb, err := r.embedModel.GetTextEmbedding(ctx, text)
		if err != nil {
			return fmt.Errorf("failed to embed object %s: %w", node.ID, err)
		}
		r.nodeEmbeddings[node.ID] = emb
	}

	return nil
}

// RetrieveTypedObjects retrieves the most relevant typed objects for a query.
func (r *TypedObjectRetriever[T]) RetrieveTypedObjects(ctx context.Context, query string) ([]T, error) {
	// Get query embedding
	queryEmb, err := r.embedModel.GetQueryEmbedding(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to embed query: %w", err)
	}

	// Calculate similarities
	type scoredObject struct {
		id    string
		score float64
	}

	var scores []scoredObject
	for nodeID, nodeEmb := range r.nodeEmbeddings {
		score, err := embedding.CosineSimilarity(queryEmb, nodeEmb)
		if err != nil {
			continue
		}
		scores = append(scores, scoredObject{id: nodeID, score: score})
	}

	// Sort by score descending
	for i := 0; i < len(scores)-1; i++ {
		for j := i + 1; j < len(scores); j++ {
			if scores[j].score > scores[i].score {
				scores[i], scores[j] = scores[j], scores[i]
			}
		}
	}

	// Get top K objects
	k := r.topK
	if k > len(scores) {
		k = len(scores)
	}

	result := make([]T, 0, k)
	for i := 0; i < k; i++ {
		obj, err := r.mapping.GetTypedObject(scores[i].id)
		if err != nil {
			continue
		}
		result = append(result, obj)
	}

	return result, nil
}

// RetrieveObjects implements ObjectRetriever.
func (r *TypedObjectRetriever[T]) RetrieveObjects(ctx context.Context, query string) ([]interface{}, error) {
	objects, err := r.RetrieveTypedObjects(ctx, query)
	if err != nil {
		return nil, err
	}

	result := make([]interface{}, len(objects))
	for i, obj := range objects {
		result[i] = obj
	}

	return result, nil
}

// ObjectIndex combines a mapping with retrieval capabilities.
type ObjectIndex[T any] struct {
	// Mapping is the object-to-node mapping.
	Mapping *TypedObjectNodeMapping[T]
	// Retriever is the object retriever.
	Retriever *TypedObjectRetriever[T]
}

// NewObjectIndex creates a new ObjectIndex.
func NewObjectIndex[T any](embedModel embedding.EmbeddingModel, opts ...TypedObjectNodeMappingOption[T]) *ObjectIndex[T] {
	mapping := NewTypedObjectNodeMapping(opts...)
	retriever := NewTypedObjectRetriever(mapping, embedModel)

	return &ObjectIndex[T]{
		Mapping:   mapping,
		Retriever: retriever,
	}
}

// Add adds objects to the index.
func (idx *ObjectIndex[T]) Add(ctx context.Context, objects ...T) error {
	if err := idx.Mapping.AddTypedObjects(objects...); err != nil {
		return err
	}
	return idx.Retriever.BuildIndex(ctx)
}

// Retrieve retrieves objects matching the query.
func (idx *ObjectIndex[T]) Retrieve(ctx context.Context, query string) ([]T, error) {
	return idx.Retriever.RetrieveTypedObjects(ctx, query)
}

// Get retrieves an object by ID.
func (idx *ObjectIndex[T]) Get(id string) (T, error) {
	return idx.Mapping.GetTypedObject(id)
}

// All returns all objects in the index.
func (idx *ObjectIndex[T]) All() []T {
	return idx.Mapping.GetTypedObjects()
}
