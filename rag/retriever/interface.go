// Package retriever provides retrieval implementations for RAG systems.
package retriever

import (
	"context"

	"github.com/aqua777/go-llamaindex/prompts"
	"github.com/aqua777/go-llamaindex/schema"
)

// Retriever is the interface for all retrievers.
type Retriever interface {
	// Retrieve retrieves nodes given a query.
	Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error)
}

// BaseRetriever provides common functionality for retrievers.
type BaseRetriever struct {
	// ObjectMap maps index IDs to objects for recursive retrieval.
	ObjectMap map[string]interface{}
	// Verbose enables verbose logging.
	Verbose bool
	// PromptMixin for prompt management.
	*prompts.BasePromptMixin
}

// NewBaseRetriever creates a new BaseRetriever.
func NewBaseRetriever() *BaseRetriever {
	return &BaseRetriever{
		ObjectMap:       make(map[string]interface{}),
		Verbose:         false,
		BasePromptMixin: prompts.NewBasePromptMixin(),
	}
}

// BaseRetrieverOption is a functional option for BaseRetriever.
type BaseRetrieverOption func(*BaseRetriever)

// WithObjectMap sets the object map for recursive retrieval.
func WithObjectMap(objectMap map[string]interface{}) BaseRetrieverOption {
	return func(br *BaseRetriever) {
		br.ObjectMap = objectMap
	}
}

// WithVerbose enables verbose logging.
func WithVerbose(verbose bool) BaseRetrieverOption {
	return func(br *BaseRetriever) {
		br.Verbose = verbose
	}
}

// NewBaseRetrieverWithOptions creates a new BaseRetriever with options.
func NewBaseRetrieverWithOptions(opts ...BaseRetrieverOption) *BaseRetriever {
	br := NewBaseRetriever()
	for _, opt := range opts {
		opt(br)
	}
	return br
}

// HandleRecursiveRetrieval handles IndexNode references by retrieving from nested objects.
// It checks if nodes have references to other retrievers/indices and follows them.
func (br *BaseRetriever) HandleRecursiveRetrieval(
	ctx context.Context,
	query schema.QueryBundle,
	nodes []schema.NodeWithScore,
) ([]schema.NodeWithScore, error) {
	var retrievedNodes []schema.NodeWithScore
	seen := make(map[string]bool)

	for _, n := range nodes {
		score := n.Score
		if score == 0 {
			score = 1.0
		}

		// Check if there's an object reference in the object map via node ID
		nodeID := n.Node.ID
		if obj, exists := br.ObjectMap[nodeID]; exists {
			// Retrieve from the object
			objNodes, err := br.retrieveFromObject(ctx, obj, query, score)
			if err != nil {
				return nil, err
			}
			for _, objNode := range objNodes {
				hash := objNode.Node.GenerateHash()
				if !seen[hash] {
					retrievedNodes = append(retrievedNodes, objNode)
					seen[hash] = true
				}
			}
			continue
		}

		// Regular node - add if not seen
		hash := n.Node.GenerateHash()
		if !seen[hash] {
			retrievedNodes = append(retrievedNodes, n)
			seen[hash] = true
		}
	}

	return retrievedNodes, nil
}

// retrieveFromObject retrieves nodes from an object based on its type.
func (br *BaseRetriever) retrieveFromObject(
	ctx context.Context,
	obj interface{},
	query schema.QueryBundle,
	score float64,
) ([]schema.NodeWithScore, error) {
	switch v := obj.(type) {
	case schema.NodeWithScore:
		return []schema.NodeWithScore{v}, nil
	case *schema.Node:
		return []schema.NodeWithScore{{Node: *v, Score: score}}, nil
	case schema.Node:
		return []schema.NodeWithScore{{Node: v, Score: score}}, nil
	case Retriever:
		return v.Retrieve(ctx, query)
	default:
		// Return empty if we can't retrieve from this object
		return nil, nil
	}
}

// AddObject adds an object to the object map.
func (br *BaseRetriever) AddObject(indexID string, obj interface{}) {
	br.ObjectMap[indexID] = obj
}

// GetObject retrieves an object from the object map.
func (br *BaseRetriever) GetObject(indexID string) interface{} {
	return br.ObjectMap[indexID]
}
