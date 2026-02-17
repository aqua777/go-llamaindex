package store

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/aqua777/go-llamaindex/schema"
)

// SimpleVectorStore is a simple in-memory vector store.
type SimpleVectorStore struct {
	mu    sync.RWMutex
	nodes map[string]schema.Node
}

// NewSimpleVectorStore creates a new SimpleVectorStore.
func NewSimpleVectorStore() *SimpleVectorStore {
	return &SimpleVectorStore{
		nodes: make(map[string]schema.Node),
	}
}

func (s *SimpleVectorStore) Add(ctx context.Context, nodes []schema.Node) ([]string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var ids []string
	for _, node := range nodes {
		if node.ID == "" {
			return nil, errors.New("node ID cannot be empty")
		}
		s.nodes[node.ID] = node
		ids = append(ids, node.ID)
	}
	return ids, nil
}

func (s *SimpleVectorStore) Query(ctx context.Context, query schema.VectorStoreQuery) ([]schema.NodeWithScore, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	type scoreResult struct {
		id    string
		score float64
	}

	var scores []scoreResult

	for id, node := range s.nodes {
		// Apply filters if present
		if !matchesFilters(node.Metadata, query.Filters) {
			continue
		}

		if len(node.Embedding) == 0 {
			continue // Skip nodes without embeddings
		}

		score, err := cosineSimilarity(query.Embedding, node.Embedding)
		if err != nil {
			return nil, fmt.Errorf("failed to calculate similarity for node %s: %w", id, err)
		}
		scores = append(scores, scoreResult{id: id, score: score})
	}

	// Simple sort for top K
	for i := 0; i < len(scores); i++ {
		for j := i + 1; j < len(scores); j++ {
			if scores[j].score > scores[i].score {
				scores[i], scores[j] = scores[j], scores[i]
			}
		}
	}

	topK := query.TopK
	if topK > len(scores) {
		topK = len(scores)
	}

	var result []schema.NodeWithScore
	for i := 0; i < topK; i++ {
		node, ok := s.nodes[scores[i].id]
		if !ok {
			continue
		}
		result = append(result, schema.NodeWithScore{
			Node:  node,
			Score: scores[i].score,
		})
	}

	return result, nil
}

// Delete removes a node from the store by ID.
func (s *SimpleVectorStore) Delete(ctx context.Context, refDocID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	delete(s.nodes, refDocID)
	return nil
}

// PersistPath returns the storage path. Returns "" for in-memory stores.
func (s *SimpleVectorStore) PersistPath() string {
	return ""
}

// DeleteByFilter removes all nodes matching the metadata filters.
func (s *SimpleVectorStore) DeleteByFilter(ctx context.Context, filters *schema.MetadataFilters) (int, error) {
	if filters == nil || len(filters.Filters) == 0 {
		return 0, errors.New("filters cannot be nil or empty for bulk delete")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	var toDelete []string
	for id, node := range s.nodes {
		if matchesFilters(node.Metadata, filters) {
			toDelete = append(toDelete, id)
		}
	}

	for _, id := range toDelete {
		delete(s.nodes, id)
	}

	return len(toDelete), nil
}

// Count returns total nodes matching the optional filters.
func (s *SimpleVectorStore) Count(ctx context.Context, filters *schema.MetadataFilters) (int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if filters == nil || len(filters.Filters) == 0 {
		return len(s.nodes), nil
	}

	count := 0
	for _, node := range s.nodes {
		if matchesFilters(node.Metadata, filters) {
			count++
		}
	}
	return count, nil
}

func cosineSimilarity(a, b []float32) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("vector lengths do not match")
	}

	var dotProduct, normA, normB float64
	for i := 0; i < len(a); i++ {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0, nil
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB)), nil
}

// matchesFilters checks if node metadata matches the filter criteria.
// Supports AND/OR conditions and comparison operators.
func matchesFilters(metadata map[string]interface{}, filters *schema.MetadataFilters) bool {
	if filters == nil || len(filters.Filters) == 0 {
		return true
	}

	// Evaluate each filter based on condition (AND by default)
	condition := filters.Condition
	if condition == "" {
		condition = schema.FilterConditionAnd
	}

	for _, filter := range filters.Filters {
		match := evaluateFilter(metadata, filter)

		if condition == schema.FilterConditionAnd && !match {
			return false
		}
		if condition == schema.FilterConditionOr && match {
			return true
		}
	}

	// For AND: all passed; for OR: none matched
	return condition == schema.FilterConditionAnd
}

// evaluateFilter checks if a single filter matches the metadata.
func evaluateFilter(metadata map[string]interface{}, f schema.MetadataFilter) bool {
	val, ok := metadata[f.Key]
	if !ok {
		return f.Operator == schema.FilterOperatorIsEmpty
	}

	valStr := fmt.Sprintf("%v", val)
	filterValStr := fmt.Sprintf("%v", f.Value)

	switch f.Operator {
	case schema.FilterOperatorEq:
		return valStr == filterValStr
	case schema.FilterOperatorNe:
		return valStr != filterValStr
	case schema.FilterOperatorIsEmpty:
		return false // Key exists, so not empty
	default:
		return false
	}
}
