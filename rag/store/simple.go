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
		if query.Filters != nil {
			match := true
			for _, filter := range query.Filters.Filters {
				if filter.Operator == schema.FilterOperatorEq {
					if val, ok := node.Metadata[filter.Key]; !ok || fmt.Sprintf("%v", val) != fmt.Sprintf("%v", filter.Value) {
						match = false
						break
					}
				}
				// Add more operators as needed
			}
			if !match {
				continue
			}
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

func cosineSimilarity(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("vector lengths do not match")
	}

	var dotProduct, normA, normB float64
	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0, nil
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB)), nil
}
