package store

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/schema"
)

// SimpleVectorStore is a simple in-memory vector store.
type SimpleVectorStore struct {
	mu        sync.RWMutex
	nodes     map[string]schema.Node
	bm25model *embedding.BM25 // lazily fitted; nil until first sparse/hybrid query
	bm25dirty bool            // true after any Add/Delete; triggers re-fit on next query
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
	s.bm25dirty = true
	return ids, nil
}

func (s *SimpleVectorStore) Query(ctx context.Context, query schema.VectorStoreQuery) ([]schema.NodeWithScore, error) {
	switch query.Mode {
	case schema.QueryModeDefault, "":
		return s.queryDense(query)
	case schema.QueryModeSparse:
		return s.querySparse(query)
	case schema.QueryModeHybrid:
		return s.queryHybridMode(query)
	default:
		return nil, fmt.Errorf("unsupported query mode: %q", query.Mode)
	}
}

// queryDense performs cosine-similarity-based dense retrieval.
func (s *SimpleVectorStore) queryDense(query schema.VectorStoreQuery) ([]schema.NodeWithScore, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var scores []scoreResult

	queryEmbedding := query.GetEmbedding()
	if len(queryEmbedding) == 0 {
		return nil, fmt.Errorf("query embedding is empty")
	}

	for id, node := range s.nodes {
		if !MatchesFilters(node.Metadata, query.Filters) {
			continue
		}

		if len(node.Embedding) == 0 {
			continue
		}

		score, err := cosineSimilarity(queryEmbedding, node.Embedding)
		if err != nil {
			return nil, fmt.Errorf("failed to calculate similarity for node %s: %w", id, err)
		}
		scores = append(scores, scoreResult{id: id, score: score})
	}

	sortScoresDesc(scores)

	topK := query.GetTopK()
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

// querySparse performs BM25-based sparse retrieval.
// Must promote to write lock when re-fit is needed.
func (s *SimpleVectorStore) querySparse(query schema.VectorStoreQuery) ([]schema.NodeWithScore, error) {
	if query.QueryStr == "" {
		return nil, fmt.Errorf("QueryStr must not be empty for sparse query mode")
	}

	s.mu.RLock()
	needsFit := s.bm25dirty || s.bm25model == nil
	s.mu.RUnlock()

	if needsFit {
		s.mu.Lock()
		if err := s.ensureBM25Fitted(); err != nil {
			s.mu.Unlock()
			return nil, err
		}
		result, err := s.queryWithBM25(query)
		s.mu.Unlock()
		return result, err
	}

	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.queryWithBM25(query)
}

// queryHybridMode performs hybrid dense+sparse retrieval.
// Must promote to write lock when re-fit is needed.
func (s *SimpleVectorStore) queryHybridMode(query schema.VectorStoreQuery) ([]schema.NodeWithScore, error) {
	if len(query.GetEmbedding()) == 0 {
		return nil, fmt.Errorf("query embedding is required for hybrid query mode")
	}
	if query.QueryStr == "" {
		return nil, fmt.Errorf("QueryStr must not be empty for hybrid query mode")
	}
	if query.Alpha != nil && (*query.Alpha < 0 || *query.Alpha > 1) {
		return nil, fmt.Errorf("alpha must be in [0, 1], got %v", *query.Alpha)
	}

	s.mu.RLock()
	needsFit := s.bm25dirty || s.bm25model == nil
	s.mu.RUnlock()

	if needsFit {
		s.mu.Lock()
		if err := s.ensureBM25Fitted(); err != nil {
			s.mu.Unlock()
			return nil, err
		}
		result, err := s.queryHybrid(query)
		s.mu.Unlock()
		return result, err
	}

	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.queryHybrid(query)
}

// ensureBM25Fitted re-fits the BM25 model over the current node corpus when
// bm25dirty is true.
//
// Must be called under write lock (mu.Lock).
func (s *SimpleVectorStore) ensureBM25Fitted() error {
	if !s.bm25dirty && s.bm25model != nil {
		return nil
	}

	texts := make([]string, 0, len(s.nodes))
	for _, node := range s.nodes {
		texts = append(texts, node.Text)
	}

	if s.bm25model == nil {
		s.bm25model = embedding.NewBM25()
	}

	if len(texts) > 0 {
		s.bm25model.Fit(texts)
	}

	s.bm25dirty = false
	return nil
}

// queryWithBM25 scores every node passing metadata filters against
// query.QueryStr using the fitted BM25 model and returns the top-K results.
//
// Must be called under lock after ensureBM25Fitted.
func (s *SimpleVectorStore) queryWithBM25(query schema.VectorStoreQuery) ([]schema.NodeWithScore, error) {
	if query.QueryStr == "" {
		return nil, fmt.Errorf("QueryStr must not be empty for sparse query mode")
	}

	var scores []scoreResult
	for id, node := range s.nodes {
		if !MatchesFilters(node.Metadata, query.Filters) {
			continue
		}
		score := s.bm25model.Score(query.QueryStr, node.Text)
		scores = append(scores, scoreResult{id: id, score: score})
	}

	sortScoresDesc(scores)

	topK := query.GetTopK()
	if topK > len(scores) {
		topK = len(scores)
	}

	result := make([]schema.NodeWithScore, 0, topK)
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

// queryHybrid combines cosine similarity (dense) and BM25 (sparse) scores:
//
//	hybridScore = alpha * denseScore + (1 - alpha) * normalizedBM25Score
//
// BM25 scores are MinMax-normalized over the filtered candidate set before
// combining. alpha defaults to 0.5 when query.Alpha is nil.
//
// Must be called under lock after ensureBM25Fitted.
func (s *SimpleVectorStore) queryHybrid(query schema.VectorStoreQuery) ([]schema.NodeWithScore, error) {
	alpha := 0.5
	if query.Alpha != nil {
		alpha = *query.Alpha
	}

	queryEmbedding := query.GetEmbedding()

	type candidate struct {
		id         string
		denseScore float64
		bm25Score  float64
	}

	var candidates []candidate
	for id, node := range s.nodes {
		if !MatchesFilters(node.Metadata, query.Filters) {
			continue
		}

		bm25Score := s.bm25model.Score(query.QueryStr, node.Text)

		var denseScore float64
		if len(node.Embedding) > 0 && len(queryEmbedding) > 0 {
			var err error
			denseScore, err = cosineSimilarity(queryEmbedding, node.Embedding)
			if err != nil {
				return nil, fmt.Errorf("failed to calculate cosine similarity for node %s: %w", id, err)
			}
		}

		candidates = append(candidates, candidate{id: id, denseScore: denseScore, bm25Score: bm25Score})
	}

	// MinMax normalize BM25 scores.
	normalizedBM25 := minMaxNormalize(candidates, func(c candidate) float64 { return c.bm25Score })

	scores := make([]scoreResult, len(candidates))
	for i, c := range candidates {
		hybrid := alpha*c.denseScore + (1-alpha)*normalizedBM25[i]
		scores[i] = scoreResult{id: c.id, score: hybrid}
	}

	sortScoresDesc(scores)

	topK := query.GetTopK()
	if topK > len(scores) {
		topK = len(scores)
	}

	result := make([]schema.NodeWithScore, 0, topK)
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

// minMaxNormalize returns a slice of values normalized to [0, 1].
// When all values are equal (including all-zero), returns all zeros.
func minMaxNormalize[T any](items []T, getValue func(T) float64) []float64 {
	if len(items) == 0 {
		return nil
	}

	minVal := getValue(items[0])
	maxVal := getValue(items[0])
	for _, item := range items[1:] {
		v := getValue(item)
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	result := make([]float64, len(items))
	span := maxVal - minVal
	if span == 0 {
		return result
	}

	for i, item := range items {
		result[i] = (getValue(item) - minVal) / span
	}
	return result
}

// Delete removes a node from the store by ID.
func (s *SimpleVectorStore) Delete(ctx context.Context, refDocID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	delete(s.nodes, refDocID)
	s.bm25dirty = true
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
		if MatchesFilters(node.Metadata, filters) {
			toDelete = append(toDelete, id)
		}
	}

	for _, id := range toDelete {
		delete(s.nodes, id)
	}
	s.bm25dirty = true

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
		if MatchesFilters(node.Metadata, filters) {
			count++
		}
	}
	return count, nil
}

// sortScoresDesc sorts a slice of {id, score} pairs in descending score order
// using a selection sort. Equal scores are broken by node ID ascending so that
// results are deterministic regardless of map iteration order.
type scoreResult struct {
	id    string
	score float64
}

func sortScoresDesc(scores []scoreResult) {
	for i := 0; i < len(scores); i++ {
		for j := i + 1; j < len(scores); j++ {
			less := scores[j].score > scores[i].score ||
				(scores[j].score == scores[i].score && scores[j].id < scores[i].id)
			if less {
				scores[i], scores[j] = scores[j], scores[i]
			}
		}
	}
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

