package chromem

import (
	"context"
	"fmt"
	"runtime"

	"github.com/aqua777/go-llamaindex/schema"
	"github.com/philippgille/chromem-go"
)

// ChromemStore is a vector store implementation using chromem-go.
type ChromemStore struct {
	db         *chromem.DB
	collection *chromem.Collection
}

// NewChromemStore creates a new ChromemStore.
// If persistPath is empty, the store will be in-memory only.
func NewChromemStore(persistPath string, collectionName string) (*ChromemStore, error) {
	var db *chromem.DB
	if persistPath != "" {
		var err error
		db, err = chromem.NewPersistentDB(persistPath, false)
		if err != nil {
			return nil, fmt.Errorf("failed to create persistent chromem db: %w", err)
		}
	} else {
		db = chromem.NewDB()
	}

	// We pass nil for embedding function because we handle embeddings externally in the RAG pipeline
	// and pass them explicitly to Add/Query.
	collection, err := db.GetOrCreateCollection(collectionName, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to get or create collection: %w", err)
	}

	return &ChromemStore{
		db:         db,
		collection: collection,
	}, nil
}

// Add adds nodes to the store.
func (s *ChromemStore) Add(ctx context.Context, nodes []schema.Node) ([]string, error) {
	docs := make([]chromem.Document, len(nodes))
	ids := make([]string, len(nodes))

	for i, node := range nodes {
		if len(node.Embedding) == 0 {
			return nil, fmt.Errorf("node %s has no embedding", node.ID)
		}

		// Map Metadata to map[string]string as chromem supports flexible metadata but let's be safe
		// chromem-go Document.Metadata is map[string]string.
		// Our Node.Metadata is map[string]interface{}.
		// We need to convert values to strings.
		meta := make(map[string]string)
		for k, v := range node.Metadata {
			meta[k] = fmt.Sprintf("%v", v)
		}

		// Add Node Type to metadata if not present, for reconstruction
		if _, ok := meta["_node_type"]; !ok {
			meta["_node_type"] = string(node.Type)
		}

		// Use generic float32 for chromem
		embedding32 := make([]float32, len(node.Embedding))
		for j, v := range node.Embedding {
			embedding32[j] = float32(v)
		}

		docs[i] = chromem.Document{
			ID:        node.ID,
			Content:   node.Text,
			Metadata:  meta,
			Embedding: embedding32,
		}
		ids[i] = node.ID
	}

	// chromem-go's AddDocuments handles concurrency.
	// We can pass runtime.NumCPU() for concurrency.
	err := s.collection.AddDocuments(ctx, docs, runtime.NumCPU())
	if err != nil {
		return nil, fmt.Errorf("failed to add documents to chromem collection: %w", err)
	}

	return ids, nil
}

// Query finds the top-k most similar nodes to the query embedding.
func (s *ChromemStore) Query(ctx context.Context, query schema.VectorStoreQuery) ([]schema.NodeWithScore, error) {
	// Convert embedding to float32
	queryEmbedding32 := make([]float32, len(query.Embedding))
	for i, v := range query.Embedding {
		queryEmbedding32[i] = float32(v)
	}

	var where map[string]string
	var whereDocument map[string]string

	// Map MetadataFilters to chromem's where/whereDocument if possible
	// Chromem currently supports exact match filtering via map[string]string for metadata
	// and contains match for document content.
	if query.Filters != nil {
		for _, f := range query.Filters.Filters {
			if f.Operator == schema.FilterOperatorEq {
				// Only EQ is directly supported by simple map[string]string where clause
				// Also assume value is string-able
				if where == nil {
					where = make(map[string]string)
				}
				where[f.Key] = fmt.Sprintf("%v", f.Value)
			}
			// For now, we only support EQ. Other operators would require post-filtering.
		}
	}

	res, err := s.collection.QueryEmbedding(ctx, queryEmbedding32, query.TopK, where, whereDocument)
	if err != nil {
		return nil, fmt.Errorf("failed to query chromem collection: %w", err)
	}

	nodes := make([]schema.NodeWithScore, len(res))
	for i, doc := range res {
		// Reconstruct Node
		// Convert metadata back
		meta := make(map[string]interface{})
		var nodeType schema.NodeType = schema.ObjectTypeText // default

		for k, v := range doc.Metadata {
			if k == "_node_type" {
				nodeType = schema.NodeType(v)
				continue
			}
			meta[k] = v // Keep as string for now
		}

		nodes[i] = schema.NodeWithScore{
			Node: schema.Node{
				ID:       doc.ID,
				Text:     doc.Content,
				Type:     nodeType,
				Metadata: meta,
			},
			// chromem returns Cosine Similarity (0-1 for normalized vectors?)
			// The `Similarity` field in result.
			Score: float64(doc.Similarity),
		}
	}

	return nodes, nil
}
