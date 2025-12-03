package index

import (
	"context"
	"fmt"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/rag/queryengine"
	"github.com/aqua777/go-llamaindex/rag/retriever"
	"github.com/aqua777/go-llamaindex/rag/store"
	"github.com/aqua777/go-llamaindex/rag/synthesizer"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/storage"
	"github.com/aqua777/go-llamaindex/storage/indexstore"
)

// VectorStoreIndex is an index backed by a vector store.
type VectorStoreIndex struct {
	*BaseIndex
	// vectorStore is the underlying vector store.
	vectorStore store.VectorStore
	// insertBatchSize is the batch size for inserting nodes.
	insertBatchSize int
	// storeNodesOverride forces storing nodes in docstore even if vector store stores text.
	storeNodesOverride bool
}

// VectorStoreIndexOption configures VectorStoreIndex creation.
type VectorStoreIndexOption func(*VectorStoreIndex)

// WithVectorStore sets the vector store.
func WithVectorStore(vs store.VectorStore) VectorStoreIndexOption {
	return func(vsi *VectorStoreIndex) {
		vsi.vectorStore = vs
	}
}

// WithInsertBatchSize sets the batch size for inserting nodes.
func WithInsertBatchSize(size int) VectorStoreIndexOption {
	return func(vsi *VectorStoreIndex) {
		vsi.insertBatchSize = size
	}
}

// WithStoreNodesOverride forces storing nodes in docstore.
func WithStoreNodesOverride(override bool) VectorStoreIndexOption {
	return func(vsi *VectorStoreIndex) {
		vsi.storeNodesOverride = override
	}
}

// WithVectorIndexStorageContext sets the storage context.
func WithVectorIndexStorageContext(sc *storage.StorageContext) VectorStoreIndexOption {
	return func(vsi *VectorStoreIndex) {
		vsi.storageContext = sc
	}
}

// WithVectorIndexEmbedModel sets the embedding model.
func WithVectorIndexEmbedModel(model EmbeddingModel) VectorStoreIndexOption {
	return func(vsi *VectorStoreIndex) {
		vsi.embedModel = model
	}
}

// NewVectorStoreIndex creates a new VectorStoreIndex.
func NewVectorStoreIndex(ctx context.Context, nodes []schema.Node, opts ...VectorStoreIndexOption) (*VectorStoreIndex, error) {
	indexStruct := indexstore.NewVectorStoreIndex()

	vsi := &VectorStoreIndex{
		BaseIndex:          NewBaseIndex(indexStruct),
		insertBatchSize:    2048,
		storeNodesOverride: false,
	}

	for _, opt := range opts {
		opt(vsi)
	}

	// Get vector store from storage context if not set
	if vsi.vectorStore == nil && vsi.storageContext != nil {
		vsi.vectorStore = vsi.storageContext.VectorStore()
	}

	// Build index from nodes
	if len(nodes) > 0 {
		if err := vsi.buildIndexFromNodes(ctx, nodes); err != nil {
			return nil, err
		}
	}

	// Add index struct to store
	if err := vsi.storageContext.IndexStore.AddIndexStruct(ctx, indexStruct); err != nil {
		return nil, err
	}

	return vsi, nil
}

// NewVectorStoreIndexFromDocuments creates a VectorStoreIndex from documents.
func NewVectorStoreIndexFromDocuments(
	ctx context.Context,
	documents []schema.Document,
	opts ...VectorStoreIndexOption,
) (*VectorStoreIndex, error) {
	// Convert documents to nodes
	var nodes []schema.Node
	for _, doc := range documents {
		node := schema.NewTextNode(doc.Text)
		node.Metadata = doc.Metadata
		if doc.ID != "" {
			node.ID = doc.ID
		}
		nodes = append(nodes, *node)
	}

	return NewVectorStoreIndex(ctx, nodes, opts...)
}

// NewVectorStoreIndexFromVectorStore creates a VectorStoreIndex from an existing vector store.
func NewVectorStoreIndexFromVectorStore(
	ctx context.Context,
	vs store.VectorStore,
	opts ...VectorStoreIndexOption,
) (*VectorStoreIndex, error) {
	opts = append([]VectorStoreIndexOption{WithVectorStore(vs)}, opts...)
	return NewVectorStoreIndex(ctx, nil, opts...)
}

// buildIndexFromNodes builds the index from nodes.
func (vsi *VectorStoreIndex) buildIndexFromNodes(ctx context.Context, nodes []schema.Node) error {
	// Filter out nodes without content
	var contentNodes []schema.Node
	for _, node := range nodes {
		if node.GetContent(schema.MetadataModeEmbed) != "" {
			contentNodes = append(contentNodes, node)
		}
	}

	if len(contentNodes) == 0 {
		return nil
	}

	return vsi.addNodesToIndex(ctx, contentNodes)
}

// addNodesToIndex adds nodes to the index.
func (vsi *VectorStoreIndex) addNodesToIndex(ctx context.Context, nodes []schema.Node) error {
	if vsi.vectorStore == nil {
		return fmt.Errorf("vector store not configured")
	}

	// Process in batches
	for i := 0; i < len(nodes); i += vsi.insertBatchSize {
		end := i + vsi.insertBatchSize
		if end > len(nodes) {
			end = len(nodes)
		}
		batch := nodes[i:end]

		// Generate embeddings
		nodesWithEmbeddings, err := vsi.getNodesWithEmbeddings(ctx, batch)
		if err != nil {
			return err
		}

		// Add to vector store
		newIDs, err := vsi.vectorStore.Add(ctx, nodesWithEmbeddings)
		if err != nil {
			return err
		}

		// Update index struct
		for j, node := range nodesWithEmbeddings {
			textID := newIDs[j]
			vsi.indexStruct.AddNode(node.ID, textID)
		}

		// Store nodes in docstore if needed
		if vsi.storeNodesOverride {
			for _, node := range nodesWithEmbeddings {
				// Clear embedding to avoid duplication
				nodeCopy := node
				nodeCopy.Embedding = nil
				if err := vsi.storageContext.DocStore.AddDocuments(ctx, []schema.BaseNode{&nodeCopy}, true); err != nil {
					return err
				}
			}
		}
	}

	return nil
}

// getNodesWithEmbeddings generates embeddings for nodes.
func (vsi *VectorStoreIndex) getNodesWithEmbeddings(ctx context.Context, nodes []schema.Node) ([]schema.Node, error) {
	if vsi.embedModel == nil {
		return nil, fmt.Errorf("embedding model not configured")
	}

	result := make([]schema.Node, len(nodes))
	for i, node := range nodes {
		embedding, err := vsi.embedModel.GetTextEmbedding(ctx, node.GetContent(schema.MetadataModeEmbed))
		if err != nil {
			return nil, err
		}

		nodeCopy := node
		nodeCopy.Embedding = embedding
		result[i] = nodeCopy
	}

	return result, nil
}

// VectorStore returns the underlying vector store.
func (vsi *VectorStoreIndex) VectorStore() store.VectorStore {
	return vsi.vectorStore
}

// AsRetriever returns a retriever for this index.
func (vsi *VectorStoreIndex) AsRetriever(opts ...RetrieverOption) retriever.Retriever {
	config := &RetrieverConfig{
		SimilarityTopK: 10,
		EmbedModel:     vsi.embedModel,
	}

	for _, opt := range opts {
		opt(config)
	}

	return &VectorIndexRetriever{
		index:          vsi,
		similarityTopK: config.SimilarityTopK,
		embedModel:     config.EmbedModel,
		filters:        config.Filters,
	}
}

// AsQueryEngine returns a query engine for this index.
func (vsi *VectorStoreIndex) AsQueryEngine(opts ...QueryEngineOption) queryengine.QueryEngine {
	config := &QueryEngineConfig{
		ResponseMode: synthesizer.ResponseModeCompact,
	}
	config.SimilarityTopK = 10

	for _, opt := range opts {
		opt(config)
	}

	// Create retriever
	ret := vsi.AsRetriever(
		WithSimilarityTopK(config.SimilarityTopK),
	)

	// Create synthesizer
	var synth synthesizer.Synthesizer
	if config.Synthesizer != nil {
		synth = config.Synthesizer
	} else if config.LLM != nil {
		synth, _ = synthesizer.GetSynthesizer(config.ResponseMode, config.LLM)
	} else {
		// Use a default mock LLM for now - in production, this should be configured
		synth = synthesizer.NewSimpleSynthesizer(llm.NewMockLLM(""))
	}

	return queryengine.NewRetrieverQueryEngine(ret, synth)
}

// InsertNodes inserts nodes into the index.
func (vsi *VectorStoreIndex) InsertNodes(ctx context.Context, nodes []schema.Node) error {
	return vsi.addNodesToIndex(ctx, nodes)
}

// DeleteNodes removes nodes from the index.
func (vsi *VectorStoreIndex) DeleteNodes(ctx context.Context, nodeIDs []string) error {
	if vsi.vectorStore == nil {
		return fmt.Errorf("vector store not configured")
	}

	// Delete from vector store
	for _, nodeID := range nodeIDs {
		if err := vsi.vectorStore.Delete(ctx, nodeID); err != nil {
			return err
		}
		vsi.indexStruct.DeleteNode(nodeID)
	}

	// Update index store
	return vsi.storageContext.IndexStore.AddIndexStruct(ctx, vsi.indexStruct)
}

// RefreshDocuments refreshes the index with updated documents.
func (vsi *VectorStoreIndex) RefreshDocuments(ctx context.Context, documents []schema.Document) ([]bool, error) {
	refreshed := make([]bool, len(documents))

	for i, doc := range documents {
		// Check if document exists and has changed
		existingHash, err := vsi.storageContext.DocStore.GetDocumentHash(ctx, doc.ID)
		if err != nil || existingHash == "" {
			// Document doesn't exist, insert it
			node := schema.NewTextNode(doc.Text)
			node.Metadata = doc.Metadata
			if doc.ID != "" {
				node.ID = doc.ID
			}
			if err := vsi.InsertNodes(ctx, []schema.Node{*node}); err != nil {
				return refreshed, err
			}
			refreshed[i] = true
		} else if existingHash != doc.GetHash() {
			// Document has changed, update it
			if err := vsi.DeleteNodes(ctx, []string{doc.ID}); err != nil {
				return refreshed, err
			}
			node := schema.NewTextNode(doc.Text)
			node.Metadata = doc.Metadata
			if doc.ID != "" {
				node.ID = doc.ID
			}
			if err := vsi.InsertNodes(ctx, []schema.Node{*node}); err != nil {
				return refreshed, err
			}
			refreshed[i] = true
		}
	}

	return refreshed, nil
}

// VectorIndexRetriever retrieves nodes from a VectorStoreIndex.
type VectorIndexRetriever struct {
	index          *VectorStoreIndex
	similarityTopK int
	embedModel     EmbeddingModel
	filters        *schema.MetadataFilters
}

// Retrieve retrieves nodes for a query.
func (r *VectorIndexRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	if r.index.vectorStore == nil {
		return nil, fmt.Errorf("vector store not configured")
	}

	// Generate query embedding
	queryEmbedding, err := r.embedModel.GetQueryEmbedding(ctx, query.QueryString)
	if err != nil {
		return nil, err
	}

	// Build vector store query
	vsQuery := schema.NewVectorStoreQuery(queryEmbedding, r.similarityTopK)
	if r.filters != nil {
		vsQuery.Filters = r.filters
	}

	// Query vector store
	results, err := r.index.vectorStore.Query(ctx, *vsQuery)
	if err != nil {
		return nil, err
	}

	return results, nil
}

// Ensure VectorStoreIndex implements Index.
var _ Index = (*VectorStoreIndex)(nil)

// Ensure VectorIndexRetriever implements Retriever.
var _ retriever.Retriever = (*VectorIndexRetriever)(nil)
