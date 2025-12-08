package index

import (
	"context"
	"fmt"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/rag/queryengine"
	"github.com/aqua777/go-llamaindex/rag/retriever"
	"github.com/aqua777/go-llamaindex/rag/synthesizer"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/storage"
	"github.com/aqua777/go-llamaindex/storage/docstore"
	"github.com/aqua777/go-llamaindex/storage/indexstore"
)

// SummaryRetrieverMode specifies how the summary index retrieves nodes.
type SummaryRetrieverMode string

const (
	// SummaryRetrieverModeDefault returns all nodes.
	SummaryRetrieverModeDefault SummaryRetrieverMode = "default"
	// SummaryRetrieverModeEmbedding uses embeddings to select top-k nodes.
	SummaryRetrieverModeEmbedding SummaryRetrieverMode = "embedding"
	// SummaryRetrieverModeLLM uses an LLM to select relevant nodes.
	SummaryRetrieverModeLLM SummaryRetrieverMode = "llm"
)

// SummaryIndex is a simple index that stores nodes in a list.
// During query time, it iterates through all nodes to synthesize an answer.
type SummaryIndex struct {
	*BaseIndex
}

// SummaryIndexOption configures SummaryIndex creation.
type SummaryIndexOption func(*SummaryIndex)

// WithSummaryIndexStorageContext sets the storage context.
func WithSummaryIndexStorageContext(sc *storage.StorageContext) SummaryIndexOption {
	return func(si *SummaryIndex) {
		si.storageContext = sc
	}
}

// WithSummaryIndexEmbedModel sets the embedding model.
func WithSummaryIndexEmbedModel(model EmbeddingModel) SummaryIndexOption {
	return func(si *SummaryIndex) {
		si.embedModel = model
	}
}

// NewSummaryIndex creates a new SummaryIndex.
func NewSummaryIndex(ctx context.Context, nodes []schema.Node, opts ...SummaryIndexOption) (*SummaryIndex, error) {
	indexStruct := indexstore.NewListIndex()

	si := &SummaryIndex{
		BaseIndex: NewBaseIndex(indexStruct),
	}

	for _, opt := range opts {
		opt(si)
	}

	// Build index from nodes
	if len(nodes) > 0 {
		if err := si.buildIndexFromNodes(ctx, nodes); err != nil {
			return nil, err
		}
	}

	// Add index struct to store
	if err := si.storageContext.IndexStore.AddIndexStruct(ctx, indexStruct); err != nil {
		return nil, err
	}

	return si, nil
}

// NewSummaryIndexFromDocuments creates a SummaryIndex from documents.
func NewSummaryIndexFromDocuments(
	ctx context.Context,
	documents []schema.Document,
	opts ...SummaryIndexOption,
) (*SummaryIndex, error) {
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

	return NewSummaryIndex(ctx, nodes, opts...)
}

// buildIndexFromNodes builds the index from nodes.
func (si *SummaryIndex) buildIndexFromNodes(ctx context.Context, nodes []schema.Node) error {
	for _, node := range nodes {
		si.indexStruct.AddToList(node.ID)

		// Add to docstore
		if err := si.storageContext.DocStore.AddDocuments(ctx, []schema.BaseNode{&node}, true); err != nil {
			return err
		}
	}

	return nil
}

// AsRetriever returns a retriever for this index.
func (si *SummaryIndex) AsRetriever(opts ...RetrieverOption) retriever.Retriever {
	config := &RetrieverConfig{
		SimilarityTopK: 0, // 0 means return all nodes
		EmbedModel:     si.embedModel,
	}

	for _, opt := range opts {
		opt(config)
	}

	return &SummaryIndexRetriever{
		index:          si,
		mode:           SummaryRetrieverModeDefault,
		similarityTopK: config.SimilarityTopK,
		embedModel:     config.EmbedModel,
	}
}

// AsQueryEngine returns a query engine for this index.
func (si *SummaryIndex) AsQueryEngine(opts ...QueryEngineOption) queryengine.QueryEngine {
	config := &QueryEngineConfig{
		ResponseMode: synthesizer.ResponseModeTreeSummarize,
	}

	for _, opt := range opts {
		opt(config)
	}

	// Create retriever
	ret := si.AsRetriever()

	// Create synthesizer
	var synth synthesizer.Synthesizer
	if config.Synthesizer != nil {
		synth = config.Synthesizer
	} else if config.LLM != nil {
		synth, _ = synthesizer.GetSynthesizer(config.ResponseMode, config.LLM)
	} else {
		// Use a default mock LLM for now
		synth = synthesizer.NewSimpleSynthesizer(llm.NewMockLLM(""))
	}

	return queryengine.NewRetrieverQueryEngine(ret, synth)
}

// InsertNodes inserts nodes into the index.
func (si *SummaryIndex) InsertNodes(ctx context.Context, nodes []schema.Node) error {
	for _, node := range nodes {
		si.indexStruct.AddToList(node.ID)

		// Add to docstore
		if err := si.storageContext.DocStore.AddDocuments(ctx, []schema.BaseNode{&node}, true); err != nil {
			return err
		}
	}

	// Update index store
	return si.storageContext.IndexStore.AddIndexStruct(ctx, si.indexStruct)
}

// DeleteNodes removes nodes from the index.
func (si *SummaryIndex) DeleteNodes(ctx context.Context, nodeIDs []string) error {
	// Get current nodes
	currentNodeIDs := si.indexStruct.Nodes

	// Filter out deleted nodes
	var newNodeIDs []string
	deleteSet := make(map[string]bool)
	for _, id := range nodeIDs {
		deleteSet[id] = true
	}

	for _, id := range currentNodeIDs {
		if !deleteSet[id] {
			newNodeIDs = append(newNodeIDs, id)
		}
	}

	si.indexStruct.Nodes = newNodeIDs

	// Delete from docstore
	for _, nodeID := range nodeIDs {
		if err := si.storageContext.DocStore.DeleteDocument(ctx, nodeID, false); err != nil {
			// Continue even if delete fails
		}
	}

	// Update index store
	return si.storageContext.IndexStore.AddIndexStruct(ctx, si.indexStruct)
}

// RefreshDocuments refreshes the index with updated documents.
func (si *SummaryIndex) RefreshDocuments(ctx context.Context, documents []schema.Document) ([]bool, error) {
	refreshed := make([]bool, len(documents))

	for i, doc := range documents {
		// Check if document exists and has changed
		existingHash, err := si.storageContext.DocStore.GetDocumentHash(ctx, doc.ID)
		if err != nil || existingHash == "" {
			// Document doesn't exist, insert it
			node := schema.NewTextNode(doc.Text)
			node.Metadata = doc.Metadata
			if doc.ID != "" {
				node.ID = doc.ID
			}
			if err := si.InsertNodes(ctx, []schema.Node{*node}); err != nil {
				return refreshed, err
			}
			refreshed[i] = true
		} else if existingHash != doc.GetHash() {
			// Document has changed, update it
			if err := si.DeleteNodes(ctx, []string{doc.ID}); err != nil {
				return refreshed, err
			}
			node := schema.NewTextNode(doc.Text)
			node.Metadata = doc.Metadata
			if doc.ID != "" {
				node.ID = doc.ID
			}
			if err := si.InsertNodes(ctx, []schema.Node{*node}); err != nil {
				return refreshed, err
			}
			refreshed[i] = true
		}
	}

	return refreshed, nil
}

// GetNodes returns all nodes in the index.
func (si *SummaryIndex) GetNodes(ctx context.Context) ([]schema.Node, error) {
	nodeIDs := si.indexStruct.Nodes
	if len(nodeIDs) == 0 {
		return nil, nil
	}

	nodes, err := docstore.GetNodes(ctx, si.storageContext.DocStore, nodeIDs, false)
	if err != nil {
		return nil, err
	}

	// Convert BaseNode to Node
	result := make([]schema.Node, len(nodes))
	for i, n := range nodes {
		if node, ok := n.(*schema.Node); ok {
			result[i] = *node
		}
	}

	return result, nil
}

// SummaryIndexRetriever retrieves nodes from a SummaryIndex.
type SummaryIndexRetriever struct {
	index          *SummaryIndex
	mode           SummaryRetrieverMode
	similarityTopK int
	embedModel     EmbeddingModel
	llm            llm.LLM
}

// SummaryIndexRetrieverOption configures the retriever.
type SummaryIndexRetrieverOption func(*SummaryIndexRetriever)

// WithSummaryRetrieverMode sets the retriever mode.
func WithSummaryRetrieverMode(mode SummaryRetrieverMode) SummaryIndexRetrieverOption {
	return func(r *SummaryIndexRetriever) {
		r.mode = mode
	}
}

// WithSummaryRetrieverLLM sets the LLM for LLM mode.
func WithSummaryRetrieverLLM(l llm.LLM) SummaryIndexRetrieverOption {
	return func(r *SummaryIndexRetriever) {
		r.llm = l
	}
}

// Retrieve retrieves nodes for a query.
func (r *SummaryIndexRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	nodeIDs := r.index.indexStruct.Nodes
	if len(nodeIDs) == 0 {
		return nil, nil
	}

	// Get nodes from docstore
	nodes, err := docstore.GetNodes(ctx, r.index.storageContext.DocStore, nodeIDs, false)
	if err != nil {
		return nil, err
	}

	switch r.mode {
	case SummaryRetrieverModeDefault:
		return r.retrieveDefault(nodes)
	case SummaryRetrieverModeEmbedding:
		return r.retrieveEmbedding(ctx, query, nodes)
	case SummaryRetrieverModeLLM:
		return r.retrieveLLM(ctx, query, nodes)
	default:
		return r.retrieveDefault(nodes)
	}
}

// retrieveDefault returns all nodes.
func (r *SummaryIndexRetriever) retrieveDefault(nodes []schema.BaseNode) ([]schema.NodeWithScore, error) {
	results := make([]schema.NodeWithScore, len(nodes))
	for i, n := range nodes {
		if node, ok := n.(*schema.Node); ok {
			results[i] = schema.NodeWithScore{Node: *node, Score: 1.0}
		}
	}
	return results, nil
}

// retrieveEmbedding uses embeddings to select top-k nodes.
func (r *SummaryIndexRetriever) retrieveEmbedding(ctx context.Context, query schema.QueryBundle, nodes []schema.BaseNode) ([]schema.NodeWithScore, error) {
	if r.embedModel == nil {
		return nil, fmt.Errorf("embedding model not configured for embedding mode")
	}

	// Get query embedding
	queryEmbedding, err := r.embedModel.GetQueryEmbedding(ctx, query.QueryString)
	if err != nil {
		return nil, err
	}

	// Score each node
	type scoredNode struct {
		node  schema.Node
		score float64
	}
	var scored []scoredNode

	for _, n := range nodes {
		node, ok := n.(*schema.Node)
		if !ok {
			continue
		}

		// Get node embedding
		nodeEmbedding, err := r.embedModel.GetTextEmbedding(ctx, node.GetContent(schema.MetadataModeEmbed))
		if err != nil {
			continue
		}

		// Calculate cosine similarity
		score := cosineSimilarity(queryEmbedding, nodeEmbedding)
		scored = append(scored, scoredNode{node: *node, score: score})
	}

	// Sort by score descending
	for i := 0; i < len(scored)-1; i++ {
		for j := i + 1; j < len(scored); j++ {
			if scored[j].score > scored[i].score {
				scored[i], scored[j] = scored[j], scored[i]
			}
		}
	}

	// Return top-k
	k := r.similarityTopK
	if k <= 0 || k > len(scored) {
		k = len(scored)
	}

	results := make([]schema.NodeWithScore, k)
	for i := 0; i < k; i++ {
		results[i] = schema.NodeWithScore{Node: scored[i].node, Score: scored[i].score}
	}

	return results, nil
}

// retrieveLLM uses an LLM to select relevant nodes.
func (r *SummaryIndexRetriever) retrieveLLM(ctx context.Context, query schema.QueryBundle, nodes []schema.BaseNode) ([]schema.NodeWithScore, error) {
	if r.llm == nil {
		return nil, fmt.Errorf("LLM not configured for LLM mode")
	}

	// For simplicity, return all nodes - in production, this would use the LLM
	// to determine which nodes are relevant
	return r.retrieveDefault(nodes)
}

// cosineSimilarity calculates the cosine similarity between two vectors.
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (sqrt(normA) * sqrt(normB))
}

// sqrt calculates square root.
func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 100; i++ {
		z = (z + x/z) / 2
	}
	return z
}

// Ensure SummaryIndex implements Index.
var _ Index = (*SummaryIndex)(nil)

// Ensure SummaryIndexRetriever implements Retriever.
var _ retriever.Retriever = (*SummaryIndexRetriever)(nil)

// ListIndex is an alias for SummaryIndex (legacy name).
type ListIndex = SummaryIndex

// NewListIndex is an alias for NewSummaryIndex.
var NewListIndex = NewSummaryIndex
