package index

import (
	"context"
	"fmt"
	"sort"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/prompts"
	"github.com/aqua777/go-llamaindex/rag/queryengine"
	"github.com/aqua777/go-llamaindex/rag/retriever"
	"github.com/aqua777/go-llamaindex/rag/synthesizer"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/storage"
	"github.com/aqua777/go-llamaindex/storage/docstore"
	"github.com/aqua777/go-llamaindex/storage/indexstore"
)

// TreeRetrieverMode specifies how the tree index retrieves nodes.
type TreeRetrieverMode string

const (
	// TreeRetrieverModeSelectLeaf traverses the tree to select leaf nodes.
	TreeRetrieverModeSelectLeaf TreeRetrieverMode = "select_leaf"
	// TreeRetrieverModeSelectLeafEmbedding uses embeddings to select leaf nodes.
	TreeRetrieverModeSelectLeafEmbedding TreeRetrieverMode = "select_leaf_embedding"
	// TreeRetrieverModeAllLeaf returns all leaf nodes.
	TreeRetrieverModeAllLeaf TreeRetrieverMode = "all_leaf"
	// TreeRetrieverModeRoot returns root nodes.
	TreeRetrieverModeRoot TreeRetrieverMode = "root"
)

// Default prompts for tree index.
const (
	DefaultSummaryPromptTemplate = `Write a summary of the following. Try to use only the information provided. 
Try to include as many key details as possible.

{context_str}

SUMMARY:`

	DefaultInsertPromptTemplate = `Context information is below. It is provided in a numbered list (1 to {num_chunks}), 
where each item in the list corresponds to a summary.

---------------------
{context_list}
---------------------

Given the context information, here is a new piece of information: {new_chunk_text}

Answer with the number corresponding to the summary that should be updated. 
The answer should be the number corresponding to the summary that is most relevant to the question.
`

	DefaultQueryPromptTemplate = `Some choices are given below. It is provided in a numbered list (1 to {num_chunks}), 
where each item in the list corresponds to a summary.

---------------------
{context_list}
---------------------

Using only the choices above and not prior knowledge, return the choice that is most relevant to the question: '{query_str}'

Provide choice in the following format: 'ANSWER: <number>' and explain why this summary was selected over the others.
`
)

// TreeIndex is a tree-structured index where each node is a summary of its children.
// During index construction, the tree is built bottom-up until we have root nodes.
type TreeIndex struct {
	*BaseIndex
	llm             llm.LLM
	numChildren     int
	buildTree       bool
	summaryTemplate *prompts.PromptTemplate
	insertPrompt    *prompts.PromptTemplate
}

// TreeIndexOption configures TreeIndex creation.
type TreeIndexOption func(*TreeIndex)

// WithTreeIndexStorageContext sets the storage context.
func WithTreeIndexStorageContext(sc *storage.StorageContext) TreeIndexOption {
	return func(ti *TreeIndex) {
		ti.storageContext = sc
	}
}

// WithTreeIndexEmbedModel sets the embedding model.
func WithTreeIndexEmbedModel(model EmbeddingModel) TreeIndexOption {
	return func(ti *TreeIndex) {
		ti.embedModel = model
	}
}

// WithTreeIndexLLM sets the LLM for summarization.
func WithTreeIndexLLM(l llm.LLM) TreeIndexOption {
	return func(ti *TreeIndex) {
		ti.llm = l
	}
}

// WithTreeIndexNumChildren sets the number of children per node.
func WithTreeIndexNumChildren(n int) TreeIndexOption {
	return func(ti *TreeIndex) {
		if n >= 2 {
			ti.numChildren = n
		}
	}
}

// WithTreeIndexBuildTree sets whether to build the tree during construction.
func WithTreeIndexBuildTree(build bool) TreeIndexOption {
	return func(ti *TreeIndex) {
		ti.buildTree = build
	}
}

// WithTreeIndexSummaryTemplate sets the summary prompt template.
func WithTreeIndexSummaryTemplate(tmpl *prompts.PromptTemplate) TreeIndexOption {
	return func(ti *TreeIndex) {
		ti.summaryTemplate = tmpl
	}
}

// WithTreeIndexInsertPrompt sets the insert prompt template.
func WithTreeIndexInsertPrompt(tmpl *prompts.PromptTemplate) TreeIndexOption {
	return func(ti *TreeIndex) {
		ti.insertPrompt = tmpl
	}
}

// NewTreeIndex creates a new TreeIndex.
func NewTreeIndex(ctx context.Context, nodes []schema.Node, opts ...TreeIndexOption) (*TreeIndex, error) {
	indexStruct := indexstore.NewTreeIndex()

	ti := &TreeIndex{
		BaseIndex:       NewBaseIndex(indexStruct),
		numChildren:     10,
		buildTree:       true,
		summaryTemplate: prompts.NewPromptTemplate(DefaultSummaryPromptTemplate, prompts.PromptTypeSummary),
		insertPrompt:    prompts.NewPromptTemplate(DefaultInsertPromptTemplate, prompts.PromptTypeTreeInsert),
	}

	for _, opt := range opts {
		opt(ti)
	}

	// Build index from nodes
	if len(nodes) > 0 {
		if err := ti.buildIndexFromNodes(ctx, nodes); err != nil {
			return nil, err
		}
	}

	// Add index struct to store
	if err := ti.storageContext.IndexStore.AddIndexStruct(ctx, indexStruct); err != nil {
		return nil, err
	}

	return ti, nil
}

// NewTreeIndexFromDocuments creates a TreeIndex from documents.
func NewTreeIndexFromDocuments(
	ctx context.Context,
	documents []schema.Document,
	opts ...TreeIndexOption,
) (*TreeIndex, error) {
	var nodes []schema.Node
	for _, doc := range documents {
		node := schema.NewTextNode(doc.Text)
		node.Metadata = doc.Metadata
		if doc.ID != "" {
			node.ID = doc.ID
		}
		nodes = append(nodes, *node)
	}

	return NewTreeIndex(ctx, nodes, opts...)
}

// buildIndexFromNodes builds the tree index from nodes.
func (ti *TreeIndex) buildIndexFromNodes(ctx context.Context, nodes []schema.Node) error {
	// Insert all leaf nodes
	for i, node := range nodes {
		ti.indexStruct.AllNodes[i] = node.ID
		ti.indexStruct.NodeIDToChildrenIDs[node.ID] = []string{}

		// Add to docstore
		if err := ti.storageContext.DocStore.AddDocuments(ctx, []schema.BaseNode{&node}, true); err != nil {
			return err
		}
	}

	if !ti.buildTree {
		// If not building tree, all nodes are root nodes
		for i, node := range nodes {
			ti.indexStruct.RootNodes[i] = node.ID
		}
		return nil
	}

	// Build tree bottom-up
	return ti.buildTreeFromNodes(ctx, ti.indexStruct.AllNodes, 0)
}

// buildTreeFromNodes recursively builds the tree from current level nodes.
func (ti *TreeIndex) buildTreeFromNodes(ctx context.Context, curNodeIDs map[int]string, level int) error {
	if len(curNodeIDs) <= ti.numChildren {
		// Current nodes become root nodes
		ti.indexStruct.RootNodes = curNodeIDs
		return nil
	}

	// Get sorted node list
	sortedNodes, err := ti.getSortedNodes(ctx, curNodeIDs)
	if err != nil {
		return err
	}

	// Create parent nodes by summarizing groups of children
	newNodeDict := make(map[int]string)
	nextIndex := len(ti.indexStruct.AllNodes)

	for i := 0; i < len(sortedNodes); i += ti.numChildren {
		end := i + ti.numChildren
		if end > len(sortedNodes) {
			end = len(sortedNodes)
		}
		chunk := sortedNodes[i:end]

		// Combine text from children
		var combinedText string
		childIDs := make([]string, len(chunk))
		for j, node := range chunk {
			combinedText += node.GetContent(schema.MetadataModeLLM) + "\n"
			childIDs[j] = node.ID
		}

		// Generate summary
		summary, err := ti.generateSummary(ctx, combinedText)
		if err != nil {
			return err
		}

		// Create parent node
		parentNode := schema.NewTextNode(summary)
		ti.indexStruct.AllNodes[nextIndex] = parentNode.ID
		ti.indexStruct.NodeIDToChildrenIDs[parentNode.ID] = childIDs
		newNodeDict[nextIndex] = parentNode.ID
		nextIndex++

		// Add to docstore
		if err := ti.storageContext.DocStore.AddDocuments(ctx, []schema.BaseNode{parentNode}, true); err != nil {
			return err
		}
	}

	// Update root nodes
	ti.indexStruct.RootNodes = newNodeDict

	// Recursively build if we have more than numChildren nodes
	if len(newNodeDict) > ti.numChildren {
		return ti.buildTreeFromNodes(ctx, newNodeDict, level+1)
	}

	return nil
}

// generateSummary generates a summary using the LLM.
func (ti *TreeIndex) generateSummary(ctx context.Context, text string) (string, error) {
	if ti.llm == nil {
		// Return truncated text if no LLM
		if len(text) > 500 {
			return text[:500] + "...", nil
		}
		return text, nil
	}

	prompt := ti.summaryTemplate.Format(map[string]string{
		"context_str": text,
	})

	response, err := ti.llm.Complete(ctx, prompt)
	if err != nil {
		return "", err
	}

	return response, nil
}

// getSortedNodes returns nodes sorted by their index.
func (ti *TreeIndex) getSortedNodes(ctx context.Context, nodeIDs map[int]string) ([]schema.Node, error) {
	// Get sorted indices
	indices := make([]int, 0, len(nodeIDs))
	for idx := range nodeIDs {
		indices = append(indices, idx)
	}
	sort.Ints(indices)

	// Get nodes in order
	ids := make([]string, len(indices))
	for i, idx := range indices {
		ids[i] = nodeIDs[idx]
	}

	baseNodes, err := docstore.GetNodes(ctx, ti.storageContext.DocStore, ids, false)
	if err != nil {
		return nil, err
	}

	nodes := make([]schema.Node, len(baseNodes))
	for i, bn := range baseNodes {
		if n, ok := bn.(*schema.Node); ok {
			nodes[i] = *n
		}
	}

	return nodes, nil
}

// AsRetriever returns a retriever for this index.
func (ti *TreeIndex) AsRetriever(opts ...RetrieverOption) retriever.Retriever {
	config := &RetrieverConfig{
		SimilarityTopK: 1,
		EmbedModel:     ti.embedModel,
	}

	for _, opt := range opts {
		opt(config)
	}

	return NewTreeAllLeafRetriever(ti)
}

// AsRetrieverWithMode returns a retriever with the specified mode.
func (ti *TreeIndex) AsRetrieverWithMode(mode TreeRetrieverMode, opts ...TreeRetrieverOption) (retriever.Retriever, error) {
	// Validate mode
	if err := ti.validateRetrieverMode(mode); err != nil {
		return nil, err
	}

	switch mode {
	case TreeRetrieverModeAllLeaf:
		return NewTreeAllLeafRetriever(ti, opts...), nil
	case TreeRetrieverModeRoot:
		return NewTreeRootRetriever(ti, opts...), nil
	case TreeRetrieverModeSelectLeaf:
		return NewTreeSelectLeafRetriever(ti, opts...), nil
	case TreeRetrieverModeSelectLeafEmbedding:
		return NewTreeSelectLeafEmbeddingRetriever(ti, opts...), nil
	default:
		return nil, fmt.Errorf("unknown retriever mode: %s", mode)
	}
}

// validateRetrieverMode checks if the mode is valid for this index.
func (ti *TreeIndex) validateRetrieverMode(mode TreeRetrieverMode) error {
	// Only select modes require tree traversal
	requiresTree := mode == TreeRetrieverModeSelectLeaf ||
		mode == TreeRetrieverModeSelectLeafEmbedding

	if requiresTree && !ti.buildTree {
		return fmt.Errorf("index was constructed without building tree, but mode %s requires tree", mode)
	}
	return nil
}

// AsQueryEngine returns a query engine for this index.
func (ti *TreeIndex) AsQueryEngine(opts ...QueryEngineOption) queryengine.QueryEngine {
	config := &QueryEngineConfig{
		ResponseMode: synthesizer.ResponseModeTreeSummarize,
	}

	for _, opt := range opts {
		opt(config)
	}

	ret := ti.AsRetriever()

	var synth synthesizer.Synthesizer
	if config.Synthesizer != nil {
		synth = config.Synthesizer
	} else if config.LLM != nil {
		synth, _ = synthesizer.GetSynthesizer(config.ResponseMode, config.LLM)
	} else if ti.llm != nil {
		synth, _ = synthesizer.GetSynthesizer(config.ResponseMode, ti.llm)
	} else {
		synth = synthesizer.NewSimpleSynthesizer(llm.NewMockLLM(""))
	}

	return queryengine.NewRetrieverQueryEngine(ret, synth)
}

// InsertNodes inserts nodes into the index.
func (ti *TreeIndex) InsertNodes(ctx context.Context, nodes []schema.Node) error {
	inserter := NewTreeIndexInserter(ti)
	return inserter.Insert(ctx, nodes)
}

// DeleteNodes removes nodes from the index.
// Note: Delete is not fully implemented for tree index as it requires tree restructuring.
func (ti *TreeIndex) DeleteNodes(ctx context.Context, nodeIDs []string) error {
	return fmt.Errorf("delete not implemented for tree index")
}

// RefreshDocuments refreshes the index with updated documents.
func (ti *TreeIndex) RefreshDocuments(ctx context.Context, documents []schema.Document) ([]bool, error) {
	refreshed := make([]bool, len(documents))

	for i, doc := range documents {
		existingHash, err := ti.storageContext.DocStore.GetDocumentHash(ctx, doc.ID)
		if err != nil || existingHash == "" {
			// Document doesn't exist, insert it
			node := schema.NewTextNode(doc.Text)
			node.Metadata = doc.Metadata
			if doc.ID != "" {
				node.ID = doc.ID
			}
			if err := ti.InsertNodes(ctx, []schema.Node{*node}); err != nil {
				return refreshed, err
			}
			refreshed[i] = true
		}
		// Note: Updates require delete + insert, which is not fully supported
	}

	return refreshed, nil
}

// GetAllNodes returns all nodes in the index.
func (ti *TreeIndex) GetAllNodes(ctx context.Context) ([]schema.Node, error) {
	return ti.getSortedNodes(ctx, ti.indexStruct.AllNodes)
}

// GetRootNodes returns the root nodes of the tree.
func (ti *TreeIndex) GetRootNodes(ctx context.Context) ([]schema.Node, error) {
	return ti.getSortedNodes(ctx, ti.indexStruct.RootNodes)
}

// GetLeafNodes returns all leaf nodes (nodes with no children).
func (ti *TreeIndex) GetLeafNodes(ctx context.Context) ([]schema.Node, error) {
	leafIDs := make(map[int]string)
	idx := 0
	for _, nodeID := range ti.indexStruct.AllNodes {
		children := ti.indexStruct.NodeIDToChildrenIDs[nodeID]
		if len(children) == 0 {
			leafIDs[idx] = nodeID
			idx++
		}
	}
	return ti.getSortedNodes(ctx, leafIDs)
}

// GetChildren returns the children of a node.
func (ti *TreeIndex) GetChildren(nodeID string) []string {
	if nodeID == "" {
		// Return root node IDs
		ids := make([]string, 0, len(ti.indexStruct.RootNodes))
		for _, id := range ti.indexStruct.RootNodes {
			ids = append(ids, id)
		}
		return ids
	}
	return ti.indexStruct.NodeIDToChildrenIDs[nodeID]
}

// GetChildrenDict returns children as a map of index to node ID.
func (ti *TreeIndex) GetChildrenDict(nodeID string) map[int]string {
	if nodeID == "" {
		return ti.indexStruct.RootNodes
	}

	children := ti.indexStruct.NodeIDToChildrenIDs[nodeID]
	result := make(map[int]string)
	for i, childID := range children {
		// Find the index of this child in AllNodes
		for idx, id := range ti.indexStruct.AllNodes {
			if id == childID {
				result[idx] = childID
				break
			}
		}
		// Fallback to sequential index if not found
		if _, ok := result[i]; !ok {
			result[i] = childID
		}
	}
	return result
}

// LLM returns the LLM used by this index.
func (ti *TreeIndex) LLM() llm.LLM {
	return ti.llm
}

// NumChildren returns the number of children per node.
func (ti *TreeIndex) NumChildren() int {
	return ti.numChildren
}

// Ensure TreeIndex implements Index.
var _ Index = (*TreeIndex)(nil)
