package index

import (
	"context"
	"fmt"
	"regexp"
	"sort"
	"strconv"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/prompts"
	"github.com/aqua777/go-llamaindex/rag/retriever"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/storage/docstore"
)

// TreeRetrieverOption configures tree retrievers.
type TreeRetrieverOption func(interface{})

// WithTreeRetrieverLLM sets the LLM for tree retrievers.
func WithTreeRetrieverLLM(l llm.LLM) TreeRetrieverOption {
	return func(r interface{}) {
		switch ret := r.(type) {
		case *TreeSelectLeafRetriever:
			ret.llm = l
		}
	}
}

// WithTreeRetrieverEmbedModel sets the embedding model.
func WithTreeRetrieverEmbedModel(model EmbeddingModel) TreeRetrieverOption {
	return func(r interface{}) {
		switch ret := r.(type) {
		case *TreeSelectLeafEmbeddingRetriever:
			ret.embedModel = model
		}
	}
}

// WithTreeRetrieverChildBranchFactor sets the child branch factor.
func WithTreeRetrieverChildBranchFactor(factor int) TreeRetrieverOption {
	return func(r interface{}) {
		switch ret := r.(type) {
		case *TreeSelectLeafRetriever:
			ret.childBranchFactor = factor
		case *TreeSelectLeafEmbeddingRetriever:
			ret.childBranchFactor = factor
		}
	}
}

// TreeAllLeafRetriever returns all leaf nodes from the tree.
type TreeAllLeafRetriever struct {
	index *TreeIndex
}

// NewTreeAllLeafRetriever creates a new TreeAllLeafRetriever.
func NewTreeAllLeafRetriever(index *TreeIndex, opts ...TreeRetrieverOption) *TreeAllLeafRetriever {
	r := &TreeAllLeafRetriever{
		index: index,
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

// Retrieve returns all leaf nodes.
func (r *TreeAllLeafRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	leafNodes, err := r.index.GetLeafNodes(ctx)
	if err != nil {
		return nil, err
	}

	results := make([]schema.NodeWithScore, len(leafNodes))
	for i, node := range leafNodes {
		results[i] = schema.NodeWithScore{Node: node, Score: 1.0}
	}
	return results, nil
}

// Ensure TreeAllLeafRetriever implements Retriever.
var _ retriever.Retriever = (*TreeAllLeafRetriever)(nil)

// TreeRootRetriever returns root nodes from the tree.
type TreeRootRetriever struct {
	index *TreeIndex
}

// NewTreeRootRetriever creates a new TreeRootRetriever.
func NewTreeRootRetriever(index *TreeIndex, opts ...TreeRetrieverOption) *TreeRootRetriever {
	r := &TreeRootRetriever{
		index: index,
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

// Retrieve returns root nodes.
func (r *TreeRootRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	rootNodes, err := r.index.GetRootNodes(ctx)
	if err != nil {
		return nil, err
	}

	results := make([]schema.NodeWithScore, len(rootNodes))
	for i, node := range rootNodes {
		results[i] = schema.NodeWithScore{Node: node, Score: 1.0}
	}
	return results, nil
}

// Ensure TreeRootRetriever implements Retriever.
var _ retriever.Retriever = (*TreeRootRetriever)(nil)

// TreeSelectLeafRetriever traverses the tree using LLM to select relevant leaf nodes.
type TreeSelectLeafRetriever struct {
	index             *TreeIndex
	llm               llm.LLM
	childBranchFactor int
	queryTemplate     *prompts.PromptTemplate
}

// Default query prompt for tree select.
const DefaultTreeQueryPrompt = `Some choices are given below. It is provided in a numbered list (1 to {num_chunks}), 
where each item in the list corresponds to a summary.

---------------------
{context_list}
---------------------

Using only the choices above and not prior knowledge, return the choice that is most relevant to the question: '{query_str}'

Provide choice in the following format: 'ANSWER: <number>' and explain why this summary was selected over the others.
`

// NewTreeSelectLeafRetriever creates a new TreeSelectLeafRetriever.
func NewTreeSelectLeafRetriever(index *TreeIndex, opts ...TreeRetrieverOption) *TreeSelectLeafRetriever {
	r := &TreeSelectLeafRetriever{
		index:             index,
		llm:               index.LLM(),
		childBranchFactor: 1,
		queryTemplate:     prompts.NewPromptTemplate(DefaultTreeQueryPrompt, prompts.PromptTypeTreeSelect),
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

// Retrieve traverses the tree to find relevant leaf nodes.
func (r *TreeSelectLeafRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	if r.llm == nil {
		return nil, fmt.Errorf("LLM not configured for select leaf retriever")
	}

	nodes, err := r.retrieveLevel(ctx, r.index.indexStruct.RootNodes, query, 0)
	if err != nil {
		return nil, err
	}

	results := make([]schema.NodeWithScore, len(nodes))
	for i, node := range nodes {
		results[i] = schema.NodeWithScore{Node: node, Score: 1.0}
	}
	return results, nil
}

// retrieveLevel recursively retrieves nodes at a given level.
func (r *TreeSelectLeafRetriever) retrieveLevel(ctx context.Context, nodeIDs map[int]string, query schema.QueryBundle, level int) ([]schema.Node, error) {
	// Get sorted nodes
	nodes, err := r.getSortedNodes(ctx, nodeIDs)
	if err != nil {
		return nil, err
	}

	if len(nodes) == 0 {
		return nil, nil
	}

	// If we have few enough nodes, select from them
	var selectedNodes []schema.Node
	if len(nodes) > r.childBranchFactor {
		selectedNodes, err = r.selectNodes(ctx, nodes, query, level)
		if err != nil {
			return nil, err
		}
	} else {
		selectedNodes = nodes
	}

	// Check if selected nodes have children
	var childrenIDs map[int]string
	for _, node := range selectedNodes {
		children := r.index.GetChildrenDict(node.ID)
		if childrenIDs == nil {
			childrenIDs = make(map[int]string)
		}
		for k, v := range children {
			childrenIDs[k] = v
		}
	}

	if len(childrenIDs) == 0 {
		// Leaf level
		return selectedNodes, nil
	}

	// Recurse to children
	return r.retrieveLevel(ctx, childrenIDs, query, level+1)
}

// selectNodes uses the LLM to select relevant nodes.
func (r *TreeSelectLeafRetriever) selectNodes(ctx context.Context, nodes []schema.Node, query schema.QueryBundle, level int) ([]schema.Node, error) {
	// Build numbered context list
	contextList := buildNumberedList(nodes)

	// Format prompt
	prompt := r.queryTemplate.Format(map[string]string{
		"num_chunks":   strconv.Itoa(len(nodes)),
		"context_list": contextList,
		"query_str":    query.QueryString,
	})

	// Get LLM response
	response, err := r.llm.Complete(ctx, prompt)
	if err != nil {
		return nil, err
	}

	// Extract numbers from response
	numbers := extractNumbers(response, r.childBranchFactor)
	if len(numbers) == 0 {
		// If no numbers found, return first node
		if len(nodes) > 0 {
			return []schema.Node{nodes[0]}, nil
		}
		return nil, nil
	}

	// Select nodes by number (1-indexed)
	var selected []schema.Node
	for _, num := range numbers {
		if num >= 1 && num <= len(nodes) {
			selected = append(selected, nodes[num-1])
		}
	}

	if len(selected) == 0 && len(nodes) > 0 {
		return []schema.Node{nodes[0]}, nil
	}

	return selected, nil
}

// getSortedNodes returns nodes sorted by their index.
func (r *TreeSelectLeafRetriever) getSortedNodes(ctx context.Context, nodeIDs map[int]string) ([]schema.Node, error) {
	indices := make([]int, 0, len(nodeIDs))
	for idx := range nodeIDs {
		indices = append(indices, idx)
	}
	sort.Ints(indices)

	ids := make([]string, len(indices))
	for i, idx := range indices {
		ids[i] = nodeIDs[idx]
	}

	baseNodes, err := docstore.GetNodes(ctx, r.index.storageContext.DocStore, ids, false)
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

// Ensure TreeSelectLeafRetriever implements Retriever.
var _ retriever.Retriever = (*TreeSelectLeafRetriever)(nil)

// TreeSelectLeafEmbeddingRetriever uses embeddings to select leaf nodes.
type TreeSelectLeafEmbeddingRetriever struct {
	index             *TreeIndex
	embedModel        EmbeddingModel
	childBranchFactor int
}

// NewTreeSelectLeafEmbeddingRetriever creates a new TreeSelectLeafEmbeddingRetriever.
func NewTreeSelectLeafEmbeddingRetriever(index *TreeIndex, opts ...TreeRetrieverOption) *TreeSelectLeafEmbeddingRetriever {
	r := &TreeSelectLeafEmbeddingRetriever{
		index:             index,
		embedModel:        index.EmbedModel(),
		childBranchFactor: 1,
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

// Retrieve uses embeddings to traverse the tree and find relevant leaf nodes.
func (r *TreeSelectLeafEmbeddingRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	if r.embedModel == nil {
		return nil, fmt.Errorf("embedding model not configured for embedding retriever")
	}

	// Get query embedding
	queryEmbedding, err := r.embedModel.GetQueryEmbedding(ctx, query.QueryString)
	if err != nil {
		return nil, err
	}

	nodes, err := r.retrieveLevel(ctx, r.index.indexStruct.RootNodes, queryEmbedding, 0)
	if err != nil {
		return nil, err
	}

	results := make([]schema.NodeWithScore, len(nodes))
	for i, node := range nodes {
		results[i] = schema.NodeWithScore{Node: node.node, Score: node.score}
	}
	return results, nil
}

type scoredNode struct {
	node  schema.Node
	score float64
}

// retrieveLevel recursively retrieves nodes using embeddings.
func (r *TreeSelectLeafEmbeddingRetriever) retrieveLevel(ctx context.Context, nodeIDs map[int]string, queryEmbedding []float64, level int) ([]scoredNode, error) {
	nodes, err := r.getSortedNodes(ctx, nodeIDs)
	if err != nil {
		return nil, err
	}

	if len(nodes) == 0 {
		return nil, nil
	}

	// Score nodes by embedding similarity
	scored := make([]scoredNode, len(nodes))
	for i, node := range nodes {
		nodeEmbedding, err := r.embedModel.GetTextEmbedding(ctx, node.GetContent(schema.MetadataModeEmbed))
		if err != nil {
			scored[i] = scoredNode{node: node, score: 0}
			continue
		}
		score := cosineSimilarity(queryEmbedding, nodeEmbedding)
		scored[i] = scoredNode{node: node, score: score}
	}

	// Sort by score descending
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	// Select top nodes
	topK := r.childBranchFactor
	if topK > len(scored) {
		topK = len(scored)
	}
	selectedNodes := scored[:topK]

	// Check if selected nodes have children
	var childrenIDs map[int]string
	for _, sn := range selectedNodes {
		children := r.index.GetChildrenDict(sn.node.ID)
		if childrenIDs == nil {
			childrenIDs = make(map[int]string)
		}
		for k, v := range children {
			childrenIDs[k] = v
		}
	}

	if len(childrenIDs) == 0 {
		// Leaf level
		return selectedNodes, nil
	}

	// Recurse to children
	return r.retrieveLevel(ctx, childrenIDs, queryEmbedding, level+1)
}

// getSortedNodes returns nodes sorted by their index.
func (r *TreeSelectLeafEmbeddingRetriever) getSortedNodes(ctx context.Context, nodeIDs map[int]string) ([]schema.Node, error) {
	indices := make([]int, 0, len(nodeIDs))
	for idx := range nodeIDs {
		indices = append(indices, idx)
	}
	sort.Ints(indices)

	ids := make([]string, len(indices))
	for i, idx := range indices {
		ids[i] = nodeIDs[idx]
	}

	baseNodes, err := docstore.GetNodes(ctx, r.index.storageContext.DocStore, ids, false)
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

// Ensure TreeSelectLeafEmbeddingRetriever implements Retriever.
var _ retriever.Retriever = (*TreeSelectLeafEmbeddingRetriever)(nil)

// Helper functions

// buildNumberedList creates a numbered list of node contents.
func buildNumberedList(nodes []schema.Node) string {
	var result string
	for i, node := range nodes {
		content := node.GetContent(schema.MetadataModeLLM)
		result += fmt.Sprintf("(%d) %s\n", i+1, content)
	}
	return result
}

// extractNumbers extracts up to n numbers from a response string.
func extractNumbers(response string, n int) []int {
	// Look for patterns like "ANSWER: 1" or just numbers
	re := regexp.MustCompile(`\d+`)
	matches := re.FindAllString(response, -1)

	var numbers []int
	seen := make(map[int]bool)
	for _, match := range matches {
		num, err := strconv.Atoi(match)
		if err == nil && num > 0 && !seen[num] {
			numbers = append(numbers, num)
			seen[num] = true
			if len(numbers) >= n {
				break
			}
		}
	}
	return numbers
}
