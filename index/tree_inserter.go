package index

import (
	"context"
	"fmt"
	"regexp"
	"sort"
	"strconv"

	"github.com/aqua777/go-llamaindex/prompts"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/storage/docstore"
)

// TreeIndexInserter handles inserting nodes into a TreeIndex.
type TreeIndexInserter struct {
	index        *TreeIndex
	insertPrompt *prompts.PromptTemplate
}

// NewTreeIndexInserter creates a new TreeIndexInserter.
func NewTreeIndexInserter(index *TreeIndex) *TreeIndexInserter {
	return &TreeIndexInserter{
		index:        index,
		insertPrompt: index.insertPrompt,
	}
}

// Insert inserts nodes into the tree index.
func (ti *TreeIndexInserter) Insert(ctx context.Context, nodes []schema.Node) error {
	for _, node := range nodes {
		if err := ti.insertNode(ctx, node); err != nil {
			return err
		}
	}
	return nil
}

// insertNode inserts a single node into the tree.
func (ti *TreeIndexInserter) insertNode(ctx context.Context, node schema.Node) error {
	// Add node to docstore
	if err := ti.index.storageContext.DocStore.AddDocuments(ctx, []schema.BaseNode{&node}, true); err != nil {
		return err
	}

	// Get next index
	nextIndex := len(ti.index.indexStruct.AllNodes)
	ti.index.indexStruct.AllNodes[nextIndex] = node.ID
	ti.index.indexStruct.NodeIDToChildrenIDs[node.ID] = []string{}

	// If tree is not built, just add to root nodes
	if !ti.index.buildTree {
		ti.index.indexStruct.RootNodes[nextIndex] = node.ID
		return ti.updateIndexStore(ctx)
	}

	// If no root nodes exist, this becomes a root node
	if len(ti.index.indexStruct.RootNodes) == 0 {
		ti.index.indexStruct.RootNodes[nextIndex] = node.ID
		return ti.updateIndexStore(ctx)
	}

	// Find the best parent node to insert under
	parentNodeID, err := ti.findParentNode(ctx, node, ti.index.indexStruct.RootNodes)
	if err != nil {
		// If we can't find a parent, add as root
		ti.index.indexStruct.RootNodes[nextIndex] = node.ID
		return ti.updateIndexStore(ctx)
	}

	// Insert under parent and consolidate if needed
	return ti.insertUnderParentAndConsolidate(ctx, node, parentNodeID)
}

// findParentNode finds the best parent node for insertion.
func (ti *TreeIndexInserter) findParentNode(ctx context.Context, node schema.Node, candidateIDs map[int]string) (string, error) {
	if ti.index.llm == nil {
		// Without LLM, return first candidate
		for _, id := range candidateIDs {
			return id, nil
		}
		return "", fmt.Errorf("no candidate nodes")
	}

	// Get candidate nodes
	candidates, err := ti.getSortedNodes(ctx, candidateIDs)
	if err != nil {
		return "", err
	}

	if len(candidates) == 0 {
		return "", fmt.Errorf("no candidate nodes")
	}

	if len(candidates) == 1 {
		return candidates[0].ID, nil
	}

	// Build context list
	contextList := buildNumberedList(candidates)

	// Format prompt
	prompt := ti.insertPrompt.Format(map[string]string{
		"num_chunks":     strconv.Itoa(len(candidates)),
		"context_list":   contextList,
		"new_chunk_text": node.GetContent(schema.MetadataModeLLM),
	})

	// Get LLM response
	response, err := ti.index.llm.Complete(ctx, prompt)
	if err != nil {
		return candidates[0].ID, nil // Fallback to first candidate
	}

	// Extract number from response
	numbers := extractInsertNumbers(response)
	if len(numbers) > 0 && numbers[0] >= 1 && numbers[0] <= len(candidates) {
		selectedNode := candidates[numbers[0]-1]

		// Check if selected node has children - if so, recurse
		children := ti.index.GetChildrenDict(selectedNode.ID)
		if len(children) > 0 {
			return ti.findParentNode(ctx, node, children)
		}

		return selectedNode.ID, nil
	}

	return candidates[0].ID, nil
}

// insertUnderParentAndConsolidate inserts a node under a parent and consolidates if needed.
func (ti *TreeIndexInserter) insertUnderParentAndConsolidate(ctx context.Context, node schema.Node, parentNodeID string) error {
	// Add node as child of parent
	ti.index.indexStruct.NodeIDToChildrenIDs[parentNodeID] = append(
		ti.index.indexStruct.NodeIDToChildrenIDs[parentNodeID],
		node.ID,
	)

	// Check if parent has too many children
	children := ti.index.indexStruct.NodeIDToChildrenIDs[parentNodeID]
	if len(children) > ti.index.numChildren {
		// Need to consolidate - split children into groups and create intermediate nodes
		return ti.consolidateChildren(ctx, parentNodeID)
	}

	return ti.updateIndexStore(ctx)
}

// consolidateChildren splits children of a node into groups and creates intermediate summary nodes.
func (ti *TreeIndexInserter) consolidateChildren(ctx context.Context, parentNodeID string) error {
	children := ti.index.indexStruct.NodeIDToChildrenIDs[parentNodeID]
	if len(children) <= ti.index.numChildren {
		return nil
	}

	// Get child nodes
	childNodes, err := docstore.GetNodes(ctx, ti.index.storageContext.DocStore, children, false)
	if err != nil {
		return err
	}

	// Split into groups of numChildren
	var newChildren []string
	for i := 0; i < len(childNodes); i += ti.index.numChildren {
		end := i + ti.index.numChildren
		if end > len(childNodes) {
			end = len(childNodes)
		}
		chunk := childNodes[i:end]

		if len(chunk) == 1 {
			// Single node, keep as is
			if n, ok := chunk[0].(*schema.Node); ok {
				newChildren = append(newChildren, n.ID)
			}
			continue
		}

		// Combine text from children
		var combinedText string
		childIDs := make([]string, len(chunk))
		for j, bn := range chunk {
			if n, ok := bn.(*schema.Node); ok {
				combinedText += n.GetContent(schema.MetadataModeLLM) + "\n"
				childIDs[j] = n.ID
			}
		}

		// Generate summary for this group
		summary, err := ti.index.generateSummary(ctx, combinedText)
		if err != nil {
			return err
		}

		// Create intermediate node
		intermediateNode := schema.NewTextNode(summary)
		nextIndex := len(ti.index.indexStruct.AllNodes)
		ti.index.indexStruct.AllNodes[nextIndex] = intermediateNode.ID
		ti.index.indexStruct.NodeIDToChildrenIDs[intermediateNode.ID] = childIDs

		// Add to docstore
		if err := ti.index.storageContext.DocStore.AddDocuments(ctx, []schema.BaseNode{intermediateNode}, true); err != nil {
			return err
		}

		newChildren = append(newChildren, intermediateNode.ID)
	}

	// Update parent's children
	ti.index.indexStruct.NodeIDToChildrenIDs[parentNodeID] = newChildren

	// If parent is a root node and now has too many children, we need to create a new root
	isRoot := false
	for _, rootID := range ti.index.indexStruct.RootNodes {
		if rootID == parentNodeID {
			isRoot = true
			break
		}
	}

	if isRoot && len(newChildren) > ti.index.numChildren {
		// Recurse to consolidate again
		return ti.consolidateChildren(ctx, parentNodeID)
	}

	return ti.updateIndexStore(ctx)
}

// updateIndexStore updates the index store with current state.
func (ti *TreeIndexInserter) updateIndexStore(ctx context.Context) error {
	return ti.index.storageContext.IndexStore.AddIndexStruct(ctx, ti.index.indexStruct)
}

// getSortedNodes returns nodes sorted by their index.
func (ti *TreeIndexInserter) getSortedNodes(ctx context.Context, nodeIDs map[int]string) ([]schema.Node, error) {
	indices := make([]int, 0, len(nodeIDs))
	for idx := range nodeIDs {
		indices = append(indices, idx)
	}
	sort.Ints(indices)

	ids := make([]string, len(indices))
	for i, idx := range indices {
		ids[i] = nodeIDs[idx]
	}

	baseNodes, err := docstore.GetNodes(ctx, ti.index.storageContext.DocStore, ids, false)
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

// extractInsertNumbers extracts numbers from an insert response.
func extractInsertNumbers(response string) []int {
	re := regexp.MustCompile(`\d+`)
	matches := re.FindAllString(response, -1)

	var numbers []int
	for _, match := range matches {
		num, err := strconv.Atoi(match)
		if err == nil && num > 0 {
			numbers = append(numbers, num)
		}
	}
	return numbers
}
