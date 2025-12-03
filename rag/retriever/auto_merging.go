package retriever

import (
	"context"
	"sort"

	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/storage"
)

// AutoMergingRetriever merges child nodes into parent nodes when enough children are retrieved.
// This is useful for hierarchical document structures where you want to return
// larger context when multiple related chunks are retrieved.
type AutoMergingRetriever struct {
	*BaseRetriever
	// VectorRetriever is the underlying retriever for initial retrieval.
	VectorRetriever Retriever
	// StorageContext provides access to the document store for parent nodes.
	StorageContext *storage.StorageContext
	// SimpleRatioThresh is the threshold ratio of children to trigger merging.
	// If more than this ratio of a parent's children are retrieved, merge into parent.
	SimpleRatioThresh float64
}

// AutoMergingRetrieverOption is a functional option for AutoMergingRetriever.
type AutoMergingRetrieverOption func(*AutoMergingRetriever)

// WithSimpleRatioThresh sets the merge threshold ratio.
func WithSimpleRatioThresh(thresh float64) AutoMergingRetrieverOption {
	return func(amr *AutoMergingRetriever) {
		amr.SimpleRatioThresh = thresh
	}
}

// NewAutoMergingRetriever creates a new AutoMergingRetriever.
func NewAutoMergingRetriever(
	vectorRetriever Retriever,
	storageContext *storage.StorageContext,
	opts ...AutoMergingRetrieverOption,
) *AutoMergingRetriever {
	amr := &AutoMergingRetriever{
		BaseRetriever:     NewBaseRetriever(),
		VectorRetriever:   vectorRetriever,
		StorageContext:    storageContext,
		SimpleRatioThresh: 0.5,
	}

	for _, opt := range opts {
		opt(amr)
	}

	return amr
}

// Retrieve retrieves nodes and attempts to merge them into parent nodes.
func (amr *AutoMergingRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	// Get initial nodes from vector retriever
	initialNodes, err := amr.VectorRetriever.Retrieve(ctx, query)
	if err != nil {
		return nil, err
	}

	// Try merging until no more changes
	curNodes := initialNodes
	for {
		mergedNodes, changed := amr.tryMerging(ctx, curNodes)
		if !changed {
			break
		}
		curNodes = mergedNodes
	}

	// Sort by score descending
	sort.Slice(curNodes, func(i, j int) bool {
		return curNodes[i].Score > curNodes[j].Score
	})

	return curNodes, nil
}

// tryMerging attempts to merge nodes into their parents.
func (amr *AutoMergingRetriever) tryMerging(ctx context.Context, nodes []schema.NodeWithScore) ([]schema.NodeWithScore, bool) {
	// First try filling in gaps
	filledNodes, changed1 := amr.fillInNodes(ctx, nodes)
	// Then try merging into parents
	mergedNodes, changed2 := amr.getParentsAndMerge(ctx, filledNodes)
	return mergedNodes, changed1 || changed2
}

// fillInNodes fills in gaps between consecutive nodes.
func (amr *AutoMergingRetriever) fillInNodes(ctx context.Context, nodes []schema.NodeWithScore) ([]schema.NodeWithScore, bool) {
	if len(nodes) < 2 {
		return nodes, false
	}

	var newNodes []schema.NodeWithScore
	changed := false

	for i, node := range nodes {
		newNodes = append(newNodes, node)

		if i >= len(nodes)-1 {
			continue
		}

		// Check if there's a gap that can be filled
		curNode := node.Node
		nextNode := nodes[i+1].Node

		// Get next relationship from current node
		nextRel := curNode.Relationships.GetNext()
		if nextRel == nil {
			continue
		}

		// Get prev relationship from next node
		prevRel := nextNode.Relationships.GetPrevious()
		if prevRel == nil {
			continue
		}

		// If the next of current equals the prev of next, there's a node in between
		if nextRel.NodeID == prevRel.NodeID && nextRel.NodeID != nextNode.ID {
			// Fetch the middle node from docstore
			if amr.StorageContext != nil && amr.StorageContext.DocStore != nil {
				middleNode, err := amr.StorageContext.DocStore.GetDocument(ctx, nextRel.NodeID, false)
				if err == nil && middleNode != nil {
					// Calculate average score
					avgScore := (node.Score + nodes[i+1].Score) / 2
					if nodePtr, ok := middleNode.(*schema.Node); ok {
						newNodes = append(newNodes, schema.NodeWithScore{
							Node:  *nodePtr,
							Score: avgScore,
						})
						changed = true
					}
				}
			}
		}
	}

	return newNodes, changed
}

// getParentsAndMerge merges child nodes into parent nodes when threshold is met.
func (amr *AutoMergingRetriever) getParentsAndMerge(ctx context.Context, nodes []schema.NodeWithScore) ([]schema.NodeWithScore, bool) {
	if amr.StorageContext == nil || amr.StorageContext.DocStore == nil {
		return nodes, false
	}

	// Track parent nodes and their retrieved children
	parentChildren := make(map[string][]schema.NodeWithScore)
	parentNodes := make(map[string]schema.BaseNode)

	for _, node := range nodes {
		// Get parent relationship
		parentRel := node.Node.Relationships.GetParent()
		if parentRel == nil {
			continue
		}

		parentID := parentRel.NodeID

		// Fetch parent node if not cached
		if _, exists := parentNodes[parentID]; !exists {
			parentNode, err := amr.StorageContext.DocStore.GetDocument(ctx, parentID, false)
			if err != nil || parentNode == nil {
				continue
			}
			parentNodes[parentID] = parentNode
		}

		parentChildren[parentID] = append(parentChildren[parentID], node)
	}

	// Determine which nodes to merge
	nodeIDsToDelete := make(map[string]bool)
	nodesToAdd := make(map[string]schema.NodeWithScore)

	for parentID, parent := range parentNodes {
		children := parentChildren[parentID]

		// Get total number of children from parent
		childRels := parent.GetRelationships().GetChildren()
		totalChildren := len(childRels)
		if totalChildren == 0 {
			totalChildren = 1
		}

		// Calculate ratio
		ratio := float64(len(children)) / float64(totalChildren)

		// If ratio exceeds threshold, merge
		if ratio > amr.SimpleRatioThresh {
			// Mark children for deletion
			for _, child := range children {
				nodeIDsToDelete[child.Node.ID] = true
			}

			// Calculate average score
			avgScore := 0.0
			for _, child := range children {
				avgScore += child.Score
			}
			avgScore /= float64(len(children))

			// Add parent node
			if parentNode, ok := parent.(*schema.Node); ok {
				nodesToAdd[parentID] = schema.NodeWithScore{
					Node:  *parentNode,
					Score: avgScore,
				}
			}
		}
	}

	// Build new node list
	var newNodes []schema.NodeWithScore
	for _, node := range nodes {
		if !nodeIDsToDelete[node.Node.ID] {
			newNodes = append(newNodes, node)
		}
	}

	// Add parent nodes
	for _, parentNode := range nodesToAdd {
		newNodes = append(newNodes, parentNode)
	}

	changed := len(nodeIDsToDelete) > 0
	return newNodes, changed
}

// Ensure AutoMergingRetriever implements Retriever.
var _ Retriever = (*AutoMergingRetriever)(nil)
