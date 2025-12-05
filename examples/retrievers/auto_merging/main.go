// Package main demonstrates the AutoMergingRetriever for hierarchical document retrieval.
// This example corresponds to Python's retrievers/auto_merging_retriever.ipynb
package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/rag/retriever"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/storage"
	"github.com/aqua777/go-llamaindex/storage/docstore"
)

func main() {
	ctx := context.Background()

	fmt.Println("=== Auto-Merging Retriever Demo ===")
	fmt.Println()
	fmt.Println("The AutoMergingRetriever merges child nodes into parent nodes when")
	fmt.Println("enough children are retrieved, providing broader context automatically.")
	fmt.Println()

	separator := strings.Repeat("=", 70)

	// 1. Create hierarchical document structure
	fmt.Println(separator)
	fmt.Println("=== Creating Hierarchical Document Structure ===")
	fmt.Println(separator)
	fmt.Println()

	// Create parent and child nodes with relationships
	parentNodes, childNodes := createHierarchicalDocuments()

	fmt.Printf("Created %d parent nodes and %d child nodes\n", len(parentNodes), len(childNodes))
	fmt.Println()

	// Display hierarchy
	fmt.Println("Document Hierarchy:")
	for _, parent := range parentNodes {
		fmt.Printf("  📁 %s: %s\n", parent.ID, truncate(parent.Text, 50))
		childRels := parent.Relationships.GetChildren()
		for _, childRel := range childRels {
			for _, child := range childNodes {
				if child.ID == childRel.NodeID {
					fmt.Printf("     └── 📄 %s: %s\n", child.ID, truncate(child.Text, 40))
				}
			}
		}
	}
	fmt.Println()

	// 2. Create storage context with parent nodes
	fmt.Println(separator)
	fmt.Println("=== Setting Up Storage Context ===")
	fmt.Println(separator)
	fmt.Println()

	storageCtx := createStorageContext(parentNodes)
	fmt.Println("Storage context created with parent nodes in document store")
	fmt.Println()

	// 3. Create mock vector retriever that returns child nodes
	fmt.Println(separator)
	fmt.Println("=== Creating Vector Retriever ===")
	fmt.Println(separator)
	fmt.Println()

	vectorRetriever := NewMockVectorRetriever(childNodes)
	fmt.Println("Mock vector retriever created with child nodes")
	fmt.Println()

	// 4. Create AutoMergingRetriever with different thresholds
	fmt.Println(separator)
	fmt.Println("=== Testing Different Merge Thresholds ===")
	fmt.Println(separator)
	fmt.Println()

	query := schema.QueryBundle{QueryString: "machine learning applications"}

	// Test with 0.3 threshold (more aggressive merging)
	fmt.Println("--- Threshold: 0.3 (Aggressive Merging) ---")
	fmt.Println("Merges when >30% of children are retrieved")
	fmt.Println()

	amr03 := retriever.NewAutoMergingRetriever(
		vectorRetriever,
		storageCtx,
		retriever.WithSimpleRatioThresh(0.3),
	)

	results03, err := amr03.Retrieve(ctx, query)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		printResults("Threshold 0.3", results03)
	}

	// Test with 0.5 threshold (default)
	fmt.Println("--- Threshold: 0.5 (Default) ---")
	fmt.Println("Merges when >50% of children are retrieved")
	fmt.Println()

	amr05 := retriever.NewAutoMergingRetriever(
		vectorRetriever,
		storageCtx,
		retriever.WithSimpleRatioThresh(0.5),
	)

	results05, err := amr05.Retrieve(ctx, query)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		printResults("Threshold 0.5", results05)
	}

	// Test with 0.8 threshold (conservative merging)
	fmt.Println("--- Threshold: 0.8 (Conservative Merging) ---")
	fmt.Println("Merges when >80% of children are retrieved")
	fmt.Println()

	amr08 := retriever.NewAutoMergingRetriever(
		vectorRetriever,
		storageCtx,
		retriever.WithSimpleRatioThresh(0.8),
	)

	results08, err := amr08.Retrieve(ctx, query)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		printResults("Threshold 0.8", results08)
	}

	// 5. Explain the merging behavior
	fmt.Println(separator)
	fmt.Println("=== How Auto-Merging Works ===")
	fmt.Println(separator)
	fmt.Println()

	fmt.Println("1. Initial Retrieval:")
	fmt.Println("   - Vector retriever returns relevant child nodes")
	fmt.Println()
	fmt.Println("2. Gap Filling:")
	fmt.Println("   - If consecutive nodes have a gap, the middle node is fetched")
	fmt.Println("   - Uses next/previous relationships to detect gaps")
	fmt.Println()
	fmt.Println("3. Parent Merging:")
	fmt.Println("   - Groups children by parent node")
	fmt.Println("   - If (retrieved children / total children) > threshold:")
	fmt.Println("     - Remove child nodes from results")
	fmt.Println("     - Add parent node with averaged score")
	fmt.Println()
	fmt.Println("4. Iteration:")
	fmt.Println("   - Repeat until no more changes")
	fmt.Println("   - Allows multi-level merging in deep hierarchies")
	fmt.Println()

	// 6. Use cases
	fmt.Println(separator)
	fmt.Println("=== Use Cases ===")
	fmt.Println(separator)
	fmt.Println()

	fmt.Println("1. Book/Document Structure:")
	fmt.Println("   Book → Chapter → Section → Paragraph")
	fmt.Println("   When multiple paragraphs from same section match, return the section")
	fmt.Println()
	fmt.Println("2. Code Documentation:")
	fmt.Println("   Module → Class → Method → Docstring")
	fmt.Println("   When multiple methods match, return the class documentation")
	fmt.Println()
	fmt.Println("3. Legal Documents:")
	fmt.Println("   Contract → Article → Clause → Sub-clause")
	fmt.Println("   When multiple sub-clauses match, return the full clause")
	fmt.Println()

	fmt.Println("=== Auto-Merging Retriever Demo Complete ===")
}

// createHierarchicalDocuments creates a hierarchical document structure.
func createHierarchicalDocuments() ([]schema.Node, []schema.Node) {
	// Parent nodes (chapters)
	parent1 := schema.Node{
		ID:   "chapter-1",
		Text: "Chapter 1: Introduction to Machine Learning. This chapter covers the fundamentals of ML including supervised learning, unsupervised learning, and reinforcement learning approaches.",
		Type: schema.ObjectTypeText,
		Metadata: map[string]interface{}{
			"type":  "chapter",
			"title": "Introduction to Machine Learning",
		},
	}

	parent2 := schema.Node{
		ID:   "chapter-2",
		Text: "Chapter 2: Deep Learning Architectures. This chapter explores neural network architectures including CNNs, RNNs, Transformers, and their applications in various domains.",
		Type: schema.ObjectTypeText,
		Metadata: map[string]interface{}{
			"type":  "chapter",
			"title": "Deep Learning Architectures",
		},
	}

	// Child nodes (sections) for Chapter 1
	child1_1 := schema.Node{
		ID:   "section-1-1",
		Text: "Supervised learning uses labeled data to train models. Common algorithms include linear regression, decision trees, and support vector machines.",
		Type: schema.ObjectTypeText,
		Metadata: map[string]interface{}{
			"type":   "section",
			"parent": "chapter-1",
		},
	}

	child1_2 := schema.Node{
		ID:   "section-1-2",
		Text: "Unsupervised learning finds patterns in unlabeled data. Clustering and dimensionality reduction are key techniques in this area.",
		Type: schema.ObjectTypeText,
		Metadata: map[string]interface{}{
			"type":   "section",
			"parent": "chapter-1",
		},
	}

	child1_3 := schema.Node{
		ID:   "section-1-3",
		Text: "Reinforcement learning trains agents through rewards and penalties. Applications include game playing, robotics, and autonomous systems.",
		Type: schema.ObjectTypeText,
		Metadata: map[string]interface{}{
			"type":   "section",
			"parent": "chapter-1",
		},
	}

	// Child nodes (sections) for Chapter 2
	child2_1 := schema.Node{
		ID:   "section-2-1",
		Text: "Convolutional Neural Networks (CNNs) excel at image processing tasks. They use convolutional layers to detect spatial patterns.",
		Type: schema.ObjectTypeText,
		Metadata: map[string]interface{}{
			"type":   "section",
			"parent": "chapter-2",
		},
	}

	child2_2 := schema.Node{
		ID:   "section-2-2",
		Text: "Recurrent Neural Networks (RNNs) process sequential data. LSTMs and GRUs address the vanishing gradient problem.",
		Type: schema.ObjectTypeText,
		Metadata: map[string]interface{}{
			"type":   "section",
			"parent": "chapter-2",
		},
	}

	// Set up relationships
	// Parent 1 -> Children 1-3
	parent1.Relationships = schema.NodeRelationships{}
	parent1.Relationships.SetChildren([]schema.RelatedNodeInfo{
		{NodeID: child1_1.ID, NodeType: schema.ObjectTypeText},
		{NodeID: child1_2.ID, NodeType: schema.ObjectTypeText},
		{NodeID: child1_3.ID, NodeType: schema.ObjectTypeText},
	})

	// Parent 2 -> Children 2-1, 2-2
	parent2.Relationships = schema.NodeRelationships{}
	parent2.Relationships.SetChildren([]schema.RelatedNodeInfo{
		{NodeID: child2_1.ID, NodeType: schema.ObjectTypeText},
		{NodeID: child2_2.ID, NodeType: schema.ObjectTypeText},
	})

	// Children -> Parent relationships
	child1_1.Relationships = schema.NodeRelationships{}
	child1_1.Relationships.SetParent(schema.RelatedNodeInfo{NodeID: parent1.ID, NodeType: schema.ObjectTypeText})

	child1_2.Relationships = schema.NodeRelationships{}
	child1_2.Relationships.SetParent(schema.RelatedNodeInfo{NodeID: parent1.ID, NodeType: schema.ObjectTypeText})

	child1_3.Relationships = schema.NodeRelationships{}
	child1_3.Relationships.SetParent(schema.RelatedNodeInfo{NodeID: parent1.ID, NodeType: schema.ObjectTypeText})

	child2_1.Relationships = schema.NodeRelationships{}
	child2_1.Relationships.SetParent(schema.RelatedNodeInfo{NodeID: parent2.ID, NodeType: schema.ObjectTypeText})

	child2_2.Relationships = schema.NodeRelationships{}
	child2_2.Relationships.SetParent(schema.RelatedNodeInfo{NodeID: parent2.ID, NodeType: schema.ObjectTypeText})

	// Set sibling relationships (next/previous)
	child1_1.Relationships.SetNext(schema.RelatedNodeInfo{NodeID: child1_2.ID, NodeType: schema.ObjectTypeText})
	child1_2.Relationships.SetPrevious(schema.RelatedNodeInfo{NodeID: child1_1.ID, NodeType: schema.ObjectTypeText})
	child1_2.Relationships.SetNext(schema.RelatedNodeInfo{NodeID: child1_3.ID, NodeType: schema.ObjectTypeText})
	child1_3.Relationships.SetPrevious(schema.RelatedNodeInfo{NodeID: child1_2.ID, NodeType: schema.ObjectTypeText})

	child2_1.Relationships.SetNext(schema.RelatedNodeInfo{NodeID: child2_2.ID, NodeType: schema.ObjectTypeText})
	child2_2.Relationships.SetPrevious(schema.RelatedNodeInfo{NodeID: child2_1.ID, NodeType: schema.ObjectTypeText})

	parents := []schema.Node{parent1, parent2}
	children := []schema.Node{child1_1, child1_2, child1_3, child2_1, child2_2}

	return parents, children
}

// createStorageContext creates a storage context with parent nodes.
func createStorageContext(parentNodes []schema.Node) *storage.StorageContext {
	docStore := docstore.NewSimpleDocumentStore()

	ctx := context.Background()
	for _, node := range parentNodes {
		nodeCopy := node
		_ = docStore.AddDocuments(ctx, []schema.BaseNode{&nodeCopy}, true)
	}

	return &storage.StorageContext{
		DocStore: docStore,
	}
}

// MockVectorRetriever simulates a vector retriever returning child nodes.
type MockVectorRetriever struct {
	*retriever.BaseRetriever
	nodes []schema.Node
}

// NewMockVectorRetriever creates a new mock vector retriever.
func NewMockVectorRetriever(nodes []schema.Node) *MockVectorRetriever {
	return &MockVectorRetriever{
		BaseRetriever: retriever.NewBaseRetriever(),
		nodes:         nodes,
	}
}

// Retrieve returns nodes matching the query.
func (m *MockVectorRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	queryLower := strings.ToLower(query.QueryString)
	var results []schema.NodeWithScore

	for _, node := range m.nodes {
		textLower := strings.ToLower(node.Text)
		score := 0.0

		// Simple keyword matching for demo
		keywords := []string{"machine", "learning", "neural", "deep", "supervised", "unsupervised"}
		for _, kw := range keywords {
			if strings.Contains(queryLower, kw) || strings.Contains(textLower, kw) {
				score += 0.15
			}
		}

		if score > 0.2 {
			results = append(results, schema.NodeWithScore{
				Node:  node,
				Score: score,
			})
		}
	}

	return results, nil
}

// printResults prints retrieval results.
func printResults(label string, results []schema.NodeWithScore) {
	fmt.Printf("Results for %s (%d nodes):\n", label, len(results))
	for i, r := range results {
		nodeType := "unknown"
		if t, ok := r.Node.Metadata["type"]; ok {
			nodeType = fmt.Sprintf("%v", t)
		}
		fmt.Printf("  %d. [Score: %.3f] [Type: %s] %s: %s\n",
			i+1, r.Score, nodeType, r.Node.ID, truncate(r.Node.Text, 50))
	}
	fmt.Println()
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
