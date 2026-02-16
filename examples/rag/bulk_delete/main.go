// Package main demonstrates bulk delete operations using the BulkVectorStore interface.
// This example shows how to delete multiple nodes by metadata filters.
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/aqua777/go-llamaindex/rag/store"
	"github.com/aqua777/go-llamaindex/schema"
)

func main() {
	ctx := context.Background()

	// 1. Create an in-memory vector store
	// SimpleVectorStore implements BulkVectorStore interface
	fmt.Println("=== Bulk Delete Operations Demo ===")
	fmt.Println("\n1. Creating SimpleVectorStore...")

	// In real code, you might receive a VectorStore interface from elsewhere.
	// We simulate this by assigning to the interface type first.
	var vectorStore store.VectorStore = store.NewSimpleVectorStore()
	fmt.Println("   Vector store created successfully")

	// 2. Type assertion to check for BulkVectorStore support
	// This is the recommended pattern when you have a VectorStore and want
	// to use bulk operations if available.
	fmt.Println("\n2. Checking BulkVectorStore support...")
	bulkStore, ok := vectorStore.(store.BulkVectorStore)
	if !ok {
		log.Fatal("Store does not implement BulkVectorStore interface")
	}
	fmt.Println("   BulkVectorStore interface supported!")

	// 3. Create sample nodes simulating a chat application
	// Each node belongs to a chat session and has a user
	fmt.Println("\n3. Creating sample nodes...")
	nodes := []schema.Node{
		// Chat session 1 - user_alice
		{ID: "msg_1", Text: "Hello, how can I help you today?", Type: schema.ObjectTypeText,
			Metadata: map[string]interface{}{"chat_id": "chat_001", "user": "alice", "role": "assistant"},
			Embedding: []float64{0.1, 0.2, 0.3}},
		{ID: "msg_2", Text: "I need help with my order", Type: schema.ObjectTypeText,
			Metadata: map[string]interface{}{"chat_id": "chat_001", "user": "alice", "role": "user"},
			Embedding: []float64{0.2, 0.3, 0.4}},
		{ID: "msg_3", Text: "Sure, let me look that up for you", Type: schema.ObjectTypeText,
			Metadata: map[string]interface{}{"chat_id": "chat_001", "user": "alice", "role": "assistant"},
			Embedding: []float64{0.3, 0.4, 0.5}},

		// Chat session 2 - user_bob
		{ID: "msg_4", Text: "What's the weather like?", Type: schema.ObjectTypeText,
			Metadata: map[string]interface{}{"chat_id": "chat_002", "user": "bob", "role": "user"},
			Embedding: []float64{0.4, 0.5, 0.6}},
		{ID: "msg_5", Text: "It's sunny and 72 degrees", Type: schema.ObjectTypeText,
			Metadata: map[string]interface{}{"chat_id": "chat_002", "user": "bob", "role": "assistant"},
			Embedding: []float64{0.5, 0.6, 0.7}},

		// Chat session 3 - user_alice (another session)
		{ID: "msg_6", Text: "Can you recommend a restaurant?", Type: schema.ObjectTypeText,
			Metadata: map[string]interface{}{"chat_id": "chat_003", "user": "alice", "role": "user"},
			Embedding: []float64{0.6, 0.7, 0.8}},
		{ID: "msg_7", Text: "I recommend the Italian place downtown", Type: schema.ObjectTypeText,
			Metadata: map[string]interface{}{"chat_id": "chat_003", "user": "alice", "role": "assistant"},
			Embedding: []float64{0.7, 0.8, 0.9}},
	}

	// 4. Add nodes to the store
	ids, err := bulkStore.Add(ctx, nodes)
	if err != nil {
		log.Fatalf("Failed to add nodes: %v", err)
	}
	fmt.Printf("   Added %d nodes: %v\n", len(ids), ids)

	// 5. Count all nodes
	fmt.Println("\n4. Counting nodes...")
	totalCount, err := bulkStore.Count(ctx, nil)
	if err != nil {
		log.Fatalf("Failed to count nodes: %v", err)
	}
	fmt.Printf("   Total nodes in store: %d\n", totalCount)

	// 6. Count nodes with filter
	fmt.Println("\n5. Counting nodes by metadata filter...")

	// Count nodes for user alice
	aliceFilter := schema.NewMetadataFilters(
		schema.NewMetadataFilter("user", "alice"),
	)
	aliceCount, err := bulkStore.Count(ctx, aliceFilter)
	if err != nil {
		log.Fatalf("Failed to count alice's nodes: %v", err)
	}
	fmt.Printf("   Nodes for user 'alice': %d\n", aliceCount)

	// Count nodes for chat_001
	chat001Filter := schema.NewMetadataFilters(
		schema.NewMetadataFilter("chat_id", "chat_001"),
	)
	chat001Count, err := bulkStore.Count(ctx, chat001Filter)
	if err != nil {
		log.Fatalf("Failed to count chat_001 nodes: %v", err)
	}
	fmt.Printf("   Nodes in chat 'chat_001': %d\n", chat001Count)

	// 7. Delete all nodes for a specific chat session
	fmt.Println("\n6. Deleting all nodes for chat_001...")
	deletedCount, err := bulkStore.DeleteByFilter(ctx, chat001Filter)
	if err != nil {
		log.Fatalf("Failed to delete nodes: %v", err)
	}
	fmt.Printf("   Deleted %d nodes\n", deletedCount)

	// Verify deletion
	remainingCount, err := bulkStore.Count(ctx, nil)
	if err != nil {
		log.Fatalf("Failed to count remaining nodes: %v", err)
	}
	fmt.Printf("   Remaining nodes in store: %d\n", remainingCount)

	// 8. Delete all nodes for a specific user
	fmt.Println("\n7. Deleting all remaining nodes for user 'alice'...")
	aliceRemainingFilter := schema.NewMetadataFilters(
		schema.NewMetadataFilter("user", "alice"),
	)
	deletedAlice, err := bulkStore.DeleteByFilter(ctx, aliceRemainingFilter)
	if err != nil {
		log.Fatalf("Failed to delete alice's nodes: %v", err)
	}
	fmt.Printf("   Deleted %d nodes belonging to user 'alice'\n", deletedAlice)

	// 9. Show final state
	fmt.Println("\n8. Final state...")
	finalCount, err := bulkStore.Count(ctx, nil)
	if err != nil {
		log.Fatalf("Failed to count final nodes: %v", err)
	}
	fmt.Printf("   Total nodes remaining: %d\n", finalCount)

	// Count bob's messages (should still exist)
	bobFilter := schema.NewMetadataFilters(
		schema.NewMetadataFilter("user", "bob"),
	)
	bobCount, err := bulkStore.Count(ctx, bobFilter)
	if err != nil {
		log.Fatalf("Failed to count bob's nodes: %v", err)
	}
	fmt.Printf("   Nodes for user 'bob': %d\n", bobCount)

	// 10. Demonstrate error handling - empty filter protection
	fmt.Println("\n9. Demonstrating safety: empty filter rejection...")
	_, err = bulkStore.DeleteByFilter(ctx, nil)
	if err != nil {
		fmt.Printf("   Expected error received: %v\n", err)
	}

	_, err = bulkStore.DeleteByFilter(ctx, &schema.MetadataFilters{})
	if err != nil {
		fmt.Printf("   Expected error received: %v\n", err)
	}

	fmt.Println("\n=== Bulk Delete Demo Complete ===")
}
