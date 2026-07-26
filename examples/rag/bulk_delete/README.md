# Bulk Delete Operations Example

This example demonstrates how to use the `BulkVectorStore` interface for efficient metadata-based bulk deletion of vector store nodes.

## Overview

The `BulkVectorStore` interface extends the standard `VectorStore` with two additional methods:

- **`DeleteByFilter(ctx, filters)`** - Delete all nodes matching metadata filters
- **`Count(ctx, filters)`** - Count nodes matching optional metadata filters

This is useful for scenarios like:
- Deleting all messages for a specific chat session
- Removing all documents associated with a user
- Cleaning up data for a specific category or time period

## Prerequisites

- Go 1.21+

## Usage

```bash
cd examples/rag/bulk_delete
go run main.go
```

## Expected Output

```
=== Bulk Delete Operations Demo ===

1. Creating SimpleVectorStore...
   Vector store created successfully

2. Checking BulkVectorStore support...
   BulkVectorStore interface supported!

3. Creating sample nodes...
   Added 7 nodes: [msg_1 msg_2 msg_3 msg_4 msg_5 msg_6 msg_7]

4. Counting nodes...
   Total nodes in store: 7

5. Counting nodes by metadata filter...
   Nodes for user 'alice': 5
   Nodes in chat 'chat_001': 3

6. Deleting all nodes for chat_001...
   Deleted 3 nodes
   Remaining nodes in store: 4

7. Deleting all remaining nodes for user 'alice'...
   Deleted 2 nodes belonging to user 'alice'

8. Final state...
   Total nodes remaining: 2
   Nodes for user 'bob': 2

9. Demonstrating safety: empty filter rejection...
   Expected error received: filters cannot be nil or empty for bulk delete
   Expected error received: filters cannot be nil or empty for bulk delete

=== Bulk Delete Demo Complete ===
```

## Key Concepts

### Type Assertion Pattern

Not all vector stores implement `BulkVectorStore`. Use type assertion to check:

```go
bulkStore, ok := store.(store.BulkVectorStore)
if !ok {
    // Fallback to individual deletion
    return iterativeDelete(ctx, store, nodeIDs)
}
// Use bulk operations
count, err := bulkStore.DeleteByFilter(ctx, filters)
```

### Creating Metadata Filters

Use the `schema.MetadataFilters` type for filter expressions:

```go
// Simple equality filter
filter := schema.NewMetadataFilters(
    schema.NewMetadataFilter("chat_id", "chat_001"),
)

// Filter with specific operator
filter := schema.NewMetadataFilters(
    schema.NewMetadataFilterWithOp("status", "active", schema.FilterOperatorNe),
)
```

### Safety: Empty Filter Rejection

`DeleteByFilter` requires a non-empty filter to prevent accidental deletion of all data:

```go
// This will return an error
_, err := bulkStore.DeleteByFilter(ctx, nil)
// Error: filters cannot be nil or empty for bulk delete
```

### Counting Nodes

Use `Count` to verify state before/after operations:

```go
// Count all nodes
total, _ := bulkStore.Count(ctx, nil)

// Count with filter
filtered, _ := bulkStore.Count(ctx, schema.NewMetadataFilters(
    schema.NewMetadataFilter("user", "alice"),
))
```

## Components Used

- `rag/store.SimpleVectorStore` - In-memory vector store implementing `BulkVectorStore`
- `rag/store.BulkVectorStore` - Interface for bulk operations
- `schema.MetadataFilters` - Filter expressions for metadata matching
- `schema.MetadataFilter` - Individual filter with key, value, and operator

## Supported Filter Operators

The following operators are supported in `evaluateFilter`:

| Operator | Description |
|----------|-------------|
| `FilterOperatorEq` (`==`) | Equal to |
| `FilterOperatorNe` (`!=`) | Not equal to |
| `FilterOperatorIsEmpty` | Field is empty or missing |

Additional operators can be added to `evaluateFilter` in `simple.go` as needed.

## Implementing BulkVectorStore for Other Backends

When implementing `BulkVectorStore` for other vector databases:

```go
// Example for a hypothetical backend
func (s *MyVectorStore) DeleteByFilter(ctx context.Context, filters *schema.MetadataFilters) (int, error) {
    if filters == nil || len(filters.Filters) == 0 {
        return 0, errors.New("filters cannot be nil or empty for bulk delete")
    }

    // Convert MetadataFilters to backend-specific filter format
    backendFilter := convertToBackendFilter(filters)

    // Execute bulk delete
    return s.client.DeleteByFilter(ctx, backendFilter)
}
```

Note: Some backends (like chromem-go) don't return a count of deleted items. In such cases, return `-1` or query before deleting to get the count.
