# Hierarchical Node Parser Example

This example demonstrates how to use the `HierarchicalNodeParser` to split a document into a hierarchy of nodes with parent-child relationships.

## Prerequisites

- Go 1.24+
- The `go-llamaindex` module

## Running the Example

Run the example from the `golang` directory:

```bash
cd examples/nodeparser/hierarchical-parser
go run main.go
```

## Expected Output

You should see a list of nodes printed with their IDs, text lengths, and parent/child relationships. The root document will have no parent, and its children will be the largest chunks. Those chunks will have smaller chunks as children, and so on.

```
Total nodes: 14

--- Node 1 (hierarchy_level=0) ---
id: 47f0398c-3670-43f4-a9e3-0f5c0454d90c
text_len: 6369
PARENT -> (none)
CHILD -> 93410ea9-d26d-4dc0-8b54-a67c4b653d31, ebe2c4d0-8e34-474a-8e7c-fdda0b857be5, 862f4cbb-7851-4100-85c5-6b82cd481913

...
```