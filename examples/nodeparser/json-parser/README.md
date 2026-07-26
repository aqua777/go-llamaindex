# JSON Node Parser Example

Demonstrates parsing JSON documents into `schema.Node` values using leaf extraction and JSON path metadata (`JSONNodeParser`).

## Prerequisites

- Go 1.24+
- This repo’s `golang` module on `GOPATH` / workspace (examples use `replace` to the parent module).

## Run

From the `golang/examples` directory:

```bash
go run ./nodeparser/json-parser/
```

Or from this folder:

```bash
go run main.go
```

## Expected behavior

The program parses a small JSON sample (object with nested object, array, and scalar fields) and prints how many nodes were extracted. For each node it prints `json_path` (dot/bracket paths to each leaf value) and the quoted `text` for that value.
