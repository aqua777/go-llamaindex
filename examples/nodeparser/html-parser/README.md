# HTML Node Parser Example

Demonstrates parsing HTML documents into `schema.Node` values using tag-based extraction (`HTMLNodeParser`).

## Prerequisites

- Go 1.24+
- This repo’s `golang` module on `GOPATH` / workspace (examples use `replace` to the parent module).

## Run

From the `golang/examples` directory:

```bash
go run ./nodeparser/html-parser/
```

Or from this folder:

```bash
go run main.go
```

## Expected behavior

The program parses a small HTML sample with default tags (e.g. `p`, `h1`, `section`, `li`, `b`, `i`) and prints each node’s text and metadata, including the `html_tag` key.
