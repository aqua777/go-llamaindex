# Markdown Element Node Parser Example

Demonstrates parsing Markdown into `schema.Node` values using `MarkdownElementNodeParser` (text blocks and tables as distinct nodes with `markdown_element` metadata).

## Prerequisites

- Go 1.24+
- This repo’s `golang` module on `GOPATH` / workspace (examples use `replace` to the parent module).

## Run

From the `golang/examples` directory:

```bash
go run ./nodeparser/markdown-element-parser/
```

Or from this folder:

```bash
go run main.go
```

## Expected behavior

The program parses a small Markdown sample with headings, paragraphs, and a table, then prints each node’s `markdown_element` kind and text.
