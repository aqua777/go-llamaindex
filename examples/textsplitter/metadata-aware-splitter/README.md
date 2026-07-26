# Metadata-Aware Sentence Splitter Example

Demonstrates `SentenceSplitter` as `MetadataAwareTextSplitter`: `SplitTextMetadataAware` reserves tokenizer budget for metadata so chunk sizes stay valid for RAG-style context windows.

## Prerequisites

- Go 1.24+
- This repo’s `golang` module on `GOPATH` / workspace (examples use `replace` to the parent module).

## Run

From the `golang/examples` directory:

```bash
go run ./textsplitter/metadata-aware-splitter/
```

Or from this folder:

```bash
go run main.go
```

## Expected behavior

The program builds long body text and large metadata, compares chunk counts with plain `SplitText` versus `SplitTextMetadataAware`, and prints `SUCCESS` when the metadata-aware split completes without error.
