# Response Synthesis Strategies Example

This example demonstrates different response synthesis strategies in Go, corresponding to the Python `low_level/response_synthesis.ipynb` example.

## Overview

Learn how to use different synthesizers:
1. **SimpleSynthesizer** - Concatenates all context and makes one LLM call
2. **RefineSynthesizer** - Iteratively refines response across chunks
3. **TreeSummarizeSynthesizer** - Recursively summarizes in a tree structure

## Prerequisites

- Go 1.21+
- OpenAI API key (set `OPENAI_API_KEY` environment variable)

## Usage

```bash
export OPENAI_API_KEY=your-api-key
cd examples/rag/response_synthesis
go run main.go
```

## Components Used

- `rag/synthesizer.SimpleSynthesizer` - Single-call synthesis
- `rag/synthesizer.RefineSynthesizer` - Iterative refinement
- `rag/synthesizer.TreeSummarizeSynthesizer` - Tree-based summarization
- `prompts.BasePromptTemplate` - Custom prompt templates
