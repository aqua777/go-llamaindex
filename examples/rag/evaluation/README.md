# RAG Evaluation Example

This example demonstrates RAG evaluation metrics in Go, corresponding to the Python `low_level/evaluation.ipynb` example.

## Overview

Learn how to evaluate RAG systems using:
1. **Faithfulness Evaluator** - Checks if response is supported by context
2. **Relevancy Evaluator** - Checks if response is relevant to query and context
3. **Correctness Evaluator** - Compares response to reference answer
4. **Semantic Similarity Evaluator** - Measures semantic similarity between texts

## Prerequisites

- Go 1.21+
- OpenAI API key (set `OPENAI_API_KEY` environment variable)

## Usage

```bash
export OPENAI_API_KEY=your-api-key
cd examples/rag/evaluation
go run main.go
```

## Components Used

- `evaluation.FaithfulnessEvaluator` - Faithfulness checking
- `evaluation.RelevancyEvaluator` - Relevancy evaluation
- `evaluation.CorrectnessEvaluator` - Correctness scoring
- `evaluation.SemanticSimilarityEvaluator` - Semantic similarity
- `evaluation.BatchRunner` - Batch evaluation
