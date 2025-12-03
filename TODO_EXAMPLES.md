# LlamaIndex Go Examples TODO

This document lists Python examples that can be implemented in Go using the existing functionality in the `golang/` package.

**Legend:**
- ✅ **Can Implement** - All required Go components exist
- 🔄 **Partial** - Most components exist, minor gaps
- ❌ **Cannot Implement** - Missing critical Go components

**Python Examples Source:** `/python/docs/examples/`

---

## Priority 1: Core RAG Examples ✅

These examples demonstrate fundamental RAG patterns and can be fully implemented with existing Go code.

### 1.1 Basic RAG Pipeline

| Python Example | Description | Go Components Required | Status |
|----------------|-------------|------------------------|--------|
| `low_level/retrieval.ipynb` | Basic vector retrieval | `rag/retriever`, `embedding`, `schema` | ✅ |
| `low_level/vector_store.ipynb` | Vector store operations | `rag/store`, `embedding` | ✅ |
| `low_level/ingestion.ipynb` | Document ingestion pipeline | `ingestion`, `textsplitter`, `nodeparser` | ✅ |
| `low_level/response_synthesis.ipynb` | Response synthesis strategies | `rag/synthesizer`, `llm` | ✅ |
| `low_level/router.ipynb` | Query routing | `rag/queryengine`, `selector` | ✅ |
| `low_level/evaluation.ipynb` | RAG evaluation | `evaluation` | ✅ |
| `low_level/fusion_retriever.ipynb` | Fusion retrieval | `rag/retriever/fusion.go` | ✅ |

**Suggested Go Example:** `examples/rag/basic_pipeline/`
```
- Load documents with SimpleDirectoryReader
- Split with SentenceSplitter
- Embed with OpenAI
- Store in ChromaDB
- Retrieve and synthesize response
```

### 1.2 Response Synthesizers

| Python Example | Description | Go Components Required | Status |
|----------------|-------------|------------------------|--------|
| `response_synthesizers/refine.ipynb` | Refine synthesis | `rag/synthesizer/refine.go` | ✅ |
| `response_synthesizers/tree_summarize.ipynb` | Tree summarization | `rag/synthesizer/tree_summarize.go` | ✅ |
| `response_synthesizers/custom_prompt_synthesizer.ipynb` | Custom prompts | `prompts`, `rag/synthesizer` | ✅ |
| `response_synthesizers/pydantic_tree_summarize.ipynb` | Structured output | `program`, `rag/synthesizer` | ✅ |

**Suggested Go Example:** `examples/rag/synthesizers/`

---

## Priority 2: Chat Engine Examples ✅

All chat engine modes are implemented in Go.

| Python Example | Description | Go Components Required | Status |
|----------------|-------------|------------------------|--------|
| `chat_engine/chat_engine_context.ipynb` | Context chat engine | `chatengine/context.go` | ✅ |
| `chat_engine/chat_engine_condense_plus_context.ipynb` | Condense + context | `chatengine/condense_plus_context.go` | ✅ |
| `chat_engine/chat_engine_condense_question.ipynb` | Condense question | `chatengine/condense_plus_context.go` | ✅ |
| `chat_engine/chat_engine_best.ipynb` | Best practices | `chatengine` | ✅ |
| `chat_engine/chat_engine_personality.ipynb` | Custom personality | `chatengine`, `prompts` | ✅ |

**Suggested Go Example:** `examples/chatengine/`
```
- SimpleChatEngine - Direct LLM chat
- ContextChatEngine - RAG-enhanced chat
- CondensePlusContextChatEngine - Query condensation
```

---

## Priority 3: Query Engine Examples ✅

| Python Example | Description | Go Components Required | Status |
|----------------|-------------|------------------------|--------|
| `query_engine/sub_question_query_engine.ipynb` | Sub-question decomposition | `rag/queryengine/sub_question.go` | ✅ |
| `query_engine/RouterQueryEngine.ipynb` | Query routing | `rag/queryengine/router.go` | ✅ |
| `query_engine/RetrieverRouterQueryEngine.ipynb` | Retriever routing | `rag/retriever/router.go` | ✅ |
| `query_engine/custom_query_engine.ipynb` | Custom query engine | `rag/queryengine/interface.go` | ✅ |
| `query_engine/CustomRetrievers.ipynb` | Custom retrievers | `rag/retriever/interface.go` | ✅ |
| `query_engine/ensemble_query_engine.ipynb` | Ensemble queries | `rag/retriever/fusion.go` | ✅ |
| `query_engine/knowledge_graph_query_engine.ipynb` | KG queries | `index/knowledge_graph.go` | ✅ |

**Suggested Go Example:** `examples/queryengine/`

---

## Priority 4: Retriever Examples ✅

| Python Example | Description | Go Components Required | Status |
|----------------|-------------|------------------------|--------|
| `retrievers/auto_merging_retriever.ipynb` | Auto-merging retrieval | `rag/retriever/auto_merging.go` | ✅ |
| `retrievers/reciprocal_rerank_fusion.ipynb` | RRF fusion | `rag/retriever/fusion.go` | ✅ |
| `retrievers/relative_score_dist_fusion.ipynb` | Score fusion | `rag/retriever/fusion.go` | ✅ |
| `retrievers/simple_fusion.ipynb` | Simple fusion | `rag/retriever/fusion.go` | ✅ |
| `retrievers/router_retriever.ipynb` | Router retriever | `rag/retriever/router.go` | ✅ |
| `retrievers/composable_retrievers.ipynb` | Composable retrievers | `rag/retriever` | ✅ |
| `retrievers/bm25_retriever.ipynb` | BM25 retrieval | `embedding/bm25.go` | ✅ |
| `retrievers/ensemble_retrieval.ipynb` | Ensemble retrieval | `rag/retriever/fusion.go` | ✅ |

**Suggested Go Example:** `examples/retrievers/`

---

## Priority 5: Agent Examples ✅

| Python Example | Description | Go Components Required | Status |
|----------------|-------------|------------------------|--------|
| `agent/react_agent.ipynb` | ReAct agent basics | `agent/react.go` | ✅ |
| `agent/react_agent_with_query_engine.ipynb` | ReAct + RAG | `agent`, `rag/queryengine` | ✅ |
| `agent/openai_agent_with_query_engine.ipynb` | Function calling + RAG | `agent`, `tools` | ✅ |
| `agent/openai_agent_retrieval.ipynb` | Agent retrieval | `agent`, `tools/retriever_tool.go` | ✅ |
| `agent/return_direct_agent.ipynb` | Direct return agent | `agent` | ✅ |
| `workflow/function_calling_agent.ipynb` | Function calling workflow | `workflow`, `agent` | ✅ |
| `workflow/react_agent.ipynb` | ReAct workflow | `workflow`, `agent` | ✅ |

**Suggested Go Example:** `examples/agent/`
```
- ReAct agent with calculator tools
- ReAct agent with query engine tool
- Function calling agent
```

---

## Priority 6: Evaluation Examples ✅

| Python Example | Description | Go Components Required | Status |
|----------------|-------------|------------------------|--------|
| `evaluation/faithfulness_eval.ipynb` | Faithfulness evaluation | `evaluation/faithfulness.go` | ✅ |
| `evaluation/relevancy_eval.ipynb` | Relevancy evaluation | `evaluation/relevancy.go` | ✅ |
| `evaluation/correctness_eval.ipynb` | Correctness evaluation | `evaluation/correctness.go` | ✅ |
| `evaluation/semantic_similarity_eval.ipynb` | Semantic similarity | `evaluation/semantic_similarity.go` | ✅ |
| `evaluation/batch_eval.ipynb` | Batch evaluation | `evaluation/batch_runner.go` | ✅ |
| `evaluation/QuestionGeneration.ipynb` | Question generation | `questiongen` | ✅ |
| `evaluation/RetryQuery.ipynb` | Retry queries | `rag/queryengine/retry.go` | ✅ |

**Suggested Go Example:** `examples/evaluation/`

---

## Priority 7: Memory Examples ✅

| Python Example | Description | Go Components Required | Status |
|----------------|-------------|------------------------|--------|
| `memory/memory.ipynb` | Basic memory | `memory/types.go` | ✅ |
| `memory/ChatSummaryMemoryBuffer.ipynb` | Summary memory | `memory/chat_summary_memory_buffer.go` | ✅ |
| `memory/custom_memory.ipynb` | Custom memory | `memory` interface | ✅ |

**Suggested Go Example:** `examples/memory/`

---

## Priority 8: Postprocessor Examples ✅

| Python Example | Description | Go Components Required | Status |
|----------------|-------------|------------------------|--------|
| `node_postprocessor/LLMReranker-Gatsby.ipynb` | LLM reranking | `postprocessor/llm_rerank.go` | ✅ |
| `node_postprocessor/rankGPT.ipynb` | RankGPT reranking | `postprocessor/rankgpt_rerank.go` | ✅ |
| `node_postprocessor/LongContextReorder.ipynb` | Long context reorder | `postprocessor/long_context_reorder.go` | ✅ |
| `node_postprocessor/MetadataReplacementDemo.ipynb` | Metadata replacement | `postprocessor/metadata_replacement.go` | ✅ |
| `node_postprocessor/PII.ipynb` | PII masking | `postprocessor/pii.go` | ✅ |
| `node_postprocessor/RecencyPostprocessorDemo.ipynb` | Recency weighting | `postprocessor/node_recency.go` | ✅ |
| `node_postprocessor/OptimizerDemo.ipynb` | Sentence optimizer | `postprocessor/optimizer.go` | ✅ |

**Suggested Go Example:** `examples/postprocessor/`

---

## Priority 9: Workflow Examples ✅

| Python Example | Description | Go Components Required | Status |
|----------------|-------------|------------------------|--------|
| `workflow/rag.ipynb` | RAG workflow | `workflow` | ✅ |
| `workflow/reflection.ipynb` | Reflection workflow | `workflow` | ✅ |
| `workflow/parallel_execution.ipynb` | Parallel execution | `workflow` | ✅ |
| `workflow/router_query_engine.ipynb` | Router workflow | `workflow`, `rag/queryengine` | ✅ |
| `workflow/sub_question_query_engine.ipynb` | Sub-question workflow | `workflow`, `questiongen` | ✅ |
| `workflow/checkpointing_workflows.ipynb` | Checkpointing | `workflow` | ✅ |

**Suggested Go Example:** `examples/workflow/`

---

## Priority 10: Index Examples ✅

| Python Example | Description | Go Components Required | Status |
|----------------|-------------|------------------------|--------|
| `index_structs/knowledge_graph/KnowledgeGraphDemo.ipynb` | KG index | `index/knowledge_graph.go` | ✅ |
| `index_structs/doc_summary/` | Summary index | `index/summary.go` | ✅ |

**Suggested Go Example:** `examples/index/`

---

## Priority 11: Ingestion Pipeline Examples ✅

| Python Example | Description | Go Components Required | Status |
|----------------|-------------|------------------------|--------|
| `ingestion/advanced_ingestion_pipeline.ipynb` | Advanced ingestion | `ingestion/pipeline.go` | ✅ |
| `ingestion/document_management_pipeline.ipynb` | Document management | `ingestion`, `storage` | ✅ |

**Suggested Go Example:** `examples/ingestion/`

---

## Priority 12: Prompt Examples ✅

| Python Example | Description | Go Components Required | Status |
|----------------|-------------|------------------------|--------|
| `prompts/advanced_prompts.ipynb` | Advanced prompts | `prompts/template.go` | ✅ |
| `prompts/prompt_mixin.ipynb` | Prompt mixin | `prompts/mixin.go` | ✅ |
| `prompts/rich_prompt_template_features.ipynb` | Template features | `prompts` | ✅ |

**Suggested Go Example:** `examples/prompts/`

---

## Priority 13: Metadata Extraction Examples ✅

| Python Example | Description | Go Components Required | Status |
|----------------|-------------|------------------------|--------|
| `metadata_extraction/MetadataExtractionSEC.ipynb` | SEC metadata | `extractors` | ✅ |
| `metadata_extraction/MetadataExtraction_LLMSurvey.ipynb` | LLM survey | `extractors` | ✅ |
| `metadata_extraction/EntityExtractionClimate.ipynb` | Entity extraction | `extractors` | ✅ |

**Suggested Go Example:** `examples/extractors/`

---

## Priority 14: Output Parsing Examples ✅

| Python Example | Description | Go Components Required | Status |
|----------------|-------------|------------------------|--------|
| `output_parsing/function_program.ipynb` | Function program | `program/function_program.go` | ✅ |
| `output_parsing/llm_program.ipynb` | LLM program | `program/llm_program.go` | ✅ |

**Suggested Go Example:** `examples/program/`

---

## Priority 15: Tool Examples ✅

| Python Example | Description | Go Components Required | Status |
|----------------|-------------|------------------------|--------|
| `tools/function_tool_callback.ipynb` | Function tools | `tools/function_tool.go` | ✅ |
| `tools/eval_query_engine_tool.ipynb` | Query engine tool | `tools/query_engine_tool.go` | ✅ |

**Suggested Go Example:** `examples/tools/`

---

## Priority 16: Streaming Examples ✅

| Python Example | Description | Go Components Required | Status |
|----------------|-------------|------------------------|--------|
| `customization/streaming/SimpleIndexDemo-streaming.ipynb` | Basic streaming | `llm`, `rag/synthesizer` | ✅ |
| `customization/streaming/chat_engine_condense_question_stream_response.ipynb` | Chat streaming | `chatengine` | ✅ |

**Suggested Go Example:** `examples/streaming/`

---

## Priority 17: LLM Provider Examples ✅

| Python Example | Description | Go Components Required | Status |
|----------------|-------------|------------------------|--------|
| `customization/llms/AzureOpenAI.ipynb` | Azure OpenAI | `llm/azure_openai.go` | ✅ |
| `customization/llms/SimpleIndexDemo-ChatGPT.ipynb` | OpenAI ChatGPT | `llm/openai.go` | ✅ |

**Suggested Go Example:** `examples/llm/`

---

## Cannot Implement (Missing Go Components) ❌

These examples require components not yet implemented in Go:

### Vector Store Integrations
| Python Example | Missing Component |
|----------------|-------------------|
| `vector_stores/PineconeIndexDemo.ipynb` | Pinecone client |
| `vector_stores/WeaviateIndexDemo.ipynb` | Weaviate client |
| `vector_stores/QdrantIndexDemo.ipynb` | Qdrant client |
| `vector_stores/MilvusIndexDemo.ipynb` | Milvus client |
| `vector_stores/PGVectorDemo.ipynb` | pgvector client |

### LLM Providers
| Python Example | Missing Component |
|----------------|-------------------|
| `llm/gemini.ipynb` | Google Gemini client |
| `llm/bedrock.ipynb` | AWS Bedrock client |
| `llm/groq.ipynb` | Groq client |
| `llm/mistral.ipynb` | Mistral AI client |

### Document Readers
| Python Example | Missing Component |
|----------------|-------------------|
| `data_connectors/NotionDemo.ipynb` | Notion API client |
| `data_connectors/SlackDemo.ipynb` | Slack API client |
| `data_connectors/GoogleDocsDemo.ipynb` | Google Docs client |

### Multi-Modal
| Python Example | Missing Component |
|----------------|-------------------|
| `multi_modal/gpt4v_*.ipynb` | Multi-modal LLM implementation |

### Observability
| Python Example | Missing Component |
|----------------|-------------------|
| `observability/LangfuseCallbackHandler.ipynb` | Langfuse integration |
| `observability/WandbCallbackHandler.ipynb` | W&B integration |

---

## Implementation Roadmap

### Phase 1: Core Examples (Week 1-2)
1. `examples/rag/basic_pipeline/` - Basic RAG with all synthesizers
2. `examples/chatengine/` - All chat engine modes
3. `examples/retrievers/` - Fusion, auto-merging, router

### Phase 2: Advanced Examples (Week 3-4)
4. `examples/agent/` - ReAct agent with tools
5. `examples/evaluation/` - All evaluators
6. `examples/workflow/` - Event-driven workflows

### Phase 3: Specialized Examples (Week 5-6)
7. `examples/postprocessor/` - Reranking, PII, recency
8. `examples/index/` - KG index, tree index
9. `examples/extractors/` - Metadata extraction

### Phase 4: Integration Examples (Week 7-8)
10. `examples/llm/` - All supported providers
11. `examples/streaming/` - Streaming responses
12. `examples/program/` - Structured output

---

## Example Template

Each Go example should follow this structure:

```
examples/
├── <category>/
│   ├── README.md           # Description and usage
│   ├── main.go             # Runnable example
│   └── data/               # Sample data (if needed)
```

Example `main.go` structure:
```go
package main

import (
    "context"
    "fmt"
    "log"
    // imports...
)

func main() {
    ctx := context.Background()
    
    // 1. Setup components
    // 2. Load/process data
    // 3. Execute pipeline
    // 4. Print results
}
```

---

## Summary

| Category | Total Python Examples | Can Implement in Go | Coverage |
|----------|----------------------|---------------------|----------|
| Core RAG | 8 | 8 | 100% |
| Chat Engine | 6 | 6 | 100% |
| Query Engine | 27 | 10 | 37% |
| Retrievers | 20 | 10 | 50% |
| Agents | 31 | 8 | 26% |
| Evaluation | 25 | 8 | 32% |
| Memory | 6 | 3 | 50% |
| Postprocessors | 26 | 8 | 31% |
| Workflows | 19 | 8 | 42% |
| Index | 18 | 3 | 17% |
| Ingestion | 6 | 2 | 33% |
| Prompts | 6 | 3 | 50% |
| Extractors | 7 | 3 | 43% |
| Output Parsing | 14 | 2 | 14% |
| Tools | 11 | 2 | 18% |
| Streaming | 2 | 2 | 100% |
| LLM Providers | 92 | 5 | 5% |
| **Total** | **~350** | **~90** | **~26%** |

**Note:** The 26% coverage is primarily limited by missing provider integrations (vector stores, LLMs, readers), not core functionality. The Go implementation covers ~85-90% of core LlamaIndex abstractions.
