# LlamaIndex Go Examples TODO

This document lists Python examples that can be implemented in Go using the existing functionality in the `golang/` package.

**Legend:**
- ✅ **Can Implement** - All required Go components exist
- 🔄 **Partial** - Most components exist, minor gaps
- ❌ **Cannot Implement** - Missing critical Go components
- [x] **Implemented** - Go example has been created
- [ ] **Not Implemented** - Go example not yet created

**Python Examples Source:** `/python/docs/examples/`

---

## Priority 1: Core RAG Examples ✅

These examples demonstrate fundamental RAG patterns and can be fully implemented with existing Go code.

### 1.1 Basic RAG Pipeline

| Python Example | Description | Go Components Required | Status | Implemented |
|----------------|-------------|------------------------|--------|-------------|
| `low_level/retrieval.ipynb` | Basic vector retrieval | `rag/retriever`, `embedding`, `schema` | ✅ | [x] `examples/rag/basic_pipeline/` |
| `low_level/vector_store.ipynb` | Vector store operations | `rag/store`, `embedding` | ✅ | [x] `examples/rag/vector_store/` |
| `low_level/ingestion.ipynb` | Document ingestion pipeline | `ingestion`, `textsplitter`, `nodeparser` | ✅ | [x] `examples/rag/ingestion/` |
| `low_level/response_synthesis.ipynb` | Response synthesis strategies | `rag/synthesizer`, `llm` | ✅ | [x] `examples/rag/response_synthesis/` |
| `low_level/router.ipynb` | Query routing | `rag/queryengine`, `selector` | ✅ | [x] `examples/rag/router/` |
| `low_level/evaluation.ipynb` | RAG evaluation | `evaluation` | ✅ | [x] `examples/rag/evaluation/` |
| `low_level/fusion_retriever.ipynb` | Fusion retrieval | `rag/retriever/fusion.go` | ✅ | [x] `examples/rag/fusion_retriever/` |

**Go Examples:** `examples/rag/basic_pipeline/`, `examples/rag/vector_store/`, `examples/rag/ingestion/`, `examples/rag/response_synthesis/`, `examples/rag/router/`, `examples/rag/evaluation/`, `examples/rag/fusion_retriever/`
```
- Load documents with SimpleDirectoryReader
- Split with SentenceSplitter
- Embed with OpenAI
- Store in ChromaDB
- Retrieve and synthesize response
```

### 1.2 Response Synthesizers

| Python Example | Description | Go Components Required | Status | Implemented |
|----------------|-------------|------------------------|--------|-------------|
| `response_synthesizers/refine.ipynb` | Refine synthesis | `rag/synthesizer/refine.go` | ✅ | [x] `examples/rag/response_synthesis/` |
| `response_synthesizers/tree_summarize.ipynb` | Tree summarization | `rag/synthesizer/tree_summarize.go` | ✅ | [x] `examples/rag/response_synthesis/` |
| `response_synthesizers/custom_prompt_synthesizer.ipynb` | Custom prompts | `prompts`, `rag/synthesizer` | ✅ | [x] `examples/rag/response_synthesis/` |
| `response_synthesizers/pydantic_tree_summarize.ipynb` | Structured output | `program`, `rag/synthesizer` | ✅ | [x] `examples/rag/pydantic_tree_summarize/` |

**Go Example:** `examples/rag/response_synthesis/` (covers refine, tree_summarize, custom prompts, pydantic_tree_summarize)

---

## Priority 2: Chat Engine Examples ✅

All chat engine modes are implemented in Go.

| Python Example | Description | Go Components Required | Status | Implemented |
|----------------|-------------|------------------------|--------|-------------|
| `chat_engine/chat_engine_context.ipynb` | Context chat engine | `chatengine/context.go` | ✅ | [x] `examples/chatengine/` |
| `chat_engine/chat_engine_condense_plus_context.ipynb` | Condense + context | `chatengine/condense_plus_context.go` | ✅ | [x] `examples/chatengine/` |
| `chat_engine/chat_engine_condense_question.ipynb` | Condense question | `chatengine/condense_plus_context.go` | ✅ | [x] `examples/chatengine/` |
| `chat_engine/chat_engine_best.ipynb` | Best practices | `chatengine` | ✅ | [x] `examples/chatengine/` |
| `chat_engine/chat_engine_personality.ipynb` | Custom personality | `chatengine`, `prompts` | ✅ | [x] `examples/chatengine/` |

**Go Example:** `examples/chatengine/` (covers all chat engine modes)
```
- SimpleChatEngine - Direct LLM chat
- ContextChatEngine - RAG-enhanced chat
- CondensePlusContextChatEngine - Query condensation
- Custom Personality - Themed assistant
```

---

## Priority 3: Query Engine Examples ✅

| Python Example | Description | Go Components Required | Status | Implemented |
|----------------|-------------|------------------------|--------|-------------|
| `query_engine/sub_question_query_engine.ipynb` | Sub-question decomposition | `rag/queryengine/sub_question.go` | ✅ | [x] `examples/queryengine/sub_question/` |
| `query_engine/RouterQueryEngine.ipynb` | Query routing | `rag/queryengine/router.go` | ✅ | [x] `examples/rag/router/` |
| `query_engine/RetrieverRouterQueryEngine.ipynb` | Retriever routing | `rag/retriever/router.go` | ✅ | [x] `examples/queryengine/retriever_router/` |
| `query_engine/custom_query_engine.ipynb` | Custom query engine | `rag/queryengine/interface.go` | ✅ | [x] `examples/queryengine/custom_query_engine/` |
| `query_engine/CustomRetrievers.ipynb` | Custom retrievers | `rag/retriever/interface.go` | ✅ | [x] `examples/queryengine/custom_retriever/` |
| `query_engine/ensemble_query_engine.ipynb` | Ensemble queries | `rag/retriever/fusion.go` | ✅ | [x] `examples/rag/fusion_retriever/` |
| `query_engine/knowledge_graph_query_engine.ipynb` | KG queries | `index/knowledge_graph.go` | ✅ | [x] `examples/queryengine/knowledge_graph/` |

**Go Example:** `examples/queryengine/` (covers sub_question, retriever_router, custom_query_engine, custom_retriever, knowledge_graph)

---

## Priority 4: Retriever Examples ✅

| Python Example | Description | Go Components Required | Status | Implemented |
|----------------|-------------|------------------------|--------|-------------|
| `retrievers/auto_merging_retriever.ipynb` | Auto-merging retrieval | `rag/retriever/auto_merging.go` | ✅ | [x] `examples/retrievers/auto_merging/` |
| `retrievers/reciprocal_rerank_fusion.ipynb` | RRF fusion | `rag/retriever/fusion.go` | ✅ | [x] `examples/rag/fusion_retriever/` |
| `retrievers/relative_score_dist_fusion.ipynb` | Score fusion | `rag/retriever/fusion.go` | ✅ | [x] `examples/rag/fusion_retriever/` |
| `retrievers/simple_fusion.ipynb` | Simple fusion | `rag/retriever/fusion.go` | ✅ | [x] `examples/rag/fusion_retriever/` |
| `retrievers/router_retriever.ipynb` | Router retriever | `rag/retriever/router.go` | ✅ | [x] `examples/retrievers/router/` |
| `retrievers/composable_retrievers.ipynb` | Composable retrievers | `rag/retriever` | ✅ | [x] `examples/retrievers/composable/` |
| `retrievers/bm25_retriever.ipynb` | BM25 retrieval | `embedding/bm25.go` | ✅ | [x] `examples/retrievers/bm25/` |
| `retrievers/ensemble_retrieval.ipynb` | Ensemble retrieval | `rag/retriever/fusion.go` | ✅ | [x] `examples/rag/fusion_retriever/` |

**Go Example:** `examples/retrievers/` (covers auto_merging, router, composable, bm25)

---

## Priority 5: Agent Examples ✅

| Python Example | Description | Go Components Required | Status | Implemented |
|----------------|-------------|------------------------|--------|-------------|
| `agent/react_agent.ipynb` | ReAct agent basics | `agent/react.go` | ✅ | [x] `examples/agent/react_agent/` |
| `agent/react_agent_with_query_engine.ipynb` | ReAct + RAG | `agent`, `rag/queryengine` | ✅ | [x] `examples/agent/react_agent_with_query_engine/` |
| `agent/openai_agent_with_query_engine.ipynb` | Function calling + RAG | `agent`, `tools` | ✅ | [x] `examples/agent/function_calling_agent/` |
| `agent/openai_agent_retrieval.ipynb` | Agent retrieval | `agent`, `tools/retriever_tool.go` | ✅ | [x] `examples/agent/agent_retrieval/` |
| `agent/return_direct_agent.ipynb` | Direct return agent | `agent` | ✅ | [x] `examples/agent/agent_retrieval/` |
| `workflow/function_calling_agent.ipynb` | Function calling workflow | `workflow`, `agent` | ✅ | [x] `examples/agent/workflow_agent/` |
| `workflow/react_agent.ipynb` | ReAct workflow | `workflow`, `agent` | ✅ | [x] `examples/agent/workflow_agent/` |

**Suggested Go Example:** `examples/agent/`
```
- ReAct agent with calculator tools
- ReAct agent with query engine tool
- Function calling agent
```

---

## Priority 6: Evaluation Examples ✅

| Python Example | Description | Go Components Required | Status | Implemented |
|----------------|-------------|------------------------|--------|-------------|
| `evaluation/faithfulness_eval.ipynb` | Faithfulness evaluation | `evaluation/faithfulness.go` | ✅ | [x] `examples/rag/evaluation/` |
| `evaluation/relevancy_eval.ipynb` | Relevancy evaluation | `evaluation/relevancy.go` | ✅ | [x] `examples/rag/evaluation/` |
| `evaluation/correctness_eval.ipynb` | Correctness evaluation | `evaluation/correctness.go` | ✅ | [x] `examples/rag/evaluation/` |
| `evaluation/semantic_similarity_eval.ipynb` | Semantic similarity | `evaluation/semantic_similarity.go` | ✅ | [x] `examples/evaluation/semantic_similarity/` |
| `evaluation/batch_eval.ipynb` | Batch evaluation | `evaluation/batch_runner.go` | ✅ | [x] `examples/evaluation/batch_eval/` |
| `evaluation/QuestionGeneration.ipynb` | Question generation | `questiongen` | ✅ | [x] `examples/evaluation/question_generation/` |
| `evaluation/RetryQuery.ipynb` | Retry queries | `rag/queryengine/retry.go` | ✅ | [x] `examples/evaluation/retry_query/` |

**Go Example:** `examples/rag/evaluation/` (covers faithfulness, relevancy, correctness)
**Go Example:** `examples/evaluation/` (covers semantic_similarity, batch_eval, question_generation, retry_query)

---

## Priority 7: Memory Examples ✅

| Python Example | Description | Go Components Required | Status | Implemented |
|----------------|-------------|------------------------|--------|-------------|
| `memory/memory.ipynb` | Basic memory | `memory/types.go` | ✅ | [x] `examples/memory/basic_memory/` |
| `memory/ChatSummaryMemoryBuffer.ipynb` | Summary memory | `memory/chat_summary_memory_buffer.go` | ✅ | [x] `examples/memory/chat_summary_memory/` |
| `memory/custom_memory.ipynb` | Custom memory | `memory` interface | ✅ | [x] `examples/memory/custom_memory/` |

**Go Example:** `examples/memory/` (covers basic_memory, chat_summary_memory, custom_memory)

---

## Priority 8: Postprocessor Examples ✅

| Python Example | Description | Go Components Required | Status | Implemented |
|----------------|-------------|------------------------|--------|-------------|
| `node_postprocessor/LLMReranker-Gatsby.ipynb` | LLM reranking | `postprocessor/llm_rerank.go` | ✅ | [x] `examples/postprocessor/llm_rerank/` |
| `node_postprocessor/rankGPT.ipynb` | RankGPT reranking | `postprocessor/rankgpt_rerank.go` | ✅ | [x] `examples/postprocessor/llm_rerank/` |
| `node_postprocessor/LongContextReorder.ipynb` | Long context reorder | `postprocessor/long_context_reorder.go` | ✅ | [x] `examples/postprocessor/long_context_reorder/` |
| `node_postprocessor/MetadataReplacementDemo.ipynb` | Metadata replacement | `postprocessor/metadata_replacement.go` | ✅ | [x] `examples/postprocessor/metadata_replacement/` |
| `node_postprocessor/PII.ipynb` | PII masking | `postprocessor/pii.go` | ✅ | [x] `examples/postprocessor/pii_masking/` |
| `node_postprocessor/RecencyPostprocessorDemo.ipynb` | Recency weighting | `postprocessor/node_recency.go` | ✅ | [x] `examples/postprocessor/recency_postprocessor/` |
| `node_postprocessor/OptimizerDemo.ipynb` | Sentence optimizer | `postprocessor/optimizer.go` | ✅ | [x] `examples/postprocessor/sentence_optimizer/` |

**Go Example:** `examples/postprocessor/` (covers llm_rerank, long_context_reorder, metadata_replacement, pii_masking, recency_postprocessor, sentence_optimizer)

---

## Priority 9: Workflow Examples ✅

| Python Example | Description | Go Components Required | Status | Implemented |
|----------------|-------------|------------------------|--------|-------------|
| `workflow/rag.ipynb` | RAG workflow | `workflow` | ✅ | [x] `examples/workflow/rag_workflow/` |
| `workflow/reflection.ipynb` | Reflection workflow | `workflow` | ✅ | [x] `examples/workflow/reflection_workflow/` |
| `workflow/parallel_execution.ipynb` | Parallel execution | `workflow` | ✅ | [x] `examples/workflow/parallel_workflow/` |
| `workflow/router_query_engine.ipynb` | Router workflow | `workflow`, `rag/queryengine` | ✅ | [x] `examples/workflow/router_workflow/` |
| `workflow/sub_question_query_engine.ipynb` | Sub-question workflow | `workflow`, `questiongen` | ✅ | [x] `examples/workflow/sub_question_workflow/` |
| `workflow/checkpointing_workflows.ipynb` | Checkpointing | `workflow` | ✅ | [x] `examples/workflow/checkpointing/` |

**Go Example:** `examples/workflow/` (covers rag_workflow, reflection_workflow, parallel_workflow, router_workflow, sub_question_workflow, checkpointing)

---

## Priority 10: Index Examples ✅

| Python Example | Description | Go Components Required | Status | Implemented |
|----------------|-------------|------------------------|--------|-------------|
| `index_structs/knowledge_graph/KnowledgeGraphDemo.ipynb` | KG index | `index/knowledge_graph.go` | ✅ | [x] `examples/index/knowledge_graph/` |
| `index_structs/doc_summary/` | Summary index | `index/summary.go` | ✅ | [x] `examples/index/summary_index/` |

**Go Example:** `examples/index/` (covers knowledge_graph, summary_index)

---

## Priority 11: Ingestion Pipeline Examples ✅

| Python Example | Description | Go Components Required | Status | Implemented |
|----------------|-------------|------------------------|--------|-------------|
| `ingestion/advanced_ingestion_pipeline.ipynb` | Advanced ingestion | `ingestion/pipeline.go` | ✅ | [x] `examples/rag/ingestion/` |
| `ingestion/document_management_pipeline.ipynb` | Document management | `ingestion`, `storage` | ✅ | [x] `examples/rag/ingestion/document_management/` |

**Go Example:** `examples/rag/ingestion/` (covers advanced_ingestion_pipeline, document_management_pipeline)

---

## Priority 12: Prompt Examples ✅

| Python Example | Description | Go Components Required | Status | Implemented |
|----------------|-------------|------------------------|--------|-------------|
| `prompts/advanced_prompts.ipynb` | Advanced prompts | `prompts/template.go` | ✅ | [x] `examples/prompts/advanced_prompts/` |
| `prompts/prompt_mixin.ipynb` | Prompt mixin | `prompts/mixin.go` | ✅ | [x] `examples/prompts/prompt_mixin/` |
| `prompts/rich_prompt_template_features.ipynb` | Template features | `prompts` | ✅ | [x] `examples/prompts/template_features/` |

**Go Example:** `examples/prompts/` (covers advanced_prompts, prompt_mixin, template_features)

---

## Priority 13: Metadata Extraction Examples ✅

| Python Example | Description | Go Components Required | Status | Implemented |
|----------------|-------------|------------------------|--------|-------------|
| `metadata_extraction/MetadataExtractionSEC.ipynb` | SEC metadata | `extractors` | ✅ | [x] `examples/extractors/metadata_extraction/` |
| `metadata_extraction/MetadataExtraction_LLMSurvey.ipynb` | LLM survey | `extractors` | ✅ | [x] `examples/extractors/metadata_extraction/` |
| `metadata_extraction/EntityExtractionClimate.ipynb` | Entity extraction | `extractors` | ✅ | [x] `examples/extractors/entity_extraction/` |

**Go Example:** `examples/extractors/` (covers metadata_extraction, entity_extraction)

---

## Priority 14: Output Parsing Examples ✅

| Python Example | Description | Go Components Required | Status | Implemented |
|----------------|-------------|------------------------|--------|-------------|
| `output_parsing/function_program.ipynb` | Function program | `program/function_program.go` | ✅ | [x] `examples/program/function_program/` |
| `output_parsing/llm_program.ipynb` | LLM program | `program/llm_program.go` | ✅ | [x] `examples/program/llm_program/` |

**Go Example:** `examples/program/` (covers function_program, llm_program)

---

## Priority 15: Tool Examples ✅

| Python Example | Description | Go Components Required | Status | Implemented |
|----------------|-------------|------------------------|--------|-------------|
| `tools/function_tool_callback.ipynb` | Function tools | `tools/function_tool.go` | ✅ | [x] `examples/tools/function_tool/` |
| `tools/eval_query_engine_tool.ipynb` | Query engine tool | `tools/query_engine_tool.go` | ✅ | [x] `examples/tools/query_engine_tool/` |

**Go Example:** `examples/tools/` (covers function_tool, query_engine_tool)

---

## Priority 16: Streaming Examples ✅

| Python Example | Description | Go Components Required | Status | Implemented |
|----------------|-------------|------------------------|--------|-------------|
| `customization/streaming/SimpleIndexDemo-streaming.ipynb` | Basic streaming | `llm`, `rag/synthesizer` | ✅ | [x] `examples/streaming/basic_streaming/` |
| `customization/streaming/chat_engine_condense_question_stream_response.ipynb` | Chat streaming | `chatengine` | ✅ | [x] `examples/streaming/chat_streaming/` |

**Go Example:** `examples/streaming/` (covers basic_streaming, chat_streaming)

---

## Priority 17: LLM Provider Examples ✅

| Python Example | Description | Go Components Required | Status | Implemented |
|----------------|-------------|------------------------|--------|-------------|
| `customization/llms/AzureOpenAI.ipynb` | Azure OpenAI | `llm/azure_openai.go` | ✅ | [x] `examples/llm/azure_openai/` |
| `customization/llms/SimpleIndexDemo-ChatGPT.ipynb` | OpenAI ChatGPT | `llm/openai.go` | ✅ | [x] `examples/llm/openai_llm/` |

**Go Example:** `examples/llm/` (covers openai_llm, azure_openai)

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
