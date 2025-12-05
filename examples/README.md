# Go LlamaIndex Examples

This directory contains comprehensive examples demonstrating the capabilities of `go-llamaindex`. Each example is self-contained with a `main.go` file and category-level `README.md` documentation.

## Prerequisites

- Go 1.21+
- OpenAI API key (set `OPENAI_API_KEY` environment variable)
- For specific examples: Azure OpenAI, Cohere, or other provider credentials

## Running Examples

```bash
cd <example-directory>
go run main.go
```

---

## Table of Contents

| Category | Description | Examples |
|----------|-------------|----------|
| [RAG](#rag-examples) | Retrieval-Augmented Generation pipelines | 11 |
| [Agents](#agent-examples) | ReAct and function-calling agents | 5 |
| [Workflows](#workflow-examples) | Event-driven workflow patterns | 6 |
| [Query Engines](#query-engine-examples) | Query processing and routing | 5 |
| [Retrievers](#retriever-examples) | Document retrieval strategies | 4 |
| [Chat Engines](#chat-engine-examples) | Conversational AI with context | 1 |
| [Evaluation](#evaluation-examples) | RAG quality assessment | 4 |
| [Postprocessors](#postprocessor-examples) | Result filtering and reranking | 6 |
| [Index](#index-examples) | Document indexing strategies | 2 |
| [LLM](#llm-examples) | LLM provider integrations | 2 |
| [Memory](#memory-examples) | Conversation memory management | 3 |
| [Program](#program-examples) | Structured LLM output | 2 |
| [Prompts](#prompt-examples) | Prompt templates and engineering | 3 |
| [Tools](#tool-examples) | Agent tools and function calling | 2 |
| [Streaming](#streaming-examples) | Real-time response streaming | 2 |
| [Extractors](#extractor-examples) | Metadata and entity extraction | 2 |
| [Text Splitter](#text-splitter-examples) | Document chunking strategies | 1 |

---

## RAG Examples

Core Retrieval-Augmented Generation patterns.

| Example | Description | Path |
|---------|-------------|------|
| **Basic Pipeline** | End-to-end RAG with retrieval and synthesis | [`rag/basic_pipeline/`](rag/basic_pipeline/) |
| **Basic RAG** | Simple RAG implementation | [`rag/basic_rag/`](rag/basic_rag/) |
| **Vector Store** | Vector database operations | [`rag/vector_store/`](rag/vector_store/) |
| **Ingestion** | Document loading and processing | [`rag/ingestion/`](rag/ingestion/) |
| **Response Synthesis** | Refine, tree summarize, compact modes | [`rag/response_synthesis/`](rag/response_synthesis/) |
| **Pydantic Tree Summarize** | Structured output with tree summarization | [`rag/pydantic_tree_summarize/`](rag/pydantic_tree_summarize/) |
| **Router** | Query routing to specialized engines | [`rag/router/`](rag/router/) |
| **Fusion Retriever** | Multi-retriever fusion (RRF, score-based) | [`rag/fusion_retriever/`](rag/fusion_retriever/) |
| **Evaluation** | RAG pipeline quality metrics | [`rag/evaluation/`](rag/evaluation/) |
| **Metadata & Streaming** | Metadata handling with streaming | [`rag/metadata_and_streaming/`](rag/metadata_and_streaming/) |
| **System** | System-level RAG configuration | [`rag/system/`](rag/system/) |

📖 See [`rag/basic_pipeline/README.md`](rag/basic_pipeline/README.md) for detailed documentation.

---

## Agent Examples

Autonomous agents with reasoning and tool use.

| Example | Description | Path |
|---------|-------------|------|
| **ReAct Agent** | Reasoning + Acting agent with tools | [`agent/react_agent/`](agent/react_agent/) |
| **ReAct + Query Engine** | ReAct agent with RAG integration | [`agent/react_agent_with_query_engine/`](agent/react_agent_with_query_engine/) |
| **Function Calling Agent** | OpenAI function calling agent | [`agent/function_calling_agent/`](agent/function_calling_agent/) |
| **Agent Retrieval** | Agent with retriever tool | [`agent/agent_retrieval/`](agent/agent_retrieval/) |
| **Workflow Agent** | Agent as workflow component | [`agent/workflow_agent/`](agent/workflow_agent/) |

📖 See [`agent/README.md`](agent/README.md) for detailed documentation.

---

## Workflow Examples

Event-driven workflow orchestration patterns.

| Example | Description | Path |
|---------|-------------|------|
| **RAG Workflow** | RAG as event-driven workflow | [`workflow/rag_workflow/`](workflow/rag_workflow/) |
| **Router Workflow** | Query routing workflow | [`workflow/router_workflow/`](workflow/router_workflow/) |
| **Parallel Workflow** | Concurrent step execution | [`workflow/parallel_workflow/`](workflow/parallel_workflow/) |
| **Reflection Workflow** | Self-correcting workflow | [`workflow/reflection_workflow/`](workflow/reflection_workflow/) |
| **Sub-Question Workflow** | Query decomposition workflow | [`workflow/sub_question_workflow/`](workflow/sub_question_workflow/) |
| **Checkpointing** | Workflow state persistence | [`workflow/checkpointing/`](workflow/checkpointing/) |

📖 See [`workflow/README.md`](workflow/README.md) for detailed documentation.

---

## Query Engine Examples

Query processing, routing, and customization.

| Example | Description | Path |
|---------|-------------|------|
| **Sub-Question** | Query decomposition into sub-questions | [`queryengine/sub_question/`](queryengine/sub_question/) |
| **Retriever Router** | Route queries to specialized retrievers | [`queryengine/retriever_router/`](queryengine/retriever_router/) |
| **Custom Query Engine** | Build custom query engines | [`queryengine/custom_query_engine/`](queryengine/custom_query_engine/) |
| **Custom Retriever** | Implement custom retrievers | [`queryengine/custom_retriever/`](queryengine/custom_retriever/) |
| **Knowledge Graph** | Graph-based query engine | [`queryengine/knowledge_graph/`](queryengine/knowledge_graph/) |

📖 See [`queryengine/README.md`](queryengine/README.md) for detailed documentation.

---

## Retriever Examples

Document retrieval strategies and patterns.

| Example | Description | Path |
|---------|-------------|------|
| **Auto-Merging** | Hierarchical document merging | [`retrievers/auto_merging/`](retrievers/auto_merging/) |
| **Router** | Query-based retriever routing | [`retrievers/router/`](retrievers/router/) |
| **Composable** | Combine multiple retrievers | [`retrievers/composable/`](retrievers/composable/) |
| **BM25** | Sparse keyword-based retrieval | [`retrievers/bm25/`](retrievers/bm25/) |

📖 See [`retrievers/README.md`](retrievers/README.md) for detailed documentation.

---

## Chat Engine Examples

Conversational AI with memory and context.

| Example | Description | Path |
|---------|-------------|------|
| **Chat Engine** | All chat modes (simple, context, condense) | [`chatengine/`](chatengine/) |

📖 See [`chatengine/main.go`](chatengine/main.go) for implementation details.

---

## Evaluation Examples

RAG quality assessment and metrics.

| Example | Description | Path |
|---------|-------------|------|
| **Semantic Similarity** | Embedding-based evaluation | [`evaluation/semantic_similarity/`](evaluation/semantic_similarity/) |
| **Batch Evaluation** | Evaluate multiple queries | [`evaluation/batch_eval/`](evaluation/batch_eval/) |
| **Question Generation** | Generate evaluation questions | [`evaluation/question_generation/`](evaluation/question_generation/) |
| **Retry Query** | Query retry with feedback | [`evaluation/retry_query/`](evaluation/retry_query/) |

📖 See [`evaluation/README.md`](evaluation/README.md) for detailed documentation.

---

## Postprocessor Examples

Result filtering, reranking, and transformation.

| Example | Description | Path |
|---------|-------------|------|
| **LLM Rerank** | LLM-based result reranking | [`postprocessor/llm_rerank/`](postprocessor/llm_rerank/) |
| **Long Context Reorder** | Optimize for long context windows | [`postprocessor/long_context_reorder/`](postprocessor/long_context_reorder/) |
| **Metadata Replacement** | Replace node content with metadata | [`postprocessor/metadata_replacement/`](postprocessor/metadata_replacement/) |
| **PII Masking** | Mask personally identifiable information | [`postprocessor/pii_masking/`](postprocessor/pii_masking/) |
| **Recency Postprocessor** | Prioritize recent documents | [`postprocessor/recency_postprocessor/`](postprocessor/recency_postprocessor/) |
| **Sentence Optimizer** | Optimize sentence-level retrieval | [`postprocessor/sentence_optimizer/`](postprocessor/sentence_optimizer/) |

📖 See [`postprocessor/README.md`](postprocessor/README.md) for detailed documentation.

---

## Index Examples

Document indexing and organization strategies.

| Example | Description | Path |
|---------|-------------|------|
| **Summary Index** | Hierarchical summarization index | [`index/summary_index/`](index/summary_index/) |
| **Knowledge Graph** | Graph-based document index | [`index/knowledge_graph/`](index/knowledge_graph/) |

📖 See [`index/README.md`](index/README.md) for detailed documentation.

---

## LLM Examples

LLM provider integrations and configurations.

| Example | Description | Path |
|---------|-------------|------|
| **OpenAI LLM** | OpenAI GPT models | [`llm/openai_llm/`](llm/openai_llm/) |
| **Azure OpenAI** | Azure-hosted OpenAI models | [`llm/azure_openai/`](llm/azure_openai/) |

📖 See [`llm/README.md`](llm/README.md) for detailed documentation.

---

## Memory Examples

Conversation memory and context management.

| Example | Description | Path |
|---------|-------------|------|
| **Basic Memory** | Simple conversation buffer | [`memory/basic_memory/`](memory/basic_memory/) |
| **Chat Summary Memory** | Summarized conversation history | [`memory/chat_summary_memory/`](memory/chat_summary_memory/) |
| **Custom Memory** | Implement custom memory backends | [`memory/custom_memory/`](memory/custom_memory/) |

📖 See [`memory/README.md`](memory/README.md) for detailed documentation.

---

## Program Examples

Structured LLM output with type safety.

| Example | Description | Path |
|---------|-------------|------|
| **LLM Program** | Text-based structured output | [`program/llm_program/`](program/llm_program/) |
| **Function Program** | Function calling structured output | [`program/function_program/`](program/function_program/) |

📖 See [`program/README.md`](program/README.md) for detailed documentation.

---

## Prompt Examples

Prompt templates and engineering patterns.

| Example | Description | Path |
|---------|-------------|------|
| **Template Features** | Prompt template variables and formatting | [`prompts/template_features/`](prompts/template_features/) |
| **Advanced Prompts** | Complex prompt patterns | [`prompts/advanced_prompts/`](prompts/advanced_prompts/) |
| **Prompt Mixin** | Composable prompt components | [`prompts/prompt_mixin/`](prompts/prompt_mixin/) |

📖 See [`prompts/README.md`](prompts/README.md) for detailed documentation.

---

## Tool Examples

Agent tools and function definitions.

| Example | Description | Path |
|---------|-------------|------|
| **Function Tool** | Define tools from Go functions | [`tools/function_tool/`](tools/function_tool/) |
| **Query Engine Tool** | Wrap query engines as tools | [`tools/query_engine_tool/`](tools/query_engine_tool/) |

📖 See [`tools/README.md`](tools/README.md) for detailed documentation.

---

## Streaming Examples

Real-time response streaming.

| Example | Description | Path |
|---------|-------------|------|
| **Basic Streaming** | Stream LLM responses | [`streaming/basic_streaming/`](streaming/basic_streaming/) |
| **Chat Streaming** | Stream chat completions | [`streaming/chat_streaming/`](streaming/chat_streaming/) |

📖 See [`streaming/README.md`](streaming/README.md) for detailed documentation.

---

## Extractor Examples

Metadata and entity extraction from documents.

| Example | Description | Path |
|---------|-------------|------|
| **Metadata Extraction** | Extract document metadata | [`extractors/metadata_extraction/`](extractors/metadata_extraction/) |
| **Entity Extraction** | Extract named entities | [`extractors/entity_extraction/`](extractors/entity_extraction/) |

📖 See [`extractors/README.md`](extractors/README.md) for detailed documentation.

---

## Text Splitter Examples

Document chunking and splitting strategies.

| Example | Description | Path |
|---------|-------------|------|
| **Sentence Splitter** | Split by sentence boundaries | [`textsplitter/sentence-splitter/`](textsplitter/sentence-splitter/) |

📖 See [`textsplitter/README.md`](textsplitter/README.md) for detailed documentation.

---

## Quick Start

1. **Set up environment:**
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

2. **Run a basic example:**
   ```bash
   cd rag/basic_pipeline
   go run main.go
   ```

3. **Explore more examples:**
   - Start with `rag/` for core RAG patterns
   - Try `agent/` for autonomous agents
   - Check `workflow/` for complex orchestration

## Architecture Overview

```
examples/
├── rag/                 # Core RAG patterns
├── agent/               # Autonomous agents
├── workflow/            # Event-driven workflows
├── queryengine/         # Query processing
├── retrievers/          # Retrieval strategies
├── chatengine/          # Conversational AI
├── evaluation/          # Quality metrics
├── postprocessor/       # Result processing
├── index/               # Document indexing
├── llm/                 # LLM providers
├── memory/              # Conversation memory
├── program/             # Structured output
├── prompts/             # Prompt engineering
├── tools/               # Agent tools
├── streaming/           # Real-time streaming
├── extractors/          # Metadata extraction
└── textsplitter/        # Document chunking
```

## Contributing

When adding new examples:
1. Create a new directory under the appropriate category
2. Include a `main.go` with runnable code
3. Update the category `README.md` if it exists
4. Add an entry to this file's table of contents
