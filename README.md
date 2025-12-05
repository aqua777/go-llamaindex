# `go-llamaindex`

**⚠️ EXPERIMENTAL AI-GENERATED CODE ⚠️**

This is an AI-generated conversion of [LlamaIndex](https://github.com/run-llama/llamaindex) concepts to Go. It has not been thoroughly tested, audited, or optimized for production use. Use at your own risk and thoroughly test any functionality before deploying in production environments.

---

## Overview

`go-llamaindex` is a comprehensive Go implementation of the LlamaIndex framework for building LLM-powered applications. It provides a complete toolkit for document processing, embedding generation, vector storage, retrieval, agents, workflows, and evaluation—all in idiomatic Go.

## Installation

```bash
go get github.com/aqua777/go-llamaindex
```

## Examples

For runnable examples covering all features, see the **[examples/](examples/)** directory with 60+ examples across 16 categories including RAG pipelines, agents, workflows, evaluation, and more.

---

## Implemented Components

### Core Schema & Base Types

**Package:** `schema/`

- **Node System** — Node relationships (`SOURCE`, `PREVIOUS`, `NEXT`, `PARENT`, `CHILD`), `RelatedNodeInfo`, SHA256-based hashing
- **MetadataMode** — Modes: `ALL`, `EMBED`, `LLM`, `NONE` with exclusion key support
- **MediaResource** — Fields: `Data`, `Text`, `Path`, `URL`, `MimeType`, `Embeddings`
- **ImageNode** — Image data (base64, path, URL)
- **IndexNode** — `IndexID` field for recursive retrieval
- **BaseComponent** — `ToJSON()`, `FromJSON()`, `ToDict()`, `FromDict()`, `ClassName()`
- **TransformComponent** — `Transform(nodes []Node) []Node`

---

### LLM Interface & Providers

**Package:** `llm/`

- **LLM Interface** — `Complete()`, `Chat()`, `Stream()`
- **LLMMetadata** — `ContextWindow`, `NumOutputTokens`, `IsChat`, `IsFunctionCalling`, `IsMultiModal`
- **ChatMessage Types** — `MessageRole` (system, user, assistant, tool), `ContentBlock` (text, image, tool call, tool result), multi-modal support
- **Tool Calling** — `ToolCall`, `ToolResult`, `ToolMetadata`, `LLMWithToolCalling` interface, `ToolChoice` enum
- **Structured Output** — `ResponseFormat` with `json_object` and `json_schema` types, `LLMWithStructuredOutput` interface

**Providers:**
- OpenAI
- Anthropic
- Ollama
- Cohere
- Azure OpenAI
- Mistral AI
- Groq
- DeepSeek
- AWS Bedrock

---

### Embedding Interface & Providers

**Package:** `embedding/`

- **EmbeddingModel Interface** — `GetTextEmbedding()`, `GetQueryEmbedding()`, `GetTextEmbeddingsBatch()`
- **EmbeddingInfo** — `Dimensions`, `MaxTokens`, `TokenizerName`, `IsMultiModal`
- **Similarity Functions** — `CosineSimilarity`, `EuclideanDistance`, `DotProduct`, `TopKSimilar`, normalization utilities
- **MultiModal Embedding** — `MultiModalEmbeddingModel` interface with `GetImageEmbedding()`
- **Sparse Embeddings** — `SparseEmbedding`, BM25 and BM25Plus models, `HybridEmbeddingModel` interface

**Providers:**
- OpenAI
- Ollama
- Cohere
- HuggingFace
- Azure OpenAI
- AWS Bedrock

---

### Text Processing

**Packages:** `textsplitter/`, `nodeparser/`, `validation/`

**Text Splitters:**
- **SentenceSplitter** — Sentence-aware splitting with metadata-conscious mode
- **TokenTextSplitter** — Token-based splitting with custom tokenizer support
- **MarkdownSplitter** — Preserves code blocks, splits by headers
- **SentenceWindowSplitter** — Configurable context window around sentences

**Tokenization:**
- **TikToken Integration** — `cl100k_base`, `p50k_base`, `r50k_base`, `o200k_base` encodings

**Node Parsers:**
- **SentenceNodeParser** — Wraps `SentenceSplitter` with event callbacks
- **SimpleNodeParser** — One node per document

**Validation:**
- `RequirePositive()`, `RequireNonNegative()`, `RequireNotEmpty()`
- Splitter-specific validation functions

---

### Storage Layer

**Package:** `storage/`

**Key-Value Store:**
- `KVStore` interface with `Put`, `Get`, `Delete`, `GetAll`
- `SimpleKVStore` (in-memory) and `FileKVStore` (file-based)

**Document Store:**
- `DocStore` interface with document management and hash tracking
- `KVDocumentStore` and `SimpleDocumentStore` implementations

**Index Store:**
- `IndexStore` interface supporting VectorStore, List, KeywordTable, Tree, KG types
- `KVIndexStore` and `SimpleIndexStore` implementations

**Chat Store:**
- `ChatStore` interface for conversation history
- `SimpleChatStore` implementation

**Vector Store:**
- `VectorStore` interface with `Add()` and `Query()`
- Query modes: `Default`, `Sparse`, `Hybrid`, `MMR`
- Filter operators: `EQ`, `GT`, `LT`, `NE`, `IN`, `NIN`, `TEXT_MATCH`, `CONTAINS`, etc.
- Implementations: `SimpleVectorStore` (in-memory), `ChromemStore` (persistent)

**Storage Context:**
- Combines `DocStore`, `IndexStore`, `VectorStores`
- Persistence and JSON serialization support

---

### Prompt System

**Package:** `prompts/`

- **PromptTemplate** — Template string with `{variable}` placeholders, `Format()`, `PartialFormat()`
- **ChatPromptTemplate** — System/user/assistant message templates
- **PromptType Enum** — `Summary`, `QuestionAnswer`, `Refine`, `TreeInsert`, `TreeSelect`, `KeywordExtract`
- **PromptMixin Interface** — `GetPrompts()`, `UpdatePrompts()`
- **Default Prompts** — `DefaultSummaryPrompt`, `DefaultTextQAPrompt`, `DefaultRefinePrompt`, etc.

---

### Retrieval System

**Package:** `rag/retriever/`

- **Retriever Interface** — `Retrieve(ctx, query) ([]NodeWithScore, error)`
- **VectorRetriever** — Vector store queries with embedding support
- **FusionRetriever** — Combines retrievers with `ReciprocalRank`, `RelativeScore`, `DistBasedScore`, `Simple` modes
- **AutoMergingRetriever** — Merges child nodes into parents with configurable threshold
- **RouterRetriever** — Routes queries via `Selector` interface

---

### Response Synthesis

**Package:** `rag/synthesizer/`

- **Synthesizer Interface** — `Synthesize()`, `GetResponse()`
- **SimpleSynthesizer** — Single LLM call with merged chunks
- **RefineSynthesizer** — Iterative refinement across chunks
- **CompactAndRefineSynthesizer** — Compacts before refining
- **TreeSummarizeSynthesizer** — Recursive bottom-up summarization
- **AccumulateSynthesizer** — Per-chunk responses concatenated
- **ResponseMode Enum** — `Refine`, `Compact`, `SimpleSummarize`, `TreeSummarize`, `Accumulate`

---

### Query Engine

**Package:** `rag/queryengine/`

- **QueryEngine Interface** — `Query(ctx, query) (*Response, error)`
- **RetrieverQueryEngine** — Combines retriever and synthesizer
- **SubQuestionQueryEngine** — Decomposes complex queries
- **RouterQueryEngine** — Routes to appropriate engines
- **RetryQueryEngine** — Retries on failure
- **TransformQueryEngine** — Query transformation with `IdentityTransform`, `HyDETransform`

---

### Index Abstractions

**Package:** `index/`

- **BaseIndex Interface** — `AsRetriever()`, `AsQueryEngine()`, `InsertNodes()`, `DeleteNodes()`, `RefreshDocuments()`
- **VectorStoreIndex** — Embedding generation and batch insertion
- **SummaryIndex** (ListIndex) — List structure with Default/Embedding/LLM retriever modes
- **KeywordTableIndex** — Keyword extraction with stop word removal
- **TreeIndex** — Hierarchical summarization with `TreeAllLeafRetriever`, `TreeRootRetriever`, `TreeSelectLeafRetriever`
- **KnowledgeGraphIndex** — Triplet extraction with keyword/embedding/hybrid retrieval modes

---

### Tools & Function Calling

**Package:** `tools/`

- **Tool Interface** — `Call()`, `Metadata()`
- **ToolMetadata** — `Name`, `Description`, `Parameters` (JSON Schema), OpenAI conversions
- **FunctionTool** — Automatic schema generation from function signatures
- **QueryEngineTool** — Wraps query engine as tool
- **RetrieverTool** — Wraps retriever as tool

---

### Memory System

**Package:** `memory/`

- **Memory Interface** — `Get`, `GetAll`, `Put`, `PutMessages`, `Set`, `Reset`
- **SimpleMemory** — Stores all messages
- **ChatMemoryBuffer** — Fixed-size buffer with token limit
- **ChatSummaryMemoryBuffer** — LLM-based summarization of older messages
- **VectorMemory** — Vector-based memory retrieval

---

### Chat Engine

**Package:** `chatengine/`

- **ChatEngine Interface** — `Chat`, `ChatWithHistory`, `StreamChat`, `Reset`, `ChatHistory`
- **SimpleChatEngine** — Direct LLM chat
- **ContextChatEngine** — RAG-enhanced with retriever
- **CondensePlusContextChatEngine** — Query condensation + context retrieval

---

### Agent System

**Package:** `agent/`

- **Agent Interface** — `AgentState`, `AgentStep`, `ToolSelection`, `ToolCallResult`, `AgentOutput`
- **ReAct Agent** — Thought-action-observation loop
- **FunctionCallingReActAgent** — OpenAI function calling integration
- **Output Parser** — `ActionReasoningStep`, `ObservationReasoningStep`, `ResponseReasoningStep`
- **Formatter** — ReAct chat formatter with system templates

---

### Evaluation Framework

**Package:** `evaluation/`

- **Evaluator Interface** — `EvaluationResult`, `EvaluateInput`, `EvaluatorRegistry`
- **FaithfulnessEvaluator** — Checks response support by context
- **RelevancyEvaluator** — Context and answer relevancy
- **CorrectnessEvaluator** — 1-5 scoring with reference comparison
- **SemanticSimilarityEvaluator** — Cosine, dot product, euclidean similarity
- **BatchEvalRunner** — Concurrent evaluation

---

### Callbacks & Instrumentation

**Package:** `callbacks/`

- **CBEventType Enum** — `Chunking`, `NodeParsing`, `Embedding`, `LLM`, `Query`, `Retrieve`, `Synthesize`, `Tree`, `SubQuestion`, `FunctionCall`, `Reranking`, `AgentStep`
- **CallbackHandler Interface** — `OnEventStart`, `OnEventEnd`, `StartTrace`, `EndTrace`
- **CallbackManager** — Thread-safe event dispatch
- **Handlers:** `LoggingHandler`, `TokenCountingHandler`, `EventCollectorHandler`

---

### Document Readers

**Package:** `rag/reader/`

- **Reader Interface** — `LoadData()`, `LazyReader`, `FileReader`, `ReaderWithContext`
- **SimpleDirectoryReader** — Recursive traversal, extension filtering
- **JSONReader** — Object, array, JSONL support
- **HTMLReader** — Script/style removal, entity decoding, metadata extraction
- **MarkdownReader** — YAML frontmatter, header-based splitting
- **PDFReader** — PDF extraction via `ledongthuc/pdf`
- **CSVReader** — CSV/TSV with streaming support for large files
- **ExcelReader** — Multi-sheet support, column selection by name/index/letter
- **DocxReader** — Paragraphs, tables, document properties, optional image extraction

---

### Ingestion Pipeline

**Package:** `ingestion/`

- **IngestionPipeline** — Transformation chains via `TransformComponent`
- **Caching** — Document deduplication
- **DocstoreStrategy** — `UPSERTS`, `DUPLICATES_ONLY`, `UPSERTS_AND_DELETE`

---

### Postprocessors

**Package:** `postprocessor/`

- **SimilarityPostprocessor** — Filter by score threshold
- **KeywordPostprocessor** — Required/excluded keywords
- **MetadataReplacementPostprocessor** — Replace content with metadata
- **LongContextReorder** — Reorder for long context models
- **TopKPostprocessor** — Limit returned nodes
- **LLMRerank** — LLM-based reranking
- **RankGPTRerank** — Conversational ranking with sliding window
- **PIIPostprocessor** — Email, phone, SSN, credit card, IP masking
- **NodeRecencyPostprocessor** — Time-based weighting (linear, exponential, step)

---

### Metadata Extractors

**Package:** `extractors/`

- **MetadataExtractor Interface** — `BaseExtractor`, `LLMExtractor`, `ExtractorChain`
- **TitleExtractor**
- **SummaryExtractor**
- **KeywordsExtractor**
- **QuestionsAnsweredExtractor**

---

### Workflow System

**Package:** `workflow/`

- **Workflow Types** — `Workflow`, `Event`, `Context`, `StateStore`, `EventFactory`, `Handler`
- **Workflow Engine** — `Run`, `RunStream`, retry support
- **Step Decorators** — Logging, timing, conditional, fallback, chain, middleware
- **Common Events** — `Start`, `Stop`, `Error`, `InputRequired`, `HumanResponse`

---

### Structured Programs

**Package:** `program/`

- **Program Interface** — `OutputParser`, `JSONOutputParser`, `PydanticOutputParser`
- **FunctionProgram** — Function-based structured output via tool calling
- **LLMProgram** — LLM-based structured output with parsing

---

### Object Index

**Package:** `objects/`

- **ObjectNodeMapping Interface** — `BaseObjectNodeMapping`, `SimpleObjectNodeMapping`
- **ToolNodeMapping** — `ToolRetriever` for tool-based retrieval
- **TypedObjectNodeMapping** — Generic typed mapping

---

### Advanced Features

- **Selectors** (`selector/`) — `LLMSingleSelector`, `LLMMultiSelector`, `SelectionOutputParser`
- **Question Generation** (`questiongen/`) — `LLMQuestionGenerator` with few-shot prompts
- **Output Parsers** (`outputparser/`) — `JSONOutputParser`, `ListOutputParser`, `BooleanOutputParser`
- **Graph Store** (`graphstore/`) — `GraphStore` interface, `Triplet`, `EntityNode`, `Relation`, `SimpleGraphStore`

---

## Dependencies

- [go-openai](https://github.com/sashabaranov/go-openai) — OpenAI API client
- [chromem-go](https://github.com/philippgille/chromem-go) — Vector database
- [sentences](https://github.com/neurosnap/sentences) — Sentence tokenization
- [tiktoken-go](https://github.com/pkoukk/tiktoken-go) — Token counting
- [excelize](https://github.com/xuri/excelize) — Excel file support
- [ledongthuc/pdf](https://github.com/ledongthuc/pdf) — PDF extraction

---

## Limitations

As an AI-generated conversion, this implementation may have:

- Incomplete feature coverage compared to Python/TypeScript LlamaIndex
- Potential bugs or edge cases not yet discovered
- Performance characteristics not optimized
- Limited testing and validation

---

## Contributing

This is experimental code. Contributions are welcome but expect significant changes as the codebase matures.

## License

See individual dependency licenses.
