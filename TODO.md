# LlamaIndex Go Implementation TODO

This document tracks the implementation status of the Go LlamaIndex port. Components are organized into two sections: what still needs to be built, and what's already done.

---

# To Be Implemented

## Provider Integrations (Major Gap)

The Go implementation has ~5% coverage of Python/TypeScript provider ecosystem. This is the primary gap.

### LLM Providers Needed

Python has 100+ LLM integrations. Go currently has 9 (OpenAI, Anthropic, Ollama, Cohere, Azure OpenAI, Mistral AI, Groq, DeepSeek, AWS Bedrock).

**High Priority:**
- [ ] **Google Vertex AI / Gemini** - `llm/google.go`
  - Python: `llama-index-integrations/llms/llama-index-llms-gemini/`
  - Python: `llama-index-integrations/llms/llama-index-llms-google-genai/`
  - TypeScript: `packages/providers/google/`
- [x] **AWS Bedrock** - `llm/bedrock.go` ✅
  - Python: `llama-index-integrations/llms/llama-index-llms-bedrock/`
  - Python: `llama-index-integrations/llms/llama-index-llms-bedrock-converse/`
  - TypeScript: `packages/providers/aws/`
- [x] **Mistral AI** - `llm/mistral.go` ✅
  - Python: `llama-index-integrations/llms/llama-index-llms-mistralai/`
  - TypeScript: `packages/providers/mistral/`
- [ ] **Together AI** - `llm/together.go`
  - Python: `llama-index-integrations/llms/llama-index-llms-together/`
  - TypeScript: `packages/providers/together/`
- [x] **Groq** - `llm/groq.go` ✅
  - Python: `llama-index-integrations/llms/llama-index-llms-groq/`
  - TypeScript: `packages/providers/groq/`
- [ ] **Fireworks AI** - `llm/fireworks.go`
  - Python: `llama-index-integrations/llms/llama-index-llms-fireworks/`
  - TypeScript: `packages/providers/fireworks/`

**Medium Priority:**
- [ ] **Replicate** - `llm/replicate.go`
  - Python: `llama-index-integrations/llms/llama-index-llms-replicate/`
  - TypeScript: `packages/providers/replicate/`
- [ ] **Perplexity** - `llm/perplexity.go`
  - Python: `llama-index-integrations/llms/llama-index-llms-perplexity/`
  - TypeScript: `packages/providers/perplexity/`
- [x] **DeepSeek** - `llm/deepseek.go` ✅
  - Python: `llama-index-integrations/llms/llama-index-llms-deepseek/`
  - TypeScript: `packages/providers/deepseek/`
- [ ] **vLLM** - `llm/vllm.go`
  - Python: `llama-index-integrations/llms/llama-index-llms-vllm/`
  - TypeScript: `packages/providers/vllm/`
- [ ] **LiteLLM** - `llm/litellm.go`
  - Python: `llama-index-integrations/llms/llama-index-llms-litellm/`

### Embedding Providers Needed

Python has 70+ embedding integrations. Go currently has 6 (OpenAI, Ollama, Cohere, HuggingFace, Azure OpenAI, AWS Bedrock).

**High Priority:**
- [ ] **Google Vertex AI Embeddings** - `embedding/google.go`
- [x] **AWS Bedrock Embeddings** - `llm/bedrock/embedding.go` ✅
- [ ] **Voyage AI** - `embedding/voyage.go`
- [ ] **Jina AI** - `embedding/jina.go`
- [ ] **Mixedbread** - `embedding/mixedbread.go`

### Vector Store Integrations Needed

Python has 50+ vector store integrations. Go currently has 2 (Chromem-go, Simple in-memory).

**Critical Priority:**
- [ ] **Pinecone** - `rag/store/pinecone/`
- [ ] **Weaviate** - `rag/store/weaviate/`
- [ ] **Qdrant** - `rag/store/qdrant/`
- [ ] **Milvus** - `rag/store/milvus/`
- [ ] **pgvector** - `rag/store/pgvector/`
- [ ] **Chroma** - `rag/store/chroma/`

**Medium Priority:**
- [ ] **Redis** - `rag/store/redis/`
- [ ] **Elasticsearch** - `rag/store/elasticsearch/`
- [ ] **MongoDB Atlas** - `rag/store/mongodb/`
- [ ] **LanceDB** - `rag/store/lancedb/`
- [ ] **Supabase** - `rag/store/supabase/`

### Document Readers Needed

Python has 100+ reader integrations. Go currently has 5 (Directory, JSON, HTML, Markdown, PDF).

**High Priority:**
- [ ] **CSV/Excel Reader** - `rag/reader/csv_reader.go`, `rag/reader/excel_reader.go`
- [ ] **Docx Reader** - `rag/reader/docx_reader.go`
- [ ] **Notion Reader** - `rag/reader/notion_reader.go`
- [ ] **Slack Reader** - `rag/reader/slack_reader.go`
- [ ] **S3 Reader** - `rag/reader/s3_reader.go`
- [ ] **Web/URL Reader** - `rag/reader/web_reader.go`

**Medium Priority:**
- [ ] **Database Readers** (PostgreSQL, MySQL) - `rag/reader/database_reader.go`
- [ ] **GitHub Reader** - `rag/reader/github_reader.go`
- [ ] **Confluence Reader** - `rag/reader/confluence_reader.go`
- [ ] **Google Drive Reader** - `rag/reader/gdrive_reader.go`

---

## Observability Integrations Needed

Go has basic logging handler only. Python has integrations with:

- [ ] **LangSmith** - `callbacks/langsmith.go`
- [ ] **Weights & Biases** - `callbacks/wandb.go`
- [ ] **Arize Phoenix** - `callbacks/arize.go`
- [ ] **OpenTelemetry** - `callbacks/otel.go`
- [ ] **Datadog** - `callbacks/datadog.go`

---

## Advanced Features (Nice-to-Have)

### Multi-Modal Enhancements
- [ ] **Vision LLM support** - Full image understanding in chat
- [ ] **Audio processing** - Speech-to-text integration
- [ ] **Video processing** - Frame extraction and analysis

### Voice Agents
- [ ] **Voice Agent Interface** - `voice/types.go`
- [ ] **Real-time conversation** - WebSocket-based voice chat
- Python Reference: `voice_agents/`

### LlamaPacks
- [ ] **Pack System** - Pre-built RAG templates
- [ ] **Pack Registry** - Download and use community packs

### Additional Index Types
- [ ] **DocumentSummaryIndex** - Per-document summaries for retrieval
- [ ] **PropertyGraphIndex** - Property graph-based retrieval

---

## Production Hardening

### Testing & CI/CD
- [ ] **CI/CD Pipeline** - GitHub Actions for automated testing
- [ ] **Integration Tests** - Tests with real providers (API key gated)
- [ ] **Benchmarks** - Performance benchmarks vs Python/TypeScript
- [ ] **Load Testing** - Concurrent request handling

### Documentation
- [ ] **API Documentation Site** - GoDoc-based documentation
- [ ] **Migration Guide** - From Python/TypeScript to Go
- [ ] **Best Practices Guide** - Production deployment patterns

### Versioning
- [ ] **Semantic Versioning** - Tagged releases
- [ ] **Changelog** - Automated changelog generation
- [ ] **Deprecation Policy** - Clear deprecation timeline

---

# Already Implemented

## Core Schema & Base Types ✅

**Package:** `schema/`

- **Node System** - `schema/schema.go`, `schema/node_relationship.go`
  - Node relationships: `SOURCE`, `PREVIOUS`, `NEXT`, `PARENT`, `CHILD`
  - `RelatedNodeInfo` struct with `NodeID`, `NodeType`, `Metadata`, `Hash`
  - SHA256-based hash generation

- **MetadataMode** - `schema/metadata_mode.go`
  - Modes: `ALL`, `EMBED`, `LLM`, `NONE`
  - `ExcludedEmbedMetadataKeys`, `ExcludedLLMMetadataKeys` fields

- **MediaResource** - `schema/media_resource.go`
  - Fields: `Data`, `Text`, `Path`, `URL`, `MimeType`, `Embeddings`

- **ImageNode** - `schema/image_node.go`
  - Image data (base64, path, URL)

- **IndexNode** - `schema/index_node.go`
  - `IndexID` field for recursive retrieval

- **BaseComponent** - `schema/component.go`
  - `ToJSON()`, `FromJSON()`, `ToDict()`, `FromDict()` methods
  - `ClassName()` for type identification

- **TransformComponent** - `schema/component.go`
  - `Transform(nodes []Node) []Node` method

---

## LLM Interface & Providers ✅

**Package:** `llm/`

- **LLM Interface** - `llm/interface.go`
  - `Complete(ctx, prompt) (string, error)`
  - `Chat(ctx, messages) (string, error)`
  - `Stream(ctx, prompt) (<-chan string, error)`

- **LLMMetadata** - `llm/types.go`
  - `ContextWindow`, `NumOutputTokens`, `IsChat`, `IsFunctionCalling`
  - `ModelName`, `IsMultiModal`, `SystemRole`
  - Pre-defined metadata for GPT-3.5, GPT-4, GPT-4-Turbo, GPT-4o

- **ChatMessage Types** - `llm/types.go`
  - `MessageRole`: `system`, `user`, `assistant`, `tool`
  - `ContentBlock`: `TextBlock`, `ImageBlock`, `ToolCallBlock`, `ToolResultBlock`
  - Multi-modal message support with `NewMultiModalMessage()`

- **Tool Calling Support** - `llm/tool_types.go`
  - `ToolCall`, `ToolResult`, `ToolMetadata` types
  - `LLMWithToolCalling` interface with `ChatWithTools()`, `SupportsToolCalling()`
  - `ToolChoice` enum: `auto`, `none`, `required`
  - `ChatCompletionOptions` for fine-grained control

- **Structured Output** - `llm/tool_types.go`
  - `ResponseFormat` with `json_object` and `json_schema` types
  - `LLMWithStructuredOutput` interface with `ChatWithFormat()`

**Providers Implemented:**
- **OpenAI** - `llm/openai.go`
- **Anthropic** - `llm/anthropic.go`
- **Ollama** - `llm/ollama.go`
- **Cohere** - `llm/cohere.go`
- **Azure OpenAI** - `llm/azure_openai.go`
- **Mistral AI** - `llm/mistral.go`
- **Groq** - `llm/groq.go`
- **DeepSeek** - `llm/deepseek.go`
- **AWS Bedrock** - `llm/bedrock.go`

---

## Embedding Interface & Providers ✅

**Package:** `embedding/`

- **EmbeddingModel Interface** - `embedding/interface.go`
  - `GetTextEmbedding(ctx, text) ([]float64, error)`
  - `GetQueryEmbedding(ctx, query) ([]float64, error)`
  - `GetTextEmbeddingsBatch(ctx, texts, callback) ([][]float64, error)`

- **EmbeddingInfo** - `embedding/types.go`
  - `Dimensions`, `MaxTokens`, `TokenizerName`, `IsMultiModal`
  - Pre-defined info for OpenAI (ada, 3-small, 3-large) and Ollama models

- **Similarity Functions** - `embedding/similarity.go`
  - `CosineSimilarity`, `EuclideanDistance`, `EuclideanSimilarity`, `DotProduct`
  - `Normalize`, `Magnitude`, `Add`, `Subtract`, `Scale`, `Mean`
  - `TopKSimilar` for finding most similar vectors
  - `SimilarityType` enum: `cosine`, `euclidean`, `dot_product`

- **MultiModal Embedding** - `embedding/interface.go`
  - `MultiModalEmbeddingModel` interface with `GetImageEmbedding()`
  - `ImageType` supporting URL, Base64, and file path inputs

- **Sparse Embeddings** - `embedding/sparse.go`, `embedding/bm25.go`
  - `SparseEmbedding`, `SparseEmbeddingModel`, `HybridEmbeddingModel` interfaces
  - BM25 and BM25Plus sparse embedding models

**Providers Implemented:**
- **OpenAI** - `embedding/openai.go`
- **Ollama** - `embedding/ollama.go`
- **Cohere** - `embedding/cohere.go`
- **HuggingFace** - `embedding/huggingface.go`
- **Azure OpenAI** - `embedding/azure_openai.go`
- **AWS Bedrock** - `llm/bedrock/embedding.go`

---

## Text Processing ✅

**Package:** `textsplitter/`, `nodeparser/`, `validation/`

### Text Splitters

- **SentenceSplitter** - `textsplitter/sentence_splitter.go`
  - Sentence-aware splitting with configurable chunk size and overlap
  - `SplitTextMetadataAware()` for metadata-conscious splitting

- **TokenTextSplitter** - `textsplitter/token_splitter.go`
  - Token-based splitting with configurable chunk size and overlap
  - Custom tokenizer support (SimpleTokenizer, TikToken)

- **MarkdownSplitter** - `textsplitter/markdown_splitter.go`
  - Preserves code blocks (``` and ~~~)
  - Splits by headers while maintaining structure

- **SentenceWindowSplitter** - `textsplitter/sentence_window_splitter.go`
  - Configurable window size for context around sentences
  - `SplitTextWithWindows()` returns sentences with context

### Tokenization

- **TikToken Integration** - `textsplitter/tokenizer_tiktoken.go`
  - `TikTokenTokenizer` using `tiktoken-go` library
  - Encoding constants: `cl100k_base`, `p50k_base`, `r50k_base`, `o200k_base`
  - `GetEncodingForModel()` maps models to encodings

### Node Parsers

- **NodeParser Interface** - `nodeparser/interface.go`
  - `GetNodesFromDocuments(documents []Document) []*Node`
  - `ParseNodes(nodes []*Node) []*Node`

- **SentenceNodeParser** - `nodeparser/sentence_parser.go`
  - Wraps `SentenceSplitter` with node management
  - Event callbacks for progress tracking

- **SimpleNodeParser** - `nodeparser/sentence_parser.go`
  - Creates one node per document (no splitting)

### Validation

- **Validation Package** - `validation/validation.go`
  - `Validator` type for collecting errors
  - `RequirePositive()`, `RequireNonNegative()`, `RequireNotEmpty()`
  - `ValidateChunkParams()` for chunk_size/chunk_overlap validation

- **Splitter Validation** - `validation/splitter_validation.go`
  - `ValidateSentenceSplitterConfig()`, `ValidateTokenSplitterConfig()`
  - `ValidateMarkdownSplitterConfig()`, `ValidateSentenceWindowSplitterConfig()`

---

## Storage Layer ✅

**Package:** `storage/`

### Key-Value Store

- **KVStore Interface** - `storage/kvstore/interface.go`
  - `Put`, `Get`, `Delete`, `GetAll` methods
  - `PersistableKVStore` extends with `Persist(ctx, path) error`

- **SimpleKVStore** - In-memory implementation with optional persistence
- **FileKVStore** - File-based persistence (auto-persists on write)

### Document Store

- **DocStore Interface** - `storage/docstore/interface.go`
  - `Docs`, `AddDocuments`, `GetDocument`, `DeleteDocument`, `DocumentExists`
  - `GetRefDocInfo`, `GetAllRefDocInfo`, `DeleteRefDoc`
  - Hash methods: `SetDocumentHash`, `GetDocumentHash`, `GetAllDocumentHashes`

- **KVDocumentStore** - `storage/docstore/kv_docstore.go`
- **SimpleDocumentStore** - `storage/docstore/simple_docstore.go`

### Index Store

- **IndexStore Interface** - `storage/indexstore/interface.go`
  - `IndexStructs`, `AddIndexStruct`, `GetIndexStruct`, `DeleteIndexStruct`

- **IndexStruct** - Supports VectorStore, List, KeywordTable, Tree, KG types
- **KVIndexStore** - `storage/indexstore/kv_indexstore.go`
- **SimpleIndexStore** - `storage/indexstore/simple_indexstore.go`

### Chat Store

- **ChatStore Interface** - `storage/chatstore/interface.go`
  - `SetMessages`, `GetMessages`, `AddMessage`, `DeleteMessages`, `GetKeys`

- **SimpleChatStore** - `storage/chatstore/simple_chatstore.go`

### Vector Store

- **VectorStore Interface** - `rag/store/interface.go`
  - `Add(ctx, nodes) ([]string, error)`
  - `Query(ctx, query) ([]NodeWithScore, error)`

- **VectorStoreQueryMode** - `schema/schema.go`
  - `QueryModeDefault`, `QueryModeSparse`, `QueryModeHybrid`, `QueryModeMMR`

- **FilterOperator** - `schema/schema.go`
  - Basic: `EQ`, `GT`, `LT`, `NE`, `GTE`, `LTE`
  - Array: `IN`, `NIN`, `ANY`, `ALL`
  - Text: `TEXT_MATCH`, `TEXT_MATCH_INSENSITIVE`
  - Special: `CONTAINS`, `IS_EMPTY`

- **Implementations:**
  - `store.SimpleVectorStore` - In-memory store
  - `chromem.ChromemStore` - Persistent store using chromem-go

### Storage Context

- **StorageContext** - `storage/context.go`
  - Combines `DocStore`, `IndexStore`, `VectorStores`
  - `NewStorageContext()`, `StorageContextFromPersistDir()`
  - `Persist()`, `ToJSON()`, `StorageContextFromJSON()`

---

## Prompt System ✅

**Package:** `prompts/`

- **PromptTemplate** - `prompts/template.go`
  - Template string with `{variable}` placeholders
  - `Format()`, `FormatMessages()`, `PartialFormat()`

- **ChatPromptTemplate** - `prompts/template.go`
  - System, user, assistant message templates
  - `FormatMessages()` returns `[]llm.ChatMessage`

- **PromptType Enum** - `prompts/prompt_type.go`
  - `PromptTypeSummary`, `PromptTypeQuestionAnswer`, `PromptTypeRefine`
  - `PromptTypeTreeInsert`, `PromptTypeTreeSelect`, `PromptTypeKeywordExtract`

- **PromptMixin Interface** - `prompts/mixin.go`
  - `GetPrompts() PromptDictType`
  - `UpdatePrompts(prompts PromptDictType)`

- **Default Prompt Library** - `prompts/default_prompts.go`
  - `DefaultSummaryPrompt`, `DefaultTreeSummarizePrompt`
  - `DefaultTextQAPrompt`, `DefaultRefinePrompt`
  - `DefaultKeywordExtractPrompt`, `DefaultKGTripletExtractPrompt`

---

## Retrieval System ✅

**Package:** `rag/retriever/`

- **Retriever Interface** - `rag/retriever/interface.go`
  - `Retrieve(ctx, query) ([]NodeWithScore, error)`

- **BaseRetriever** - Full base implementation
  - `ObjectMap` for recursive retrieval
  - `HandleRecursiveRetrieval()` for IndexNode references

- **VectorRetriever** - `rag/retriever/vector.go`
  - Queries vector store with embeddings
  - Supports query modes (default, hybrid, MMR, etc.)

- **FusionRetriever** - `rag/retriever/fusion.go`
  - Combines multiple retrievers
  - Fusion modes: `ReciprocalRank`, `RelativeScore`, `DistBasedScore`, `Simple`

- **AutoMergingRetriever** - `rag/retriever/auto_merging.go`
  - Merges child nodes into parent nodes
  - `SimpleRatioThresh` controls merge threshold

- **RouterRetriever** - `rag/retriever/router.go`
  - Routes queries to appropriate retrievers
  - `Selector` interface for routing decisions

---

## Response Synthesis ✅

**Package:** `rag/synthesizer/`

- **Synthesizer Interface** - `rag/synthesizer/interface.go`
  - `Synthesize(ctx, query, nodes) (*Response, error)`
  - `GetResponse(ctx, query, textChunks) (string, error)`

- **SimpleSynthesizer** - `rag/synthesizer/simple.go`
  - Merges all chunks, single LLM call

- **RefineSynthesizer** - `rag/synthesizer/refine.go`
  - Iteratively refines response across chunks

- **CompactAndRefineSynthesizer** - `rag/synthesizer/compact.go`
  - Compacts chunks before refining

- **TreeSummarizeSynthesizer** - `rag/synthesizer/tree_summarize.go`
  - Recursive bottom-up summarization

- **AccumulateSynthesizer** - `rag/synthesizer/accumulate.go`
  - Generates response per chunk, concatenates

- **ResponseMode Enum** - `rag/synthesizer/response_mode.go`
  - `Refine`, `Compact`, `SimpleSummarize`, `TreeSummarize`, `Accumulate`

- **Response Types** - `rag/synthesizer/response.go`
  - `Response`, `StreamingResponse`, `ResponseType` interface

---

## Query Engine ✅

**Package:** `rag/queryengine/`

- **QueryEngine Interface** - `rag/queryengine/interface.go`
  - `Query(ctx, query) (*Response, error)`

- **RetrieverQueryEngine** - `rag/queryengine/interface.go`
  - Combines retriever and synthesizer

- **SubQuestionQueryEngine** - `rag/queryengine/sub_question.go`
  - Decomposes complex queries into sub-questions

- **RouterQueryEngine** - `rag/queryengine/router.go`
  - Routes queries to appropriate engines

- **RetryQueryEngine** - `rag/queryengine/retry.go`
  - Retries queries on failure

- **TransformQueryEngine** - `rag/queryengine/transform.go`
  - Transforms queries before execution
  - `IdentityTransform`, `HyDETransform` implementations

---

## Index Abstractions ✅

**Package:** `index/`

- **BaseIndex Interface** - `index/interface.go`
  - `AsRetriever()`, `AsQueryEngine()`
  - `InsertNodes()`, `DeleteNodes()`, `RefreshDocuments()`

- **VectorStoreIndex** - `index/vector_store.go`
  - Embedding generation and storage
  - Batch insertion with configurable batch size

- **SummaryIndex** (ListIndex) - `index/summary.go`
  - Stores nodes in a list structure
  - Multiple retriever modes: Default, Embedding, LLM

- **KeywordTableIndex** - `index/keyword_table.go`
  - SimpleKeywordExtractor with stop word removal
  - Keyword-to-node mapping

- **TreeIndex** - `index/tree.go`
  - Hierarchical summarization index
  - Bottom-up tree building
  - Tree retrievers: `TreeAllLeafRetriever`, `TreeRootRetriever`, `TreeSelectLeafRetriever`

- **KnowledgeGraphIndex** - `index/knowledge_graph.go`
  - Triplet extraction from documents
  - KGTableRetriever (keyword, embedding, hybrid modes)

---

## Tools & Function Calling ✅

**Package:** `tools/`

- **Tool Interface** - `tools/types.go`
  - `Call(ctx, input) (*ToolOutput, error)`
  - `Metadata() *ToolMetadata`

- **ToolMetadata** - `tools/types.go`
  - `Name`, `Description`, `Parameters` (JSON Schema)
  - `ToOpenAITool()`, `ToOpenAIFunction()` conversions

- **FunctionTool** - `tools/function_tool.go`
  - Automatic schema generation from function signature
  - Type conversion for function arguments

- **QueryEngineTool** - `tools/query_engine_tool.go`
  - Wraps query engine as tool

- **RetrieverTool** - `tools/retriever_tool.go`
  - Wraps retriever as tool

---

## Memory System ✅

**Package:** `memory/`

- **Memory Interface** - `memory/types.go`
  - `Get`, `GetAll`, `Put`, `PutMessages`, `Set`, `Reset`

- **SimpleMemory** - `memory/types.go`
  - Basic memory that stores all messages

- **ChatMemoryBuffer** - `memory/chat_memory_buffer.go`
  - Fixed-size message buffer with token limit enforcement

- **ChatSummaryMemoryBuffer** - `memory/chat_summary_memory_buffer.go`
  - Summarizes older messages using LLM

- **VectorMemory** - `memory/vector_memory.go`
  - Vector-based memory retrieval

---

## Chat Engine ✅

**Package:** `chatengine/`

- **ChatEngine Interface** - `chatengine/types.go`
  - `Chat`, `ChatWithHistory`, `StreamChat`, `Reset`, `ChatHistory`

- **SimpleChatEngine** - `chatengine/simple.go`
  - Direct LLM chat without knowledge base

- **ContextChatEngine** - `chatengine/context.go`
  - RAG-enhanced chat with retriever

- **CondensePlusContextChatEngine** - `chatengine/condense_plus_context.go`
  - Query condensation + context retrieval

---

## Agent System ✅

**Package:** `agent/`

- **Agent Interface** - `agent/types.go`
  - `AgentState`, `AgentStep`, `ToolSelection`, `ToolCallResult`, `AgentOutput`

- **ReAct Agent** - `agent/react.go`
  - Thought-action-observation loop
  - `FunctionCallingReActAgent`, `SimpleAgent`

- **Output Parser** - `agent/output_parser.go`
  - `ActionReasoningStep`, `ObservationReasoningStep`, `ResponseReasoningStep`

- **Formatter** - `agent/formatter.go`
  - ReAct chat formatter with system header templates

---

## Evaluation Framework ✅

**Package:** `evaluation/`

- **Evaluator Interface** - `evaluation/types.go`
  - `EvaluationResult`, `EvaluateInput`, `EvaluatorRegistry`

- **FaithfulnessEvaluator** - `evaluation/faithfulness.go`
  - Checks if response is supported by context

- **RelevancyEvaluator** - `evaluation/relevancy.go`
  - `ContextRelevancyEvaluator`, `AnswerRelevancyEvaluator`

- **CorrectnessEvaluator** - `evaluation/correctness.go`
  - Scores 1-5 with reference comparison

- **SemanticSimilarityEvaluator** - `evaluation/semantic_similarity.go`
  - Cosine, dot product, euclidean similarity

- **BatchEvalRunner** - `evaluation/batch_runner.go`
  - Concurrent evaluation support

---

## Callbacks & Instrumentation ✅

**Package:** `callbacks/`

- **CBEventType Enum** - `callbacks/schema.go`
  - `Chunking`, `NodeParsing`, `Embedding`, `LLM`, `Query`, `Retrieve`
  - `Synthesize`, `Tree`, `SubQuestion`, `FunctionCall`, `Reranking`, `AgentStep`

- **CallbackHandler Interface** - `callbacks/handler.go`
  - `OnEventStart`, `OnEventEnd`, `StartTrace`, `EndTrace`

- **CallbackManager** - `callbacks/manager.go`
  - Event dispatch system, thread-safe

- **Handler Implementations:**
  - `LoggingHandler` - Logs events with timestamps
  - `TokenCountingHandler` - Tracks token usage
  - `EventCollectorHandler` - Collects events for testing

---

## Document Readers ✅

**Package:** `rag/reader/`

- **Reader Interface** - `rag/reader/interface.go`
  - `LoadData() ([]Node, error)`
  - `LazyReader`, `FileReader`, `ReaderWithContext`

- **SimpleDirectoryReader** - `rag/reader/simple_directory_reader.go`
  - Recursive directory traversal, extension filtering

- **JSONReader** - `rag/reader/json_reader.go`
  - JSON object, array, and JSONL support

- **HTMLReader** - `rag/reader/html_reader.go`
  - Script/style removal, entity decoding, metadata extraction

- **MarkdownReader** - `rag/reader/markdown_reader.go`
  - YAML frontmatter, header-based splitting

- **PDFReader** - `rag/reader/pdf_reader.go`
  - PDF extraction using `ledongthuc/pdf` library

---

## Ingestion Pipeline ✅

**Package:** `ingestion/`

- **IngestionPipeline** - `ingestion/pipeline.go`
  - Chain of transformations via `TransformComponent`
  - Caching support, document deduplication
  - `DocstoreStrategy`: `UPSERTS`, `DUPLICATES_ONLY`, `UPSERTS_AND_DELETE`

- **IngestionCache** - `ingestion/cache.go`
  - In-memory cache with persistence support

---

## Postprocessors ✅

**Package:** `postprocessor/`

- **NodePostprocessor Interface** - `postprocessor/types.go`
  - `PostprocessNodes(ctx, nodes, queryBundle) ([]NodeWithScore, error)`

- **SimilarityPostprocessor** - `postprocessor/similarity.go`
  - Filter by similarity score threshold

- **KeywordPostprocessor** - `postprocessor/keyword.go`
  - Filter by required/excluded keywords

- **MetadataReplacementPostprocessor** - `postprocessor/metadata_replacement.go`
  - Replace node content with metadata value

- **LongContextReorder** - `postprocessor/long_context_reorder.go`
  - Reorder nodes for long context models

- **TopKPostprocessor** - `postprocessor/top_k.go`
  - Limit number of returned nodes

- **LLMRerank** - `postprocessor/llm_rerank.go`
  - LLM-based reranking with choice-select prompt

- **RankGPTRerank** - `postprocessor/rankgpt_rerank.go`
  - Conversational ranking, sliding window

- **PIIPostprocessor** - `postprocessor/pii.go`
  - Email, phone, SSN, credit card, IP detection/masking

- **NodeRecencyPostprocessor** - `postprocessor/node_recency.go`
  - Time-based weighting (linear, exponential, step)

---

## Metadata Extractors ✅

**Package:** `extractors/`

- **MetadataExtractor Interface** - `extractors/types.go`
  - `BaseExtractor`, `LLMExtractor`, `ExtractorChain`

- **TitleExtractor** - `extractors/title.go`
- **SummaryExtractor** - `extractors/summary.go`
- **KeywordsExtractor** - `extractors/keywords.go`
- **QuestionsAnsweredExtractor** - `extractors/questions.go`

---

## Workflow System ✅

**Package:** `workflow/`

- **Workflow Types** - `workflow/types.go`
  - `Workflow`, `Event`, `Context`, `StateStore`, `EventFactory`, `Handler`

- **Workflow Engine** - `workflow/workflow.go`
  - `Run`, `RunStream`, retry support

- **Step Decorators** - `workflow/decorators.go`
  - Logging, timing, conditional, fallback, chain, middleware

- **Common Events** - `workflow/events.go`
  - `Start`, `Stop`, `Error`, `InputRequired`, `HumanResponse`

---

## Structured Programs ✅

**Package:** `program/`

- **Program Interface** - `program/types.go`
  - `OutputParser`, `JSONOutputParser`, `PydanticOutputParser`

- **FunctionProgram** - `program/function_program.go`
  - Function-based structured output using LLM tool calling

- **LLMProgram** - `program/llm_program.go`
  - LLM-based structured output with parsing

---

## Object Index ✅

**Package:** `objects/`

- **ObjectNodeMapping Interface** - `objects/types.go`
  - `BaseObjectNodeMapping`, `SimpleObjectNodeMapping`

- **ToolNodeMapping** - `objects/tool_mapping.go`
  - `ToolRetriever` for tool-based retrieval

- **TypedObjectNodeMapping** - `objects/base_mapping.go`
  - Generic typed mapping and retrieval

---

## Advanced Features ✅

### Selectors - `selector/`
- `LLMSingleSelector`, `LLMMultiSelector`
- `SelectionOutputParser` for JSON parsing

### Question Generation - `questiongen/`
- `LLMQuestionGenerator` with few-shot prompts
- `SubQuestionOutputParser`

### Output Parsers - `outputparser/`
- `JSONOutputParser`, `ListOutputParser`, `BooleanOutputParser`

### Graph Store - `graphstore/`
- `GraphStore` interface, `Triplet`, `EntityNode`, `Relation` types
- `SimpleGraphStore` - In-memory with persistence

---

## References

- **Python llama-index-core**: `/python/llama-index-core/llama_index/core/`
- **TypeScript @llamaindex/core**: `/typescript/packages/core/src/`
- **Current Go Implementation**: `/golang/`
