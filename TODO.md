# LlamaIndex Go Implementation TODO

This document outlines the components needed to create a comprehensive Go implementation of LlamaIndex, organized from foundational to advanced components. Each section includes references to Python (`llama-index-core`) and TypeScript (`@llamaindex/core`) implementations.

## Implementation Status Legend

- ✅ **Implemented** - Component exists and is functional
- 🔄 **Partial** - Basic implementation exists, needs enhancement
- ❌ **Not Started** - Component needs to be implemented

---

## Phase 1: Core Schema & Base Types ✅

Foundation types that all other components depend on.

### 1.1 Node System Enhancement ✅

**Current State:** Fully implemented in `schema/` package

**Implemented:**

- [x] **Node Relationships** - `schema/node_relationship.go`
  - `SOURCE`, `PREVIOUS`, `NEXT`, `PARENT`, `CHILD` relationships
  - `RelatedNodeInfo` struct with `NodeID`, `NodeType`, `Metadata`, `Hash`

- [x] **MetadataMode** - `schema/metadata_mode.go`
  - Modes: `ALL`, `EMBED`, `LLM`, `NONE`
  - `ExcludedEmbedMetadataKeys`, `ExcludedLLMMetadataKeys` fields in `Node`

- [x] **Node Hash Generation** - `schema/schema.go:288-300`
  - SHA256-based hash of content + metadata

- [x] **MediaResource** - `schema/media_resource.go`
  - Fields: `Data`, `Text`, `Path`, `URL`, `MimeType`, `Embeddings`

- [x] **ImageNode** - `schema/image_node.go`
  - Image data (base64, path, URL)

- [x] **IndexNode** - `schema/index_node.go`
  - `IndexID` field for recursive retrieval

### 1.2 Base Component Interface ✅

- [x] **BaseComponent** - `schema/component.go:10-20`
  - `ToJSON()`, `FromJSON()`, `ToDict()`, `FromDict()` methods
  - `ClassName()` for type identification

- [x] **TransformComponent** - `schema/component.go:69-76`
  - `Transform(nodes []Node) []Node` method

---

## Phase 2: LLM & Embedding Interfaces

### 2.1 LLM Interface Enhancement ✅

**Current State:** Fully implemented in `llm/` package

**Implemented:**

- [x] **LLMMetadata** - `llm/types.go`
  - `ContextWindow`, `NumOutputTokens`, `IsChat`, `IsFunctionCalling`
  - `ModelName`, `IsMultiModal`, `SystemRole`
  - Pre-defined metadata for GPT-3.5, GPT-4, GPT-4-Turbo, GPT-4o

- [x] **ChatMessage Types** - `llm/types.go`
  - `MessageRole`: `system`, `user`, `assistant`, `tool`
  - `ContentBlock`: `TextBlock`, `ImageBlock`, `ToolCallBlock`, `ToolResultBlock`
  - Multi-modal message support with `NewMultiModalMessage()`

- [x] **Tool Calling Support** - `llm/tool_types.go`
  - `ToolCall`, `ToolResult`, `ToolMetadata` types
  - `LLMWithToolCalling` interface with `ChatWithTools()`, `SupportsToolCalling()`
  - `ToolChoice` enum: `auto`, `none`, `required`
  - `ChatCompletionOptions` for fine-grained control

- [x] **Structured Output** - `llm/tool_types.go`
  - `ResponseFormat` with `json_object` and `json_schema` types
  - `LLMWithStructuredOutput` interface with `ChatWithFormat()`

### 2.2 Embedding Interface Enhancement 🔄

**Current State:** Basic `EmbeddingModel` interface in `embedding/interface.go`

**Required Enhancements:**

- [ ] **EmbeddingInfo** - Model metadata
  - `Dimensions`, `MaxTokens`, `Tokenizer`
  - TypeScript: `embeddings/base.ts:EmbeddingInfo`

- [ ] **Batch Embedding** - Efficient bulk processing
  - `GetTextEmbeddingsBatch(texts []string) ([][]float64, error)`
  - Progress callback support
  - TypeScript: `embeddings/base.ts:101-111`

- [ ] **Similarity Functions** - Vector comparison utilities
  - Cosine, Euclidean, Dot Product similarity
  - TypeScript: `embeddings/utils.ts`

- [ ] **MultiModal Embedding** - Image embedding support
  - `GetImageEmbedding(image ImageType) ([]float64, error)`
  - Python: `embeddings/multi_modal_base.py`

---

## Phase 3: Text Processing

### 3.1 Node Parser Interface ❌

**Current State:** `TextSplitter` interface exists in `textsplitter/`

**Required Components:**

- [ ] **NodeParser Interface** - Base interface for all parsers
  - `ParseNodes(nodes []Node) []Node`
  - `IncludeMetadata`, `IncludePrevNextRel` options
  - Python: `node_parser/interface.py:50-74`
  - TypeScript: `node-parser/base.ts:12-121`

- [ ] **Post-Processing** - Node relationship establishment
  - Set `PREVIOUS`/`NEXT` relationships
  - Calculate `StartCharIdx`/`EndCharIdx`
  - Merge parent metadata
  - Python: `node_parser/interface.py:84-155`
  - TypeScript: `node-parser/base.ts:27-77`

### 3.2 Text Splitters 🔄

**Current State:** `SentenceSplitter` exists

**Additional Splitters Needed:**

- [ ] **TokenTextSplitter** - Token-based splitting
  - Respects token limits
  - TypeScript: `node-parser/token-text-splitter.ts`

- [ ] **MarkdownSplitter** - Markdown-aware splitting
  - Preserves headers, code blocks
  - TypeScript: `node-parser/markdown.ts`

- [ ] **SentenceWindowSplitter** - Context window around sentences
  - Configurable window size
  - TypeScript: `node-parser/sentence-window.ts`

- [ ] **MetadataAwareTextSplitter** - Metadata-conscious splitting
  - Accounts for metadata in chunk size
  - TypeScript: `node-parser/base.ts:139-171`

---

## Phase 4: Storage Layer

### 4.1 Key-Value Store ❌

- [ ] **KVStore Interface**
  - `Put(key string, value interface{}) error`
  - `Get(key string) (interface{}, error)`
  - `Delete(key string) error`
  - `GetAll() (map[string]interface{}, error)`
  - Python: `storage/kvstore/`
  - TypeScript: `storage/kv-store/`

- [ ] **SimpleKVStore** - In-memory implementation
- [ ] **FileKVStore** - File-based persistence

### 4.2 Document Store ❌

- [ ] **DocStore Interface**
  - `AddDocuments(docs []Document) error`
  - `GetDocument(docID string) (*Document, error)`
  - `DeleteDocument(docID string) error`
  - `DocumentExists(docID string) bool`
  - `GetRefDocInfo(refDocID string) (*RefDocInfo, error)`
  - Python: `storage/docstore/`
  - TypeScript: `storage/doc-store/`

- [ ] **SimpleDocumentStore** - KVStore-backed implementation
- [ ] **RefDocInfo** - Reference document tracking

### 4.3 Index Store ❌

- [ ] **IndexStore Interface**
  - `AddIndexStruct(indexStruct IndexStruct) error`
  - `GetIndexStruct(structID string) (*IndexStruct, error)`
  - `DeleteIndexStruct(structID string) error`
  - Python: `storage/index_store/`
  - TypeScript: `storage/index-store/`

### 4.4 Chat Store ❌

- [ ] **ChatStore Interface**
  - `SetMessages(key string, messages []ChatMessage) error`
  - `GetMessages(key string) ([]ChatMessage, error)`
  - `AddMessage(key string, message ChatMessage) error`
  - `DeleteMessages(key string) error`
  - Python: `storage/chat_store/`
  - TypeScript: `storage/chat-store/`

### 4.5 Vector Store Enhancement 🔄

**Current State:** `VectorStore` interface and `SimpleVectorStore`, `ChromemStore` exist

**Required Enhancements:**

- [ ] **VectorStoreQueryMode** - Query mode support
  - `DEFAULT`, `SPARSE`, `HYBRID`, `MMR`
  - Python: `vector_stores/types.py:45-60`
  - TypeScript: `vector-store/index.ts:26-39`

- [ ] **Enhanced Filtering** - Additional filter operators
  - `TEXT_MATCH`, `CONTAINS`, `IS_EMPTY`, `ANY`, `ALL`
  - Nested filter conditions (`AND`, `OR`, `NOT`)
  - Python: `vector_stores/types.py:63-91`
  - TypeScript: `vector-store/index.ts:41-73`

### 4.6 Storage Context ❌

- [ ] **StorageContext** - Unified storage management
  - Combines DocStore, IndexStore, VectorStore
  - Persistence/loading utilities
  - Python: `storage/storage_context.py`

---

## Phase 5: Prompt System

### 5.1 Prompt Templates ❌

- [ ] **PromptTemplate** - Basic template with variable substitution
  - Template string with `{variable}` placeholders
  - Partial formatting support
  - Python: `prompts/base.py`
  - TypeScript: `prompts/prompt.ts`

- [ ] **ChatPromptTemplate** - Message-based prompts
  - System, user, assistant message templates
  - Python: `prompts/chat_prompts.py`

- [ ] **PromptType Enum** - Standard prompt categories
  - `SUMMARY`, `TREE_SUMMARIZE`, `QUESTION_ANSWER`, `REFINE`, etc.
  - Python: `prompts/prompt_type.py`
  - TypeScript: `prompts/prompt-type.ts`

### 5.2 Prompt Mixin ❌

- [ ] **PromptMixin Interface** - Prompt management for components
  - `GetPrompts() map[string]PromptTemplate`
  - `UpdatePrompts(prompts map[string]PromptTemplate)`
  - Python: `prompts/mixin.py`
  - TypeScript: `prompts/mixin.ts`

### 5.3 Default Prompts ❌

- [ ] **Default Prompt Library**
  - QA prompts, summarization prompts, refinement prompts
  - Python: `prompts/default_prompts.py`

---

## Phase 6: Retrieval System

### 6.1 Retriever Enhancement 🔄

**Current State:** Basic `Retriever` interface and `VectorStoreRetriever` exist

**Required Enhancements:**

- [ ] **BaseRetriever** - Full base implementation
  - Object map for recursive retrieval
  - Callback manager integration
  - Python: `base/base_retriever.py`
  - TypeScript: `retriever/index.ts`

- [ ] **Recursive Retrieval** - Handle IndexNode references
  - Retrieve from nested indices/retrievers
  - Python: `base/base_retriever.py:116-146`
  - TypeScript: `retriever/index.ts:58-79`

### 6.2 Advanced Retrievers ❌

- [ ] **FusionRetriever** - Combine multiple retrievers
  - Reciprocal Rank Fusion
  - Python: `retrievers/fusion_retriever.py`

- [ ] **AutoMergingRetriever** - Hierarchical node merging
  - Python: `retrievers/auto_merging_retriever.py`

- [ ] **RouterRetriever** - Route queries to appropriate retriever
  - Python: `retrievers/router_retriever.py`

---

## Phase 7: Response Synthesis

### 7.1 Synthesizer Enhancement 🔄

**Current State:** Basic `Synthesizer` interface and `SimpleSynthesizer` exist

**Required Enhancements:**

- [ ] **BaseSynthesizer** - Full base implementation
  - PromptHelper integration
  - Streaming support
  - Python: `response_synthesizers/base.py`
  - TypeScript: `response-synthesizers/base-synthesizer.ts`

### 7.2 Response Synthesizer Strategies ❌

- [ ] **CompactAndRefine** - Compact context, then refine
  - Python: `response_synthesizers/compact_and_refine.py`

- [ ] **TreeSummarize** - Hierarchical summarization
  - Python: `response_synthesizers/tree_summarize.py`

- [ ] **Accumulate** - Accumulate responses from each chunk
  - Python: `response_synthesizers/accumulate.py`

- [ ] **SimpleSummarize** - Single-pass summarization
  - Python: `response_synthesizers/simple_summarize.py`

- [ ] **ResponseMode Enum**
  - `REFINE`, `COMPACT`, `TREE_SUMMARIZE`, `SIMPLE_SUMMARIZE`, `ACCUMULATE`
  - Python: `response_synthesizers/type.py`

### 7.3 Response Types ❌

- [ ] **Response** - Standard response with metadata
  - `Response`, `SourceNodes`, `Metadata`
  - Python: `base/response/schema.py`

- [ ] **StreamingResponse** - Async streaming response
  - Response generator/channel
  - Python: `base/response/schema.py`

---

## Phase 8: Query Engine

### 8.1 Query Engine Enhancement 🔄

**Current State:** Basic `QueryEngine` interface and `RetrieverQueryEngine` exist

**Required Enhancements:**

- [ ] **BaseQueryEngine** - Full base implementation
  - `Retrieve()`, `Synthesize()` separate methods
  - Callback integration
  - Python: `base/base_query_engine.py`
  - TypeScript: `query-engine/base.ts`

### 8.2 Advanced Query Engines ❌

- [ ] **SubQuestionQueryEngine** - Decompose complex queries
  - Python: `query_engine/sub_question_query_engine.py`

- [ ] **RouterQueryEngine** - Route to appropriate engine
  - Python: `query_engine/router_query_engine.py`

- [ ] **RetryQueryEngine** - Retry on failure
  - Python: `query_engine/retry_query_engine.py`

- [ ] **TransformQueryEngine** - Transform queries before execution
  - Python: `query_engine/transform_query_engine.py`

---

## Phase 9: Index Abstractions

### 9.1 Base Index ❌

- [ ] **BaseIndex Interface**
  - `AsRetriever() Retriever`
  - `AsQueryEngine() QueryEngine`
  - `Insert(nodes []Node) error`
  - `Delete(nodeIDs []string) error`
  - `Refresh(documents []Document) error`
  - Python: `indices/base.py`

### 9.2 Index Types ❌

- [ ] **VectorStoreIndex** - Vector-based retrieval
  - Python: `indices/vector_store/`

- [ ] **SummaryIndex** (ListIndex) - Sequential document index
  - Python: `indices/list/`

- [ ] **KeywordTableIndex** - Keyword-based retrieval
  - Python: `indices/keyword_table/`

- [ ] **TreeIndex** - Hierarchical summarization
  - Python: `indices/tree/`

---

## Phase 10: Tools & Function Calling

### 10.1 Tool System ❌

- [ ] **BaseTool Interface**
  - `Call(input interface{}) (ToolOutput, error)`
  - `Metadata() ToolMetadata`
  - Python: `tools/types.py`
  - TypeScript: `llms/type.ts:BaseTool`

- [ ] **ToolMetadata** - Tool description and schema
  - `Name`, `Description`, `Parameters` (JSON Schema)
  - `ToOpenAITool()` conversion
  - Python: `tools/types.py:23-90`
  - TypeScript: `llms/type.ts:ToolMetadata`

- [ ] **ToolOutput** - Tool execution result
  - `Content`, `ToolName`, `RawInput`, `RawOutput`, `IsError`
  - Python: `tools/types.py:93-150`

- [ ] **FunctionTool** - Wrap Go functions as tools
  - Schema generation from function signature
  - TypeScript: `tools/function-tool.ts`

### 10.2 Specialized Tools ❌

- [ ] **QueryEngineTool** - Wrap query engine as tool
  - Python: `tools/query_engine.py`

- [ ] **RetrieverTool** - Wrap retriever as tool
  - Python: `tools/retriever_tool.py`

---

## Phase 11: Memory System

### 11.1 Chat Memory ❌

- [ ] **BaseMemory Interface**
  - `Get() []ChatMessage`
  - `Put(message ChatMessage) error`
  - `Reset() error`
  - Python: `memory/types.py`
  - TypeScript: `memory/types.ts`

- [ ] **ChatMemoryBuffer** - Fixed-size message buffer
  - Token limit enforcement
  - Python: `memory/chat_memory_buffer.py`

- [ ] **ChatSummaryMemoryBuffer** - Summarize old messages
  - Python: `memory/chat_summary_memory_buffer.py`

### 11.2 Memory Blocks ❌

- [ ] **MemoryBlock Interface** - Composable memory components
  - Python: `memory/memory_blocks/`
  - TypeScript: `memory/block/`

- [ ] **VectorMemory** - Vector-based memory retrieval
  - Python: `memory/vector_memory.py`

---

## Phase 12: Chat Engine

### 12.1 Chat Engine Interface ❌

- [ ] **BaseChatEngine Interface**
  - `Chat(message string) (Response, error)`
  - `ChatStream(message string) (StreamingResponse, error)`
  - `Reset() error`
  - Python: `chat_engine/types.py`
  - TypeScript: `chat-engine/type.ts`

### 12.2 Chat Engine Implementations ❌

- [ ] **SimpleChatEngine** - Direct LLM chat
  - Python: `chat_engine/simple.py`
  - TypeScript: `chat-engine/simple-chat-engine.ts`

- [ ] **ContextChatEngine** - RAG-enhanced chat
  - Python: `chat_engine/context.py`
  - TypeScript: `chat-engine/context-chat-engine.ts`

- [ ] **CondensePlusContextChatEngine** - Query condensation + context
  - Python: `chat_engine/condense_plus_context.py`

---

## Phase 13: Callbacks & Instrumentation

### 13.1 Callback System ❌

- [ ] **CallbackManager** - Event dispatch system
  - Register/unregister handlers
  - Event types: `LLM_START`, `LLM_END`, `RETRIEVE_START`, `RETRIEVE_END`, etc.
  - Python: `callbacks/base.py`
  - TypeScript: `global/settings/callback-manager.ts`

- [ ] **BaseCallbackHandler** - Handler interface
  - `OnEventStart()`, `OnEventEnd()`
  - Python: `callbacks/base_handler.py`

### 13.2 Event Types ❌

- [ ] **CBEventType Enum**
  - `CHUNKING`, `EMBEDDING`, `LLM`, `QUERY`, `RETRIEVE`, `SYNTHESIZE`, etc.
  - Python: `callbacks/schema.py`

---

## Phase 14: Document Readers

### 14.1 Reader Interface ❌

- [ ] **BaseReader Interface**
  - `LoadData() ([]Document, error)`
  - `LazyLoadData() <-chan Document`
  - Python: `readers/base.py`

### 14.2 Reader Implementations 🔄

**Current State:** `SimpleDirectoryReader` exists

**Additional Readers Needed:**

- [ ] **JSONReader** - JSON file parsing
  - Python: `readers/json.py`

- [ ] **PDFReader** - PDF document extraction
- [ ] **HTMLReader** - HTML content extraction
- [ ] **MarkdownReader** - Markdown parsing

---

## Phase 15: Ingestion Pipeline

### 15.1 Pipeline Components ❌

- [ ] **IngestionPipeline** - Document processing pipeline
  - Chain of transformations
  - Caching support
  - Python: `ingestion/pipeline.py`

- [ ] **IngestionCache** - Deduplication cache
  - Python: `ingestion/cache.py`

---

## Phase 16: Postprocessors

### 16.1 Node Postprocessors ❌

- [ ] **BaseNodePostprocessor Interface**
  - `PostprocessNodes(nodes []NodeWithScore, query QueryBundle) []NodeWithScore`
  - Python: `postprocessor/`
  - TypeScript: `postprocessor/`

- [ ] **SimilarityPostprocessor** - Filter by similarity threshold
- [ ] **KeywordNodePostprocessor** - Filter by keywords
- [ ] **MetadataReplacementPostprocessor** - Replace node content with metadata
- [ ] **LongContextReorder** - Reorder for long context models

---

## Phase 17: Advanced Features

### 17.1 Selectors ❌

- [ ] **BaseSelector Interface** - Route queries to components
  - Python: `selectors/`

- [ ] **LLMSingleSelector** - LLM-based single selection
- [ ] **LLMMultiSelector** - LLM-based multi selection

### 17.2 Question Generation ❌

- [ ] **BaseQuestionGenerator Interface**
  - Generate sub-questions from complex queries
  - Python: `question_gen/`

### 17.3 Output Parsers ❌

- [ ] **BaseOutputParser Interface**
  - Parse and validate LLM outputs
  - Python: `output_parsers/`
  - TypeScript: `schema/type.ts`

---

## Implementation Priority Order

### Tier 1: Foundation (Weeks 1-2)
1. ~~Node System Enhancement (1.1)~~ ✅
2. ~~Base Component Interface (1.2)~~ ✅
3. LLM Interface Enhancement (2.1)
4. Embedding Interface Enhancement (2.2)

### Tier 2: Storage & Processing (Weeks 3-4)
5. Key-Value Store (4.1)
6. Document Store (4.2)
7. Node Parser Interface (3.1)
8. Additional Text Splitters (3.2)

### Tier 3: Prompts & Retrieval (Weeks 5-6)
9. Prompt Templates (5.1)
10. Prompt Mixin (5.2)
11. Retriever Enhancement (6.1)
12. Vector Store Enhancement (4.5)

### Tier 4: Response & Query (Weeks 7-8)
13. Synthesizer Enhancement (7.1)
14. Response Synthesizer Strategies (7.2)
15. Query Engine Enhancement (8.1)
16. Base Index (9.1)

### Tier 5: Tools & Memory (Weeks 9-10)
17. Tool System (10.1)
18. Chat Memory (11.1)
19. Chat Engine (12.1-12.2)

### Tier 6: Advanced Features (Weeks 11-12)
20. Callbacks & Instrumentation (13.1-13.2)
21. Ingestion Pipeline (15.1)
22. Postprocessors (16.1)
23. Advanced Query Engines (8.2)

---

## Testing Requirements

Each component should include:

1. **Unit Tests** - Using `testify/suite`
2. **Mock Implementations** - For interface testing
3. **Integration Tests** - With real providers (skip if API key not set)
4. **Example Usage** - In `examples/` directory

## Documentation Requirements

1. **GoDoc Comments** - All exported types and functions
2. **CLAUDE.md Updates** - Document new patterns and conventions
3. **README Updates** - Usage examples and feature list
4. **Example Programs** - Runnable examples for each major feature

---

## References

- **Python llama-index-core**: `/python/llama-index-core/llama_index/core/`
- **TypeScript @llamaindex/core**: `/typescript/packages/core/src/`
- **Current Go Implementation**: `/golang/`
