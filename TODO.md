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

### 2.2 Embedding Interface Enhancement ✅

**Current State:** Fully implemented in `embedding/` package

**Implemented:**

- [x] **EmbeddingInfo** - `embedding/types.go`
  - `Dimensions`, `MaxTokens`, `TokenizerName`, `IsMultiModal`
  - Pre-defined info for OpenAI (ada, 3-small, 3-large) and Ollama models

- [x] **Batch Embedding** - `embedding/interface.go`, `embedding/openai.go`
  - `GetTextEmbeddingsBatch(ctx, texts, callback) ([][]float64, error)`
  - `ProgressCallback` for progress tracking
  - Automatic chunking for large batches (2048 limit)

- [x] **Similarity Functions** - `embedding/similarity.go`
  - `CosineSimilarity`, `EuclideanDistance`, `EuclideanSimilarity`, `DotProduct`
  - `Normalize`, `Magnitude`, `Add`, `Subtract`, `Scale`, `Mean`
  - `TopKSimilar` for finding most similar vectors
  - `SimilarityType` enum: `cosine`, `euclidean`, `dot_product`

- [x] **MultiModal Embedding** - `embedding/interface.go`
  - `MultiModalEmbeddingModel` interface with `GetImageEmbedding()`
  - `ImageType` supporting URL, Base64, and file path inputs

---

## Phase 3: Text Processing

### 3.1 Node Parser Interface ✅

**Current State:** Fully implemented in `nodeparser/` package

**Implemented:**

- [x] **NodeParser Interface** - `nodeparser/interface.go`
  - `GetNodesFromDocuments(documents []Document) []*Node`
  - `ParseNodes(nodes []*Node) []*Node`
  - `NodeParserWithOptions` for fluent configuration
  - `NodeParserOptions` with `IncludeMetadata`, `IncludePrevNextRel`, `IDFunc`

- [x] **Post-Processing** - `nodeparser/base.go`
  - `PostProcessNodes()` establishes `PREVIOUS`/`NEXT` relationships
  - Calculates `StartCharIdx`/`EndCharIdx` for each node
  - `mergeMetadata()` copies parent metadata to child nodes
  - `BuildNodesFromSplits()` helper for text splitter integration

- [x] **Implementations** - `nodeparser/sentence_parser.go`
  - `SentenceNodeParser` wraps `SentenceSplitter` with node management
  - `SimpleNodeParser` creates one node per document (no splitting)
  - Event callbacks for progress tracking

### 3.2 Text Splitters ✅

**Current State:** Fully implemented in `textsplitter/` package

**Implemented:**

- [x] **TokenTextSplitter** - `textsplitter/token_splitter.go`
  - Token-based splitting with configurable chunk size and overlap
  - Custom tokenizer support (SimpleTokenizer, TikToken)
  - `SplitTextMetadataAware()` for metadata-conscious splitting

- [x] **MarkdownSplitter** - `textsplitter/markdown_splitter.go`
  - Preserves code blocks (``` and ~~~)
  - Splits by headers while maintaining structure
  - Handles large code blocks gracefully
  - `SplitTextMetadataAware()` support

- [x] **SentenceWindowSplitter** - `textsplitter/sentence_window_splitter.go`
  - Configurable window size for context around sentences
  - `SplitTextWithWindows()` returns sentences with context
  - `SplitTextForNodes()` returns data with metadata for node creation
  - Custom metadata keys support

- [x] **MetadataAwareTextSplitter** - All splitters support this
  - `SentenceSplitter.SplitTextMetadataAware()` (existing)
  - `TokenTextSplitter.SplitTextMetadataAware()`
  - `MarkdownSplitter.SplitTextMetadataAware()`

### 3.3 Tokenization ✅

**Current State:** Full TikToken integration in `textsplitter/` package

**Implemented:**

- [x] **TikToken Integration** - `textsplitter/tokenizer_tiktoken.go`
  - `TikTokenTokenizer` using `tiktoken-go` library
  - `TikTokenTokenizerByEncoding` for specific encodings
  - Encoding constants: `cl100k_base`, `p50k_base`, `r50k_base`, `o200k_base`
  - `GetEncodingForModel()` maps models to encodings
  - `DefaultTokenizer()` singleton for shared usage
  - `CountTokens()`, `EncodeToIDs()`, `Decode()` methods

- [x] **Model Encoding Map** - Supports GPT-4o, GPT-4, GPT-3.5, embedding models

### 3.4 Input Validation ✅

**Current State:** Validation package in `validation/`

**Implemented:**

- [x] **Validation Package** - `validation/validation.go`
  - `Validator` type for collecting errors
  - `ValidationError` and `ValidationErrors` types
  - `RequirePositive()`, `RequireNonNegative()`, `RequireNotEmpty()`
  - `RequireLessThan()`, `RequireLessOrEqual()`, `RequireNotNil()`
  - `ValidateChunkParams()` for chunk_size/chunk_overlap validation

- [x] **Splitter Validation** - `validation/splitter_validation.go`
  - `ValidateSentenceSplitterConfig()`
  - `ValidateTokenSplitterConfig()`
  - `ValidateMarkdownSplitterConfig()`
  - `ValidateSentenceWindowSplitterConfig()`
  - `ValidateMetadataAwareSplit()` for metadata size checks
  - `GetEffectiveChunkSize()` helper

- [x] **Validated Constructors** - In splitter files
  - `NewSentenceSplitterWithValidation()`
  - `NewTokenTextSplitterWithValidation()`
  - `Validate()` methods on splitter instances

---

## Phase 4: Storage Layer

### 4.1 Key-Value Store ✅

- [x] **KVStore Interface** - `storage/kvstore/interface.go`
  - `Put(ctx, key, val, collection) error`
  - `Get(ctx, key, collection) (StoredValue, error)`
  - `Delete(ctx, key, collection) (bool, error)`
  - `GetAll(ctx, collection) (map[string]StoredValue, error)`
  - `PersistableKVStore` extends with `Persist(ctx, path) error`

- [x] **SimpleKVStore** - In-memory implementation with optional persistence
- [x] **FileKVStore** - File-based persistence (auto-persists on write)

### 4.2 Document Store ✅

- [x] **DocStore Interface** - `storage/docstore/interface.go`
  - `Docs(ctx) (map[string]BaseNode, error)`
  - `AddDocuments(ctx, docs, allowUpdate) error`
  - `GetDocument(ctx, docID, raiseError) (BaseNode, error)`
  - `DeleteDocument(ctx, docID, raiseError) error`
  - `DocumentExists(ctx, docID) (bool, error)`
  - `GetRefDocInfo(ctx, refDocID) (*RefDocInfo, error)`
  - `GetAllRefDocInfo(ctx) (map[string]*RefDocInfo, error)`
  - `DeleteRefDoc(ctx, refDocID, raiseError) error`
  - Hash methods: `SetDocumentHash`, `GetDocumentHash`, `GetAllDocumentHashes`

- [x] **KVDocumentStore** - KVStore-backed implementation (`storage/docstore/kv_docstore.go`)
- [x] **SimpleDocumentStore** - In-memory with persistence (`storage/docstore/simple_docstore.go`)
- [x] **RefDocInfo** - Reference document tracking with NodeIDs and Metadata

### 4.3 Index Store ✅

- [x] **IndexStore Interface** - `storage/indexstore/interface.go`
  - `IndexStructs(ctx) ([]*IndexStruct, error)`
  - `AddIndexStruct(ctx, indexStruct) error`
  - `GetIndexStruct(ctx, structID) (*IndexStruct, error)`
  - `DeleteIndexStruct(ctx, key) error`

- [x] **IndexStruct** - Supports multiple index types (VectorStore, List, KeywordTable, Tree, KG)
- [x] **KVIndexStore** - KVStore-backed implementation (`storage/indexstore/kv_indexstore.go`)
- [x] **SimpleIndexStore** - In-memory with persistence (`storage/indexstore/simple_indexstore.go`)

### 4.4 Chat Store ✅

- [x] **ChatStore Interface** - `storage/chatstore/interface.go`
  - `SetMessages(ctx, key, messages) error`
  - `GetMessages(ctx, key) ([]ChatMessage, error)`
  - `AddMessage(ctx, key, message, idx) error`
  - `DeleteMessages(ctx, key) ([]ChatMessage, error)`
  - `DeleteMessage(ctx, key, idx) (*ChatMessage, error)`
  - `DeleteLastMessage(ctx, key) (*ChatMessage, error)`
  - `GetKeys(ctx) ([]string, error)`

- [x] **SimpleChatStore** - In-memory with persistence (`storage/chatstore/simple_chatstore.go`)
- [x] Uses existing `llm.ChatMessage` type from `llm/types.go`

### 4.5 Vector Store Enhancement ✅

**Enhanced in `schema/schema.go`:**

- [x] **VectorStoreQueryMode** - Query mode support
  - `QueryModeDefault`, `QueryModeSparse`, `QueryModeHybrid`, `QueryModeMMR`
  - `QueryModeTextSearch`, `QueryModeSemanticHybrid`, `QueryModeSVM`

- [x] **Enhanced FilterOperator** - Additional filter operators
  - Basic: `EQ`, `GT`, `LT`, `NE`, `GTE`, `LTE`
  - Array: `IN`, `NIN`, `ANY`, `ALL`
  - Text: `TEXT_MATCH`, `TEXT_MATCH_INSENSITIVE`
  - Special: `CONTAINS`, `IS_EMPTY`

- [x] **FilterCondition** - Nested filter conditions
  - `FilterConditionAnd`, `FilterConditionOr`, `FilterConditionNot`
  - `MetadataFilters.Nested` for complex filter trees

- [x] **Enhanced VectorStoreQuery** - Extended query options
  - `Mode`, `Alpha` (hybrid), `MMRThreshold`
  - `DocIDs`, `NodeIDs`, `SparseTopK`, `HybridTopK`
  - Builder methods: `WithMode()`, `WithFilters()`, `WithAlpha()`, `WithMMRThreshold()`

- [x] **VectorStoreQueryResult** - Structured query results

### 4.6 Storage Context ✅

- [x] **StorageContext** - Unified storage management (`storage/context.go`)
  - Combines `DocStore`, `IndexStore`, `VectorStores` (map by namespace)
  - `NewStorageContext()` - Create with defaults
  - `NewStorageContextFromOptions()` - Create with custom stores
  - `StorageContextFromPersistDir()` - Load from disk
  - `Persist()` - Save to disk
  - `ToDict()` / `StorageContextFromDict()` - Dictionary serialization
  - `ToJSON()` / `StorageContextFromJSON()` - JSON serialization
  - `VectorStore()` / `SetVectorStore()` / `AddVectorStore()` / `GetVectorStore()` - Vector store management

---

## Phase 5: Prompt System ✅

### 5.1 Prompt Templates ✅

- [x] **PromptTemplate** - Basic template with variable substitution (`prompts/template.go`)
  - Template string with `{variable}` placeholders
  - `Format()`, `FormatMessages()`, `PartialFormat()`
  - `GetTemplate()`, `GetTemplateVars()`, `GetPromptType()`, `GetMetadata()`

- [x] **ChatPromptTemplate** - Message-based prompts (`prompts/template.go`)
  - System, user, assistant message templates
  - `FormatMessages()` returns `[]llm.ChatMessage`
  - `NewChatPromptTemplate()`, `ChatPromptTemplateFromMessages()`

- [x] **PromptType Enum** - Standard prompt categories (`prompts/prompt_type.go`)
  - `PromptTypeSummary`, `PromptTypeQuestionAnswer`, `PromptTypeRefine`
  - `PromptTypeTreeInsert`, `PromptTypeTreeSelect`, `PromptTypeKeywordExtract`
  - `PromptTypeKnowledgeTripletExtract`, `PromptTypeChoiceSelect`, `PromptTypeCustom`, etc.

### 5.2 Prompt Mixin ✅

- [x] **PromptMixin Interface** - Prompt management for components (`prompts/mixin.go`)
  - `GetPrompts() PromptDictType`
  - `UpdatePrompts(prompts PromptDictType)`
  - Supports nested modules with "module:prompt" naming

- [x] **BasePromptMixin** - Base implementation
  - `SetPrompt()`, `GetPrompt()`, `AddModule()`

### 5.3 Default Prompts ✅

- [x] **Default Prompt Library** (`prompts/default_prompts.go`)
  - `DefaultSummaryPrompt`, `DefaultTreeSummarizePrompt`
  - `DefaultTextQAPrompt`, `DefaultRefinePrompt`
  - `DefaultInsertPrompt`, `DefaultQueryPrompt`
  - `DefaultKeywordExtractPrompt`, `DefaultKGTripletExtractPrompt`
  - `DefaultChoiceSelectPrompt`, `DefaultRankGPTRerankPrompt`
  - `GetDefaultPrompt(promptType)` helper function

---

## Phase 6: Retrieval System ✅

### 6.1 Retriever Enhancement ✅

- [x] **Retriever Interface** (`rag/retriever/interface.go`)
  - `Retrieve(ctx, query) ([]NodeWithScore, error)`

- [x] **BaseRetriever** - Full base implementation
  - `ObjectMap` for recursive retrieval
  - `HandleRecursiveRetrieval()` for IndexNode references
  - `AddObject()`, `GetObject()` for object management
  - `BasePromptMixin` integration

- [x] **VectorRetriever** (`rag/retriever/vector.go`)
  - Queries vector store with embeddings
  - Supports query modes (default, hybrid, MMR, etc.)
  - `WithTopK()`, `WithQueryMode()` options

### 6.2 Advanced Retrievers ✅

- [x] **FusionRetriever** (`rag/retriever/fusion.go`)
  - Combines multiple retrievers
  - Fusion modes:
    - `FusionModeReciprocalRank` - Reciprocal Rank Fusion (RRF)
    - `FusionModeRelativeScore` - Relative score normalization
    - `FusionModeDistBasedScore` - Distance-based score fusion
    - `FusionModeSimple` - Max score for duplicates
  - `WithRetrieverWeights()` for weighted fusion

- [x] **AutoMergingRetriever** (`rag/retriever/auto_merging.go`)
  - Merges child nodes into parent nodes
  - `SimpleRatioThresh` controls merge threshold
  - Fills gaps between consecutive nodes
  - Uses `StorageContext.DocStore` for parent lookup

- [x] **RouterRetriever** (`rag/retriever/router.go`)
  - Routes queries to appropriate retrievers
  - `RetrieverTool` wraps retrievers with metadata
  - `Selector` interface for routing decisions
  - `SimpleSelector` (all), `SingleSelector` (first)

---

## Phase 7: Response Synthesis ✅

### 7.1 Synthesizer Enhancement ✅

- [x] **Synthesizer Interface** (`rag/synthesizer/interface.go`)
  - `Synthesize(ctx, query, nodes) (*Response, error)`
  - `GetResponse(ctx, query, textChunks) (string, error)`

- [x] **BaseSynthesizer** - Full base implementation
  - `LLM`, `Streaming`, `Verbose` fields
  - `GetMetadataForResponse()`, `PrepareResponseOutput()`
  - `BasePromptMixin` integration
  - `WithStreaming()`, `WithSynthesizerVerbose()` options

### 7.2 Response Synthesizer Strategies ✅

- [x] **SimpleSynthesizer** (`rag/synthesizer/simple.go`)
  - Merges all chunks, single LLM call
  - `WithTextQATemplate()` option

- [x] **RefineSynthesizer** (`rag/synthesizer/refine.go`)
  - Iteratively refines response across chunks
  - `WithRefineTextQATemplate()`, `WithRefineTemplate()` options

- [x] **CompactAndRefineSynthesizer** (`rag/synthesizer/compact.go`)
  - Compacts chunks before refining
  - `WithMaxChunkSize()`, `WithChunkSeparator()` options

- [x] **TreeSummarizeSynthesizer** (`rag/synthesizer/tree_summarize.go`)
  - Recursive bottom-up summarization
  - `WithSummaryTemplate()`, `WithTreeMaxChunkSize()` options

- [x] **AccumulateSynthesizer** (`rag/synthesizer/accumulate.go`)
  - Generates response per chunk, concatenates
  - `WithAccumulateTextQATemplate()`, `WithAccumulateSeparator()` options

- [x] **CompactAccumulateSynthesizer** (`rag/synthesizer/accumulate.go`)
  - Compacts chunks before accumulating

- [x] **ResponseMode Enum** (`rag/synthesizer/response_mode.go`)
  - `ResponseModeRefine`, `ResponseModeCompact`, `ResponseModeSimpleSummarize`
  - `ResponseModeTreeSummarize`, `ResponseModeAccumulate`, `ResponseModeCompactAccumulate`
  - `ResponseModeGeneration`, `ResponseModeNoText`, `ResponseModeContextOnly`

- [x] **GetSynthesizer Factory** (`rag/synthesizer/factory.go`)
  - Returns synthesizer for given response mode

### 7.3 Response Types ✅

- [x] **Response** (`rag/synthesizer/response.go`)
  - `Response`, `SourceNodes`, `Metadata` fields
  - `String()`, `GetFormattedSources()` methods

- [x] **StreamingResponse** (`rag/synthesizer/response.go`)
  - `ResponseChan` for streaming tokens
  - `String()`, `GetResponse()`, `GetFormattedSources()` methods

- [x] **ResponseType Interface**
  - Common interface for all response types

---

## Phase 8: Query Engine ✅

### 8.1 Query Engine Enhancement ✅

- [x] **QueryEngine Interface** (`rag/queryengine/interface.go`)
  - `Query(ctx, query) (*Response, error)`

- [x] **QueryEngineWithRetrieval Interface**
  - `Retrieve(ctx, query) ([]NodeWithScore, error)`
  - `Synthesize(ctx, query, nodes) (*Response, error)`

- [x] **BaseQueryEngine** - Full base implementation
  - `Verbose` field, `BasePromptMixin` integration
  - `WithQueryEngineVerbose()` option

- [x] **RetrieverQueryEngine** (`rag/queryengine/interface.go`)
  - Combines retriever and synthesizer
  - Implements `QueryEngineWithRetrieval`

- [x] **QueryEngineTool** (`rag/queryengine/tool.go`)
  - Wraps query engine with metadata for routing
  - `ToolMetadata` struct

### 8.2 Advanced Query Engines ✅

- [x] **SubQuestionQueryEngine** (`rag/queryengine/sub_question.go`)
  - Decomposes complex queries into sub-questions
  - `QuestionGenerator` interface
  - `LLMQuestionGenerator` implementation
  - `SubQuestion`, `SubQuestionAnswerPair` types

- [x] **RouterQueryEngine** (`rag/queryengine/router.go`)
  - Routes queries to appropriate engines
  - `QueryEngineSelector` interface
  - `SingleSelector`, `MultiSelector` implementations
  - `WithRouterSelector()`, `WithRouterSummarizer()` options

- [x] **RetryQueryEngine** (`rag/queryengine/retry.go`)
  - Retries queries on failure
  - `WithMaxRetries()`, `WithRetryDelay()` options

- [x] **TransformQueryEngine** (`rag/queryengine/transform.go`)
  - Transforms queries before execution
  - `QueryTransform` interface
  - `IdentityTransform`, `HyDETransform` implementations

---

## Phase 9: Index Abstractions ✅

### 9.1 Base Index ✅

- [x] **BaseIndex Interface** - `index/interface.go`
  - `AsRetriever() Retriever`
  - `AsQueryEngine() QueryEngine`
  - `InsertNodes(nodes []Node) error`
  - `DeleteNodes(nodeIDs []string) error`
  - `RefreshDocuments(documents []Document) error`
  - Python: `indices/base.py`

- [x] **Index Interface** - Core interface for all index types
  - `IndexID() string`
  - `IndexStruct() *IndexStruct`
  - `StorageContext() *StorageContext`

- [x] **RetrieverConfig/QueryEngineConfig** - Configuration options
  - Functional options pattern for flexible configuration

### 9.2 Index Types ✅

- [x] **VectorStoreIndex** - Vector-based retrieval - `index/vector_store.go`
  - Embedding generation and storage
  - Batch insertion with configurable batch size
  - Integration with VectorStore interface
  - VectorIndexRetriever for similarity search
  - Python: `indices/vector_store/`

- [x] **SummaryIndex** (ListIndex) - Sequential document index - `index/summary.go`
  - Stores nodes in a list structure
  - Multiple retriever modes: Default, Embedding, LLM
  - SummaryIndexRetriever with mode selection
  - Python: `indices/list/`

- [x] **KeywordTableIndex** - Keyword-based retrieval - `index/keyword_table.go`
  - SimpleKeywordExtractor with stop word removal
  - Keyword-to-node mapping
  - KeywordTableRetriever with keyword matching
  - Python: `indices/keyword_table/`

- [ ] **TreeIndex** - Hierarchical summarization (future)
  - Python: `indices/tree/`

### 9.3 Tests ✅

- [x] **Index Tests** - `index/index_test.go`
  - MockEmbeddingModel for testing
  - VectorStoreIndex tests (create, insert, delete, retrieve)
  - SummaryIndex tests (create, insert, delete, retrieve)
  - KeywordTableIndex tests (create, insert, delete, retrieve)
  - SimpleKeywordExtractor tests
  - Interface compliance tests

---

## Phase 10: Tools & Function Calling ✅

### 10.1 Tool System ✅

- [x] **Tool Interface** - `tools/types.go`
  - `Call(ctx context.Context, input interface{}) (*ToolOutput, error)`
  - `Metadata() *ToolMetadata`
  - Python: `tools/types.py`

- [x] **ToolMetadata** - Tool description and schema - `tools/types.go`
  - `Name`, `Description`, `Parameters` (JSON Schema)
  - `ToOpenAITool()` conversion to OpenAI function calling format
  - `ToOpenAIFunction()` legacy format
  - `GetParametersJSON()` for schema serialization
  - Python: `tools/types.py:23-90`

- [x] **ToolOutput** - Tool execution result - `tools/types.go`
  - `Content`, `ToolName`, `RawInput`, `RawOutput`, `IsError`, `Error`
  - `NewToolOutput()`, `NewToolOutputWithInput()`, `NewErrorToolOutput()` constructors
  - Python: `tools/types.py:93-150`

- [x] **BaseTool** - Base implementation for tools - `tools/types.go`
  - Common metadata handling
  - `ToolSpec` for declarative tool definition

- [x] **FunctionTool** - Wrap Go functions as tools - `tools/function_tool.go`
  - Automatic schema generation from function signature
  - Support for context.Context parameter
  - Type conversion for function arguments
  - `typeToJSONSchema()` for Go type to JSON Schema conversion
  - `structToJSONSchema()` for struct types
  - Python: `tools/function_tool.py`

### 10.2 Specialized Tools ✅

- [x] **QueryEngineTool** - Wrap query engine as tool - `tools/query_engine_tool.go`
  - Executes queries against a QueryEngine
  - Configurable input resolution
  - `NewQueryEngineTool()`, `NewQueryEngineToolFromDefaults()` constructors
  - Python: `tools/query_engine.py`

- [x] **RetrieverTool** - Wrap retriever as tool - `tools/retriever_tool.go`
  - Retrieves documents from a Retriever
  - Support for NodePostprocessors
  - Formats retrieved content for LLM consumption
  - Python: `tools/retriever_tool.py`

### 10.3 Tests ✅

- [x] **Tool Tests** - `tools/tools_test.go`
  - ToolMetadata tests (creation, JSON schema, OpenAI format)
  - ToolOutput tests (creation, error handling)
  - ToolSpec tests
  - FunctionTool tests (simple functions, context, map input, errors)
  - QueryEngineTool tests
  - RetrieverTool tests
  - Type conversion tests
  - Interface compliance tests

---

## Phase 11: Memory System ✅

### 11.1 Chat Memory ✅

- [x] **Memory Interface** - `memory/types.go`
  - `Get(ctx, input) ([]ChatMessage, error)` - Retrieve messages, optionally filtered
  - `GetAll(ctx) ([]ChatMessage, error)` - Retrieve all messages
  - `Put(ctx, message) error` - Add a message
  - `PutMessages(ctx, messages) error` - Add multiple messages
  - `Set(ctx, messages) error` - Replace all messages
  - `Reset(ctx) error` - Clear all messages
  - Python: `memory/types.py`

- [x] **BaseMemory** - Base implementation - `memory/types.go`
  - Common functionality for chat store interaction
  - Configurable chat store and key
  - `TokenizerFunc` type for token counting
  - `DefaultTokenizer` (~4 chars per token)

- [x] **SimpleMemory** - Basic memory that stores all messages - `memory/types.go`
  - Simple implementation that returns all messages on Get

- [x] **ChatMemoryBuffer** - Fixed-size message buffer - `memory/chat_memory_buffer.go`
  - Token limit enforcement
  - Trims older messages when limit exceeded
  - Skips assistant/tool messages at start
  - `GetWithInitialTokenCount` for accounting system prompts
  - `NewChatMemoryBufferFromDefaults` with LLM context window support
  - Python: `memory/chat_memory_buffer.py`

- [x] **ChatSummaryMemoryBuffer** - Summarize old messages - `memory/chat_summary_memory_buffer.go`
  - Summarizes older messages using LLM
  - Keeps recent messages in full text
  - Configurable summarization prompt
  - `countInitialTokens` option
  - `NewChatSummaryMemoryBufferFromDefaults` constructor
  - Python: `memory/chat_summary_memory_buffer.py`

### 11.2 Vector Memory ✅

- [x] **VectorMemory** - Vector-based memory retrieval - `memory/vector_memory.go`
  - Stores messages in vector store for semantic retrieval
  - `batchByUserMessage` groups user/assistant pairs
  - Retrieves relevant messages based on query similarity
  - `NewVectorMemoryFromDefaults` constructor
  - Python: `memory/vector_memory.py`

### 11.3 Tests ✅

- [x] **Memory Tests** - `memory/memory_test.go`
  - SimpleMemory tests (put, get, set, reset)
  - ChatMemoryBuffer tests (token limits, trimming)
  - ChatSummaryMemoryBuffer tests (summarization)
  - VectorMemory tests (put, get, batching)
  - BaseMemory tests
  - DefaultTokenizer tests
  - Interface compliance tests

---

## Phase 12: Chat Engine ✅

### 12.1 Chat Engine Interface ✅

- [x] **ChatEngine Interface** - `chatengine/types.go`
  - `Chat(ctx, message) (*ChatResponse, error)` - Send message and get response
  - `ChatWithHistory(ctx, message, history) (*ChatResponse, error)` - Chat with explicit history
  - `StreamChat(ctx, message) (*StreamingChatResponse, error)` - Streaming chat
  - `Reset(ctx) error` - Clear conversation state
  - `ChatHistory(ctx) ([]ChatMessage, error)` - Get chat history
  - Python: `chat_engine/types.py`

- [x] **ChatResponse** - Chat response struct - `chatengine/types.go`
  - `Response` - Text response
  - `SourceNodes` - Source nodes used for response
  - `Sources` - Tool sources (retriever output)
  - `Metadata` - Additional metadata

- [x] **StreamingChatResponse** - Streaming response - `chatengine/types.go`
  - `ResponseChan` - Channel for streaming tokens
  - `Consume()` - Read all tokens and return full response
  - `IsDone()` - Check if streaming is complete

- [x] **BaseChatEngine** - Base implementation - `chatengine/types.go`
  - Common LLM and prefix messages handling
  - `WithLLM`, `WithSystemPrompt`, `WithPrefixMessages` options

- [x] **ChatMode** - Chat engine modes enum - `chatengine/types.go`
  - `ChatModeSimple`, `ChatModeContext`, `ChatModeCondensePlusContext`

### 12.2 Chat Engine Implementations ✅

- [x] **SimpleChatEngine** - Direct LLM chat - `chatengine/simple.go`
  - Chat with LLM without knowledge base
  - Memory integration for conversation history
  - Streaming support
  - `NewSimpleChatEngineFromDefaults` constructor
  - Python: `chat_engine/simple.py`

- [x] **ContextChatEngine** - RAG-enhanced chat - `chatengine/context.go`
  - Retrieves context from retriever
  - Injects context into system prompt
  - Returns source nodes with response
  - Configurable context template
  - `NewContextChatEngineFromDefaults` constructor
  - Python: `chat_engine/context.py`

- [x] **CondensePlusContextChatEngine** - Query condensation + context - `chatengine/condense_plus_context.go`
  - Condenses conversation history + latest message to standalone question
  - Retrieves context using condensed question
  - `skipCondense` option to bypass condensation
  - `verbose` mode for debugging
  - Configurable condense and context templates
  - `NewCondensePlusContextChatEngineFromDefaults` constructor
  - Python: `chat_engine/condense_plus_context.py`

### 12.3 Tests ✅

- [x] **Chat Engine Tests** - `chatengine/chatengine_test.go`
  - ChatResponse and StreamingChatResponse tests
  - SimpleChatEngine tests (chat, streaming, reset, history)
  - ContextChatEngine tests (chat with retriever, source nodes)
  - CondensePlusContextChatEngine tests (condensing, skip condense)
  - BaseChatEngine tests
  - Interface compliance tests
  - Custom memory integration tests

---

## Phase 13: Callbacks & Instrumentation ✅

### 13.1 Event Types ✅

- [x] **CBEventType Enum** - `callbacks/schema.go`
  - `CBEventTypeChunking`, `CBEventTypeNodeParsing`, `CBEventTypeEmbedding`
  - `CBEventTypeLLM`, `CBEventTypeQuery`, `CBEventTypeRetrieve`
  - `CBEventTypeSynthesize`, `CBEventTypeTree`, `CBEventTypeSubQuestion`
  - `CBEventTypeTemplating`, `CBEventTypeFunctionCall`, `CBEventTypeReranking`
  - `CBEventTypeException`, `CBEventTypeAgentStep`
  - `IsLeafEvent()` helper function
  - Python: `callbacks/schema.py`

- [x] **EventPayload Enum** - `callbacks/schema.go`
  - `EventPayloadDocuments`, `EventPayloadChunks`, `EventPayloadNodes`
  - `EventPayloadPrompt`, `EventPayloadMessages`, `EventPayloadCompletion`
  - `EventPayloadResponse`, `EventPayloadQueryStr`, `EventPayloadEmbeddings`
  - `EventPayloadException`, etc.

- [x] **CBEvent** - Event struct - `callbacks/schema.go`
  - `EventType`, `Payload`, `Time`, `ID`
  - `NewCBEvent()` constructor

- [x] **EventStats** - Statistics struct - `callbacks/schema.go`
  - `TotalSecs`, `AverageSecs`, `TotalCount`

### 13.2 Callback Handler ✅

- [x] **CallbackHandler Interface** - `callbacks/handler.go`
  - `OnEventStart(eventType, payload, eventID, parentID) string`
  - `OnEventEnd(eventType, payload, eventID)`
  - `StartTrace(traceID)`, `EndTrace(traceID, traceMap)`
  - `EventStartsToIgnore()`, `EventEndsToIgnore()`
  - Python: `callbacks/base_handler.py`

- [x] **BaseCallbackHandler** - Base implementation - `callbacks/handler.go`
  - Default no-op implementations
  - `ShouldIgnoreEventStart()`, `ShouldIgnoreEventEnd()` helpers
  - Configurable event ignore lists

### 13.3 Callback Manager ✅

- [x] **CallbackManager** - Event dispatch system - `callbacks/manager.go`
  - `OnEventStart()`, `OnEventEnd()` - Dispatch to handlers
  - `AddHandler()`, `RemoveHandler()`, `SetHandlers()` - Handler management
  - `StartTrace()`, `EndTrace()` - Trace management
  - `TraceMap()` - Get event parent-child relationships
  - Thread-safe with mutex
  - Python: `callbacks/base.py`

- [x] **EventContext** - Event wrapper - `callbacks/manager.go`
  - `OnStart()`, `OnEnd()` - Trigger event start/end
  - `EventID()`, `IsStarted()`, `IsFinished()` - State queries
  - Prevents duplicate start/end calls

- [x] **Helper Methods** - `callbacks/manager.go`
  - `Event()` - Create EventContext
  - `WithEvent()` - Execute function within event context
  - `WithTrace()` - Execute function within trace context

### 13.4 Handler Implementations ✅

- [x] **LoggingHandler** - Logs events - `callbacks/handlers.go`
  - Logs event start/end with timestamps
  - Verbose mode with payload details
  - Duration tracking
  - Configurable writer

- [x] **TokenCountingHandler** - Tracks token usage - `callbacks/handlers.go`
  - `TotalLLMTokens()`, `PromptTokens()`, `CompletionTokens()`
  - `TotalEmbedTokens()`
  - `LLMEventCount()`, `EmbedEventCount()`
  - `Reset()` to clear counters

- [x] **EventCollectorHandler** - Collects events - `callbacks/handlers.go`
  - `StartEvents()`, `EndEvents()` - Get collected events
  - `GetEventsByType()` - Filter by event type
  - `Clear()` - Clear collected events
  - Useful for testing and debugging

### 13.5 Tests ✅

- [x] **Callback Tests** - `callbacks/callbacks_test.go`
  - CBEventType and EventPayload tests
  - CBEvent and EventStats tests
  - BaseCallbackHandler tests
  - CallbackManager tests (handlers, events, traces)
  - EventContext tests
  - LoggingHandler tests
  - TokenCountingHandler tests
  - EventCollectorHandler tests
  - Event ignoring tests
  - Trace stack tests
  - Concurrent access tests
  - Interface compliance tests

---

## Phase 14: Document Readers

### 14.1 Reader Interface ✅

- [x] **Reader Interface** - `rag/reader/interface.go`
  - `LoadData() ([]Node, error)`
  - `LazyReader` with `LazyLoadData() (<-chan Node, <-chan error)`
  - `FileReader` with `LoadFromFile(path string)`
  - `ReaderWithContext` for cancellation support
  - `ReaderMetadata` for reader information
  - `ReaderOptions` for common configuration
  - `ReaderError` for structured error handling

### 14.2 Reader Implementations ✅

**Current State:** Multiple readers implemented

**Implemented:**

- [x] **SimpleDirectoryReader** - `rag/reader/simple_directory_reader.go`
  - Recursive directory traversal
  - Extension filtering

- [x] **JSONReader** - `rag/reader/json_reader.go`
  - Single JSON object and array support
  - JSON Lines (JSONL) format support
  - Configurable text content key
  - Metadata key extraction

- [x] **HTMLReader** - `rag/reader/html_reader.go`
  - Script/style tag removal
  - HTML entity decoding
  - Metadata extraction (title, description, language)
  - Tag-specific content extraction
  - Whitespace normalization

- [x] **MarkdownReader** - `rag/reader/markdown_reader.go`
  - YAML frontmatter extraction
  - Header-based document splitting
  - Hyperlink/image removal options
  - Multiple markdown extensions support

**Additional Readers Needed:**

- [ ] **PDFReader** - PDF document extraction (requires external library)

---

## Phase 15: Ingestion Pipeline ✅

### 15.1 Pipeline Components ✅

- [x] **IngestionPipeline** - `ingestion/pipeline.go`
  - Chain of transformations via `TransformComponent` interface
  - Caching support with `IngestionCache`
  - Document deduplication with docstore strategies
  - Vector store integration
  - Configurable via functional options pattern
  - `DocstoreStrategy` enum: `UPSERTS`, `DUPLICATES_ONLY`, `UPSERTS_AND_DELETE`
  - `Run()` method for synchronous execution
  - `RunTransformations()` standalone function

- [x] **IngestionCache** - `ingestion/cache.go`
  - In-memory cache with collection support
  - `Put()` and `Get()` for node caching
  - `Persist()` and `LoadFromPath()` for persistence
  - `Clear()` and `HasKey()` utility methods
  - Thread-safe with `sync.RWMutex`

### 15.2 Tests ✅

- [x] **Ingestion Tests** - `ingestion/ingestion_test.go`
  - IngestionCache tests (put, get, clear, persist, load)
  - IngestionPipeline tests (run with documents, nodes, transformations)
  - Cache hit/miss tests
  - Docstore deduplication tests
  - Vector store integration tests
  - Transformation hash tests

---

## Phase 16: Postprocessors ✅

### 16.1 Node Postprocessors ✅

- [x] **NodePostprocessor Interface** - `postprocessor/types.go`
  - `PostprocessNodes(ctx, nodes []NodeWithScore, queryBundle) ([]NodeWithScore, error)`
  - `Name() string`

- [x] **BaseNodePostprocessor** - `postprocessor/types.go`
  - Default no-op implementation
  - Configurable via functional options

- [x] **PostprocessorChain** - `postprocessor/types.go`
  - Chain multiple postprocessors together
  - Sequential execution

- [x] **SimilarityPostprocessor** - `postprocessor/similarity.go`
  - Filter nodes by similarity score threshold
  - Configurable cutoff via `WithSimilarityCutoff()`

- [x] **KeywordPostprocessor** - `postprocessor/keyword.go`
  - Filter by required keywords (all must match)
  - Filter by excluded keywords (any match excludes)
  - Case-sensitive/insensitive matching

- [x] **MetadataReplacementPostprocessor** - `postprocessor/metadata_replacement.go`
  - Replace node content with metadata value
  - Configurable target metadata key

- [x] **LongContextReorder** - `postprocessor/long_context_reorder.go`
  - Reorder nodes for long context models
  - Based on paper: https://arxiv.org/abs/2307.03172
  - Places higher-scored nodes at start and end

- [x] **TopKPostprocessor** - `postprocessor/top_k.go`
  - Limit number of returned nodes
  - Sorts by score descending

### 16.2 Tests ✅

- [x] **Postprocessor Tests** - `postprocessor/postprocessor_test.go`
  - BaseNodePostprocessor tests
  - SimilarityPostprocessor tests (cutoff filtering)
  - KeywordPostprocessor tests (required/excluded, case sensitivity)
  - MetadataReplacementPostprocessor tests
  - LongContextReorder tests
  - TopKPostprocessor tests
  - PostprocessorChain tests
  - Interface compliance tests

---

## Phase 17: Advanced Features ✅

### 17.1 Selectors ✅

- [x] **Selector Interface** - `selector/types.go`
  - `Select(ctx, choices []ToolMetadata, query string) (*SelectorResult, error)`
  - `ToolMetadata` struct with Name and Description
  - `SingleSelection` struct with Index and Reason
  - `SelectorResult` with helper methods (Ind, Inds, Reason, Reasons)
  - `BuildChoicesText()` helper function

- [x] **BaseSelector** - `selector/types.go`
  - Default no-op implementation
  - Configurable via functional options

- [x] **LLMSingleSelector** - `selector/llm_selector.go`
  - LLM-based single selection from choices
  - Configurable prompt template
  - SelectionOutputParser for JSON parsing

- [x] **LLMMultiSelector** - `selector/llm_selector.go`
  - LLM-based multi selection from choices
  - Configurable max outputs
  - Converts 1-indexed LLM output to 0-indexed

### 17.2 Question Generation ✅

- [x] **QuestionGenerator Interface** - `questiongen/types.go`
  - `Generate(ctx, tools []ToolMetadata, query string) ([]SubQuestion, error)`
  - `SubQuestion` struct with SubQuestion and ToolName
  - `SubQuestionList` wrapper for JSON parsing

- [x] **BaseQuestionGenerator** - `questiongen/types.go`
  - Default no-op implementation
  - Configurable via functional options

- [x] **LLMQuestionGenerator** - `questiongen/llm_generator.go`
  - LLM-based sub-question generation
  - Few-shot prompt template with examples
  - SubQuestionOutputParser for JSON parsing
  - `buildToolsText()` helper function

### 17.3 Output Parsers ✅

- [x] **OutputParser Interface** - `outputparser/types.go`
  - `Parse(output string) (*StructuredOutput, error)`
  - `Format(promptTemplate string) string`
  - `StructuredOutput` with RawOutput and ParsedOutput
  - `OutputParserError` for structured errors

- [x] **BaseOutputParser** - `outputparser/types.go`
  - Default pass-through implementation
  - Configurable via functional options

- [x] **JSONOutputParser** - `outputparser/json_parser.go`
  - Parses JSON objects and arrays
  - Extracts JSON from code blocks
  - Handles surrounding text

- [x] **ListOutputParser** - `outputparser/json_parser.go`
  - Parses newline-separated lists
  - Configurable separator
  - Filters empty items

- [x] **BooleanOutputParser** - `outputparser/json_parser.go`
  - Parses boolean values from text
  - Configurable true/false value sets
  - Handles surrounding text

### 17.4 Tests ✅

- [x] **Selector Tests** - `selector/selector_test.go`
  - ToolMetadata, SingleSelection, SelectorResult tests
  - BaseSelector tests
  - SelectionOutputParser tests
  - LLMSingleSelector and LLMMultiSelector tests
  - Interface compliance tests

- [x] **Question Generator Tests** - `questiongen/questiongen_test.go`
  - SubQuestion, SubQuestionList tests
  - BaseQuestionGenerator tests
  - SubQuestionOutputParser tests
  - LLMQuestionGenerator tests
  - Interface compliance tests

- [x] **Output Parser Tests** - `outputparser/outputparser_test.go`
  - StructuredOutput, OutputParserError tests
  - BaseOutputParser tests
  - JSONOutputParser tests
  - ListOutputParser tests
  - BooleanOutputParser tests
  - extractJSON tests
  - Interface compliance tests

---

## Implementation Priority Order

### Tier 1: Foundation (Weeks 1-2)
1. ~~Node System Enhancement (1.1)~~ ✅
2. ~~Base Component Interface (1.2)~~ ✅
3. ~~LLM Interface Enhancement (2.1)~~ ✅
4. ~~Embedding Interface Enhancement (2.2)~~ ✅

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

## Deferred Features

The following features were deferred during initial implementation and should be completed:

### TreeIndex (Phase 9.2)
- **Priority**: Medium
- **Complexity**: High
- **Description**: Hierarchical summarization index that organizes nodes in a tree structure
- **Requirements**:
  - Tree building with recursive LLM-based summarization
  - `TreeSelectLeafRetriever` - Select relevant leaf nodes via tree traversal
  - `TreeAllLeafRetriever` - Return all leaf nodes
  - `TreeRootRetriever` - Return root summary
  - Support for different tree building strategies (top-down, bottom-up)
- **Python Reference**: `indices/tree/`
- **Depends On**: LLM integration for summarization

---

## References

- **Python llama-index-core**: `/python/llama-index-core/llama_index/core/`
- **TypeScript @llamaindex/core**: `/typescript/packages/core/src/`
- **Current Go Implementation**: `/golang/`
