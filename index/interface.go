// Package index provides index abstractions for LlamaIndex.
package index

import (
	"context"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/prompts"
	"github.com/aqua777/go-llamaindex/rag/queryengine"
	"github.com/aqua777/go-llamaindex/rag/retriever"
	"github.com/aqua777/go-llamaindex/rag/synthesizer"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/storage"
	"github.com/aqua777/go-llamaindex/storage/indexstore"
)

// EmbeddingModel is an alias for embedding.EmbeddingModel.
type EmbeddingModel = embedding.EmbeddingModel

// Index is the base interface for all index types.
type Index interface {
	// IndexID returns the unique identifier for this index.
	IndexID() string

	// IndexStruct returns the underlying index structure.
	IndexStruct() *indexstore.IndexStruct

	// StorageContext returns the storage context.
	StorageContext() *storage.StorageContext

	// AsRetriever returns a retriever for this index.
	AsRetriever(opts ...RetrieverOption) retriever.Retriever

	// AsQueryEngine returns a query engine for this index.
	AsQueryEngine(opts ...QueryEngineOption) queryengine.QueryEngine

	// InsertNodes inserts nodes into the index.
	InsertNodes(ctx context.Context, nodes []schema.Node) error

	// DeleteNodes removes nodes from the index.
	DeleteNodes(ctx context.Context, nodeIDs []string) error

	// RefreshDocuments refreshes the index with updated documents.
	RefreshDocuments(ctx context.Context, documents []schema.Document) ([]bool, error)
}

// RetrieverOption configures retriever creation.
type RetrieverOption func(*RetrieverConfig)

// RetrieverConfig holds retriever configuration.
type RetrieverConfig struct {
	SimilarityTopK int
	EmbedModel     EmbeddingModel
	Filters        *schema.MetadataFilters
}

// WithSimilarityTopK sets the number of top results to return.
func WithSimilarityTopK(k int) RetrieverOption {
	return func(c *RetrieverConfig) {
		c.SimilarityTopK = k
	}
}

// WithRetrieverEmbedModel sets the embedding model for retrieval.
func WithRetrieverEmbedModel(model EmbeddingModel) RetrieverOption {
	return func(c *RetrieverConfig) {
		c.EmbedModel = model
	}
}

// WithRetrieverFilters sets metadata filters for retrieval.
func WithRetrieverFilters(filters *schema.MetadataFilters) RetrieverOption {
	return func(c *RetrieverConfig) {
		c.Filters = filters
	}
}

// QueryEngineOption configures query engine creation.
type QueryEngineOption func(*QueryEngineConfig)

// QueryEngineConfig holds query engine configuration.
type QueryEngineConfig struct {
	LLM          llm.LLM
	ResponseMode synthesizer.ResponseMode
	Synthesizer  synthesizer.Synthesizer
	RetrieverConfig
}

// WithQueryEngineLLM sets the LLM for the query engine.
func WithQueryEngineLLM(l llm.LLM) QueryEngineOption {
	return func(c *QueryEngineConfig) {
		c.LLM = l
	}
}

// WithResponseMode sets the response synthesis mode.
func WithResponseMode(mode synthesizer.ResponseMode) QueryEngineOption {
	return func(c *QueryEngineConfig) {
		c.ResponseMode = mode
	}
}

// WithQueryEngineSynthesizer sets a custom synthesizer.
func WithQueryEngineSynthesizer(s synthesizer.Synthesizer) QueryEngineOption {
	return func(c *QueryEngineConfig) {
		c.Synthesizer = s
	}
}

// WithQueryEngineTopK sets the number of top results for retrieval.
func WithQueryEngineTopK(k int) QueryEngineOption {
	return func(c *QueryEngineConfig) {
		c.SimilarityTopK = k
	}
}

// BaseIndex provides common functionality for all index types.
type BaseIndex struct {
	// indexStruct is the underlying index structure.
	indexStruct *indexstore.IndexStruct
	// storageContext manages storage components.
	storageContext *storage.StorageContext
	// embedModel is the embedding model for this index.
	embedModel EmbeddingModel
	// showProgress enables progress display.
	showProgress bool
	// PromptMixin for prompt management.
	*prompts.BasePromptMixin
}

// BaseIndexOption configures BaseIndex creation.
type BaseIndexOption func(*BaseIndex)

// WithStorageContext sets the storage context.
func WithStorageContext(sc *storage.StorageContext) BaseIndexOption {
	return func(bi *BaseIndex) {
		bi.storageContext = sc
	}
}

// WithEmbedModel sets the embedding model.
func WithEmbedModel(model EmbeddingModel) BaseIndexOption {
	return func(bi *BaseIndex) {
		bi.embedModel = model
	}
}

// WithShowProgress enables progress display.
func WithShowProgress(show bool) BaseIndexOption {
	return func(bi *BaseIndex) {
		bi.showProgress = show
	}
}

// NewBaseIndex creates a new BaseIndex.
func NewBaseIndex(indexStruct *indexstore.IndexStruct, opts ...BaseIndexOption) *BaseIndex {
	bi := &BaseIndex{
		indexStruct:     indexStruct,
		storageContext:  storage.NewStorageContext(),
		showProgress:    false,
		BasePromptMixin: prompts.NewBasePromptMixin(),
	}

	for _, opt := range opts {
		opt(bi)
	}

	return bi
}

// IndexID returns the unique identifier for this index.
func (bi *BaseIndex) IndexID() string {
	return bi.indexStruct.IndexID
}

// IndexStruct returns the underlying index structure.
func (bi *BaseIndex) IndexStruct() *indexstore.IndexStruct {
	return bi.indexStruct
}

// StorageContext returns the storage context.
func (bi *BaseIndex) StorageContext() *storage.StorageContext {
	return bi.storageContext
}

// Summary returns the index summary.
func (bi *BaseIndex) Summary() string {
	return bi.indexStruct.Summary
}

// SetSummary sets the index summary.
func (bi *BaseIndex) SetSummary(ctx context.Context, summary string) error {
	bi.indexStruct.Summary = summary
	return bi.storageContext.IndexStore.AddIndexStruct(ctx, bi.indexStruct)
}

// DocStore returns the document store.
func (bi *BaseIndex) DocStore() interface{} {
	return bi.storageContext.DocStore
}

// EmbedModel returns the embedding model.
func (bi *BaseIndex) EmbedModel() EmbeddingModel {
	return bi.embedModel
}
