package retriever

import (
	"context"
	"fmt"
	"sort"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/rag/store"
	"github.com/aqua777/go-llamaindex/schema"
)

// BM25Retriever retrieves nodes from a fixed in-memory corpus using BM25 scoring.
//
// It does not delegate to a VectorStore. The corpus is provided at construction
// time; BM25 is fitted once over those node texts. Subsequent calls to Retrieve
// score the query against every node in the corpus that passes metadata filters,
// then return the top-K results sorted by score descending.
//
// The retriever is safe for concurrent reads after construction. The corpus is
// immutable after construction; to update the corpus, create a new BM25Retriever.
type BM25Retriever struct {
	*BaseRetriever
	nodes []schema.Node
	bm25  *embedding.BM25
	topK  int
}

// bm25RetrieverBuilder accumulates options before the retriever is fully
// constructed, so that WithBM25Options can forward parameters to the BM25
// constructor before the model is created.
type bm25RetrieverBuilder struct {
	topK      int
	bm25Opts  []embedding.BM25Option
	bm25Model *embedding.BM25
}

// BM25RetrieverOption is a functional option for BM25Retriever.
type BM25RetrieverOption func(*bm25RetrieverBuilder)

// WithBM25TopK sets the maximum number of results returned by Retrieve.
// topK must be greater than zero; if not, Retrieve will return an error.
func WithBM25TopK(topK int) BM25RetrieverOption {
	return func(b *bm25RetrieverBuilder) {
		b.topK = topK
	}
}

// WithBM25Options passes BM25 construction options through to the underlying
// embedding.BM25 model created during NewBM25Retriever. Has no effect when
// WithBM25Model is also provided (caller-supplied model is used as-is).
func WithBM25Options(opts ...embedding.BM25Option) BM25RetrieverOption {
	return func(b *bm25RetrieverBuilder) {
		b.bm25Opts = append(b.bm25Opts, opts...)
	}
}

// WithBM25Model replaces the internally created BM25 model with a pre-fitted
// caller-supplied model. The provided model must already have been fitted on a
// corpus that covers the nodes passed to NewBM25Retriever; no re-fitting is
// performed when this option is used.
func WithBM25Model(model *embedding.BM25) BM25RetrieverOption {
	return func(b *bm25RetrieverBuilder) {
		b.bm25Model = model
	}
}

// NewBM25Retriever creates a BM25Retriever over the supplied node corpus.
//
// Unless WithBM25Model is provided, a new embedding.BM25 is created with any
// BM25Options forwarded via WithBM25Options, and it is immediately fitted on
// the text of every node in corpus.
//
// Args:
//
//	corpus: Nodes to index. May not be nil; may be empty, in which case
//	        Retrieve returns an error (unfitted model, no corpus).
//	opts:   Zero or more BM25RetrieverOption values.
//
// Returns:
//
//	A ready-to-use BM25Retriever with a fitted BM25 model.
func NewBM25Retriever(corpus []schema.Node, opts ...BM25RetrieverOption) *BM25Retriever {
	builder := &bm25RetrieverBuilder{
		topK: 10,
	}

	for _, opt := range opts {
		opt(builder)
	}

	var model *embedding.BM25
	if builder.bm25Model != nil {
		model = builder.bm25Model
	} else {
		model = embedding.NewBM25(builder.bm25Opts...)
		texts := nodeTexts(corpus)
		if len(texts) > 0 {
			model.Fit(texts)
		}
	}

	return &BM25Retriever{
		BaseRetriever: NewBaseRetriever(),
		nodes:         corpus,
		bm25:          model,
		topK:          builder.topK,
	}
}

// Retrieve returns up to topK nodes from the corpus ranked by BM25 score for
// the given query, in descending score order.
//
// Processing steps:
//  1. Return an error if the corpus was empty at construction (unfitted BM25).
//  2. Filter corpus to nodes satisfying query.Filters (same MatchesFilters logic
//     as SimpleVectorStore).
//  3. Score each surviving node with bm25.Score(query.QueryString, node.Text).
//  4. Sort scored nodes descending.
//  5. Return top topK results (or all if fewer than topK remain).
//  6. Delegate to BaseRetriever.HandleRecursiveRetrieval for IndexNode resolution.
//
// Args:
//
//	ctx:   Request context; forwarded to HandleRecursiveRetrieval.
//	query: QueryString drives BM25 scoring; Filters applied before scoring.
//
// Returns:
//
//	[]schema.NodeWithScore sorted by score descending.
//
// Raises:
//
//	error: When BM25 model is not fitted (empty corpus at construction).
//	error: When topK is not greater than zero.
//	error: When HandleRecursiveRetrieval fails.
func (br *BM25Retriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	if len(br.nodes) == 0 {
		return nil, fmt.Errorf("bm25 retriever: corpus is empty, model is not fitted")
	}

	if br.topK <= 0 {
		return nil, fmt.Errorf("bm25 retriever: topK must be greater than zero, got %d", br.topK)
	}

	candidates := filterNodes(br.nodes, query.Filters)
	scored := scoreCandidates(br.bm25, query.QueryString, candidates)
	sortNodesByScore(scored)
	top := limitTopK(scored, br.topK)

	return br.HandleRecursiveRetrieval(ctx, query, top)
}

// filterNodes returns only those nodes that pass the metadata filters.
func filterNodes(nodes []schema.Node, filters *schema.MetadataFilters) []schema.Node {
	if filters == nil || len(filters.Filters) == 0 {
		return nodes
	}
	result := make([]schema.Node, 0, len(nodes))
	for _, n := range nodes {
		if store.MatchesFilters(n.Metadata, filters) {
			result = append(result, n)
		}
	}
	return result
}

// scoreCandidates scores each candidate node against the query using BM25.
func scoreCandidates(model *embedding.BM25, queryString string, candidates []schema.Node) []schema.NodeWithScore {
	result := make([]schema.NodeWithScore, 0, len(candidates))
	for _, n := range candidates {
		score := model.Score(queryString, n.Text)
		result = append(result, schema.NodeWithScore{Node: n, Score: score})
	}
	return result
}

// sortNodesByScore sorts a slice of NodeWithScore in descending order by score.
func sortNodesByScore(nodes []schema.NodeWithScore) {
	sort.Slice(nodes, func(i, j int) bool {
		return nodes[i].Score > nodes[j].Score
	})
}

// limitTopK returns the first k elements from nodes, or all elements if len(nodes) <= k.
func limitTopK(nodes []schema.NodeWithScore, k int) []schema.NodeWithScore {
	if k >= len(nodes) {
		return nodes
	}
	return nodes[:k]
}

// nodeTexts extracts the text from each node for BM25 fitting.
func nodeTexts(nodes []schema.Node) []string {
	texts := make([]string, len(nodes))
	for i, n := range nodes {
		texts[i] = n.Text
	}
	return texts
}

// Compile-time assertion: BM25Retriever implements Retriever.
var _ Retriever = (*BM25Retriever)(nil)
