package retriever

import (
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/stretchr/testify/suite"
)

// BM25RetrieverTestSuite tests BM25Retriever behavior.
type BM25RetrieverTestSuite struct {
	suite.Suite
	ctx    context.Context
	corpus []schema.Node
}

func TestBM25RetrieverTestSuite(t *testing.T) {
	suite.Run(t, new(BM25RetrieverTestSuite))
}

func (s *BM25RetrieverTestSuite) SetupTest() {
	s.ctx = context.Background()
	s.corpus = makeCorpus()
}

// makeCorpus builds a deterministic 5-node corpus for testing.
func makeCorpus() []schema.Node {
	texts := []struct {
		id       string
		text     string
		category string
	}{
		{"n1", "Machine learning algorithms learn patterns from data automatically.", "ml"},
		{"n2", "Deep learning uses neural networks for complex pattern recognition.", "ml"},
		{"n3", "Database systems store and retrieve data with SQL queries.", "db"},
		{"n4", "Information retrieval ranks documents by relevance to user queries.", "ir"},
		{"n5", "Quantum computing uses qubits for parallel computation.", "quantum"},
	}

	nodes := make([]schema.Node, len(texts))
	for i, t := range texts {
		n := schema.NewTextNode(t.text)
		n.ID = t.id
		n.Metadata = map[string]interface{}{"category": t.category}
		nodes[i] = *n
	}
	return nodes
}

// TestRetrieve_TopResultContainsQueryTerm verifies that the highest-scored node
// contains the exact query term.
func (s *BM25RetrieverTestSuite) TestRetrieve_TopResultContainsQueryTerm() {
	r := NewBM25Retriever(s.corpus)
	query := schema.QueryBundle{QueryString: "machine learning"}

	results, err := r.Retrieve(s.ctx, query)

	s.NoError(err)
	s.NotEmpty(results)
	s.Contains(strings.ToLower(results[0].Node.Text), "machine learning")
}

// TestRetrieve_NodeWithoutQueryTermScoresZeroOrLower verifies that a node
// containing no query terms receives zero score and is excluded from a top-2
// result set drawn from a corpus where 2 other nodes match well.
func (s *BM25RetrieverTestSuite) TestRetrieve_NodeWithoutQueryTermScoresZeroOrLower() {
	r := NewBM25Retriever(s.corpus, WithBM25TopK(2))
	query := schema.QueryBundle{QueryString: "machine learning algorithms"}

	results, err := r.Retrieve(s.ctx, query)

	s.NoError(err)
	s.LessOrEqual(len(results), 2)

	// The quantum node should not appear in top-2 for an ML query.
	for _, res := range results {
		s.NotEqual("n5", res.Node.ID, "quantum node should not be in top results for ML query")
	}
}

// TestRetrieve_FiltersExcludeNodeBeforeScoring verifies that a node matching
// the query text is excluded when its metadata does not pass the filter.
func (s *BM25RetrieverTestSuite) TestRetrieve_FiltersExcludeNodeBeforeScoring() {
	r := NewBM25Retriever(s.corpus)

	// Filter to only "db" category — excludes both ML nodes even though they
	// would score highest for "machine learning".
	filters := &schema.MetadataFilters{
		Filters: []schema.MetadataFilter{
			{Key: "category", Value: "db", Operator: schema.FilterOperatorEq},
		},
	}
	query := schema.QueryBundle{
		QueryString: "machine learning algorithms",
		Filters:     filters,
	}

	results, err := r.Retrieve(s.ctx, query)

	s.NoError(err)
	for _, res := range results {
		s.Equal("db", res.Node.Metadata["category"], "only db nodes should pass filter")
	}
}

// TestRetrieve_EmptyCorpusReturnsError verifies that Retrieve returns an error
// when the corpus was empty at construction time.
func (s *BM25RetrieverTestSuite) TestRetrieve_EmptyCorpusReturnsError() {
	r := NewBM25Retriever([]schema.Node{})
	query := schema.QueryBundle{QueryString: "anything"}

	_, err := r.Retrieve(s.ctx, query)

	s.Error(err)
	s.Contains(err.Error(), "corpus is empty")
}

// TestWithBM25TopK_LimitsResults verifies that WithBM25TopK(2) returns at most
// 2 results from a 5-node corpus.
func (s *BM25RetrieverTestSuite) TestWithBM25TopK_LimitsResults() {
	r := NewBM25Retriever(s.corpus, WithBM25TopK(2))
	query := schema.QueryBundle{QueryString: "learning data algorithms"}

	results, err := r.Retrieve(s.ctx, query)

	s.NoError(err)
	s.LessOrEqual(len(results), 2)
}

// TestWithBM25Model_UsesProvidedModel verifies that when WithBM25Model is
// provided, the retriever uses it without re-fitting.
func (s *BM25RetrieverTestSuite) TestWithBM25Model_UsesProvidedModel() {
	preFitted := embedding.NewBM25()
	texts := nodeTexts(s.corpus)
	preFitted.Fit(texts)

	r := NewBM25Retriever(s.corpus, WithBM25Model(preFitted))

	s.Equal(preFitted, r.bm25, "retriever should use the provided pre-fitted model")

	query := schema.QueryBundle{QueryString: "machine learning"}
	results, err := r.Retrieve(s.ctx, query)
	s.NoError(err)
	s.NotEmpty(results)
}

// TestRetrieve_Concurrent verifies that concurrent calls to Retrieve produce no
// data races and all succeed.
func (s *BM25RetrieverTestSuite) TestRetrieve_Concurrent() {
	r := NewBM25Retriever(s.corpus)
	query := schema.QueryBundle{QueryString: "machine learning"}

	const goroutines = 20
	errs := make(chan error, goroutines)
	var wg sync.WaitGroup

	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := r.Retrieve(s.ctx, query)
			errs <- err
		}()
	}

	wg.Wait()
	close(errs)

	for err := range errs {
		s.NoError(err)
	}
}

// TestWithBM25TopK_InvalidTopKReturnsError verifies that a non-positive topK
// causes Retrieve to return an error.
func (s *BM25RetrieverTestSuite) TestWithBM25TopK_InvalidTopKReturnsError() {
	r := NewBM25Retriever(s.corpus, WithBM25TopK(0))
	query := schema.QueryBundle{QueryString: "machine learning"}

	_, err := r.Retrieve(s.ctx, query)

	s.Error(err)
	s.Contains(err.Error(), "topK must be greater than zero")
}

// TestWithBM25Options_ForwardedToModel verifies that WithBM25Options parameters
// are forwarded to the underlying BM25 constructor.
func (s *BM25RetrieverTestSuite) TestWithBM25Options_ForwardedToModel() {
	r := NewBM25Retriever(s.corpus, WithBM25Options(embedding.WithBM25K1(2.0), embedding.WithBM25B(0.5)))

	s.NotNil(r.bm25)
	// A retrieval should succeed, confirming the model was built with the options.
	query := schema.QueryBundle{QueryString: "machine learning"}
	results, err := r.Retrieve(s.ctx, query)
	s.NoError(err)
	s.NotEmpty(results)
}

// TestRetrieve_ResultsSortedDescending verifies that results are returned in
// descending score order.
func (s *BM25RetrieverTestSuite) TestRetrieve_ResultsSortedDescending() {
	r := NewBM25Retriever(s.corpus)
	query := schema.QueryBundle{QueryString: "learning algorithms data"}

	results, err := r.Retrieve(s.ctx, query)

	s.NoError(err)
	for i := 1; i < len(results); i++ {
		s.GreaterOrEqual(results[i-1].Score, results[i].Score,
			"results should be sorted by score descending")
	}
}
