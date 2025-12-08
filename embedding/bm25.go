package embedding

import (
	"context"
	"math"
	"regexp"
	"strings"
	"sync"
)

// BM25 implements the BM25 sparse embedding model.
// BM25 (Best Matching 25) is a ranking function used for information retrieval.
type BM25 struct {
	// k1 controls term frequency saturation (typically 1.2-2.0).
	k1 float64
	// b controls document length normalization (typically 0.75).
	b float64
	// vocabulary maps terms to their indices.
	vocabulary map[string]int
	// idf stores inverse document frequency for each term.
	idf map[string]float64
	// avgDocLength is the average document length in the corpus.
	avgDocLength float64
	// numDocs is the total number of documents.
	numDocs int
	// docFreq stores document frequency for each term.
	docFreq map[string]int
	// mu protects concurrent access.
	mu sync.RWMutex
	// tokenizer is the function used to tokenize text.
	tokenizer func(string) []string
	// stopwords is the set of words to ignore.
	stopwords map[string]bool
}

// BM25Option configures a BM25 model.
type BM25Option func(*BM25)

// WithBM25K1 sets the k1 parameter.
func WithBM25K1(k1 float64) BM25Option {
	return func(b *BM25) {
		b.k1 = k1
	}
}

// WithBM25B sets the b parameter.
func WithBM25B(bParam float64) BM25Option {
	return func(b *BM25) {
		b.b = bParam
	}
}

// WithBM25Tokenizer sets a custom tokenizer.
func WithBM25Tokenizer(tokenizer func(string) []string) BM25Option {
	return func(b *BM25) {
		b.tokenizer = tokenizer
	}
}

// WithBM25Stopwords sets stopwords to ignore.
func WithBM25Stopwords(stopwords []string) BM25Option {
	return func(b *BM25) {
		b.stopwords = make(map[string]bool)
		for _, w := range stopwords {
			b.stopwords[strings.ToLower(w)] = true
		}
	}
}

// NewBM25 creates a new BM25 model.
func NewBM25(opts ...BM25Option) *BM25 {
	b := &BM25{
		k1:         1.5,
		b:          0.75,
		vocabulary: make(map[string]int),
		idf:        make(map[string]float64),
		docFreq:    make(map[string]int),
		tokenizer:  defaultTokenizer,
		stopwords:  defaultBM25Stopwords(),
	}

	for _, opt := range opts {
		opt(b)
	}

	return b
}

// Fit trains the BM25 model on a corpus of documents.
func (b *BM25) Fit(documents []string) {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.numDocs = len(documents)
	b.vocabulary = make(map[string]int)
	b.docFreq = make(map[string]int)
	b.idf = make(map[string]float64)

	var totalLength int
	vocabIndex := 0

	// First pass: collect vocabulary and document frequencies
	for _, doc := range documents {
		tokens := b.tokenize(doc)
		totalLength += len(tokens)

		// Track unique terms in this document
		seen := make(map[string]bool)
		for _, token := range tokens {
			if !seen[token] {
				seen[token] = true
				b.docFreq[token]++
			}

			if _, exists := b.vocabulary[token]; !exists {
				b.vocabulary[token] = vocabIndex
				vocabIndex++
			}
		}
	}

	b.avgDocLength = float64(totalLength) / float64(b.numDocs)

	// Calculate IDF for each term
	for term, df := range b.docFreq {
		// IDF with smoothing to avoid negative values
		b.idf[term] = math.Log((float64(b.numDocs)-float64(df)+0.5)/(float64(df)+0.5) + 1)
	}
}

// FitTransform fits the model and returns embeddings for all documents.
func (b *BM25) FitTransform(documents []string) []*SparseEmbedding {
	b.Fit(documents)

	embeddings := make([]*SparseEmbedding, len(documents))
	for i, doc := range documents {
		embeddings[i] = b.transform(doc)
	}

	return embeddings
}

// GetSparseEmbedding generates a sparse BM25 embedding for text.
func (b *BM25) GetSparseEmbedding(ctx context.Context, text string) (*SparseEmbedding, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	return b.transform(text), nil
}

// GetSparseQueryEmbedding generates a sparse embedding for a query.
// For BM25, query embeddings use binary term presence.
func (b *BM25) GetSparseQueryEmbedding(ctx context.Context, query string) (*SparseEmbedding, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	return b.transformQuery(query), nil
}

// GetSparseEmbeddingsBatch generates sparse embeddings for multiple texts.
func (b *BM25) GetSparseEmbeddingsBatch(ctx context.Context, texts []string) ([]*SparseEmbedding, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	embeddings := make([]*SparseEmbedding, len(texts))
	for i, text := range texts {
		embeddings[i] = b.transform(text)
	}

	return embeddings, nil
}

// transform converts a document to a sparse BM25 embedding.
func (bm *BM25) transform(text string) *SparseEmbedding {
	tokens := bm.tokenize(text)
	docLength := len(tokens)

	// Count term frequencies
	tf := make(map[string]int)
	for _, token := range tokens {
		tf[token]++
	}

	// Calculate BM25 scores
	var indices []int
	var values []float64

	for term, freq := range tf {
		idx, exists := bm.vocabulary[term]
		if !exists {
			continue
		}

		idf := bm.idf[term]
		if idf == 0 {
			continue
		}

		// BM25 formula
		tfNorm := float64(freq) * (bm.k1 + 1)
		tfDenom := float64(freq) + bm.k1*(1-bm.b+bm.b*(float64(docLength)/bm.avgDocLength))
		score := idf * (tfNorm / tfDenom)

		if score > 0 {
			indices = append(indices, idx)
			values = append(values, score)
		}
	}

	return &SparseEmbedding{
		Indices:   indices,
		Values:    values,
		Dimension: len(bm.vocabulary),
	}
}

// transformQuery converts a query to a sparse embedding.
func (b *BM25) transformQuery(query string) *SparseEmbedding {
	tokens := b.tokenize(query)

	// For queries, use IDF-weighted binary presence
	seen := make(map[string]bool)
	var indices []int
	var values []float64

	for _, token := range tokens {
		if seen[token] {
			continue
		}
		seen[token] = true

		idx, exists := b.vocabulary[token]
		if !exists {
			continue
		}

		idf := b.idf[token]
		if idf > 0 {
			indices = append(indices, idx)
			values = append(values, idf)
		}
	}

	return &SparseEmbedding{
		Indices:   indices,
		Values:    values,
		Dimension: len(b.vocabulary),
	}
}

// tokenize splits text into tokens.
func (b *BM25) tokenize(text string) []string {
	tokens := b.tokenizer(text)

	// Filter stopwords
	var filtered []string
	for _, token := range tokens {
		if !b.stopwords[token] {
			filtered = append(filtered, token)
		}
	}

	return filtered
}

// Score calculates the BM25 score between a query and document.
func (b *BM25) Score(query, document string) float64 {
	b.mu.RLock()
	defer b.mu.RUnlock()

	queryEmb := b.transformQuery(query)
	docEmb := b.transform(document)

	return queryEmb.DotProduct(docEmb)
}

// GetVocabularySize returns the vocabulary size.
func (b *BM25) GetVocabularySize() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return len(b.vocabulary)
}

// GetTermIDF returns the IDF for a term.
func (b *BM25) GetTermIDF(term string) float64 {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.idf[strings.ToLower(term)]
}

// defaultTokenizer is the default tokenization function.
func defaultTokenizer(text string) []string {
	// Convert to lowercase
	text = strings.ToLower(text)

	// Remove punctuation and split on whitespace
	re := regexp.MustCompile(`[^\w\s]`)
	text = re.ReplaceAllString(text, " ")

	// Split and filter empty strings
	words := strings.Fields(text)
	var tokens []string
	for _, w := range words {
		w = strings.TrimSpace(w)
		if len(w) > 0 {
			tokens = append(tokens, w)
		}
	}

	return tokens
}

// defaultBM25Stopwords returns default English stopwords.
func defaultBM25Stopwords() map[string]bool {
	words := []string{
		"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
		"of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
		"be", "have", "has", "had", "do", "does", "did", "will", "would",
		"could", "should", "may", "might", "must", "shall", "can", "need",
		"this", "that", "these", "those", "i", "you", "he", "she", "it",
		"we", "they", "what", "which", "who", "whom", "when", "where", "why",
		"how", "all", "each", "every", "both", "few", "more", "most", "other",
		"some", "such", "no", "nor", "not", "only", "own", "same", "so",
		"than", "too", "very", "just", "also", "now",
	}

	stopwords := make(map[string]bool)
	for _, w := range words {
		stopwords[w] = true
	}
	return stopwords
}

// BM25Plus implements BM25+ which addresses the issue of negative IDF.
type BM25Plus struct {
	*BM25
	// delta is the lower bound for term frequency normalization.
	delta float64
}

// NewBM25Plus creates a new BM25+ model.
func NewBM25Plus(delta float64, opts ...BM25Option) *BM25Plus {
	return &BM25Plus{
		BM25:  NewBM25(opts...),
		delta: delta,
	}
}

// GetSparseEmbedding generates a sparse BM25+ embedding.
func (bp *BM25Plus) GetSparseEmbedding(ctx context.Context, text string) (*SparseEmbedding, error) {
	bp.mu.RLock()
	defer bp.mu.RUnlock()

	tokens := bp.tokenize(text)
	docLength := len(tokens)

	tf := make(map[string]int)
	for _, token := range tokens {
		tf[token]++
	}

	var indices []int
	var values []float64

	for term, freq := range tf {
		idx, exists := bp.vocabulary[term]
		if !exists {
			continue
		}

		idf := bp.idf[term]
		if idf == 0 {
			continue
		}

		// BM25+ formula with delta
		tfNorm := float64(freq) * (bp.k1 + 1)
		tfDenom := float64(freq) + bp.k1*(1-bp.b+bp.b*(float64(docLength)/bp.avgDocLength))
		score := idf * ((tfNorm / tfDenom) + bp.delta)

		if score > 0 {
			indices = append(indices, idx)
			values = append(values, score)
		}
	}

	return &SparseEmbedding{
		Indices:   indices,
		Values:    values,
		Dimension: len(bp.vocabulary),
	}, nil
}

// Ensure interfaces are implemented.
var _ SparseEmbeddingModel = (*BM25)(nil)
var _ SparseEmbeddingModelWithBatch = (*BM25)(nil)
var _ SparseEmbeddingModel = (*BM25Plus)(nil)
