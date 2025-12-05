package index

import (
	"context"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/rag/queryengine"
	"github.com/aqua777/go-llamaindex/rag/retriever"
	"github.com/aqua777/go-llamaindex/rag/synthesizer"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/storage"
	"github.com/aqua777/go-llamaindex/storage/docstore"
	"github.com/aqua777/go-llamaindex/storage/indexstore"
)

// KeywordTableIndex is an index that maps keywords to nodes.
type KeywordTableIndex struct {
	*BaseIndex
	// keywordExtractor extracts keywords from text.
	keywordExtractor KeywordExtractor
	// maxKeywordsPerChunk limits keywords per node.
	maxKeywordsPerChunk int
}

// KeywordExtractor extracts keywords from text.
type KeywordExtractor interface {
	// ExtractKeywords extracts keywords from text.
	ExtractKeywords(ctx context.Context, text string, maxKeywords int) ([]string, error)
}

// SimpleKeywordExtractor extracts keywords using simple text processing.
type SimpleKeywordExtractor struct{}

// ExtractKeywords extracts keywords using simple word splitting.
func (e *SimpleKeywordExtractor) ExtractKeywords(ctx context.Context, text string, maxKeywords int) ([]string, error) {
	// Simple implementation: split by whitespace and filter
	words := strings.Fields(strings.ToLower(text))

	// Remove common stop words
	stopWords := map[string]bool{
		"the": true, "a": true, "an": true, "is": true, "are": true,
		"was": true, "were": true, "be": true, "been": true, "being": true,
		"have": true, "has": true, "had": true, "do": true, "does": true,
		"did": true, "will": true, "would": true, "could": true, "should": true,
		"may": true, "might": true, "must": true, "shall": true,
		"to": true, "of": true, "in": true, "for": true, "on": true,
		"with": true, "at": true, "by": true, "from": true, "as": true,
		"into": true, "through": true, "during": true, "before": true, "after": true,
		"above": true, "below": true, "between": true, "under": true, "again": true,
		"and": true, "but": true, "or": true, "nor": true, "so": true,
		"yet": true, "both": true, "either": true, "neither": true,
		"not": true, "only": true, "own": true, "same": true, "than": true,
		"too": true, "very": true, "just": true, "also": true,
		"this": true, "that": true, "these": true, "those": true,
		"i": true, "me": true, "my": true, "myself": true, "we": true,
		"our": true, "ours": true, "ourselves": true, "you": true, "your": true,
		"yours": true, "yourself": true, "yourselves": true, "he": true, "him": true,
		"his": true, "himself": true, "she": true, "her": true, "hers": true,
		"herself": true, "it": true, "its": true, "itself": true, "they": true,
		"them": true, "their": true, "theirs": true, "themselves": true,
		"what": true, "which": true, "who": true, "whom": true, "whose": true,
		"when": true, "where": true, "why": true, "how": true,
	}

	// Count word frequencies
	wordCount := make(map[string]int)
	for _, word := range words {
		// Clean word
		word = strings.Trim(word, ".,!?;:\"'()[]{}#@$%^&*-_=+<>/\\|`~")
		if len(word) < 3 || stopWords[word] {
			continue
		}
		wordCount[word]++
	}

	// Sort by frequency
	type wordFreq struct {
		word  string
		count int
	}
	var sorted []wordFreq
	for word, count := range wordCount {
		sorted = append(sorted, wordFreq{word, count})
	}
	for i := 0; i < len(sorted)-1; i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j].count > sorted[i].count {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	// Return top keywords
	var keywords []string
	for i := 0; i < len(sorted) && i < maxKeywords; i++ {
		keywords = append(keywords, sorted[i].word)
	}

	return keywords, nil
}

// KeywordTableIndexOption configures KeywordTableIndex creation.
type KeywordTableIndexOption func(*KeywordTableIndex)

// WithKeywordExtractor sets the keyword extractor.
func WithKeywordExtractor(extractor KeywordExtractor) KeywordTableIndexOption {
	return func(kti *KeywordTableIndex) {
		kti.keywordExtractor = extractor
	}
}

// WithMaxKeywordsPerChunk sets the maximum keywords per chunk.
func WithMaxKeywordsPerChunk(max int) KeywordTableIndexOption {
	return func(kti *KeywordTableIndex) {
		kti.maxKeywordsPerChunk = max
	}
}

// WithKeywordTableStorageContext sets the storage context.
func WithKeywordTableStorageContext(sc *storage.StorageContext) KeywordTableIndexOption {
	return func(kti *KeywordTableIndex) {
		kti.storageContext = sc
	}
}

// NewKeywordTableIndex creates a new KeywordTableIndex.
func NewKeywordTableIndex(ctx context.Context, nodes []schema.Node, opts ...KeywordTableIndexOption) (*KeywordTableIndex, error) {
	indexStruct := indexstore.NewKeywordTableIndex()

	kti := &KeywordTableIndex{
		BaseIndex:           NewBaseIndex(indexStruct),
		keywordExtractor:    &SimpleKeywordExtractor{},
		maxKeywordsPerChunk: 10,
	}

	for _, opt := range opts {
		opt(kti)
	}

	// Build index from nodes
	if len(nodes) > 0 {
		if err := kti.buildIndexFromNodes(ctx, nodes); err != nil {
			return nil, err
		}
	}

	// Add index struct to store
	if err := kti.storageContext.IndexStore.AddIndexStruct(ctx, indexStruct); err != nil {
		return nil, err
	}

	return kti, nil
}

// NewKeywordTableIndexFromDocuments creates a KeywordTableIndex from documents.
func NewKeywordTableIndexFromDocuments(
	ctx context.Context,
	documents []schema.Document,
	opts ...KeywordTableIndexOption,
) (*KeywordTableIndex, error) {
	// Convert documents to nodes
	var nodes []schema.Node
	for _, doc := range documents {
		node := schema.NewTextNode(doc.Text)
		node.Metadata = doc.Metadata
		if doc.ID != "" {
			node.ID = doc.ID
		}
		nodes = append(nodes, *node)
	}

	return NewKeywordTableIndex(ctx, nodes, opts...)
}

// buildIndexFromNodes builds the index from nodes.
func (kti *KeywordTableIndex) buildIndexFromNodes(ctx context.Context, nodes []schema.Node) error {
	for _, node := range nodes {
		// Extract keywords
		keywords, err := kti.keywordExtractor.ExtractKeywords(ctx, node.GetContent(schema.MetadataModeEmbed), kti.maxKeywordsPerChunk)
		if err != nil {
			return err
		}

		// Add to index
		kti.indexStruct.AddToTable(keywords, node.ID)

		// Add to docstore
		if err := kti.storageContext.DocStore.AddDocuments(ctx, []schema.BaseNode{&node}, true); err != nil {
			return err
		}
	}

	return nil
}

// AsRetriever returns a retriever for this index.
func (kti *KeywordTableIndex) AsRetriever(opts ...RetrieverOption) retriever.Retriever {
	config := &RetrieverConfig{
		SimilarityTopK: 10,
	}

	for _, opt := range opts {
		opt(config)
	}

	return &KeywordTableRetriever{
		index:             kti,
		maxKeywordsPerQuery: 10,
		numChunksPerQuery: config.SimilarityTopK,
	}
}

// AsQueryEngine returns a query engine for this index.
func (kti *KeywordTableIndex) AsQueryEngine(opts ...QueryEngineOption) queryengine.QueryEngine {
	config := &QueryEngineConfig{
		ResponseMode: synthesizer.ResponseModeCompact,
	}

	for _, opt := range opts {
		opt(config)
	}

	// Create retriever
	ret := kti.AsRetriever()

	// Create synthesizer
	var synth synthesizer.Synthesizer
	if config.Synthesizer != nil {
		synth = config.Synthesizer
	} else if config.LLM != nil {
		synth, _ = synthesizer.GetSynthesizer(config.ResponseMode, config.LLM)
	} else {
		synth = synthesizer.NewSimpleSynthesizer(llm.NewMockLLM(""))
	}

	return queryengine.NewRetrieverQueryEngine(ret, synth)
}

// InsertNodes inserts nodes into the index.
func (kti *KeywordTableIndex) InsertNodes(ctx context.Context, nodes []schema.Node) error {
	for _, node := range nodes {
		// Extract keywords
		keywords, err := kti.keywordExtractor.ExtractKeywords(ctx, node.GetContent(schema.MetadataModeEmbed), kti.maxKeywordsPerChunk)
		if err != nil {
			return err
		}

		// Add to index
		kti.indexStruct.AddToTable(keywords, node.ID)

		// Add to docstore
		if err := kti.storageContext.DocStore.AddDocuments(ctx, []schema.BaseNode{&node}, true); err != nil {
			return err
		}
	}

	// Update index store
	return kti.storageContext.IndexStore.AddIndexStruct(ctx, kti.indexStruct)
}

// DeleteNodes removes nodes from the index.
func (kti *KeywordTableIndex) DeleteNodes(ctx context.Context, nodeIDs []string) error {
	deleteSet := make(map[string]bool)
	for _, id := range nodeIDs {
		deleteSet[id] = true
	}

	// Remove from keyword table
	for keyword, ids := range kti.indexStruct.Table {
		var newIDs []string
		for _, id := range ids {
			if !deleteSet[id] {
				newIDs = append(newIDs, id)
			}
		}
		if len(newIDs) > 0 {
			kti.indexStruct.Table[keyword] = newIDs
		} else {
			delete(kti.indexStruct.Table, keyword)
		}
	}

	// Delete from docstore
	for _, nodeID := range nodeIDs {
		if err := kti.storageContext.DocStore.DeleteDocument(ctx, nodeID, false); err != nil {
			// Continue even if delete fails
		}
	}

	// Update index store
	return kti.storageContext.IndexStore.AddIndexStruct(ctx, kti.indexStruct)
}

// RefreshDocuments refreshes the index with updated documents.
func (kti *KeywordTableIndex) RefreshDocuments(ctx context.Context, documents []schema.Document) ([]bool, error) {
	refreshed := make([]bool, len(documents))

	for i, doc := range documents {
		existingHash, err := kti.storageContext.DocStore.GetDocumentHash(ctx, doc.ID)
		if err != nil || existingHash == "" {
			node := schema.NewTextNode(doc.Text)
			node.Metadata = doc.Metadata
			if doc.ID != "" {
				node.ID = doc.ID
			}
			if err := kti.InsertNodes(ctx, []schema.Node{*node}); err != nil {
				return refreshed, err
			}
			refreshed[i] = true
		} else if existingHash != doc.GetHash() {
			if err := kti.DeleteNodes(ctx, []string{doc.ID}); err != nil {
				return refreshed, err
			}
			node := schema.NewTextNode(doc.Text)
			node.Metadata = doc.Metadata
			if doc.ID != "" {
				node.ID = doc.ID
			}
			if err := kti.InsertNodes(ctx, []schema.Node{*node}); err != nil {
				return refreshed, err
			}
			refreshed[i] = true
		}
	}

	return refreshed, nil
}

// KeywordTableRetriever retrieves nodes from a KeywordTableIndex.
type KeywordTableRetriever struct {
	index               *KeywordTableIndex
	maxKeywordsPerQuery int
	numChunksPerQuery   int
}

// Retrieve retrieves nodes for a query.
func (r *KeywordTableRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	// Extract keywords from query
	keywords, err := r.index.keywordExtractor.ExtractKeywords(ctx, query.QueryString, r.maxKeywordsPerQuery)
	if err != nil {
		return nil, err
	}

	// Find matching node IDs
	nodeIDSet := make(map[string]int) // node ID -> match count
	for _, keyword := range keywords {
		if nodeIDs, ok := r.index.indexStruct.Table[keyword]; ok {
			for _, nodeID := range nodeIDs {
				nodeIDSet[nodeID]++
			}
		}
	}

	// Sort by match count
	type nodeMatch struct {
		nodeID string
		count  int
	}
	var matches []nodeMatch
	for nodeID, count := range nodeIDSet {
		matches = append(matches, nodeMatch{nodeID, count})
	}
	for i := 0; i < len(matches)-1; i++ {
		for j := i + 1; j < len(matches); j++ {
			if matches[j].count > matches[i].count {
				matches[i], matches[j] = matches[j], matches[i]
			}
		}
	}

	// Get top-k node IDs
	k := r.numChunksPerQuery
	if k <= 0 || k > len(matches) {
		k = len(matches)
	}

	var nodeIDs []string
	for i := 0; i < k; i++ {
		nodeIDs = append(nodeIDs, matches[i].nodeID)
	}

	if len(nodeIDs) == 0 {
		return nil, nil
	}

	// Get nodes from docstore
	nodes, err := docstore.GetNodes(ctx, r.index.storageContext.DocStore, nodeIDs, false)
	if err != nil {
		return nil, err
	}

	// Convert to NodeWithScore
	results := make([]schema.NodeWithScore, 0, len(nodes))
	for i, n := range nodes {
		if node, ok := n.(*schema.Node); ok {
			score := float64(matches[i].count) / float64(len(keywords))
			results = append(results, schema.NodeWithScore{Node: *node, Score: score})
		}
	}

	return results, nil
}

// Ensure KeywordTableIndex implements Index.
var _ Index = (*KeywordTableIndex)(nil)

// Ensure KeywordTableRetriever implements Retriever.
var _ retriever.Retriever = (*KeywordTableRetriever)(nil)
