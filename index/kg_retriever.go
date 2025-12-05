package index

import (
	"context"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/aqua777/go-llamaindex/schema"
)

const (
	// DefaultNodeScore is the default score for nodes found by keyword.
	DefaultNodeScore = 1000.0
	// GlobalExploreNodeLimit limits how many nodes to explore per keyword.
	GlobalExploreNodeLimit = 3
	// DefaultRelTextLimit is the default limit for relation texts.
	DefaultRelTextLimit = 30
)

// KGTableRetriever retrieves nodes from a KnowledgeGraphIndex.
// It supports keyword, embedding, and hybrid retrieval modes.
type KGTableRetriever struct {
	index                 *KnowledgeGraphIndex
	mode                  KGRetrieverMode
	maxKeywordsPerQuery   int
	numChunksPerQuery     int
	includeText           bool
	similarityTopK        int
	graphStoreQueryDepth  int
	useGlobalNodeTriplets bool
	maxKnowledgeSequence  int
	verbose               bool
}

// KGRetrieverOption configures KGTableRetriever.
type KGRetrieverOption func(*KGTableRetriever)

// WithKGRetrieverMode sets the retrieval mode.
func WithKGRetrieverMode(mode KGRetrieverMode) KGRetrieverOption {
	return func(r *KGTableRetriever) {
		r.mode = mode
	}
}

// WithKGRetrieverMaxKeywords sets the max keywords per query.
func WithKGRetrieverMaxKeywords(n int) KGRetrieverOption {
	return func(r *KGTableRetriever) {
		if n > 0 {
			r.maxKeywordsPerQuery = n
		}
	}
}

// WithKGRetrieverNumChunks sets the number of chunks per query.
func WithKGRetrieverNumChunks(n int) KGRetrieverOption {
	return func(r *KGTableRetriever) {
		if n > 0 {
			r.numChunksPerQuery = n
		}
	}
}

// WithKGRetrieverIncludeText sets whether to include text from nodes.
func WithKGRetrieverIncludeText(include bool) KGRetrieverOption {
	return func(r *KGTableRetriever) {
		r.includeText = include
	}
}

// WithKGRetrieverSimilarityTopK sets the top-k for embedding similarity.
func WithKGRetrieverSimilarityTopK(k int) KGRetrieverOption {
	return func(r *KGTableRetriever) {
		if k > 0 {
			r.similarityTopK = k
		}
	}
}

// WithKGRetrieverGraphDepth sets the graph traversal depth.
func WithKGRetrieverGraphDepth(depth int) KGRetrieverOption {
	return func(r *KGTableRetriever) {
		if depth > 0 {
			r.graphStoreQueryDepth = depth
		}
	}
}

// WithKGRetrieverUseGlobalTriplets enables global node triplet exploration.
func WithKGRetrieverUseGlobalTriplets(use bool) KGRetrieverOption {
	return func(r *KGTableRetriever) {
		r.useGlobalNodeTriplets = use
	}
}

// WithKGRetrieverMaxKnowledgeSequence sets the max knowledge sequence.
func WithKGRetrieverMaxKnowledgeSequence(n int) KGRetrieverOption {
	return func(r *KGTableRetriever) {
		if n > 0 {
			r.maxKnowledgeSequence = n
		}
	}
}

// WithKGRetrieverVerbose enables verbose output.
func WithKGRetrieverVerbose(verbose bool) KGRetrieverOption {
	return func(r *KGTableRetriever) {
		r.verbose = verbose
	}
}

// NewKGTableRetriever creates a new KGTableRetriever.
func NewKGTableRetriever(index *KnowledgeGraphIndex, opts ...KGRetrieverOption) *KGTableRetriever {
	r := &KGTableRetriever{
		index:                 index,
		mode:                  KGRetrieverModeKeyword,
		maxKeywordsPerQuery:   index.MaxKeywordsPerQuery(),
		numChunksPerQuery:     10,
		includeText:           true,
		similarityTopK:        2,
		graphStoreQueryDepth:  index.GraphStoreQueryDepth(),
		useGlobalNodeTriplets: false,
		maxKnowledgeSequence:  DefaultRelTextLimit,
		verbose:               false,
	}

	for _, opt := range opts {
		opt(r)
	}

	return r
}

// Retrieve retrieves nodes for the given query bundle.
func (r *KGTableRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	return r.retrieve(ctx, &query)
}

// retrieve is the internal implementation.
func (r *KGTableRetriever) retrieve(ctx context.Context, queryBundle *schema.QueryBundle) ([]schema.NodeWithScore, error) {
	return r.doRetrieve(ctx, queryBundle)
}

// doRetrieve is the core retrieval implementation.
func (r *KGTableRetriever) doRetrieve(ctx context.Context, queryBundle *schema.QueryBundle) ([]schema.NodeWithScore, error) {
	nodeVisited := make(map[string]bool)
	var relTexts []string
	curRelMap := make(map[string][][]string)
	chunkIndicesCount := make(map[string]int)

	// Extract keywords from query
	keywords, err := r.getKeywords(ctx, queryBundle.QueryString)
	if err != nil {
		return nil, err
	}

	// Keyword-based retrieval
	if r.mode != KGRetrieverModeEmbedding {
		for _, keyword := range keywords {
			subjs := map[string]bool{keyword: true}

			// Get node IDs for this keyword
			nodeIDs := r.index.SearchNodeByKeyword(keyword)
			for i, nodeID := range nodeIDs {
				if i >= GlobalExploreNodeLimit {
					break
				}
				if nodeVisited[nodeID] {
					continue
				}

				if r.includeText {
					chunkIndicesCount[nodeID]++
				}

				nodeVisited[nodeID] = true

				// Optionally expand subjects from node content
				if r.useGlobalNodeTriplets {
					doc, err := r.index.storageContext.DocStore.GetDocument(ctx, nodeID, false)
					if err == nil && doc != nil {
						extendedSubjs, _ := r.getKeywords(ctx, doc.GetContent(schema.MetadataModeLLM))
						for _, s := range extendedSubjs {
							subjs[s] = true
						}
					}
				}
			}

			// Get relation map from graph store
			subjList := make([]string, 0, len(subjs))
			for s := range subjs {
				subjList = append(subjList, s)
			}

			relMap, err := r.index.graphStore.GetRelMap(ctx, subjList, r.graphStoreQueryDepth, r.maxKnowledgeSequence)
			if err != nil {
				continue
			}

			if len(relMap) == 0 {
				continue
			}

			for subj, rels := range relMap {
				for _, rel := range rels {
					relTexts = append(relTexts, formatRelText(rel))
				}
				curRelMap[subj] = rels
			}
		}
	}

	// Embedding-based retrieval
	if r.mode != KGRetrieverModeKeyword && len(r.index.indexStruct.EmbeddingDict) > 0 {
		if r.index.embedModel != nil {
			queryEmbedding, err := r.index.embedModel.GetQueryEmbedding(ctx, queryBundle.QueryString)
			if err == nil {
				topRelTexts := r.getTopKEmbeddings(queryEmbedding, r.similarityTopK)
				relTexts = append(relTexts, topRelTexts...)
			}
		}
	}

	// Remove duplicates for hybrid mode
	if r.mode == KGRetrieverModeHybrid {
		relTexts = removeDuplicates(relTexts)
		// Remove shorter rel_texts that are substrings of longer ones
		relTexts = removeSubstrings(relTexts)
		// Truncate to max
		if len(relTexts) > r.maxKnowledgeSequence {
			relTexts = relTexts[:r.maxKnowledgeSequence]
		}
	}

	// Get nodes for keywords extracted from relation texts
	if r.includeText {
		relKeywords := extractRelTextKeywords(relTexts)
		for _, keyword := range relKeywords {
			nodeIDs := r.index.SearchNodeByKeyword(keyword)
			for _, nodeID := range nodeIDs {
				chunkIndicesCount[nodeID]++
			}
		}
	}

	// Sort chunks by count and get top ones
	sortedChunkIndices := sortByCount(chunkIndicesCount, r.numChunksPerQuery)

	// Get nodes from docstore
	var sortedNodesWithScores []schema.NodeWithScore
	for _, chunkIdx := range sortedChunkIndices {
		doc, err := r.index.storageContext.DocStore.GetDocument(ctx, chunkIdx, false)
		if err != nil || doc == nil {
			continue
		}
		if node, ok := doc.(*schema.Node); ok {
			sortedNodesWithScores = append(sortedNodesWithScores, schema.NodeWithScore{
				Node:  *node,
				Score: DefaultNodeScore,
			})
		}
	}

	// If no relationships found, return nodes found by keywords
	if len(relTexts) == 0 {
		if len(sortedNodesWithScores) == 0 {
			return []schema.NodeWithScore{
				{
					Node:  *schema.NewTextNode("No relationships found."),
					Score: 1.0,
				},
			}, nil
		}
		return sortedNodesWithScores, nil
	}

	// Add relationships as a node
	relInitialText := fmt.Sprintf(
		"The following are knowledge sequences in max depth %d in the form of directed graph like:\n"+
			"`subject -[predicate]-> object, <-[predicate_next_hop]- object_next_hop ...`",
		r.graphStoreQueryDepth,
	)

	relInfoText := relInitialText + "\n" + strings.Join(relTexts, "\n")

	relTextNode := schema.NewTextNode(relInfoText)
	relTextNode.Metadata = map[string]interface{}{
		"kg_rel_texts": relTexts,
		"kg_rel_map":   curRelMap,
	}

	sortedNodesWithScores = append(sortedNodesWithScores, schema.NodeWithScore{
		Node:  *relTextNode,
		Score: DefaultNodeScore,
	})

	return sortedNodesWithScores, nil
}

// getKeywords extracts keywords from text using the LLM.
func (r *KGTableRetriever) getKeywords(ctx context.Context, text string) ([]string, error) {
	if r.index.llm == nil {
		// Fall back to simple keyword extraction
		return simpleKeywordExtract(text, r.maxKeywordsPerQuery), nil
	}

	prompt := r.index.keywordExtractTemplate.Format(map[string]string{
		"max_keywords": fmt.Sprintf("%d", r.maxKeywordsPerQuery),
		"question":     text,
	})

	response, err := r.index.llm.Complete(ctx, prompt)
	if err != nil {
		return simpleKeywordExtract(text, r.maxKeywordsPerQuery), nil
	}

	return extractKeywordsFromResponse(response, r.maxKeywordsPerQuery), nil
}

// getTopKEmbeddings returns the top-k most similar triplet texts.
func (r *KGTableRetriever) getTopKEmbeddings(queryEmbedding []float64, topK int) []string {
	type embeddingScore struct {
		text  string
		score float64
	}

	var scores []embeddingScore
	for text, embedding := range r.index.indexStruct.EmbeddingDict {
		score := kgCosineSimilarity(queryEmbedding, embedding)
		scores = append(scores, embeddingScore{text: text, score: score})
	}

	// Sort by score descending
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	var result []string
	for i := 0; i < topK && i < len(scores); i++ {
		result = append(result, scores[i].text)
	}

	return result
}

// Name returns the retriever name.
func (r *KGTableRetriever) Name() string {
	return "KGTableRetriever"
}

// Helper functions

// formatRelText formats a relation triplet as a string.
func formatRelText(rel []string) string {
	if len(rel) >= 3 {
		return fmt.Sprintf("[%s, %s, %s]", rel[0], rel[1], rel[2])
	}
	return strings.Join(rel, ", ")
}

// extractRelTextKeywords extracts keywords from relation texts.
func extractRelTextKeywords(relTexts []string) []string {
	var keywords []string
	for _, relText := range relTexts {
		// Split by comma and extract subject and object
		parts := strings.Split(relText, ",")
		if len(parts) > 0 {
			keyword := strings.Trim(parts[0], " [](\"'")
			if keyword != "" {
				keywords = append(keywords, keyword)
			}
		}
		if len(parts) > 2 {
			keyword := strings.Trim(parts[2], " [](\"'")
			if keyword != "" {
				keywords = append(keywords, keyword)
			}
		}
	}
	return keywords
}

// extractKeywordsFromResponse extracts keywords from LLM response.
func extractKeywordsFromResponse(response string, maxKeywords int) []string {
	// Look for "KEYWORDS:" prefix
	response = strings.ToUpper(response)
	if idx := strings.Index(response, "KEYWORDS:"); idx != -1 {
		response = response[idx+9:]
	}

	// Split by comma
	parts := strings.Split(response, ",")
	var keywords []string
	for _, part := range parts {
		keyword := strings.TrimSpace(part)
		keyword = strings.Trim(keyword, `"'`)
		if keyword != "" && len(keywords) < maxKeywords {
			keywords = append(keywords, keyword)
		}
	}

	return keywords
}

// simpleKeywordExtract performs simple keyword extraction without LLM.
func simpleKeywordExtract(text string, maxKeywords int) []string {
	// Simple approach: split by whitespace and filter
	words := strings.Fields(text)
	var keywords []string

	stopwords := map[string]bool{
		"the": true, "a": true, "an": true, "is": true, "are": true,
		"was": true, "were": true, "be": true, "been": true, "being": true,
		"have": true, "has": true, "had": true, "do": true, "does": true,
		"did": true, "will": true, "would": true, "could": true, "should": true,
		"may": true, "might": true, "must": true, "shall": true,
		"of": true, "in": true, "to": true, "for": true, "with": true,
		"on": true, "at": true, "by": true, "from": true, "as": true,
		"and": true, "or": true, "but": true, "if": true, "then": true,
		"that": true, "which": true, "who": true, "whom": true, "whose": true,
		"this": true, "these": true, "those": true, "it": true, "its": true,
		"what": true, "when": true, "where": true, "why": true, "how": true,
	}

	for _, word := range words {
		// Clean the word
		word = strings.ToLower(word)
		word = regexp.MustCompile(`[^a-z0-9]`).ReplaceAllString(word, "")

		if len(word) > 2 && !stopwords[word] {
			keywords = append(keywords, word)
			if len(keywords) >= maxKeywords {
				break
			}
		}
	}

	return keywords
}

// removeDuplicates removes duplicate strings from a slice.
func removeDuplicates(strs []string) []string {
	seen := make(map[string]bool)
	var result []string
	for _, s := range strs {
		if !seen[s] {
			seen[s] = true
			result = append(result, s)
		}
	}
	return result
}

// removeSubstrings removes shorter strings that are substrings of longer ones.
func removeSubstrings(strs []string) []string {
	// Sort by length descending
	sort.Slice(strs, func(i, j int) bool {
		return len(strs[i]) > len(strs[j])
	})

	var result []string
	for i, s := range strs {
		isSubstring := false
		for j := 0; j < i; j++ {
			if strings.Contains(strs[j], s) {
				isSubstring = true
				break
			}
		}
		if !isSubstring {
			result = append(result, s)
		}
	}

	return result
}

// sortByCount sorts keys by their count and returns top n.
func sortByCount(counts map[string]int, n int) []string {
	type kv struct {
		key   string
		count int
	}

	var kvs []kv
	for k, v := range counts {
		kvs = append(kvs, kv{key: k, count: v})
	}

	sort.Slice(kvs, func(i, j int) bool {
		return kvs[i].count > kvs[j].count
	})

	var result []string
	for i := 0; i < n && i < len(kvs); i++ {
		result = append(result, kvs[i].key)
	}

	return result
}

// kgCosineSimilarity calculates cosine similarity between two vectors.
func kgCosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (kgSqrt(normA) * kgSqrt(normB))
}

// kgSqrt is a simple square root implementation.
func kgSqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 100; i++ {
		z = (z + x/z) / 2
	}
	return z
}

// KGRAGRetriever is a more advanced retriever that performs SubGraph RAG.
type KGRAGRetriever struct {
	*KGTableRetriever
	entityExtractFn   func(string) ([]string, error)
	synonymExpandFn   func(string) ([]string, error)
	maxEntities       int
	maxSynonyms       int
	withNL2GraphQuery bool
}

// KGRAGRetrieverOption configures KGRAGRetriever.
type KGRAGRetrieverOption func(*KGRAGRetriever)

// WithKGRAGEntityExtractFn sets a custom entity extraction function.
func WithKGRAGEntityExtractFn(fn func(string) ([]string, error)) KGRAGRetrieverOption {
	return func(r *KGRAGRetriever) {
		r.entityExtractFn = fn
	}
}

// WithKGRAGSynonymExpandFn sets a custom synonym expansion function.
func WithKGRAGSynonymExpandFn(fn func(string) ([]string, error)) KGRAGRetrieverOption {
	return func(r *KGRAGRetriever) {
		r.synonymExpandFn = fn
	}
}

// WithKGRAGMaxEntities sets the max entities to extract.
func WithKGRAGMaxEntities(n int) KGRAGRetrieverOption {
	return func(r *KGRAGRetriever) {
		if n > 0 {
			r.maxEntities = n
		}
	}
}

// WithKGRAGMaxSynonyms sets the max synonyms per entity.
func WithKGRAGMaxSynonyms(n int) KGRAGRetrieverOption {
	return func(r *KGRAGRetriever) {
		if n > 0 {
			r.maxSynonyms = n
		}
	}
}

// NewKGRAGRetriever creates a new KGRAGRetriever.
func NewKGRAGRetriever(index *KnowledgeGraphIndex, opts ...KGRAGRetrieverOption) *KGRAGRetriever {
	r := &KGRAGRetriever{
		KGTableRetriever:  NewKGTableRetriever(index),
		maxEntities:       5,
		maxSynonyms:       5,
		withNL2GraphQuery: false,
	}

	for _, opt := range opts {
		opt(r)
	}

	return r
}

// Name returns the retriever name.
func (r *KGRAGRetriever) Name() string {
	return "KGRAGRetriever"
}

// Ensure retrievers implement the Retriever interface.
var _ interface {
	Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error)
	Name() string
} = (*KGTableRetriever)(nil)

var _ interface {
	Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error)
	Name() string
} = (*KGRAGRetriever)(nil)
