package postprocessor

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"unicode"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/schema"
)

// DefaultRankGPTPrompt is the default prompt for RankGPT reranking.
const DefaultRankGPTPrompt = `Search Query: {query}. 
Rank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain.`

// RankGPTRerank is a RankGPT-based reranker.
// It uses a conversational approach to rank passages based on relevance.
type RankGPTRerank struct {
	*BaseNodePostprocessor
	llm            llm.LLM
	topN           int
	verbose        bool
	rerankPrompt   string
	maxWordsPerDoc int
}

// RankGPTRerankOption configures a RankGPTRerank.
type RankGPTRerankOption func(*RankGPTRerank)

// WithRankGPTLLM sets the LLM.
func WithRankGPTLLM(l llm.LLM) RankGPTRerankOption {
	return func(r *RankGPTRerank) {
		r.llm = l
	}
}

// WithRankGPTTopN sets the number of top nodes to return.
func WithRankGPTTopN(n int) RankGPTRerankOption {
	return func(r *RankGPTRerank) {
		r.topN = n
	}
}

// WithRankGPTVerbose sets verbose mode.
func WithRankGPTVerbose(verbose bool) RankGPTRerankOption {
	return func(r *RankGPTRerank) {
		r.verbose = verbose
	}
}

// WithRankGPTPrompt sets the rerank prompt.
func WithRankGPTPrompt(prompt string) RankGPTRerankOption {
	return func(r *RankGPTRerank) {
		r.rerankPrompt = prompt
	}
}

// WithRankGPTMaxWords sets the maximum words per document.
func WithRankGPTMaxWords(maxWords int) RankGPTRerankOption {
	return func(r *RankGPTRerank) {
		r.maxWordsPerDoc = maxWords
	}
}

// NewRankGPTRerank creates a new RankGPTRerank.
func NewRankGPTRerank(opts ...RankGPTRerankOption) *RankGPTRerank {
	r := &RankGPTRerank{
		BaseNodePostprocessor: NewBaseNodePostprocessor(WithPostprocessorName("RankGPTRerank")),
		topN:                  5,
		verbose:               false,
		rerankPrompt:          DefaultRankGPTPrompt,
		maxWordsPerDoc:        300,
	}

	for _, opt := range opts {
		opt(r)
	}

	return r
}

// PostprocessNodes reranks nodes using RankGPT.
func (r *RankGPTRerank) PostprocessNodes(
	ctx context.Context,
	nodes []schema.NodeWithScore,
	queryBundle *schema.QueryBundle,
) ([]schema.NodeWithScore, error) {
	if queryBundle == nil {
		return nil, fmt.Errorf("query bundle must be provided")
	}
	if len(nodes) == 0 {
		return []schema.NodeWithScore{}, nil
	}
	if r.llm == nil {
		return nil, fmt.Errorf("LLM must be provided for RankGPTRerank")
	}

	// Prepare items for ranking
	hits := make([]string, len(nodes))
	for i, node := range nodes {
		content := node.Node.GetContent(schema.MetadataModeNone)
		// Clean and truncate content
		content = strings.ReplaceAll(content, "Title: Content: ", "")
		content = strings.TrimSpace(content)
		content = truncateToWords(content, r.maxWordsPerDoc)
		hits[i] = content
	}

	// Create permutation instruction messages
	messages := r.createPermutationInstruction(queryBundle.QueryString, hits)

	// Run LLM
	response, err := r.runLLM(ctx, messages)
	if err != nil {
		return nil, fmt.Errorf("LLM call failed: %w", err)
	}

	// Parse the permutation response
	rerankedIndices := r.receivePermutation(response, len(nodes))

	if r.verbose {
		fmt.Printf("[RankGPT] Reranked order: %v\n", rerankedIndices)
	}

	// Build result list
	results := make([]schema.NodeWithScore, 0, len(rerankedIndices))
	for _, idx := range rerankedIndices {
		if idx >= 0 && idx < len(nodes) {
			results = append(results, schema.NodeWithScore{
				Node:  nodes[idx].Node,
				Score: nodes[idx].Score,
			})
		}
	}

	// Return top N
	if len(results) > r.topN {
		return results[:r.topN], nil
	}
	return results, nil
}

// createPermutationInstruction creates the chat messages for RankGPT.
func (r *RankGPTRerank) createPermutationInstruction(query string, hits []string) []llm.ChatMessage {
	num := len(hits)

	messages := []llm.ChatMessage{
		llm.NewSystemMessage("You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."),
		llm.NewUserMessage(fmt.Sprintf("I will provide you with %d passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: %s.", num, query)),
		llm.NewAssistantMessage("Okay, please provide the passages."),
	}

	// Add each passage
	for i, hit := range hits {
		rank := i + 1
		messages = append(messages, llm.NewUserMessage(fmt.Sprintf("[%d] %s", rank, hit)))
		messages = append(messages, llm.NewAssistantMessage(fmt.Sprintf("Received passage [%d].", rank)))
	}

	// Add the final ranking request
	postPrompt := strings.ReplaceAll(r.rerankPrompt, "{query}", query)
	postPrompt = strings.ReplaceAll(postPrompt, "{num}", strconv.Itoa(num))
	messages = append(messages, llm.NewUserMessage(postPrompt))

	return messages
}

// runLLM runs the LLM with the given messages.
func (r *RankGPTRerank) runLLM(ctx context.Context, messages []llm.ChatMessage) (string, error) {
	response, err := r.llm.Chat(ctx, messages)
	if err != nil {
		return "", err
	}
	return response, nil
}

// receivePermutation parses the LLM response to extract the ranking order.
func (r *RankGPTRerank) receivePermutation(response string, numHits int) []int {
	// Clean the response to extract only numbers
	cleaned := cleanResponse(response)

	// Parse numbers
	parts := strings.Fields(cleaned)
	var indices []int
	seen := make(map[int]bool)

	for _, part := range parts {
		num, err := strconv.Atoi(part)
		if err != nil {
			continue
		}
		// Convert to 0-indexed
		idx := num - 1
		if idx >= 0 && idx < numHits && !seen[idx] {
			seen[idx] = true
			indices = append(indices, idx)
		}
	}

	// Add any missing indices at the end
	for i := 0; i < numHits; i++ {
		if !seen[i] {
			indices = append(indices, i)
		}
	}

	return indices
}

// cleanResponse extracts only digits from the response.
func cleanResponse(response string) string {
	var builder strings.Builder
	for _, c := range response {
		if unicode.IsDigit(c) {
			builder.WriteRune(c)
		} else {
			builder.WriteRune(' ')
		}
	}
	return strings.TrimSpace(builder.String())
}

// truncateToWords truncates text to a maximum number of words.
func truncateToWords(text string, maxWords int) string {
	words := strings.Fields(text)
	if len(words) <= maxWords {
		return text
	}
	return strings.Join(words[:maxWords], " ")
}

// Ensure RankGPTRerank implements NodePostprocessor.
var _ NodePostprocessor = (*RankGPTRerank)(nil)

// SlidingWindowRankGPT implements sliding window reranking for large document sets.
// This is useful when you have more documents than can fit in a single LLM context.
type SlidingWindowRankGPT struct {
	*RankGPTRerank
	windowSize int
	stepSize   int
}

// SlidingWindowRankGPTOption configures a SlidingWindowRankGPT.
type SlidingWindowRankGPTOption func(*SlidingWindowRankGPT)

// WithSlidingWindowSize sets the window size.
func WithSlidingWindowSize(size int) SlidingWindowRankGPTOption {
	return func(r *SlidingWindowRankGPT) {
		r.windowSize = size
	}
}

// WithSlidingStepSize sets the step size.
func WithSlidingStepSize(size int) SlidingWindowRankGPTOption {
	return func(r *SlidingWindowRankGPT) {
		r.stepSize = size
	}
}

// NewSlidingWindowRankGPT creates a new SlidingWindowRankGPT.
func NewSlidingWindowRankGPT(baseOpts []RankGPTRerankOption, opts ...SlidingWindowRankGPTOption) *SlidingWindowRankGPT {
	r := &SlidingWindowRankGPT{
		RankGPTRerank: NewRankGPTRerank(baseOpts...),
		windowSize:    20,
		stepSize:      10,
	}

	for _, opt := range opts {
		opt(r)
	}

	return r
}

// PostprocessNodes reranks nodes using sliding window RankGPT.
func (r *SlidingWindowRankGPT) PostprocessNodes(
	ctx context.Context,
	nodes []schema.NodeWithScore,
	queryBundle *schema.QueryBundle,
) ([]schema.NodeWithScore, error) {
	if queryBundle == nil {
		return nil, fmt.Errorf("query bundle must be provided")
	}
	if len(nodes) == 0 {
		return []schema.NodeWithScore{}, nil
	}
	if r.llm == nil {
		return nil, fmt.Errorf("LLM must be provided for SlidingWindowRankGPT")
	}

	// If nodes fit in one window, use regular RankGPT
	if len(nodes) <= r.windowSize {
		return r.RankGPTRerank.PostprocessNodes(ctx, nodes, queryBundle)
	}

	// Use sliding window approach
	currentNodes := nodes

	// Process windows from end to start
	endPos := len(currentNodes)
	for endPos > r.topN {
		startPos := endPos - r.windowSize
		if startPos < 0 {
			startPos = 0
		}

		// Get window nodes
		windowNodes := currentNodes[startPos:endPos]

		// Rerank window
		rerankedWindow, err := r.RankGPTRerank.PostprocessNodes(ctx, windowNodes, queryBundle)
		if err != nil {
			return nil, err
		}

		// Replace window with reranked results
		newNodes := make([]schema.NodeWithScore, 0, len(currentNodes))
		newNodes = append(newNodes, currentNodes[:startPos]...)
		newNodes = append(newNodes, rerankedWindow...)
		currentNodes = newNodes

		// Move window
		endPos = startPos + r.stepSize
	}

	// Final pass on top portion
	if len(currentNodes) > r.topN {
		topNodes := currentNodes[:min(r.windowSize, len(currentNodes))]
		rerankedTop, err := r.RankGPTRerank.PostprocessNodes(ctx, topNodes, queryBundle)
		if err != nil {
			return nil, err
		}
		currentNodes = rerankedTop
	}

	if len(currentNodes) > r.topN {
		return currentNodes[:r.topN], nil
	}
	return currentNodes, nil
}

// min returns the minimum of two integers.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Ensure SlidingWindowRankGPT implements NodePostprocessor.
var _ NodePostprocessor = (*SlidingWindowRankGPT)(nil)

// CohereRerank is a placeholder for Cohere-based reranking.
// This would require the Cohere API client.
type CohereRerank struct {
	*BaseNodePostprocessor
	apiKey string
	model  string
	topN   int
}

// CohereRerankOption configures a CohereRerank.
type CohereRerankOption func(*CohereRerank)

// WithCohereAPIKey sets the API key.
func WithCohereAPIKey(key string) CohereRerankOption {
	return func(r *CohereRerank) {
		r.apiKey = key
	}
}

// WithCohereModel sets the model.
func WithCohereModel(model string) CohereRerankOption {
	return func(r *CohereRerank) {
		r.model = model
	}
}

// WithCohereTopN sets the number of top nodes to return.
func WithCohereTopN(n int) CohereRerankOption {
	return func(r *CohereRerank) {
		r.topN = n
	}
}

// NewCohereRerank creates a new CohereRerank.
func NewCohereRerank(opts ...CohereRerankOption) *CohereRerank {
	r := &CohereRerank{
		BaseNodePostprocessor: NewBaseNodePostprocessor(WithPostprocessorName("CohereRerank")),
		model:                 "rerank-english-v2.0",
		topN:                  5,
	}

	for _, opt := range opts {
		opt(r)
	}

	return r
}

// PostprocessNodes reranks nodes using Cohere.
// Note: This is a placeholder implementation. Full implementation would require Cohere API client.
func (r *CohereRerank) PostprocessNodes(
	ctx context.Context,
	nodes []schema.NodeWithScore,
	queryBundle *schema.QueryBundle,
) ([]schema.NodeWithScore, error) {
	if r.apiKey == "" {
		return nil, fmt.Errorf("Cohere API key must be provided")
	}

	// Placeholder: In a full implementation, this would call the Cohere rerank API
	// For now, return nodes sorted by existing score
	result := make([]schema.NodeWithScore, len(nodes))
	copy(result, nodes)

	sort.Slice(result, func(i, j int) bool {
		return result[i].Score > result[j].Score
	})

	if len(result) > r.topN {
		return result[:r.topN], nil
	}
	return result, nil
}

// Ensure CohereRerank implements NodePostprocessor.
var _ NodePostprocessor = (*CohereRerank)(nil)
