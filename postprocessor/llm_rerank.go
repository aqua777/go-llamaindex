package postprocessor

import (
	"context"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/schema"
)

// DefaultChoiceSelectPrompt is the default prompt for LLM-based reranking.
const DefaultChoiceSelectPrompt = `A list of documents is shown below. Each document has a number next to it along with a summary of the document. A question is also provided.
Respond with the numbers of the documents you should consult to answer the question, in order of relevance, as well as the relevance score. The relevance score is a number from 1-10 based on how relevant you think the document is to the question.
Do not include any documents that are not relevant to the question.
Example format:
Document 1:
<summary of document 1>

Document 2:
<summary of document 2>

...

Document 10:
<summary of document 10>

Question: <question>
Answer:
Doc: 9, Relevance: 7
Doc: 3, Relevance: 4
Doc: 7, Relevance: 3

Let's try this now:

{context_str}
Question: {query_str}
Answer:
`

// ChoiceWithRelevance represents a document choice with its relevance score.
type ChoiceWithRelevance struct {
	DocIndex  int
	Relevance float64
}

// LLMRerank is an LLM-based reranker that uses an LLM to rerank nodes.
type LLMRerank struct {
	*BaseNodePostprocessor
	llm                       llm.LLM
	topN                      int
	choiceBatchSize           int
	choiceSelectPrompt        string
	formatNodeBatchFn         func([]*schema.Node) string
	parseChoiceSelectAnswerFn func(string, int) ([]ChoiceWithRelevance, error)
}

// LLMRerankOption configures an LLMRerank.
type LLMRerankOption func(*LLMRerank)

// WithLLMRerankLLM sets the LLM.
func WithLLMRerankLLM(l llm.LLM) LLMRerankOption {
	return func(r *LLMRerank) {
		r.llm = l
	}
}

// WithLLMRerankTopN sets the number of top nodes to return.
func WithLLMRerankTopN(n int) LLMRerankOption {
	return func(r *LLMRerank) {
		r.topN = n
	}
}

// WithLLMRerankBatchSize sets the batch size for processing.
func WithLLMRerankBatchSize(size int) LLMRerankOption {
	return func(r *LLMRerank) {
		r.choiceBatchSize = size
	}
}

// WithLLMRerankPrompt sets the choice select prompt.
func WithLLMRerankPrompt(prompt string) LLMRerankOption {
	return func(r *LLMRerank) {
		r.choiceSelectPrompt = prompt
	}
}

// WithLLMRerankFormatFn sets a custom format function.
func WithLLMRerankFormatFn(fn func([]*schema.Node) string) LLMRerankOption {
	return func(r *LLMRerank) {
		r.formatNodeBatchFn = fn
	}
}

// WithLLMRerankParseFn sets a custom parse function.
func WithLLMRerankParseFn(fn func(string, int) ([]ChoiceWithRelevance, error)) LLMRerankOption {
	return func(r *LLMRerank) {
		r.parseChoiceSelectAnswerFn = fn
	}
}

// NewLLMRerank creates a new LLMRerank.
func NewLLMRerank(opts ...LLMRerankOption) *LLMRerank {
	r := &LLMRerank{
		BaseNodePostprocessor:     NewBaseNodePostprocessor(WithPostprocessorName("LLMRerank")),
		topN:                      10,
		choiceBatchSize:           10,
		choiceSelectPrompt:        DefaultChoiceSelectPrompt,
		formatNodeBatchFn:         defaultFormatNodeBatch,
		parseChoiceSelectAnswerFn: defaultParseChoiceSelectAnswer,
	}

	for _, opt := range opts {
		opt(r)
	}

	return r
}

// PostprocessNodes reranks nodes using an LLM.
func (r *LLMRerank) PostprocessNodes(
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
		return nil, fmt.Errorf("LLM must be provided for LLMRerank")
	}

	var initialResults []schema.NodeWithScore

	// Process in batches
	for idx := 0; idx < len(nodes); idx += r.choiceBatchSize {
		end := idx + r.choiceBatchSize
		if end > len(nodes) {
			end = len(nodes)
		}

		// Extract nodes for this batch
		nodesBatch := make([]*schema.Node, end-idx)
		for i, n := range nodes[idx:end] {
			node := n.Node // Copy to get addressable value
			nodesBatch[i] = &node
		}

		// Format the batch
		fmtBatchStr := r.formatNodeBatchFn(nodesBatch)

		// Create the prompt
		prompt := strings.ReplaceAll(r.choiceSelectPrompt, "{context_str}", fmtBatchStr)
		prompt = strings.ReplaceAll(prompt, "{query_str}", queryBundle.QueryString)

		// Call LLM
		rawResponse, err := r.llm.Complete(ctx, prompt)
		if err != nil {
			return nil, fmt.Errorf("LLM call failed: %w", err)
		}

		// Parse the response
		choices, err := r.parseChoiceSelectAnswerFn(rawResponse, len(nodesBatch))
		if err != nil {
			// If parsing fails, keep original order for this batch
			for i, n := range nodes[idx:end] {
				initialResults = append(initialResults, schema.NodeWithScore{
					Node:  n.Node,
					Score: float64(len(nodes) - idx - i), // Decreasing scores
				})
			}
			continue
		}

		// Add chosen nodes with relevance scores
		for _, choice := range choices {
			if choice.DocIndex >= 0 && choice.DocIndex < len(nodesBatch) {
				initialResults = append(initialResults, schema.NodeWithScore{
					Node:  *nodesBatch[choice.DocIndex],
					Score: choice.Relevance,
				})
			}
		}
	}

	// Sort by score descending
	sort.Slice(initialResults, func(i, j int) bool {
		return initialResults[i].Score > initialResults[j].Score
	})

	// Return top N
	if len(initialResults) > r.topN {
		return initialResults[:r.topN], nil
	}
	return initialResults, nil
}

// defaultFormatNodeBatch formats a batch of nodes for the LLM prompt.
func defaultFormatNodeBatch(nodes []*schema.Node) string {
	var builder strings.Builder

	for i, node := range nodes {
		content := node.GetContent(schema.MetadataModeNone)
		// Truncate long content
		if len(content) > 500 {
			content = content[:500] + "..."
		}
		builder.WriteString(fmt.Sprintf("Document %d:\n%s\n\n", i+1, content))
	}

	return builder.String()
}

// defaultParseChoiceSelectAnswer parses the LLM response to extract choices and relevance scores.
func defaultParseChoiceSelectAnswer(response string, numNodes int) ([]ChoiceWithRelevance, error) {
	var choices []ChoiceWithRelevance

	// Pattern to match "Doc: N, Relevance: M" or similar formats
	// Also handles "Doc N, Relevance M" and "Document N: Relevance M"
	patterns := []*regexp.Regexp{
		regexp.MustCompile(`(?i)Doc(?:ument)?[:\s]*(\d+)[,\s]*Relevance[:\s]*(\d+(?:\.\d+)?)`),
		regexp.MustCompile(`(?i)(\d+)[:\s]*Relevance[:\s]*(\d+(?:\.\d+)?)`),
		regexp.MustCompile(`(?i)Doc(?:ument)?[:\s]*(\d+)`),
	}

	lines := strings.Split(response, "\n")
	seenDocs := make(map[int]bool)

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		var docNum int
		var relevance float64 = 5.0 // Default relevance

		matched := false
		for _, pattern := range patterns {
			matches := pattern.FindStringSubmatch(line)
			if matches != nil {
				var err error
				docNum, err = strconv.Atoi(matches[1])
				if err != nil {
					continue
				}

				if len(matches) > 2 && matches[2] != "" {
					relevance, _ = strconv.ParseFloat(matches[2], 64)
				}

				matched = true
				break
			}
		}

		if matched && docNum >= 1 && docNum <= numNodes {
			docIdx := docNum - 1 // Convert to 0-indexed
			if !seenDocs[docIdx] {
				seenDocs[docIdx] = true
				choices = append(choices, ChoiceWithRelevance{
					DocIndex:  docIdx,
					Relevance: relevance,
				})
			}
		}
	}

	if len(choices) == 0 {
		return nil, fmt.Errorf("could not parse any choices from response")
	}

	return choices, nil
}

// Ensure LLMRerank implements NodePostprocessor.
var _ NodePostprocessor = (*LLMRerank)(nil)
