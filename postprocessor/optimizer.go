package postprocessor

import (
	"context"
	"strings"
	"unicode"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/schema"
)

// SentenceOptimizerPostprocessor optimizes node content by selecting
// the most relevant sentences based on the query.
type SentenceOptimizerPostprocessor struct {
	*BaseNodePostprocessor
	// EmbedModel is the embedding model for similarity calculation.
	EmbedModel embedding.EmbeddingModel
	// TopK is the number of top sentences to keep per node.
	TopK int
	// Threshold is the minimum similarity threshold for sentences.
	Threshold float64
	// ContextWindow is the number of surrounding sentences to include.
	ContextWindow int
	// PreserveParagraphs keeps paragraph structure when possible.
	PreserveParagraphs bool
}

// SentenceOptimizerOption configures a SentenceOptimizerPostprocessor.
type SentenceOptimizerOption func(*SentenceOptimizerPostprocessor)

// WithOptimizerEmbedModel sets the embedding model.
func WithOptimizerEmbedModel(model embedding.EmbeddingModel) SentenceOptimizerOption {
	return func(p *SentenceOptimizerPostprocessor) {
		p.EmbedModel = model
	}
}

// WithOptimizerTopK sets the number of top sentences to keep.
func WithOptimizerTopK(k int) SentenceOptimizerOption {
	return func(p *SentenceOptimizerPostprocessor) {
		p.TopK = k
	}
}

// WithOptimizerThreshold sets the minimum similarity threshold.
func WithOptimizerThreshold(threshold float64) SentenceOptimizerOption {
	return func(p *SentenceOptimizerPostprocessor) {
		p.Threshold = threshold
	}
}

// WithOptimizerContextWindow sets the context window size.
func WithOptimizerContextWindow(window int) SentenceOptimizerOption {
	return func(p *SentenceOptimizerPostprocessor) {
		p.ContextWindow = window
	}
}

// WithOptimizerPreserveParagraphs enables paragraph preservation.
func WithOptimizerPreserveParagraphs(preserve bool) SentenceOptimizerOption {
	return func(p *SentenceOptimizerPostprocessor) {
		p.PreserveParagraphs = preserve
	}
}

// NewSentenceOptimizerPostprocessor creates a new SentenceOptimizerPostprocessor.
func NewSentenceOptimizerPostprocessor(embedModel embedding.EmbeddingModel, opts ...SentenceOptimizerOption) *SentenceOptimizerPostprocessor {
	p := &SentenceOptimizerPostprocessor{
		BaseNodePostprocessor: NewBaseNodePostprocessor(WithPostprocessorName("SentenceOptimizerPostprocessor")),
		EmbedModel:            embedModel,
		TopK:                  5,
		Threshold:             0.0,
		ContextWindow:         0,
		PreserveParagraphs:    false,
	}

	for _, opt := range opts {
		opt(p)
	}

	return p
}

// PostprocessNodes optimizes node content based on query relevance.
func (p *SentenceOptimizerPostprocessor) PostprocessNodes(
	ctx context.Context,
	nodes []schema.NodeWithScore,
	queryBundle *schema.QueryBundle,
) ([]schema.NodeWithScore, error) {
	if p.EmbedModel == nil {
		return nodes, nil
	}

	query := ""
	if queryBundle != nil {
		query = queryBundle.QueryString
	}

	if query == "" {
		return nodes, nil
	}

	// Get query embedding
	queryEmb, err := p.EmbedModel.GetQueryEmbedding(ctx, query)
	if err != nil {
		return nodes, nil // Return original nodes on error
	}

	result := make([]schema.NodeWithScore, len(nodes))

	for i, nodeWithScore := range nodes {
		optimizedText, err := p.optimizeText(ctx, nodeWithScore.Node.Text, queryEmb)
		if err != nil {
			result[i] = nodeWithScore
			continue
		}

		newNode := nodeWithScore.Node
		newNode.Text = optimizedText

		// Store original text in metadata
		if newNode.Metadata == nil {
			newNode.Metadata = make(map[string]interface{})
		}
		newNode.Metadata["original_text"] = nodeWithScore.Node.Text
		newNode.Metadata["optimized"] = true

		result[i] = schema.NodeWithScore{
			Node:  newNode,
			Score: nodeWithScore.Score,
		}
	}

	return result, nil
}

// optimizeText selects the most relevant sentences from text.
func (p *SentenceOptimizerPostprocessor) optimizeText(ctx context.Context, text string, queryEmb []float64) (string, error) {
	sentences := splitSentences(text)
	if len(sentences) == 0 {
		return text, nil
	}

	// Score each sentence
	type scoredSentence struct {
		index    int
		sentence string
		score    float64
	}

	var scored []scoredSentence

	for i, sentence := range sentences {
		sentence = strings.TrimSpace(sentence)
		if sentence == "" {
			continue
		}

		sentEmb, err := p.EmbedModel.GetTextEmbedding(ctx, sentence)
		if err != nil {
			continue
		}

		score, err := embedding.CosineSimilarity(queryEmb, sentEmb)
		if err != nil {
			continue
		}

		if score >= p.Threshold {
			scored = append(scored, scoredSentence{
				index:    i,
				sentence: sentence,
				score:    score,
			})
		}
	}

	if len(scored) == 0 {
		return text, nil
	}

	// Sort by score descending
	for i := 0; i < len(scored)-1; i++ {
		for j := i + 1; j < len(scored); j++ {
			if scored[j].score > scored[i].score {
				scored[i], scored[j] = scored[j], scored[i]
			}
		}
	}

	// Select top K sentences
	k := p.TopK
	if k > len(scored) {
		k = len(scored)
	}
	selected := scored[:k]

	// Include context window
	if p.ContextWindow > 0 {
		selectedIndices := make(map[int]bool)
		for _, s := range selected {
			for j := s.index - p.ContextWindow; j <= s.index+p.ContextWindow; j++ {
				if j >= 0 && j < len(sentences) {
					selectedIndices[j] = true
				}
			}
		}

		// Rebuild selected with context
		selected = nil
		for idx := range selectedIndices {
			if idx < len(sentences) {
				selected = append(selected, scoredSentence{
					index:    idx,
					sentence: strings.TrimSpace(sentences[idx]),
				})
			}
		}
	}

	// Sort by original order
	for i := 0; i < len(selected)-1; i++ {
		for j := i + 1; j < len(selected); j++ {
			if selected[j].index < selected[i].index {
				selected[i], selected[j] = selected[j], selected[i]
			}
		}
	}

	// Join sentences
	var parts []string
	for _, s := range selected {
		if s.sentence != "" {
			parts = append(parts, s.sentence)
		}
	}

	if p.PreserveParagraphs {
		return strings.Join(parts, "\n\n"), nil
	}
	return strings.Join(parts, " "), nil
}

// splitSentences splits text into sentences.
func splitSentences(text string) []string {
	var sentences []string
	var current strings.Builder

	runes := []rune(text)
	for i := 0; i < len(runes); i++ {
		r := runes[i]
		current.WriteRune(r)

		// Check for sentence ending
		if r == '.' || r == '!' || r == '?' {
			// Check if followed by space and capital letter (or end of text)
			if i+1 >= len(runes) {
				sentences = append(sentences, current.String())
				current.Reset()
			} else if i+1 < len(runes) && (unicode.IsSpace(runes[i+1]) || runes[i+1] == '\n') {
				// Look ahead for capital letter
				j := i + 1
				for j < len(runes) && unicode.IsSpace(runes[j]) {
					j++
				}
				if j >= len(runes) || unicode.IsUpper(runes[j]) {
					sentences = append(sentences, current.String())
					current.Reset()
				}
			}
		} else if r == '\n' && i+1 < len(runes) && runes[i+1] == '\n' {
			// Paragraph break
			sentences = append(sentences, current.String())
			current.Reset()
			i++ // Skip second newline
		}
	}

	// Add remaining text
	if current.Len() > 0 {
		sentences = append(sentences, current.String())
	}

	return sentences
}

// TextCompressorPostprocessor compresses node text by removing redundancy.
type TextCompressorPostprocessor struct {
	*BaseNodePostprocessor
	// MaxLength is the maximum length of compressed text.
	MaxLength int
	// RemoveStopwords removes common stopwords.
	RemoveStopwords bool
	// Stopwords is the list of stopwords to remove.
	Stopwords map[string]bool
}

// TextCompressorOption configures a TextCompressorPostprocessor.
type TextCompressorOption func(*TextCompressorPostprocessor)

// WithCompressorMaxLength sets the maximum text length.
func WithCompressorMaxLength(length int) TextCompressorOption {
	return func(p *TextCompressorPostprocessor) {
		p.MaxLength = length
	}
}

// WithCompressorRemoveStopwords enables stopword removal.
func WithCompressorRemoveStopwords(remove bool) TextCompressorOption {
	return func(p *TextCompressorPostprocessor) {
		p.RemoveStopwords = remove
	}
}

// WithCompressorStopwords sets custom stopwords.
func WithCompressorStopwords(words []string) TextCompressorOption {
	return func(p *TextCompressorPostprocessor) {
		p.Stopwords = make(map[string]bool)
		for _, w := range words {
			p.Stopwords[strings.ToLower(w)] = true
		}
	}
}

// NewTextCompressorPostprocessor creates a new TextCompressorPostprocessor.
func NewTextCompressorPostprocessor(opts ...TextCompressorOption) *TextCompressorPostprocessor {
	p := &TextCompressorPostprocessor{
		BaseNodePostprocessor: NewBaseNodePostprocessor(WithPostprocessorName("TextCompressorPostprocessor")),
		MaxLength:             1000,
		RemoveStopwords:       false,
		Stopwords:             defaultStopwords(),
	}

	for _, opt := range opts {
		opt(p)
	}

	return p
}

// PostprocessNodes compresses node text.
func (p *TextCompressorPostprocessor) PostprocessNodes(
	ctx context.Context,
	nodes []schema.NodeWithScore,
	queryBundle *schema.QueryBundle,
) ([]schema.NodeWithScore, error) {
	result := make([]schema.NodeWithScore, len(nodes))

	for i, nodeWithScore := range nodes {
		compressedText := p.compressText(nodeWithScore.Node.Text)

		newNode := nodeWithScore.Node
		newNode.Text = compressedText

		result[i] = schema.NodeWithScore{
			Node:  newNode,
			Score: nodeWithScore.Score,
		}
	}

	return result, nil
}

// compressText compresses text by removing redundancy.
func (p *TextCompressorPostprocessor) compressText(text string) string {
	// Normalize whitespace
	text = strings.Join(strings.Fields(text), " ")

	// Remove stopwords if enabled
	if p.RemoveStopwords {
		words := strings.Fields(text)
		var filtered []string
		for _, word := range words {
			if !p.Stopwords[strings.ToLower(word)] {
				filtered = append(filtered, word)
			}
		}
		text = strings.Join(filtered, " ")
	}

	// Truncate if too long
	if p.MaxLength > 0 && len(text) > p.MaxLength {
		// Try to truncate at sentence boundary
		truncated := text[:p.MaxLength]
		lastPeriod := strings.LastIndex(truncated, ".")
		if lastPeriod > p.MaxLength/2 {
			truncated = truncated[:lastPeriod+1]
		} else {
			// Truncate at word boundary
			lastSpace := strings.LastIndex(truncated, " ")
			if lastSpace > 0 {
				truncated = truncated[:lastSpace] + "..."
			}
		}
		text = truncated
	}

	return text
}

// defaultStopwords returns a default set of English stopwords.
func defaultStopwords() map[string]bool {
	words := []string{
		"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
		"of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
		"be", "have", "has", "had", "do", "does", "did", "will", "would",
		"could", "should", "may", "might", "must", "shall", "can", "need",
		"this", "that", "these", "those", "i", "you", "he", "she", "it",
		"we", "they", "what", "which", "who", "whom", "when", "where", "why",
		"how", "all", "each", "every", "both", "few", "more", "most", "other",
		"some", "such", "no", "nor", "not", "only", "own", "same", "so",
		"than", "too", "very", "just", "also", "now", "here", "there",
	}

	stopwords := make(map[string]bool)
	for _, w := range words {
		stopwords[w] = true
	}
	return stopwords
}

// Ensure postprocessors implement NodePostprocessor.
var _ NodePostprocessor = (*SentenceOptimizerPostprocessor)(nil)
var _ NodePostprocessor = (*TextCompressorPostprocessor)(nil)
