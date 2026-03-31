package textsplitter

import (
	"context"
	"math"
	"sort"
	"strings"

	"github.com/aqua777/go-llamaindex/embedding"
)

// SemanticSplitterNodeParser groups semantically related sentences using embedding
// similarity and a percentile-based breakpoint rule (aligned with LlamaIndex Python).
type SemanticSplitterNodeParser struct {
	EmbedModel                    embedding.EmbeddingModel
	BufferSize                    int
	BreakpointPercentileThreshold int
	SentenceSplitter              SentenceSplitterStrategy
}

// NewSemanticSplitterNodeParser creates a new SemanticSplitterNodeParser.
//
// Args:
//
//	embedModel: The embedding model to use for similarity checks.
//	bufferSize: Number of sentences to include before/after when forming combined text per index.
//	breakpointPercentileThreshold: Percentile of pairwise dissimilarities used as the split threshold (0–100).
//	sentenceSplitter: Strategy to split text into sentences; if nil, RegexSplitterStrategy with DefaultChunkingRegex is used.
//
// Returns:
//
//	A configured SemanticSplitterNodeParser. Non-positive bufferSize defaults to 1.
//	Threshold is clamped to [0, 100].
func NewSemanticSplitterNodeParser(
	embedModel embedding.EmbeddingModel,
	bufferSize int,
	breakpointPercentileThreshold int,
	sentenceSplitter SentenceSplitterStrategy,
) *SemanticSplitterNodeParser {
	if bufferSize <= 0 {
		bufferSize = 1
	}
	if breakpointPercentileThreshold < 0 {
		breakpointPercentileThreshold = 0
	}
	if breakpointPercentileThreshold > 100 {
		breakpointPercentileThreshold = 100
	}
	if sentenceSplitter == nil {
		sentenceSplitter = NewRegexSplitterStrategy(DefaultChunkingRegex)
	}
	return &SemanticSplitterNodeParser{
		EmbedModel:                    embedModel,
		BufferSize:                    bufferSize,
		BreakpointPercentileThreshold: breakpointPercentileThreshold,
		SentenceSplitter:              sentenceSplitter,
	}
}

// SplitText splits the text into semantically grouped chunks.
//
// Args:
//
//	text: The text string to split.
//
// Returns:
//
//	A slice of chunks joined from original sentences (no added spaces between sentences).
//	On embedding failure, invalid embeddings for similarity, or pairwise similarity failure,
//	returns nil. Empty input, no embed model, or no sentences after trimming yields a non-nil
//	empty slice.
func (s *SemanticSplitterNodeParser) SplitText(text string) []string {
	if text == "" {
		return []string{}
	}
	if s.EmbedModel == nil {
		return []string{}
	}
	sentences := filterNonEmptySentences(s.SentenceSplitter.Split(text))
	if len(sentences) == 0 {
		return []string{}
	}
	if len(sentences) == 1 {
		return []string{strings.Join(sentences, " ")}
	}
	combined := buildCombinedSentenceGroups(sentences, s.BufferSize)
	ctx := context.Background()
	embs := make([][]float32, len(combined))
	for i, c := range combined {
		vec, err := s.EmbedModel.GetTextEmbedding(ctx, c)
		if err != nil {
			return nil
		}
		embs[i] = vec
	}
	distances, err := pairwiseDissimilarities(embs)
	if err != nil {
		return nil
	}
	chunks := buildSemanticChunksFromDistances(sentences, distances, s.BreakpointPercentileThreshold)
	return chunks
}

func filterNonEmptySentences(in []string) []string {
	var out []string
	for _, x := range in {
		if strings.TrimSpace(x) != "" {
			out = append(out, x)
		}
	}
	return out
}

// buildCombinedSentenceGroups returns one combined string per sentence index, including
// up to bufferSize sentences before and after (LlamaIndex combined_sentence).
func buildCombinedSentenceGroups(sentences []string, bufferSize int) []string {
	n := len(sentences)
	out := make([]string, n)
	for i := 0; i < n; i++ {
		var b strings.Builder
		for j := i - bufferSize; j < i; j++ {
			if j >= 0 {
				b.WriteString(sentences[j])
			}
		}
		b.WriteString(sentences[i])
		for j := i + 1; j < i+1+bufferSize; j++ {
			if j < n {
				b.WriteString(sentences[j])
			}
		}
		out[i] = b.String()
	}
	return out
}

// pairwiseDissimilarities returns 1 - cosine_similarity between consecutive combined embeddings.
func pairwiseDissimilarities(embeddings [][]float32) ([]float64, error) {
	if len(embeddings) < 2 {
		return nil, nil
	}
	d := make([]float64, len(embeddings)-1)
	for i := 0; i < len(embeddings)-1; i++ {
		sim, err := cosineSimilarityFloat32(embeddings[i], embeddings[i+1])
		if err != nil {
			return nil, err
		}
		if sim > 1 {
			sim = 1
		}
		if sim < -1 {
			sim = -1
		}
		d[i] = 1 - float64(sim)
	}
	return d, nil
}

func cosineSimilarityFloat32(a, b []float32) (float64, error) {
	return embedding.CosineSimilarity(a, b)
}

// percentileLinear returns the p-th percentile (0–100) using linear interpolation on a copy of x.
func percentileLinear(x []float64, p float64) float64 {
	if len(x) == 0 {
		return 0
	}
	cp := append([]float64(nil), x...)
	sort.Float64s(cp)
	if p <= 0 {
		return cp[0]
	}
	if p >= 100 {
		return cp[len(cp)-1]
	}
	n := len(cp)
	if n == 1 {
		return cp[0]
	}
	pos := (p / 100.0) * float64(n-1)
	lo := int(math.Floor(pos))
	hi := int(math.Ceil(pos))
	if lo == hi {
		return cp[lo]
	}
	w := pos - float64(lo)
	return cp[lo]*(1-w) + cp[hi]*w
}

func indicesWhereDissimilarityExceeds(distances []float64, threshold float64) []int {
	var idx []int
	for i, x := range distances {
		if x > threshold {
			idx = append(idx, i)
		}
	}
	return idx
}

// buildSemanticChunksFromDistances splits sentences at breakpoints where distance exceeds the
// percentile threshold (same rule as LlamaIndex _build_node_chunks).
func buildSemanticChunksFromDistances(sentences []string, distances []float64, breakpointPercentile int) []string {
	if len(sentences) == 0 {
		return nil
	}
	if len(distances) == 0 {
		return []string{strings.Join(sentences, " ")}
	}
	th := percentileLinear(distances, float64(breakpointPercentile))
	breaks := indicesWhereDissimilarityExceeds(distances, th)
	var chunks []string
	start := 0
	for _, index := range breaks {
		group := sentences[start : index+1]
		chunks = append(chunks, strings.Join(group, ""))
		start = index + 1
	}
	if start < len(sentences) {
		chunks = append(chunks, strings.Join(sentences[start:], ""))
	}
	return chunks
}
