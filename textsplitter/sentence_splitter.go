package textsplitter

import (
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/validation"
)

const (
	DefaultChunkSize     = 1024
	DefaultChunkOverlap  = 200
	DefaultParagraphSep  = "\n\n\n"
	DefaultSeparator     = " "
	DefaultChunkingRegex = `[^,.;。？！]+[,.;。？！]?|[,.;。？！]`
)

// textSplit holds intermediate split information.
type textSplit struct {
	text       string
	isSentence bool
	tokenSize  int
}

// SentenceSplitter splits text with a preference for complete sentences.
type SentenceSplitter struct {
	ChunkSize              int
	ChunkOverlap           int
	Separator              string
	ParagraphSeparator     string
	SecondaryChunkingRegex string
	Tokenizer              Tokenizer
	SplitterStrategy       SentenceSplitterStrategy

	_splitFns            []func(string) []string
	_subSentenceSplitFns []func(string) []string

	// Callbacks
	onChunkingStart func(text []string)
	onChunkingEnd   func(chunks []string)
}

// WithOnChunkingStart sets the callback for when chunking starts.
func (s *SentenceSplitter) WithOnChunkingStart(fn func(text []string)) *SentenceSplitter {
	s.onChunkingStart = fn
	return s
}

// WithOnChunkingEnd sets the callback for when chunking ends.
func (s *SentenceSplitter) WithOnChunkingEnd(fn func(chunks []string)) *SentenceSplitter {
	s.onChunkingEnd = fn
	return s
}

// NewSentenceSplitter creates a new SentenceSplitter.
// Pass 0 or empty strings to use defaults.
// If tokenizer is nil, defaults to SimpleTokenizer.
// If splitterStrategy is nil, defaults to RegexSplitterStrategy with DefaultChunkingRegex.
func NewSentenceSplitter(
	chunkSize int,
	chunkOverlap int,
	tokenizer Tokenizer,
	splitterStrategy SentenceSplitterStrategy,
) *SentenceSplitter {
	if chunkSize <= 0 {
		chunkSize = DefaultChunkSize
	}
	// chunkOverlap can be 0. We do not default it if 0 is passed.
	// To use default overlap, caller should pass DefaultChunkOverlap.

	if tokenizer == nil {
		tokenizer = NewSimpleTokenizer()
	}

	if splitterStrategy == nil {
		splitterStrategy = NewRegexSplitterStrategy(DefaultChunkingRegex)
	}

	// Note: We don't return error here for backward compatibility.
	// Use NewSentenceSplitterWithValidation for strict validation.

	s := &SentenceSplitter{
		ChunkSize:              chunkSize,
		ChunkOverlap:           chunkOverlap,
		Separator:              DefaultSeparator,
		ParagraphSeparator:     DefaultParagraphSep,
		SecondaryChunkingRegex: DefaultChunkingRegex,
		Tokenizer:              tokenizer,
		SplitterStrategy:       splitterStrategy,
		onChunkingStart:        func(text []string) {},
		onChunkingEnd:          func(chunks []string) {},
	}

	s.initSplitFns()
	return s
}

// NewSentenceSplitterWithValidation creates a new SentenceSplitter with input validation.
// Returns an error if parameters are invalid.
func NewSentenceSplitterWithValidation(
	chunkSize int,
	chunkOverlap int,
	tokenizer Tokenizer,
	splitterStrategy SentenceSplitterStrategy,
) (*SentenceSplitter, error) {
	// Validate parameters
	if err := validation.ValidateChunkParams(chunkSize, chunkOverlap); err != nil {
		return nil, fmt.Errorf("invalid sentence splitter config: %w", err)
	}

	if tokenizer == nil {
		tokenizer = NewSimpleTokenizer()
	}

	if splitterStrategy == nil {
		splitterStrategy = NewRegexSplitterStrategy(DefaultChunkingRegex)
	}

	s := &SentenceSplitter{
		ChunkSize:              chunkSize,
		ChunkOverlap:           chunkOverlap,
		Separator:              DefaultSeparator,
		ParagraphSeparator:     DefaultParagraphSep,
		SecondaryChunkingRegex: DefaultChunkingRegex,
		Tokenizer:              tokenizer,
		SplitterStrategy:       splitterStrategy,
		onChunkingStart:        func(text []string) {},
		onChunkingEnd:          func(chunks []string) {},
	}

	s.initSplitFns()
	return s, nil
}

// Validate validates the current splitter configuration.
func (s *SentenceSplitter) Validate() error {
	return validation.ValidateSentenceSplitterConfig(validation.SentenceSplitterConfig{
		ChunkSize:              s.ChunkSize,
		ChunkOverlap:           s.ChunkOverlap,
		Separator:              s.Separator,
		ParagraphSeparator:     s.ParagraphSeparator,
		SecondaryChunkingRegex: s.SecondaryChunkingRegex,
	})
}

func (s *SentenceSplitter) initSplitFns() {
	// Primary split functions:
	// 1. Paragraph separator
	// 2. Sentence Splitter Strategy (Regex or Neurosnap or custom)
	s._splitFns = []func(string) []string{
		SplitBySep(s.ParagraphSeparator),
		func(text string) []string { return s.SplitterStrategy.Split(text) },
	}

	// Sub-sentence split functions (fallback if sentences are still too big):
	// 1. Regex fallback (hardcoded default regex often used as backup)
	// 2. Separator (Word)
	// 3. Character
	// Note: Python implementation allows customizing secondary regex.
	s._subSentenceSplitFns = []func(string) []string{
		SplitByRegex(s.SecondaryChunkingRegex),
		SplitBySep(s.Separator),
		SplitByChar(),
	}
}

// SplitText splits the text into chunks.
func (s *SentenceSplitter) SplitText(text string) []string {
	return s.splitText(text, s.ChunkSize)
}

// SplitTextMetadataAware splits text into chunks, accounting for metadata length.
// This is useful for RAG applications where metadata consumes context window.
func (s *SentenceSplitter) SplitTextMetadataAware(text string, metadata string) ([]string, error) {
	metaTokens := MetadataTokenCount(s.Tokenizer, metadata)
	effective, err := EffectiveChunkSizeAfterMetadata(s.ChunkSize, metaTokens)
	if err != nil {
		return nil, err
	}
	return s.splitText(text, effective), nil
}

func (s *SentenceSplitter) splitText(text string, chunkSize int) []string {
	if text == "" {
		return []string{text}
	}

	if s.onChunkingStart != nil {
		s.onChunkingStart([]string{text})
	}

	splits := s.split(text, chunkSize)
	chunks := s.merge(splits, chunkSize)
	processedChunks := s.postprocessChunks(chunks)

	if s.onChunkingEnd != nil {
		s.onChunkingEnd(processedChunks)
	}

	return processedChunks
}

func (s *SentenceSplitter) split(text string, chunkSize int) []textSplit {
	tokenSize := s.getTokenSize(text)
	if tokenSize <= chunkSize {
		return []textSplit{{text: text, isSentence: true, tokenSize: tokenSize}}
	}

	textSplitsByFns, isSentence := s.getSplitsByFns(text)
	var textSplits []textSplit

	for _, splitStr := range textSplitsByFns {
		tokenSize := s.getTokenSize(splitStr)
		if tokenSize <= chunkSize {
			textSplits = append(textSplits, textSplit{
				text:       splitStr,
				isSentence: isSentence,
				tokenSize:  tokenSize,
			})
		} else {
			recursiveSplits := s.split(splitStr, chunkSize)
			textSplits = append(textSplits, recursiveSplits...)
		}
	}
	return textSplits
}

func (s *SentenceSplitter) merge(splits []textSplit, chunkSize int) []string {
	var chunks []string
	// current chunk buffer: list of (text, length)
	type bufItem struct {
		text string
		len  int
	}
	var curChunk []bufItem
	var lastChunk []bufItem
	curChunkLen := 0
	overlapBudget := min(s.ChunkOverlap, chunkSize)

	closeChunk := func() {
		var sb strings.Builder
		for _, item := range curChunk {
			sb.WriteString(item.text)
		}
		chunks = append(chunks, sb.String())

		lastChunk = curChunk
		curChunk = nil // reset
		curChunkLen = 0

		// Add overlap from lastChunk; cap by both ChunkOverlap and chunkSize so the
		// next chunk's buffer never exceeds the effective content window.
		ob := overlapBudget
		if len(lastChunk) > 0 {
			lastIndex := len(lastChunk) - 1
			for lastIndex >= 0 {
				item := lastChunk[lastIndex]
				if curChunkLen+item.len <= ob {
					curChunkLen += item.len
					// Prepend to curChunk
					curChunk = append([]bufItem{item}, curChunk...)
					lastIndex--
				} else {
					break
				}
			}
		}
	}

	splitIdx := 0
	for splitIdx < len(splits) {
		curSplit := splits[splitIdx]

		if curChunkLen+curSplit.tokenSize > chunkSize && len(curChunk) > 0 {
			closeChunk()
			continue
		}
		emptyBuffer := len(curChunk) == 0 && curChunkLen == 0
		if curSplit.isSentence || curChunkLen+curSplit.tokenSize <= chunkSize || emptyBuffer {
			curChunkLen += curSplit.tokenSize
			curChunk = append(curChunk, bufItem{text: curSplit.text, len: curSplit.tokenSize})
			splitIdx++
		} else {
			closeChunk()
		}
	}

	if len(curChunk) > 0 {
		var sb strings.Builder
		for _, item := range curChunk {
			sb.WriteString(item.text)
		}
		chunks = append(chunks, sb.String())
	}

	return chunks
}

func (s *SentenceSplitter) postprocessChunks(chunks []string) []string {
	var newChunks []string
	for _, chunk := range chunks {
		stripped := strings.TrimSpace(chunk)
		if stripped == "" {
			continue
		}
		newChunks = append(newChunks, stripped)
	}
	return newChunks
}

func (s *SentenceSplitter) getTokenSize(text string) int {
	return len(s.Tokenizer.Encode(text))
}

func (s *SentenceSplitter) getSplitsByFns(text string) ([]string, bool) {
	for _, splitFn := range s._splitFns {
		splits := splitFn(text)
		if len(splits) > 1 {
			return splits, true
		}
	}

	var splits []string
	for _, splitFn := range s._subSentenceSplitFns {
		splits = splitFn(text)
		if len(splits) > 1 {
			break
		}
	}
	return splits, false
}
