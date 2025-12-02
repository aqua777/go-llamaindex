package textsplitter

import (
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/validation"
)

// TokenTextSplitter splits text based on token count rather than character count.
// This is useful when working with LLMs that have token limits.
type TokenTextSplitter struct {
	// ChunkSize is the maximum number of tokens per chunk.
	ChunkSize int
	// ChunkOverlap is the number of overlapping tokens between chunks.
	ChunkOverlap int
	// Tokenizer is used to count tokens. Defaults to SimpleTokenizer.
	Tokenizer Tokenizer
	// Separator is used to split text into initial segments. Defaults to " ".
	Separator string
	// KeepSeparator determines if separators are kept in the output.
	KeepSeparator bool
}

// NewTokenTextSplitter creates a new TokenTextSplitter with default settings.
func NewTokenTextSplitter(chunkSize, chunkOverlap int) *TokenTextSplitter {
	if chunkSize <= 0 {
		chunkSize = DefaultChunkSize
	}
	return &TokenTextSplitter{
		ChunkSize:     chunkSize,
		ChunkOverlap:  chunkOverlap,
		Tokenizer:     NewSimpleTokenizer(),
		Separator:     " ",
		KeepSeparator: false,
	}
}

// NewTokenTextSplitterWithTokenizer creates a TokenTextSplitter with a custom tokenizer.
func NewTokenTextSplitterWithTokenizer(chunkSize, chunkOverlap int, tokenizer Tokenizer) *TokenTextSplitter {
	if chunkSize <= 0 {
		chunkSize = DefaultChunkSize
	}
	if tokenizer == nil {
		tokenizer = NewSimpleTokenizer()
	}
	return &TokenTextSplitter{
		ChunkSize:     chunkSize,
		ChunkOverlap:  chunkOverlap,
		Tokenizer:     tokenizer,
		Separator:     " ",
		KeepSeparator: false,
	}
}

// WithSeparator sets a custom separator.
func (s *TokenTextSplitter) WithSeparator(sep string) *TokenTextSplitter {
	s.Separator = sep
	return s
}

// WithKeepSeparator sets whether to keep separators.
func (s *TokenTextSplitter) WithKeepSeparator(keep bool) *TokenTextSplitter {
	s.KeepSeparator = keep
	return s
}

// NewTokenTextSplitterWithValidation creates a TokenTextSplitter with input validation.
// Returns an error if parameters are invalid.
func NewTokenTextSplitterWithValidation(chunkSize, chunkOverlap int, tokenizer Tokenizer) (*TokenTextSplitter, error) {
	if err := validation.ValidateChunkParams(chunkSize, chunkOverlap); err != nil {
		return nil, fmt.Errorf("invalid token splitter config: %w", err)
	}

	if tokenizer == nil {
		tokenizer = NewSimpleTokenizer()
	}

	return &TokenTextSplitter{
		ChunkSize:     chunkSize,
		ChunkOverlap:  chunkOverlap,
		Tokenizer:     tokenizer,
		Separator:     " ",
		KeepSeparator: false,
	}, nil
}

// Validate validates the current splitter configuration.
func (s *TokenTextSplitter) Validate() error {
	return validation.ValidateTokenSplitterConfig(validation.TokenSplitterConfig{
		ChunkSize:    s.ChunkSize,
		ChunkOverlap: s.ChunkOverlap,
		Separator:    s.Separator,
	})
}

// SplitText splits text into chunks based on token count.
func (s *TokenTextSplitter) SplitText(text string) []string {
	if text == "" {
		return []string{}
	}

	// Split by separator first
	var splits []string
	if s.Separator != "" {
		if s.KeepSeparator {
			splits = SplitTextKeepSeparator(text, s.Separator)
		} else {
			splits = strings.Split(text, s.Separator)
		}
	} else {
		splits = []string{text}
	}

	// Filter empty splits
	var filteredSplits []string
	for _, split := range splits {
		if split != "" {
			filteredSplits = append(filteredSplits, split)
		}
	}

	return s.mergeSplits(filteredSplits)
}

// mergeSplits merges splits into chunks respecting token limits.
func (s *TokenTextSplitter) mergeSplits(splits []string) []string {
	var chunks []string
	var currentChunk []string
	currentTokens := 0

	separator := s.Separator
	if s.KeepSeparator {
		separator = ""
	}
	sepTokens := s.tokenLength(separator)

	for _, split := range splits {
		splitTokens := s.tokenLength(split)

		// If single split exceeds chunk size, we need to handle it
		if splitTokens > s.ChunkSize {
			// Flush current chunk if not empty
			if len(currentChunk) > 0 {
				chunks = append(chunks, s.joinChunk(currentChunk, separator))
				currentChunk = nil
				currentTokens = 0
			}
			// Add oversized split as its own chunk (or split it further)
			subChunks := s.splitOversized(split)
			chunks = append(chunks, subChunks...)
			continue
		}

		// Calculate tokens if we add this split
		newTokens := currentTokens + splitTokens
		if len(currentChunk) > 0 {
			newTokens += sepTokens
		}

		if newTokens > s.ChunkSize {
			// Flush current chunk
			if len(currentChunk) > 0 {
				chunks = append(chunks, s.joinChunk(currentChunk, separator))

				// Handle overlap
				currentChunk, currentTokens = s.getOverlapChunk(currentChunk, separator)
			}
		}

		currentChunk = append(currentChunk, split)
		currentTokens = s.tokenLength(s.joinChunk(currentChunk, separator))
	}

	// Add remaining chunk
	if len(currentChunk) > 0 {
		chunks = append(chunks, s.joinChunk(currentChunk, separator))
	}

	return s.postProcess(chunks)
}

// splitOversized splits a single oversized piece into smaller chunks.
func (s *TokenTextSplitter) splitOversized(text string) []string {
	// Split by characters as last resort
	tokens := s.Tokenizer.Encode(text)
	var chunks []string

	for i := 0; i < len(tokens); i += s.ChunkSize - s.ChunkOverlap {
		end := i + s.ChunkSize
		if end > len(tokens) {
			end = len(tokens)
		}

		// We need to reconstruct text from tokens
		// Since our tokenizer returns string representations, we approximate
		// by splitting the original text proportionally
		startRatio := float64(i) / float64(len(tokens))
		endRatio := float64(end) / float64(len(tokens))

		startChar := int(startRatio * float64(len(text)))
		endChar := int(endRatio * float64(len(text)))

		if endChar > len(text) {
			endChar = len(text)
		}

		chunk := text[startChar:endChar]
		if chunk != "" {
			chunks = append(chunks, strings.TrimSpace(chunk))
		}

		if end >= len(tokens) {
			break
		}
	}

	return chunks
}

// getOverlapChunk returns the overlap portion of the current chunk.
func (s *TokenTextSplitter) getOverlapChunk(chunk []string, separator string) ([]string, int) {
	if s.ChunkOverlap <= 0 || len(chunk) == 0 {
		return nil, 0
	}

	// Work backwards to find overlap
	var overlapChunk []string
	overlapTokens := 0

	for i := len(chunk) - 1; i >= 0; i-- {
		part := chunk[i]
		partTokens := s.tokenLength(part)

		if overlapTokens+partTokens > s.ChunkOverlap {
			break
		}

		overlapChunk = append([]string{part}, overlapChunk...)
		overlapTokens += partTokens
		if len(overlapChunk) > 1 {
			overlapTokens += s.tokenLength(separator)
		}
	}

	return overlapChunk, overlapTokens
}

// joinChunk joins chunk parts with separator.
func (s *TokenTextSplitter) joinChunk(parts []string, separator string) string {
	return strings.Join(parts, separator)
}

// tokenLength returns the token count for text.
func (s *TokenTextSplitter) tokenLength(text string) int {
	return len(s.Tokenizer.Encode(text))
}

// postProcess cleans up chunks.
func (s *TokenTextSplitter) postProcess(chunks []string) []string {
	var result []string
	for _, chunk := range chunks {
		trimmed := strings.TrimSpace(chunk)
		if trimmed != "" {
			result = append(result, trimmed)
		}
	}
	return result
}

// SplitTextMetadataAware splits text accounting for metadata token usage.
func (s *TokenTextSplitter) SplitTextMetadataAware(text string, metadata string) []string {
	metadataTokens := s.tokenLength(metadata)
	effectiveChunkSize := s.ChunkSize - metadataTokens

	if effectiveChunkSize < 1 {
		effectiveChunkSize = 1
	}

	// Create a temporary splitter with reduced chunk size
	tempSplitter := &TokenTextSplitter{
		ChunkSize:     effectiveChunkSize,
		ChunkOverlap:  s.ChunkOverlap,
		Tokenizer:     s.Tokenizer,
		Separator:     s.Separator,
		KeepSeparator: s.KeepSeparator,
	}

	return tempSplitter.SplitText(text)
}
