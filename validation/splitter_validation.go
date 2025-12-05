package validation

import (
	"fmt"
)

// SentenceSplitterConfig holds configuration for SentenceSplitter validation.
type SentenceSplitterConfig struct {
	ChunkSize              int
	ChunkOverlap           int
	Separator              string
	ParagraphSeparator     string
	SecondaryChunkingRegex string
}

// ValidateSentenceSplitterConfig validates SentenceSplitter configuration.
func ValidateSentenceSplitterConfig(cfg SentenceSplitterConfig) error {
	v := NewValidator()

	v.RequirePositive(cfg.ChunkSize, "chunk_size")
	v.RequireNonNegative(cfg.ChunkOverlap, "chunk_overlap")

	if cfg.ChunkOverlap >= cfg.ChunkSize && cfg.ChunkSize > 0 {
		v.AddError("chunk_overlap", 
			fmt.Sprintf("must be less than chunk_size (%d)", cfg.ChunkSize), 
			cfg.ChunkOverlap)
	}

	// Warn if overlap is more than 50% of chunk size (not an error, but unusual)
	if cfg.ChunkSize > 0 && cfg.ChunkOverlap > cfg.ChunkSize/2 {
		// This is a warning, not an error - we don't add it to validator
	}

	return v.Error()
}

// TokenSplitterConfig holds configuration for TokenTextSplitter validation.
type TokenSplitterConfig struct {
	ChunkSize    int
	ChunkOverlap int
	Separator    string
}

// ValidateTokenSplitterConfig validates TokenTextSplitter configuration.
func ValidateTokenSplitterConfig(cfg TokenSplitterConfig) error {
	v := NewValidator()

	v.RequirePositive(cfg.ChunkSize, "chunk_size")
	v.RequireNonNegative(cfg.ChunkOverlap, "chunk_overlap")

	if cfg.ChunkOverlap >= cfg.ChunkSize && cfg.ChunkSize > 0 {
		v.AddError("chunk_overlap",
			fmt.Sprintf("must be less than chunk_size (%d)", cfg.ChunkSize),
			cfg.ChunkOverlap)
	}

	return v.Error()
}

// MarkdownSplitterConfig holds configuration for MarkdownSplitter validation.
type MarkdownSplitterConfig struct {
	ChunkSize    int
	ChunkOverlap int
}

// ValidateMarkdownSplitterConfig validates MarkdownSplitter configuration.
func ValidateMarkdownSplitterConfig(cfg MarkdownSplitterConfig) error {
	v := NewValidator()

	v.RequirePositive(cfg.ChunkSize, "chunk_size")
	v.RequireNonNegative(cfg.ChunkOverlap, "chunk_overlap")

	if cfg.ChunkOverlap >= cfg.ChunkSize && cfg.ChunkSize > 0 {
		v.AddError("chunk_overlap",
			fmt.Sprintf("must be less than chunk_size (%d)", cfg.ChunkSize),
			cfg.ChunkOverlap)
	}

	return v.Error()
}

// SentenceWindowSplitterConfig holds configuration for SentenceWindowSplitter validation.
type SentenceWindowSplitterConfig struct {
	WindowSize            int
	OriginalTextMetaKey   string
	WindowMetaKey         string
}

// ValidateSentenceWindowSplitterConfig validates SentenceWindowSplitter configuration.
func ValidateSentenceWindowSplitterConfig(cfg SentenceWindowSplitterConfig) error {
	v := NewValidator()

	v.RequireNonNegative(cfg.WindowSize, "window_size")
	v.RequireNotEmpty(cfg.OriginalTextMetaKey, "original_text_metadata_key")
	v.RequireNotEmpty(cfg.WindowMetaKey, "window_metadata_key")

	return v.Error()
}

// MetadataAwareSplitConfig holds configuration for metadata-aware splitting.
type MetadataAwareSplitConfig struct {
	ChunkSize      int
	MetadataLength int
}

// ValidateMetadataAwareSplit validates that metadata doesn't exceed chunk size.
func ValidateMetadataAwareSplit(cfg MetadataAwareSplitConfig) error {
	effectiveSize := cfg.ChunkSize - cfg.MetadataLength

	if effectiveSize <= 0 {
		return fmt.Errorf(
			"metadata length (%d) is longer than chunk size (%d); "+
				"consider increasing chunk size or decreasing metadata size",
			cfg.MetadataLength, cfg.ChunkSize)
	}

	if effectiveSize < 50 {
		// This is a warning condition - effective size is very small
		// We return nil but callers may want to log this
	}

	return nil
}

// GetEffectiveChunkSize returns the effective chunk size after accounting for metadata.
// Returns an error if the effective size would be non-positive.
func GetEffectiveChunkSize(chunkSize, metadataLength int) (int, error) {
	effective := chunkSize - metadataLength
	if effective <= 0 {
		return 0, fmt.Errorf(
			"metadata length (%d) exceeds or equals chunk size (%d)",
			metadataLength, chunkSize)
	}
	return effective, nil
}
