package textsplitter

import (
	"regexp"
	"strings"
)

// MarkdownSplitter splits markdown text while preserving structure.
// It respects headers, code blocks, and other markdown elements.
type MarkdownSplitter struct {
	// ChunkSize is the maximum size of each chunk in tokens.
	ChunkSize int
	// ChunkOverlap is the number of overlapping tokens between chunks.
	ChunkOverlap int
	// Tokenizer is used to count tokens.
	Tokenizer Tokenizer
	// HeadersToSplitOn defines which header levels trigger splits.
	// Default: ["#", "##", "###", "####", "#####", "######"]
	HeadersToSplitOn []string
	// ReturnEachLine if true, returns each line as a separate chunk.
	ReturnEachLine bool
	// StripHeaders if true, removes headers from the output.
	StripHeaders bool
}

// MarkdownHeaderType represents a markdown header.
type MarkdownHeaderType struct {
	Level  int
	Header string
	Data   string
}

// NewMarkdownSplitter creates a new MarkdownSplitter with default settings.
func NewMarkdownSplitter(chunkSize, chunkOverlap int) *MarkdownSplitter {
	if chunkSize <= 0 {
		chunkSize = DefaultChunkSize
	}
	return &MarkdownSplitter{
		ChunkSize:    chunkSize,
		ChunkOverlap: chunkOverlap,
		Tokenizer:    NewSimpleTokenizer(),
		HeadersToSplitOn: []string{
			"#", "##", "###", "####", "#####", "######",
		},
		ReturnEachLine: false,
		StripHeaders:   false,
	}
}

// WithTokenizer sets a custom tokenizer.
func (s *MarkdownSplitter) WithTokenizer(tokenizer Tokenizer) *MarkdownSplitter {
	s.Tokenizer = tokenizer
	return s
}

// WithHeadersToSplitOn sets which headers to split on.
func (s *MarkdownSplitter) WithHeadersToSplitOn(headers []string) *MarkdownSplitter {
	s.HeadersToSplitOn = headers
	return s
}

// WithStripHeaders sets whether to strip headers from output.
func (s *MarkdownSplitter) WithStripHeaders(strip bool) *MarkdownSplitter {
	s.StripHeaders = strip
	return s
}

// SplitText splits markdown text into chunks.
func (s *MarkdownSplitter) SplitText(text string) []string {
	if text == "" {
		return []string{}
	}

	// First, split by code blocks to preserve them
	sections := s.splitByCodeBlocks(text)

	var allChunks []string
	for _, section := range sections {
		if section.isCode {
			// Code blocks are kept together if possible
			if s.tokenLength(section.content) <= s.ChunkSize {
				allChunks = append(allChunks, section.content)
			} else {
				// Split large code blocks by lines
				chunks := s.splitCodeBlock(section.content)
				allChunks = append(allChunks, chunks...)
			}
		} else {
			// Split markdown content by headers
			chunks := s.splitMarkdownContent(section.content)
			allChunks = append(allChunks, chunks...)
		}
	}

	return s.postProcess(allChunks)
}

type markdownSection struct {
	content string
	isCode  bool
}

// splitByCodeBlocks separates code blocks from regular markdown.
func (s *MarkdownSplitter) splitByCodeBlocks(text string) []markdownSection {
	// Match fenced code blocks (``` or ~~~)
	codeBlockRegex := regexp.MustCompile("(?s)(```[^`]*```|~~~[^~]*~~~)")

	var sections []markdownSection
	lastEnd := 0

	matches := codeBlockRegex.FindAllStringIndex(text, -1)
	for _, match := range matches {
		// Add text before code block
		if match[0] > lastEnd {
			before := text[lastEnd:match[0]]
			if strings.TrimSpace(before) != "" {
				sections = append(sections, markdownSection{content: before, isCode: false})
			}
		}
		// Add code block
		codeBlock := text[match[0]:match[1]]
		sections = append(sections, markdownSection{content: codeBlock, isCode: true})
		lastEnd = match[1]
	}

	// Add remaining text
	if lastEnd < len(text) {
		remaining := text[lastEnd:]
		if strings.TrimSpace(remaining) != "" {
			sections = append(sections, markdownSection{content: remaining, isCode: false})
		}
	}

	// If no code blocks found, return the whole text
	if len(sections) == 0 {
		sections = append(sections, markdownSection{content: text, isCode: false})
	}

	return sections
}

// splitCodeBlock splits a large code block into smaller chunks.
func (s *MarkdownSplitter) splitCodeBlock(codeBlock string) []string {
	lines := strings.Split(codeBlock, "\n")
	if len(lines) <= 2 {
		return []string{codeBlock}
	}

	// Extract opening and closing markers
	opening := lines[0]
	closing := lines[len(lines)-1]
	codeLines := lines[1 : len(lines)-1]

	var chunks []string
	var currentLines []string
	currentTokens := s.tokenLength(opening) + s.tokenLength(closing)

	for _, line := range codeLines {
		lineTokens := s.tokenLength(line)

		if currentTokens+lineTokens > s.ChunkSize && len(currentLines) > 0 {
			// Flush current chunk
			chunk := opening + "\n" + strings.Join(currentLines, "\n") + "\n" + closing
			chunks = append(chunks, chunk)
			currentLines = nil
			currentTokens = s.tokenLength(opening) + s.tokenLength(closing)
		}

		currentLines = append(currentLines, line)
		currentTokens += lineTokens
	}

	// Add remaining
	if len(currentLines) > 0 {
		chunk := opening + "\n" + strings.Join(currentLines, "\n") + "\n" + closing
		chunks = append(chunks, chunk)
	}

	return chunks
}

// splitMarkdownContent splits markdown by headers and merges into chunks.
func (s *MarkdownSplitter) splitMarkdownContent(text string) []string {
	// Build header regex pattern
	headerPattern := s.buildHeaderPattern()
	headerRegex := regexp.MustCompile(headerPattern)

	// Split by headers
	sections := s.splitByHeaders(text, headerRegex)

	// Merge sections into chunks respecting size limits
	return s.mergeSections(sections)
}

// buildHeaderPattern builds a regex pattern for matching headers.
func (s *MarkdownSplitter) buildHeaderPattern() string {
	// Match headers at the start of a line
	// e.g., "^(#{1,6})\s+(.+)$" for all header levels
	return `(?m)^(#{1,6})\s+(.+)$`
}

// splitByHeaders splits text by markdown headers.
func (s *MarkdownSplitter) splitByHeaders(text string, headerRegex *regexp.Regexp) []markdownSection {
	var sections []markdownSection

	matches := headerRegex.FindAllStringIndex(text, -1)
	if len(matches) == 0 {
		// No headers, return as single section
		return []markdownSection{{content: text, isCode: false}}
	}

	lastEnd := 0
	for _, match := range matches {
		// Add content before this header
		if match[0] > lastEnd {
			before := text[lastEnd:match[0]]
			if strings.TrimSpace(before) != "" {
				sections = append(sections, markdownSection{content: before, isCode: false})
			}
		}
		lastEnd = match[0]
	}

	// Add content from last header to end
	if lastEnd < len(text) {
		sections = append(sections, markdownSection{content: text[lastEnd:], isCode: false})
	}

	return sections
}

// mergeSections merges sections into chunks respecting token limits.
func (s *MarkdownSplitter) mergeSections(sections []markdownSection) []string {
	var chunks []string
	var currentChunk strings.Builder
	currentTokens := 0

	for _, section := range sections {
		sectionTokens := s.tokenLength(section.content)

		// If section alone exceeds chunk size, split it further
		if sectionTokens > s.ChunkSize {
			// Flush current chunk
			if currentChunk.Len() > 0 {
				chunks = append(chunks, currentChunk.String())
				currentChunk.Reset()
				currentTokens = 0
			}
			// Split large section by paragraphs or lines
			subChunks := s.splitLargeSection(section.content)
			chunks = append(chunks, subChunks...)
			continue
		}

		// Check if adding this section exceeds limit
		if currentTokens+sectionTokens > s.ChunkSize && currentChunk.Len() > 0 {
			chunks = append(chunks, currentChunk.String())
			currentChunk.Reset()
			currentTokens = 0
		}

		currentChunk.WriteString(section.content)
		currentTokens += sectionTokens
	}

	// Add remaining
	if currentChunk.Len() > 0 {
		chunks = append(chunks, currentChunk.String())
	}

	return chunks
}

// splitLargeSection splits a large section by paragraphs or lines.
func (s *MarkdownSplitter) splitLargeSection(text string) []string {
	// Try splitting by paragraphs first
	paragraphs := strings.Split(text, "\n\n")
	if len(paragraphs) > 1 {
		return s.mergeSections(s.toSections(paragraphs))
	}

	// Fall back to splitting by lines
	lines := strings.Split(text, "\n")
	return s.mergeSections(s.toSections(lines))
}

// toSections converts strings to sections.
func (s *MarkdownSplitter) toSections(texts []string) []markdownSection {
	sections := make([]markdownSection, 0, len(texts))
	for _, t := range texts {
		if strings.TrimSpace(t) != "" {
			sections = append(sections, markdownSection{content: t, isCode: false})
		}
	}
	return sections
}

// tokenLength returns the token count for text.
func (s *MarkdownSplitter) tokenLength(text string) int {
	return len(s.Tokenizer.Encode(text))
}

// postProcess cleans up chunks.
func (s *MarkdownSplitter) postProcess(chunks []string) []string {
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
func (s *MarkdownSplitter) SplitTextMetadataAware(text string, metadata string) []string {
	metadataTokens := s.tokenLength(metadata)
	effectiveChunkSize := s.ChunkSize - metadataTokens

	if effectiveChunkSize < 1 {
		effectiveChunkSize = 1
	}

	tempSplitter := &MarkdownSplitter{
		ChunkSize:        effectiveChunkSize,
		ChunkOverlap:     s.ChunkOverlap,
		Tokenizer:        s.Tokenizer,
		HeadersToSplitOn: s.HeadersToSplitOn,
		ReturnEachLine:   s.ReturnEachLine,
		StripHeaders:     s.StripHeaders,
	}

	return tempSplitter.SplitText(text)
}
