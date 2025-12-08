package textsplitter

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ============================================================================
// TokenTextSplitter Tests
// ============================================================================

func TestTokenTextSplitter_Basic(t *testing.T) {
	splitter := NewTokenTextSplitter(10, 2)

	text := "This is a test sentence. Here is another one. And a third."
	chunks := splitter.SplitText(text)

	require.NotEmpty(t, chunks)
	// All chunks should be non-empty
	for _, chunk := range chunks {
		assert.NotEmpty(t, chunk)
	}
}

func TestTokenTextSplitter_EmptyText(t *testing.T) {
	splitter := NewTokenTextSplitter(10, 0)

	chunks := splitter.SplitText("")
	assert.Empty(t, chunks)
}

func TestTokenTextSplitter_SmallText(t *testing.T) {
	splitter := NewTokenTextSplitter(100, 0)

	text := "Short text"
	chunks := splitter.SplitText(text)

	require.Len(t, chunks, 1)
	assert.Equal(t, "Short text", chunks[0])
}

func TestTokenTextSplitter_WithCustomSeparator(t *testing.T) {
	splitter := NewTokenTextSplitter(20, 0).WithSeparator("\n")

	text := "Line one\nLine two\nLine three"
	chunks := splitter.SplitText(text)

	require.NotEmpty(t, chunks)
}

func TestTokenTextSplitter_WithKeepSeparator(t *testing.T) {
	splitter := NewTokenTextSplitter(50, 0).WithSeparator(".").WithKeepSeparator(true)

	text := "First sentence. Second sentence. Third sentence."
	chunks := splitter.SplitText(text)

	require.NotEmpty(t, chunks)
}

func TestTokenTextSplitter_MetadataAware(t *testing.T) {
	splitter := NewTokenTextSplitter(20, 0)

	text := "This is some text that should be split into chunks."
	metadata := "filename: test.txt, author: John"

	chunksWithMeta := splitter.SplitTextMetadataAware(text, metadata)
	chunksWithoutMeta := splitter.SplitText(text)

	// With metadata, chunks should be smaller or equal
	assert.GreaterOrEqual(t, len(chunksWithMeta), len(chunksWithoutMeta))
}

func TestTokenTextSplitter_WithTokenizer(t *testing.T) {
	tokenizer := NewSimpleTokenizer()
	splitter := NewTokenTextSplitterWithTokenizer(10, 2, tokenizer)

	text := "One two three four five six seven eight nine ten eleven twelve"
	chunks := splitter.SplitText(text)

	require.NotEmpty(t, chunks)
}

func TestTokenTextSplitter_Overlap(t *testing.T) {
	splitter := NewTokenTextSplitter(5, 2)

	text := "one two three four five six seven eight"
	chunks := splitter.SplitText(text)

	require.Greater(t, len(chunks), 1)
}

// ============================================================================
// MarkdownSplitter Tests
// ============================================================================

func TestMarkdownSplitter_Basic(t *testing.T) {
	splitter := NewMarkdownSplitter(100, 0)

	text := `# Header 1

This is some content under header 1.

## Header 2

This is content under header 2.
`
	chunks := splitter.SplitText(text)

	require.NotEmpty(t, chunks)
}

func TestMarkdownSplitter_EmptyText(t *testing.T) {
	splitter := NewMarkdownSplitter(100, 0)

	chunks := splitter.SplitText("")
	assert.Empty(t, chunks)
}

func TestMarkdownSplitter_CodeBlocks(t *testing.T) {
	splitter := NewMarkdownSplitter(200, 0)

	text := "Some text before.\n\n```python\ndef hello():\n    print('Hello')\n```\n\nSome text after."

	chunks := splitter.SplitText(text)

	require.NotEmpty(t, chunks)

	// Check that code block is preserved
	foundCode := false
	for _, chunk := range chunks {
		if strings.Contains(chunk, "```python") {
			foundCode = true
			assert.Contains(t, chunk, "def hello()")
		}
	}
	assert.True(t, foundCode, "Code block should be preserved")
}

func TestMarkdownSplitter_MultipleCodeBlocks(t *testing.T) {
	splitter := NewMarkdownSplitter(500, 0)

	text := `# Code Examples

Here's Python:

` + "```python\nprint('hello')\n```" + `

And JavaScript:

` + "```javascript\nconsole.log('hello');\n```"

	chunks := splitter.SplitText(text)

	require.NotEmpty(t, chunks)
}

func TestMarkdownSplitter_LargeCodeBlock(t *testing.T) {
	splitter := NewMarkdownSplitter(20, 0) // Very small chunk size

	// Create a large code block
	var lines []string
	lines = append(lines, "```python")
	for i := 0; i < 50; i++ {
		lines = append(lines, "print('This is a much longer line number "+string(rune('0'+i%10))+"')")
	}
	lines = append(lines, "```")
	text := strings.Join(lines, "\n")

	chunks := splitter.SplitText(text)

	require.NotEmpty(t, chunks)
	// Large code block should be handled (may be split or kept whole depending on implementation)
	// The important thing is it doesn't crash and produces valid output
	for _, chunk := range chunks {
		assert.NotEmpty(t, strings.TrimSpace(chunk))
	}
}

func TestMarkdownSplitter_HeadersOnly(t *testing.T) {
	splitter := NewMarkdownSplitter(50, 0)

	text := `# Header 1

Content 1

## Header 2

Content 2

### Header 3

Content 3`

	chunks := splitter.SplitText(text)

	require.NotEmpty(t, chunks)
}

func TestMarkdownSplitter_NoHeaders(t *testing.T) {
	splitter := NewMarkdownSplitter(100, 0)

	text := "Just some plain text without any markdown headers."

	chunks := splitter.SplitText(text)

	require.Len(t, chunks, 1)
	assert.Equal(t, text, chunks[0])
}

func TestMarkdownSplitter_MetadataAware(t *testing.T) {
	splitter := NewMarkdownSplitter(50, 0)

	text := "# Header\n\nSome content that should be split based on available space."
	metadata := "source: document.md"

	chunks := splitter.SplitTextMetadataAware(text, metadata)

	require.NotEmpty(t, chunks)
}

func TestMarkdownSplitter_WithTokenizer(t *testing.T) {
	splitter := NewMarkdownSplitter(100, 0).WithTokenizer(NewSimpleTokenizer())

	text := "# Test\n\nContent"
	chunks := splitter.SplitText(text)

	require.NotEmpty(t, chunks)
}

// ============================================================================
// SentenceWindowSplitter Tests
// ============================================================================

func TestSentenceWindowSplitter_Basic(t *testing.T) {
	splitter := NewSentenceWindowSplitter(1)

	text := "First sentence. Second sentence. Third sentence. Fourth sentence."
	chunks := splitter.SplitText(text)

	require.NotEmpty(t, chunks)
}

func TestSentenceWindowSplitter_EmptyText(t *testing.T) {
	splitter := NewSentenceWindowSplitter(1)

	chunks := splitter.SplitText("")
	assert.Empty(t, chunks)
}

func TestSentenceWindowSplitter_Windows(t *testing.T) {
	splitter := NewSentenceWindowSplitter(1)

	text := "First. Second. Third. Fourth. Fifth."
	windows := splitter.SplitTextWithWindows(text)

	require.NotEmpty(t, windows)

	// Check first window (should include first and second)
	assert.Equal(t, "First.", windows[0].Sentence)
	assert.Contains(t, windows[0].Window, "First.")
	assert.Contains(t, windows[0].Window, "Second.")
	assert.Equal(t, 0, windows[0].StartSentence)

	// Check middle window (should include prev, current, next)
	if len(windows) > 2 {
		middleIdx := len(windows) / 2
		assert.Contains(t, windows[middleIdx].Window, windows[middleIdx].Sentence)
	}

	// Check last window
	lastIdx := len(windows) - 1
	assert.Equal(t, lastIdx, windows[lastIdx].Index)
}

func TestSentenceWindowSplitter_WindowSize(t *testing.T) {
	// Window size 0 means just the sentence itself
	splitter := NewSentenceWindowSplitter(0)

	text := "First. Second. Third."
	windows := splitter.SplitTextWithWindows(text)

	require.NotEmpty(t, windows)

	// Each window should just be the sentence
	for _, w := range windows {
		assert.Equal(t, w.Sentence, w.Window)
	}
}

func TestSentenceWindowSplitter_LargeWindowSize(t *testing.T) {
	// Window size larger than text
	splitter := NewSentenceWindowSplitter(100)

	text := "First. Second. Third."
	windows := splitter.SplitTextWithWindows(text)

	require.NotEmpty(t, windows)

	// All windows should contain all sentences
	for _, w := range windows {
		assert.Contains(t, w.Window, "First.")
		assert.Contains(t, w.Window, "Second.")
		assert.Contains(t, w.Window, "Third.")
	}
}

func TestSentenceWindowSplitter_GetWindowsText(t *testing.T) {
	splitter := NewSentenceWindowSplitter(1)

	text := "First. Second. Third."
	windowTexts := splitter.GetWindowsText(text)

	require.NotEmpty(t, windowTexts)
	assert.Len(t, windowTexts, 3)
}

func TestSentenceWindowSplitter_ForNodes(t *testing.T) {
	splitter := NewSentenceWindowSplitter(1)

	text := "First sentence. Second sentence. Third sentence."
	nodeData := splitter.SplitTextForNodes(text)

	require.NotEmpty(t, nodeData)

	for _, data := range nodeData {
		assert.NotEmpty(t, data.Text)
		assert.NotEmpty(t, data.Window)
		assert.NotNil(t, data.Metadata)
		assert.Contains(t, data.Metadata, "original_sentence")
		assert.Contains(t, data.Metadata, "window")
		assert.Contains(t, data.Metadata, "sentence_index")
	}
}

func TestSentenceWindowSplitter_CustomMetadataKeys(t *testing.T) {
	splitter := NewSentenceWindowSplitter(1).WithMetadataKeys("sent", "ctx")

	text := "First. Second."
	nodeData := splitter.SplitTextForNodes(text)

	require.NotEmpty(t, nodeData)
	assert.Contains(t, nodeData[0].Metadata, "sent")
	assert.Contains(t, nodeData[0].Metadata, "ctx")
}

func TestSentenceWindowSplitter_WithCustomSplitter(t *testing.T) {
	customSplitter := NewRegexSplitterStrategy(`[^.!?]+[.!?]`)
	splitter := NewSentenceWindowSplitter(1).WithSentenceSplitter(customSplitter)

	text := "Hello! How are you? I am fine."
	windows := splitter.SplitTextWithWindows(text)

	require.NotEmpty(t, windows)
}

// ============================================================================
// Interface Compliance Tests
// ============================================================================

func TestTextSplitterInterface(t *testing.T) {
	// Verify all splitters implement TextSplitter
	var _ TextSplitter = &TokenTextSplitter{}
	var _ TextSplitter = &MarkdownSplitter{}
	var _ TextSplitter = &SentenceWindowSplitter{}
	var _ TextSplitter = &SentenceSplitter{}
}

// ============================================================================
// Edge Cases
// ============================================================================

func TestSplitters_SingleWord(t *testing.T) {
	tokenSplitter := NewTokenTextSplitter(10, 0)
	mdSplitter := NewMarkdownSplitter(10, 0)
	windowSplitter := NewSentenceWindowSplitter(1)

	text := "Hello"

	assert.NotEmpty(t, tokenSplitter.SplitText(text))
	assert.NotEmpty(t, mdSplitter.SplitText(text))
	assert.NotEmpty(t, windowSplitter.SplitText(text))
}

func TestSplitters_WhitespaceOnly(t *testing.T) {
	tokenSplitter := NewTokenTextSplitter(10, 0)
	mdSplitter := NewMarkdownSplitter(10, 0)
	windowSplitter := NewSentenceWindowSplitter(1)

	text := "   \n\t  "

	// Should return empty or handle gracefully
	assert.Empty(t, tokenSplitter.SplitText(text))
	assert.Empty(t, mdSplitter.SplitText(text))
	assert.Empty(t, windowSplitter.SplitText(text))
}

func TestSplitters_SpecialCharacters(t *testing.T) {
	tokenSplitter := NewTokenTextSplitter(100, 0)

	text := "Hello! @#$%^&*() World"
	chunks := tokenSplitter.SplitText(text)

	require.NotEmpty(t, chunks)
	assert.Contains(t, chunks[0], "@#$%^&*()")
}

func TestSplitters_Unicode(t *testing.T) {
	tokenSplitter := NewTokenTextSplitter(100, 0)
	mdSplitter := NewMarkdownSplitter(100, 0)

	text := "Hello 世界! Привет мир! 🌍"

	tokenChunks := tokenSplitter.SplitText(text)
	mdChunks := mdSplitter.SplitText(text)

	require.NotEmpty(t, tokenChunks)
	require.NotEmpty(t, mdChunks)
}
