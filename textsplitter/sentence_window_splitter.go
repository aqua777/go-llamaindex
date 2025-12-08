package textsplitter

import (
	"strings"
)

// SentenceWindowSplitter splits text into sentences and includes surrounding
// context (window) for each sentence. This is useful for retrieval where
// you want to match on a specific sentence but return more context.
type SentenceWindowSplitter struct {
	// WindowSize is the number of sentences to include on each side.
	WindowSize int
	// Tokenizer is used to count tokens.
	Tokenizer Tokenizer
	// SentenceSplitter is used to split text into sentences.
	SentenceSplitter SentenceSplitterStrategy
	// OriginalTextMetadataKey is the metadata key for storing the original sentence.
	OriginalTextMetadataKey string
	// WindowMetadataKey is the metadata key for storing the window text.
	WindowMetadataKey string
}

// SentenceWindow represents a sentence with its surrounding context.
type SentenceWindow struct {
	// Sentence is the original sentence.
	Sentence string
	// Window is the sentence with surrounding context.
	Window string
	// Index is the sentence index in the original text.
	Index int
	// StartSentence is the index of the first sentence in the window.
	StartSentence int
	// EndSentence is the index of the last sentence in the window.
	EndSentence int
}

// NewSentenceWindowSplitter creates a new SentenceWindowSplitter.
func NewSentenceWindowSplitter(windowSize int) *SentenceWindowSplitter {
	if windowSize < 0 {
		windowSize = 3 // Default window size
	}
	return &SentenceWindowSplitter{
		WindowSize:              windowSize,
		Tokenizer:               NewSimpleTokenizer(),
		SentenceSplitter:        NewRegexSplitterStrategy(DefaultChunkingRegex),
		OriginalTextMetadataKey: "original_sentence",
		WindowMetadataKey:       "window",
	}
}

// WithTokenizer sets a custom tokenizer.
func (s *SentenceWindowSplitter) WithTokenizer(tokenizer Tokenizer) *SentenceWindowSplitter {
	s.Tokenizer = tokenizer
	return s
}

// WithSentenceSplitter sets a custom sentence splitter.
func (s *SentenceWindowSplitter) WithSentenceSplitter(splitter SentenceSplitterStrategy) *SentenceWindowSplitter {
	s.SentenceSplitter = splitter
	return s
}

// WithMetadataKeys sets custom metadata keys.
func (s *SentenceWindowSplitter) WithMetadataKeys(originalKey, windowKey string) *SentenceWindowSplitter {
	s.OriginalTextMetadataKey = originalKey
	s.WindowMetadataKey = windowKey
	return s
}

// SplitText splits text into sentences (returns just the sentences).
// Use SplitTextWithWindows for full window information.
func (s *SentenceWindowSplitter) SplitText(text string) []string {
	windows := s.SplitTextWithWindows(text)
	result := make([]string, len(windows))
	for i, w := range windows {
		result[i] = w.Sentence
	}
	return result
}

// SplitTextWithWindows splits text and returns sentences with their windows.
func (s *SentenceWindowSplitter) SplitTextWithWindows(text string) []SentenceWindow {
	if text == "" {
		return nil
	}

	// Split into sentences
	sentences := s.SentenceSplitter.Split(text)
	if len(sentences) == 0 {
		return nil
	}

	// Clean up sentences
	var cleanSentences []string
	for _, sent := range sentences {
		trimmed := strings.TrimSpace(sent)
		if trimmed != "" {
			cleanSentences = append(cleanSentences, trimmed)
		}
	}

	if len(cleanSentences) == 0 {
		return nil
	}

	// Build windows for each sentence
	windows := make([]SentenceWindow, len(cleanSentences))
	for i := range cleanSentences {
		windows[i] = s.buildWindow(cleanSentences, i)
	}

	return windows
}

// buildWindow builds a window around the sentence at the given index.
func (s *SentenceWindowSplitter) buildWindow(sentences []string, index int) SentenceWindow {
	// Calculate window bounds
	start := index - s.WindowSize
	if start < 0 {
		start = 0
	}
	end := index + s.WindowSize + 1
	if end > len(sentences) {
		end = len(sentences)
	}

	// Build window text
	windowSentences := sentences[start:end]
	windowText := strings.Join(windowSentences, " ")

	return SentenceWindow{
		Sentence:      sentences[index],
		Window:        windowText,
		Index:         index,
		StartSentence: start,
		EndSentence:   end - 1,
	}
}

// GetWindowsText returns just the window texts (for embedding the context).
func (s *SentenceWindowSplitter) GetWindowsText(text string) []string {
	windows := s.SplitTextWithWindows(text)
	result := make([]string, len(windows))
	for i, w := range windows {
		result[i] = w.Window
	}
	return result
}

// SentenceWindowNodeData contains data for creating nodes with window metadata.
type SentenceWindowNodeData struct {
	// Text is the sentence text (for embedding/matching).
	Text string
	// Window is the surrounding context.
	Window string
	// Metadata contains additional metadata.
	Metadata map[string]interface{}
}

// SplitTextForNodes returns data suitable for creating nodes with window metadata.
func (s *SentenceWindowSplitter) SplitTextForNodes(text string) []SentenceWindowNodeData {
	windows := s.SplitTextWithWindows(text)
	result := make([]SentenceWindowNodeData, len(windows))

	for i, w := range windows {
		result[i] = SentenceWindowNodeData{
			Text:   w.Sentence,
			Window: w.Window,
			Metadata: map[string]interface{}{
				s.OriginalTextMetadataKey: w.Sentence,
				s.WindowMetadataKey:       w.Window,
				"sentence_index":          w.Index,
				"window_start":            w.StartSentence,
				"window_end":              w.EndSentence,
			},
		}
	}

	return result
}
