package textsplitter

// TextSplitter is the interface for splitting text.
type TextSplitter interface {
	SplitText(text string) []string
}

// Tokenizer is the interface for tokenizing text.
// It encodes text into a list of string tokens.
type Tokenizer interface {
	Encode(text string) []string
}

// SentenceSplitterStrategy is the interface for primary sentence splitting.
type SentenceSplitterStrategy interface {
	Split(text string) []string
}
