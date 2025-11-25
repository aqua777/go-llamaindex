package textsplitter

import (
	"testing"

	"github.com/stretchr/testify/suite"
)

type SentenceSplitterTestSuite struct {
	suite.Suite
}

func TestSentenceSplitterTestSuite(t *testing.T) {
	suite.Run(t, new(SentenceSplitterTestSuite))
}

func (s *SentenceSplitterTestSuite) TestSplitText_Basic() {
	splitter := NewSentenceSplitter(100, 0, nil, nil)
	text := "Hello world. This is a test."
	chunks := splitter.SplitText(text)
	s.Len(chunks, 1)
	s.Equal("Hello world. This is a test.", chunks[0])
}

func (s *SentenceSplitterTestSuite) TestSplitText_SplitBySentence() {
	// Tokenizer counts whitespace-separated words.
	splitter := NewSentenceSplitter(3, 0, nil, nil)
	text := "Hello world. This is a test."
	chunks := splitter.SplitText(text)

	// Regex strategy splits: "Hello world.", "This is a test."
	// "Hello world." (2 tokens) fits in 3.
	// "This is a test." (4 tokens) > 3, recursively splits.
	// "This is a test." -> ["This", "is", "a", "test."]
	// Merging:
	// Chunk 1: "Hello world." (2) + "This" (1) = 3. Matches chunk size.
	// Chunk 2: "is" (1) + "a" (1) + "test." (1) = 3.
	
	s.Len(chunks, 2)
	s.Equal("Hello world. This", chunks[0])
	s.Equal("is a test.", chunks[1])
}

func (s *SentenceSplitterTestSuite) TestSplitText_Overlap() {
	splitter := NewSentenceSplitter(3, 1, nil, nil)
	text := "A B C D E"
	chunks := splitter.SplitText(text)
	
	s.Len(chunks, 2)
	s.Equal("A B C", chunks[0])
	s.Equal("C D E", chunks[1])
}

func (s *SentenceSplitterTestSuite) TestSplitText_Paragraphs() {
	text := "P1 S1. P1 S2.\n\n\nP2 S1. P2 S2."
	
	splitter := NewSentenceSplitter(3, 0, nil, nil)
	chunks := splitter.SplitText(text)
	
	s.Len(chunks, 4)
	s.Equal("P1 S1.", chunks[0])
	s.Equal("P1 S2.", chunks[1])
	s.Equal("P2 S1.", chunks[2])
	s.Equal("P2 S2.", chunks[3])
}

func (s *SentenceSplitterTestSuite) TestSplitText_RegexFallback() {
    text := "a,b c,d"
    splitter := NewSentenceSplitter(1, 0, nil, nil)
    chunks := splitter.SplitText(text)
    
    s.Len(chunks, 4)
    s.Equal("a,", chunks[0])
    s.Equal("b", chunks[1]) // trimmed
    s.Equal("c,", chunks[2])
    s.Equal("d", chunks[3])
}

func (s *SentenceSplitterTestSuite) TestTikTokenIntegration() {
	tokenizer, err := NewTikTokenTokenizer("gpt-3.5-turbo")
	if err != nil {
		s.T().Skip("Skipping TikToken test due to initialization error (network?): ", err)
		return
	}
	
	splitter := NewSentenceSplitter(10, 0, tokenizer, nil)
	text := "Hello world with tiktoken"
	chunks := splitter.SplitText(text)
	s.Len(chunks, 1)
	s.Equal("Hello world with tiktoken", chunks[0])
}

func (s *SentenceSplitterTestSuite) TestSplitTextKeepSeparator_EdgeCases() {
	// Empty string
	res := SplitTextKeepSeparator("", " ")
	s.Empty(res)

	// No separator in text
	res = SplitTextKeepSeparator("hello", " ")
	s.Len(res, 1)
	s.Equal("hello", res[0])

	// Empty separator (should avoid infinite loop, maybe return whole text or char split?)
	// Our impl returns whole text if empty sep to be safe, or empty list if empty text
	res = SplitTextKeepSeparator("hello", "")
	s.Len(res, 1)
	s.Equal("hello", res[0])

	res = SplitTextKeepSeparator("", "")
	s.Empty(res)
}

func (s *SentenceSplitterTestSuite) TestTikTokenTokenizer_Error() {
	// Invalid model name should return error
	_, err := NewTikTokenTokenizer("invalid-model-name-12345")
	s.Error(err)
}

func (s *SentenceSplitterTestSuite) TestNeurosnapSplitterStrategy_Error() {
	// Test invalid JSON data
	_, err := NewNeurosnapSplitterStrategy([]byte("invalid json"))
	s.Error(err)
	
	// Test valid but empty/minimal JSON that fits structure
	// We need a minimal valid sentences.Storage JSON structure
	// Storage struct has: AbbrevTypes, Collocations, SentStarters, OrthoContext
	// All are SetString (map[string]int)
	minimalJSON := `{"AbbrevTypes":{}, "Collocations":{}, "SentStarters":{}, "OrthoContext":{}}`
	strategy, err := NewNeurosnapSplitterStrategy([]byte(minimalJSON))
	s.NoError(err)
	s.NotNil(strategy)
	
	// Test splitting with this minimal strategy
	text := "Hello world. This is a test."
	// With no training, it might split weirdly or just work on basic punctuation if the library has defaults
	// Actually neurosnap/sentences relies heavily on the training data.
	chunks := strategy.Split(text)
	s.NotEmpty(chunks)
}
