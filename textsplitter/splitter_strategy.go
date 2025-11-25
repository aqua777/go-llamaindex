package textsplitter

import (
	_ "embed"
	"fmt"
	"os"

	"github.com/neurosnap/sentences"
)

//go:embed english.json
var englishTrainingData []byte

// RegexSplitterStrategy uses regex for sentence splitting.
type RegexSplitterStrategy struct {
	regexStr string
}

func NewRegexSplitterStrategy(regexStr string) *RegexSplitterStrategy {
	if regexStr == "" {
		regexStr = DefaultChunkingRegex
	}
	return &RegexSplitterStrategy{regexStr: regexStr}
}

func (s *RegexSplitterStrategy) Split(text string) []string {
	return SplitByRegex(s.regexStr)(text)
}

// NeurosnapSplitterStrategy uses neurosnap/sentences for sentence splitting.
type NeurosnapSplitterStrategy struct {
	tokenizer *sentences.DefaultSentenceTokenizer
}

// NewNeurosnapSplitterStrategy creates a new strategy using the provided JSON training data.
// If trainingData is nil or empty, it defaults to the embedded english.json training data.
func NewNeurosnapSplitterStrategy(trainingData []byte) (*NeurosnapSplitterStrategy, error) {
	if len(trainingData) == 0 {
		trainingData = englishTrainingData
	}

	// In v1.1.2, training data is loaded into Storage struct via LoadTraining helper
	storage, err := sentences.LoadTraining(trainingData)
	if err != nil {
		return nil, fmt.Errorf("failed to load training data: %w", err)
	}

	tokenizer := sentences.NewSentenceTokenizer(storage)
	return &NeurosnapSplitterStrategy{tokenizer: tokenizer}, nil
}

// NewNeurosnapSplitterStrategyFromFile creates a new strategy by reading training data from a file.
func NewNeurosnapSplitterStrategyFromFile(path string) (*NeurosnapSplitterStrategy, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read training data from %s: %w", path, err)
	}
	return NewNeurosnapSplitterStrategy(b)
}

func (s *NeurosnapSplitterStrategy) Split(text string) []string {
	sentences := s.tokenizer.Tokenize(text)
	result := make([]string, len(sentences))
	for i, sent := range sentences {
		result[i] = sent.Text
	}
	return result
}
