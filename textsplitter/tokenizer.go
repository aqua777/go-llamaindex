package textsplitter

import (
	"fmt"
	"strings"

	"github.com/pkoukk/tiktoken-go"
)

// SimpleTokenizer tokenizes text by splitting on whitespace.
type SimpleTokenizer struct{}

func NewSimpleTokenizer() *SimpleTokenizer {
	return &SimpleTokenizer{}
}

func (t *SimpleTokenizer) Encode(text string) []string {
	return strings.Fields(text)
}

// TikTokenTokenizer tokenizes text using OpenAI's tiktoken.
type TikTokenTokenizer struct {
	encoding *tiktoken.Tiktoken
}

func NewTikTokenTokenizer(model string) (*TikTokenTokenizer, error) {
	if model == "" {
		model = "gpt-3.5-turbo" // Default
	}
	// Use EncodingForModel to get the correct encoding for the model
	tkm, err := tiktoken.EncodingForModel(model)
	if err != nil {
		return nil, fmt.Errorf("failed to get encoding for model %s: %w", model, err)
	}
	return &TikTokenTokenizer{encoding: tkm}, nil
}

func (t *TikTokenTokenizer) Encode(text string) []string {
	// Encode returns []int. We need to map them to strings.
	// Since the interface requires []string (and splitter uses len()), we can return a list of string representations.
	// OR, since we primarily use this for length checking, we can just return dummy strings?
	// The Python code uses `len(self._tokenizer(text))`.
	// If we just return strings, we satisfy the interface.
	// However, simply converting int->string ("123") might be enough if only length matters.
	// BUT if we ever need to see the tokens, we might want real token strings?
	// Tiktoken-go doesn't expose "decode single token" easily without Decode([]int).
	// Let's just return stringified integers for now, as it's efficient enough for length check.
	// Actually, to be safe for future use (e.g. debugging), let's just return string representations of IDs.
	
	tokenIDs := t.encoding.Encode(text, nil, nil)
	tokens := make([]string, len(tokenIDs))
	for i, id := range tokenIDs {
		tokens[i] = fmt.Sprintf("%d", id)
	}
	return tokens
}


