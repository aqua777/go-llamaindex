package textsplitter

import (
	"fmt"
	"sync"

	"github.com/pkoukk/tiktoken-go"
)

// Common encoding names
const (
	EncodingCL100kBase = "cl100k_base" // GPT-4, GPT-3.5-turbo, text-embedding-ada-002
	EncodingP50kBase   = "p50k_base"   // Codex models, text-davinci-002/003
	EncodingR50kBase   = "r50k_base"   // GPT-3 models like davinci
	EncodingO200kBase  = "o200k_base"  // GPT-4o models
)

// Model to encoding mapping
var modelEncodingMap = map[string]string{
	// GPT-4o models
	"gpt-4o":            EncodingO200kBase,
	"gpt-4o-mini":       EncodingO200kBase,
	"gpt-4o-2024-05-13": EncodingO200kBase,
	// GPT-4 models
	"gpt-4":                EncodingCL100kBase,
	"gpt-4-32k":            EncodingCL100kBase,
	"gpt-4-turbo":          EncodingCL100kBase,
	"gpt-4-turbo-preview":  EncodingCL100kBase,
	"gpt-4-0125-preview":   EncodingCL100kBase,
	"gpt-4-1106-preview":   EncodingCL100kBase,
	"gpt-4-vision-preview": EncodingCL100kBase,
	// GPT-3.5 models
	"gpt-3.5-turbo":          EncodingCL100kBase,
	"gpt-3.5-turbo-16k":      EncodingCL100kBase,
	"gpt-3.5-turbo-instruct": EncodingCL100kBase,
	"gpt-35-turbo":           EncodingCL100kBase, // Azure naming
	// Embedding models
	"text-embedding-ada-002": EncodingCL100kBase,
	"text-embedding-3-small": EncodingCL100kBase,
	"text-embedding-3-large": EncodingCL100kBase,
}

// GetEncodingForModel returns the encoding name for a given model.
// Returns cl100k_base as default if model is not found.
func GetEncodingForModel(model string) string {
	if enc, ok := modelEncodingMap[model]; ok {
		return enc
	}
	return EncodingCL100kBase // Default to cl100k_base
}

// TikTokenTokenizerByEncoding creates a tokenizer using a specific encoding name.
type TikTokenTokenizerByEncoding struct {
	encoding     *tiktoken.Tiktoken
	encodingName string
}

// NewTikTokenTokenizerByEncoding creates a tokenizer using a specific encoding.
func NewTikTokenTokenizerByEncoding(encodingName string) (*TikTokenTokenizerByEncoding, error) {
	if encodingName == "" {
		encodingName = EncodingCL100kBase
	}
	enc, err := tiktoken.GetEncoding(encodingName)
	if err != nil {
		return nil, fmt.Errorf("failed to get encoding %s: %w", encodingName, err)
	}
	return &TikTokenTokenizerByEncoding{
		encoding:     enc,
		encodingName: encodingName,
	}, nil
}

// Encode tokenizes text and returns token strings.
func (t *TikTokenTokenizerByEncoding) Encode(text string) []string {
	tokenIDs := t.encoding.Encode(text, nil, nil)
	tokens := make([]string, len(tokenIDs))
	for i, id := range tokenIDs {
		tokens[i] = fmt.Sprintf("%d", id)
	}
	return tokens
}

// EncodeToIDs returns the raw token IDs.
func (t *TikTokenTokenizerByEncoding) EncodeToIDs(text string) []int {
	return t.encoding.Encode(text, nil, nil)
}

// Decode converts token IDs back to text.
func (t *TikTokenTokenizerByEncoding) Decode(tokenIDs []int) string {
	return t.encoding.Decode(tokenIDs)
}

// CountTokens returns the number of tokens in the text.
func (t *TikTokenTokenizerByEncoding) CountTokens(text string) int {
	return len(t.encoding.Encode(text, nil, nil))
}

// EncodingName returns the encoding name.
func (t *TikTokenTokenizerByEncoding) EncodingName() string {
	return t.encodingName
}

// Global default tokenizer (lazy initialized)
var (
	defaultTokenizer     Tokenizer
	defaultTokenizerOnce sync.Once
	defaultTokenizerErr  error
)

// DefaultTokenizer returns a shared default TikToken tokenizer using cl100k_base encoding.
// This is safe for concurrent use.
func DefaultTokenizer() (Tokenizer, error) {
	defaultTokenizerOnce.Do(func() {
		defaultTokenizer, defaultTokenizerErr = NewTikTokenTokenizerByEncoding(EncodingCL100kBase)
	})
	return defaultTokenizer, defaultTokenizerErr
}

// MustDefaultTokenizer returns the default tokenizer or panics on error.
func MustDefaultTokenizer() Tokenizer {
	tok, err := DefaultTokenizer()
	if err != nil {
		panic(fmt.Sprintf("failed to create default tokenizer: %v", err))
	}
	return tok
}

// TokenCounter is an interface for counting tokens.
type TokenCounter interface {
	CountTokens(text string) int
}

// Ensure TikTokenTokenizerByEncoding implements TokenCounter
var _ TokenCounter = (*TikTokenTokenizerByEncoding)(nil)

// CountTokens counts tokens using the TikTokenTokenizer.
func (t *TikTokenTokenizer) CountTokens(text string) int {
	return len(t.encoding.Encode(text, nil, nil))
}

// Ensure TikTokenTokenizer implements TokenCounter
var _ TokenCounter = (*TikTokenTokenizer)(nil)
