package textsplitter

import (
	"testing"
)

func TestSentenceSplitterValidation(t *testing.T) {
	t.Run("NewSentenceSplitterWithValidation valid", func(t *testing.T) {
		splitter, err := NewSentenceSplitterWithValidation(1024, 200, nil, nil)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if splitter == nil {
			t.Error("expected non-nil splitter")
		}
	})

	t.Run("NewSentenceSplitterWithValidation invalid chunk size", func(t *testing.T) {
		_, err := NewSentenceSplitterWithValidation(0, 200, nil, nil)
		if err == nil {
			t.Error("expected error for zero chunk size")
		}
	})

	t.Run("NewSentenceSplitterWithValidation overlap too large", func(t *testing.T) {
		_, err := NewSentenceSplitterWithValidation(100, 200, nil, nil)
		if err == nil {
			t.Error("expected error for overlap > chunk size")
		}
	})

	t.Run("Validate method", func(t *testing.T) {
		splitter := NewSentenceSplitter(1024, 200, nil, nil)
		if err := splitter.Validate(); err != nil {
			t.Errorf("unexpected validation error: %v", err)
		}

		// Manually set invalid values
		splitter.ChunkSize = 0
		if err := splitter.Validate(); err == nil {
			t.Error("expected validation error for zero chunk size")
		}
	})
}

func TestTokenSplitterValidation(t *testing.T) {
	t.Run("NewTokenTextSplitterWithValidation valid", func(t *testing.T) {
		splitter, err := NewTokenTextSplitterWithValidation(1024, 200, nil)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if splitter == nil {
			t.Error("expected non-nil splitter")
		}
	})

	t.Run("NewTokenTextSplitterWithValidation invalid chunk size", func(t *testing.T) {
		_, err := NewTokenTextSplitterWithValidation(-1, 200, nil)
		if err == nil {
			t.Error("expected error for negative chunk size")
		}
	})

	t.Run("NewTokenTextSplitterWithValidation overlap too large", func(t *testing.T) {
		_, err := NewTokenTextSplitterWithValidation(100, 150, nil)
		if err == nil {
			t.Error("expected error for overlap > chunk size")
		}
	})

	t.Run("Validate method", func(t *testing.T) {
		splitter := NewTokenTextSplitter(1024, 200)
		if err := splitter.Validate(); err != nil {
			t.Errorf("unexpected validation error: %v", err)
		}

		// Manually set invalid values
		splitter.ChunkOverlap = -1
		if err := splitter.Validate(); err == nil {
			t.Error("expected validation error for negative overlap")
		}
	})
}

func TestTikTokenTokenizer(t *testing.T) {
	t.Run("NewTikTokenTokenizerByEncoding cl100k_base", func(t *testing.T) {
		tok, err := NewTikTokenTokenizerByEncoding(EncodingCL100kBase)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		text := "Hello, world! This is a test."
		tokens := tok.Encode(text)
		if len(tokens) == 0 {
			t.Error("expected non-empty tokens")
		}

		count := tok.CountTokens(text)
		if count != len(tokens) {
			t.Errorf("CountTokens() = %d, want %d", count, len(tokens))
		}
	})

	t.Run("NewTikTokenTokenizerByEncoding default", func(t *testing.T) {
		tok, err := NewTikTokenTokenizerByEncoding("")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if tok.EncodingName() != EncodingCL100kBase {
			t.Errorf("expected default encoding %s, got %s", EncodingCL100kBase, tok.EncodingName())
		}
	})

	t.Run("GetEncodingForModel", func(t *testing.T) {
		tests := []struct {
			model    string
			expected string
		}{
			{"gpt-4", EncodingCL100kBase},
			{"gpt-4o", EncodingO200kBase},
			{"gpt-3.5-turbo", EncodingCL100kBase},
			{"unknown-model", EncodingCL100kBase}, // default
		}

		for _, tt := range tests {
			enc := GetEncodingForModel(tt.model)
			if enc != tt.expected {
				t.Errorf("GetEncodingForModel(%s) = %s, want %s", tt.model, enc, tt.expected)
			}
		}
	})

	t.Run("DefaultTokenizer", func(t *testing.T) {
		tok, err := DefaultTokenizer()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if tok == nil {
			t.Error("expected non-nil tokenizer")
		}

		// Should return same instance
		tok2, _ := DefaultTokenizer()
		if tok != tok2 {
			t.Error("expected same tokenizer instance")
		}
	})

	t.Run("MustDefaultTokenizer", func(t *testing.T) {
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("unexpected panic: %v", r)
			}
		}()

		tok := MustDefaultTokenizer()
		if tok == nil {
			t.Error("expected non-nil tokenizer")
		}
	})

	t.Run("EncodeToIDs and Decode", func(t *testing.T) {
		tok, err := NewTikTokenTokenizerByEncoding(EncodingCL100kBase)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		text := "Hello world"
		ids := tok.EncodeToIDs(text)
		if len(ids) == 0 {
			t.Error("expected non-empty token IDs")
		}

		decoded := tok.Decode(ids)
		if decoded != text {
			t.Errorf("Decode() = %q, want %q", decoded, text)
		}
	})
}
