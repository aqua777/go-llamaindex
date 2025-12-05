// Package reader provides document loading functionality for go-llamaindex.
package reader

import (
	"context"

	"github.com/aqua777/go-llamaindex/schema"
)

// Reader is the interface for document loaders.
// Implementations should load documents from various sources (files, URLs, etc.)
type Reader interface {
	// LoadData loads documents and returns them as a slice.
	LoadData() ([]schema.Node, error)
}

// ReaderWithContext is a Reader that supports context for cancellation.
type ReaderWithContext interface {
	Reader
	// LoadDataWithContext loads documents with context support.
	LoadDataWithContext(ctx context.Context) ([]schema.Node, error)
}

// LazyReader is a Reader that can load documents lazily via a channel.
type LazyReader interface {
	Reader
	// LazyLoadData returns a channel that yields documents one at a time.
	// The channel is closed when all documents have been loaded or an error occurs.
	LazyLoadData() (<-chan schema.Node, <-chan error)
}

// FileReader is a Reader that loads from file paths.
type FileReader interface {
	Reader
	// LoadFromFile loads a document from a specific file path.
	LoadFromFile(filePath string) ([]schema.Node, error)
}

// ReaderMetadata contains metadata about a reader.
type ReaderMetadata struct {
	// Name is the reader name (e.g., "JSONReader", "PDFReader")
	Name string
	// SupportedExtensions lists file extensions this reader supports
	SupportedExtensions []string
	// Description describes what this reader does
	Description string
}

// ReaderWithMetadata is a Reader that provides metadata about itself.
type ReaderWithMetadata interface {
	Reader
	// Metadata returns information about this reader.
	Metadata() ReaderMetadata
}

// ReaderOptions contains common options for readers.
type ReaderOptions struct {
	// Recursive determines if directory readers should recurse into subdirectories
	Recursive bool
	// FileExtensions filters which file extensions to process
	FileExtensions []string
	// ExcludePatterns are glob patterns for files/dirs to exclude
	ExcludePatterns []string
	// IncludeHidden determines if hidden files should be included
	IncludeHidden bool
	// NumWorkers is the number of concurrent workers for parallel loading
	NumWorkers int
	// ExtraMetadata is additional metadata to add to all loaded documents
	ExtraMetadata map[string]interface{}
}

// DefaultReaderOptions returns default reader options.
func DefaultReaderOptions() ReaderOptions {
	return ReaderOptions{
		Recursive:      true,
		FileExtensions: nil, // nil means all extensions
		ExcludePatterns: []string{},
		IncludeHidden:  false,
		NumWorkers:     1,
		ExtraMetadata:  nil,
	}
}

// ReaderError represents an error during document loading.
type ReaderError struct {
	Source  string // File path or URL that caused the error
	Message string
	Err     error
}

func (e *ReaderError) Error() string {
	if e.Err != nil {
		return e.Source + ": " + e.Message + ": " + e.Err.Error()
	}
	return e.Source + ": " + e.Message
}

func (e *ReaderError) Unwrap() error {
	return e.Err
}

// NewReaderError creates a new ReaderError.
func NewReaderError(source, message string, err error) *ReaderError {
	return &ReaderError{
		Source:  source,
		Message: message,
		Err:     err,
	}
}
