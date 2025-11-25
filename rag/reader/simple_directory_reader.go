package reader

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/aqua777/go-llamaindex/schema"
)

// SimpleDirectoryReader reads files from a directory.
type SimpleDirectoryReader struct {
	inputDir   string
	extensions []string // e.g. ".txt", ".md"
}

// NewSimpleDirectoryReader creates a new SimpleDirectoryReader.
func NewSimpleDirectoryReader(inputDir string, extensions ...string) *SimpleDirectoryReader {
	if len(extensions) == 0 {
		// Default to txt and md
		extensions = []string{".txt", ".md"}
	}
	return &SimpleDirectoryReader{
		inputDir:   inputDir,
		extensions: extensions,
	}
}

// LoadData reads files and returns a slice of Documents (Nodes with type Document).
func (r *SimpleDirectoryReader) LoadData() ([]schema.Node, error) {
	var docs []schema.Node

	err := filepath.WalkDir(r.inputDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if d.IsDir() {
			return nil
		}

		ext := strings.ToLower(filepath.Ext(path))
		match := false
		for _, e := range r.extensions {
			if ext == e {
				match = true
				break
			}
		}

		if !match {
			return nil
		}

		content, err := os.ReadFile(path)
		if err != nil {
			return fmt.Errorf("failed to read file %s: %w", path, err)
		}

		doc := schema.Node{
			ID:   path, // Use file path as ID for now
			Text: string(content),
			Type: schema.ObjectTypeDocument,
			Metadata: map[string]interface{}{
				"filename": d.Name(),
				"path":     path,
				"ext":      ext,
			},
		}
		docs = append(docs, doc)
		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to walk directory %s: %w", r.inputDir, err)
	}

	return docs, nil
}
