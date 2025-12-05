package reader

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/aqua777/go-llamaindex/schema"
)

// JSONReader reads JSON files and converts them to documents.
type JSONReader struct {
	// InputFiles is a list of JSON file paths to read
	InputFiles []string
	// InputDir is a directory containing JSON files
	InputDir string
	// Recursive determines if subdirectories should be searched
	Recursive bool
	// TextContentKey is the JSON key to use as document text content.
	// If empty, the entire JSON is serialized as text.
	TextContentKey string
	// MetadataKeys are JSON keys to extract as document metadata.
	// If empty, all non-text keys are used as metadata.
	MetadataKeys []string
	// IsJSONL indicates if files are JSON Lines format (one JSON object per line)
	IsJSONL bool
}

// NewJSONReader creates a new JSONReader for specific files.
func NewJSONReader(inputFiles ...string) *JSONReader {
	return &JSONReader{
		InputFiles: inputFiles,
		Recursive:  false,
	}
}

// NewJSONReaderFromDir creates a new JSONReader for a directory.
func NewJSONReaderFromDir(inputDir string, recursive bool) *JSONReader {
	return &JSONReader{
		InputDir:  inputDir,
		Recursive: recursive,
	}
}

// WithTextContentKey sets the key to use for document text content.
func (r *JSONReader) WithTextContentKey(key string) *JSONReader {
	r.TextContentKey = key
	return r
}

// WithMetadataKeys sets the keys to extract as metadata.
func (r *JSONReader) WithMetadataKeys(keys ...string) *JSONReader {
	r.MetadataKeys = keys
	return r
}

// WithJSONL enables JSON Lines format parsing.
func (r *JSONReader) WithJSONL(isJSONL bool) *JSONReader {
	r.IsJSONL = isJSONL
	return r
}

// LoadData loads JSON files and returns documents.
func (r *JSONReader) LoadData() ([]schema.Node, error) {
	files, err := r.getFiles()
	if err != nil {
		return nil, err
	}

	var docs []schema.Node
	for _, file := range files {
		fileDocs, err := r.loadFile(file)
		if err != nil {
			return nil, NewReaderError(file, "failed to load JSON file", err)
		}
		docs = append(docs, fileDocs...)
	}

	return docs, nil
}

// LoadFromFile loads a single JSON file.
func (r *JSONReader) LoadFromFile(filePath string) ([]schema.Node, error) {
	return r.loadFile(filePath)
}

// Metadata returns reader metadata.
func (r *JSONReader) Metadata() ReaderMetadata {
	return ReaderMetadata{
		Name:                "JSONReader",
		SupportedExtensions: []string{".json", ".jsonl"},
		Description:         "Reads JSON and JSON Lines files",
	}
}

func (r *JSONReader) getFiles() ([]string, error) {
	if len(r.InputFiles) > 0 {
		return r.InputFiles, nil
	}

	if r.InputDir == "" {
		return nil, fmt.Errorf("no input files or directory specified")
	}

	var files []string
	walkFn := func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			if !r.Recursive && path != r.InputDir {
				return filepath.SkipDir
			}
			return nil
		}
		ext := strings.ToLower(filepath.Ext(path))
		if ext == ".json" || ext == ".jsonl" {
			files = append(files, path)
		}
		return nil
	}

	if err := filepath.Walk(r.InputDir, walkFn); err != nil {
		return nil, fmt.Errorf("failed to walk directory %s: %w", r.InputDir, err)
	}

	return files, nil
}

func (r *JSONReader) loadFile(filePath string) ([]schema.Node, error) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	ext := strings.ToLower(filepath.Ext(filePath))
	isJSONL := r.IsJSONL || ext == ".jsonl"

	if isJSONL {
		return r.parseJSONL(content, filePath)
	}
	return r.parseJSON(content, filePath)
}

func (r *JSONReader) parseJSON(content []byte, filePath string) ([]schema.Node, error) {
	var data interface{}
	if err := json.Unmarshal(content, &data); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}

	// Handle array of objects
	if arr, ok := data.([]interface{}); ok {
		var docs []schema.Node
		for i, item := range arr {
			doc, err := r.createDocument(item, filePath, i)
			if err != nil {
				return nil, err
			}
			docs = append(docs, doc)
		}
		return docs, nil
	}

	// Handle single object
	doc, err := r.createDocument(data, filePath, 0)
	if err != nil {
		return nil, err
	}
	return []schema.Node{doc}, nil
}

func (r *JSONReader) parseJSONL(content []byte, filePath string) ([]schema.Node, error) {
	lines := strings.Split(string(content), "\n")
	var docs []schema.Node

	for i, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		var data interface{}
		if err := json.Unmarshal([]byte(line), &data); err != nil {
			return nil, fmt.Errorf("failed to parse JSON at line %d: %w", i+1, err)
		}

		doc, err := r.createDocument(data, filePath, i)
		if err != nil {
			return nil, err
		}
		docs = append(docs, doc)
	}

	return docs, nil
}

func (r *JSONReader) createDocument(data interface{}, filePath string, index int) (schema.Node, error) {
	var text string
	metadata := make(map[string]interface{})

	// Add file metadata
	metadata["source"] = filePath
	metadata["filename"] = filepath.Base(filePath)
	if index > 0 {
		metadata["index"] = index
	}

	// Handle map data
	if m, ok := data.(map[string]interface{}); ok {
		// Extract text content
		if r.TextContentKey != "" {
			if val, exists := m[r.TextContentKey]; exists {
				text = fmt.Sprintf("%v", val)
			} else {
				// Key not found, serialize entire object
				jsonBytes, _ := json.Marshal(data)
				text = string(jsonBytes)
			}
		} else {
			// No specific key, serialize entire object
			jsonBytes, _ := json.Marshal(data)
			text = string(jsonBytes)
		}

		// Extract metadata
		if len(r.MetadataKeys) > 0 {
			for _, key := range r.MetadataKeys {
				if val, exists := m[key]; exists {
					metadata[key] = val
				}
			}
		} else {
			// Use all keys except text content key as metadata
			for key, val := range m {
				if key != r.TextContentKey {
					metadata[key] = val
				}
			}
		}
	} else {
		// Not a map, serialize as text
		jsonBytes, _ := json.Marshal(data)
		text = string(jsonBytes)
	}

	doc := schema.Node{
		ID:       fmt.Sprintf("%s_%d", filePath, index),
		Text:     text,
		Type:     schema.ObjectTypeDocument,
		Metadata: metadata,
	}

	return doc, nil
}
