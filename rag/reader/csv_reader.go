package reader

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/aqua777/go-llamaindex/schema"
)

// CSVReader reads CSV files and converts them to documents.
type CSVReader struct {
	// InputFiles is a list of CSV file paths to read
	InputFiles []string
	// InputDir is a directory containing CSV files
	InputDir string
	// Recursive determines if subdirectories should be searched
	Recursive bool
	// Delimiter is the field delimiter (default: comma)
	Delimiter rune
	// HasHeader indicates if the first row is a header row
	HasHeader bool
	// TextColumns are column names or indices to use as document text.
	// If empty, all columns are concatenated as text.
	TextColumns []string
	// MetadataColumns are column names or indices to extract as metadata.
	// If empty, all non-text columns are used as metadata.
	MetadataColumns []string
	// ConcatRows determines if all rows should be concatenated into a single document.
	// If false (default), each row becomes a separate document.
	ConcatRows bool
	// RowSeparator is used when ConcatRows is true (default: newline)
	RowSeparator string
}

// NewCSVReader creates a new CSVReader for specific files.
func NewCSVReader(inputFiles ...string) *CSVReader {
	return &CSVReader{
		InputFiles:   inputFiles,
		Delimiter:    ',',
		HasHeader:    true,
		RowSeparator: "\n",
	}
}

// NewCSVReaderFromDir creates a new CSVReader for a directory.
func NewCSVReaderFromDir(inputDir string, recursive bool) *CSVReader {
	return &CSVReader{
		InputDir:     inputDir,
		Recursive:    recursive,
		Delimiter:    ',',
		HasHeader:    true,
		RowSeparator: "\n",
	}
}

// WithDelimiter sets the field delimiter.
func (r *CSVReader) WithDelimiter(delimiter rune) *CSVReader {
	r.Delimiter = delimiter
	return r
}

// WithHeader sets whether the first row is a header.
func (r *CSVReader) WithHeader(hasHeader bool) *CSVReader {
	r.HasHeader = hasHeader
	return r
}

// WithTextColumns sets which columns to use as document text.
func (r *CSVReader) WithTextColumns(columns ...string) *CSVReader {
	r.TextColumns = columns
	return r
}

// WithMetadataColumns sets which columns to extract as metadata.
func (r *CSVReader) WithMetadataColumns(columns ...string) *CSVReader {
	r.MetadataColumns = columns
	return r
}

// WithConcatRows sets whether to concatenate all rows into a single document.
func (r *CSVReader) WithConcatRows(concat bool) *CSVReader {
	r.ConcatRows = concat
	return r
}

// WithRowSeparator sets the separator used when concatenating rows.
func (r *CSVReader) WithRowSeparator(sep string) *CSVReader {
	r.RowSeparator = sep
	return r
}

// LoadData loads CSV files and returns documents.
func (r *CSVReader) LoadData() ([]schema.Node, error) {
	files, err := r.getFiles()
	if err != nil {
		return nil, err
	}

	var docs []schema.Node
	for _, file := range files {
		fileDocs, err := r.loadFile(file)
		if err != nil {
			return nil, NewReaderError(file, "failed to load CSV file", err)
		}
		docs = append(docs, fileDocs...)
	}

	return docs, nil
}

// LoadFromFile loads a single CSV file.
func (r *CSVReader) LoadFromFile(filePath string) ([]schema.Node, error) {
	return r.loadFile(filePath)
}

// Metadata returns reader metadata.
func (r *CSVReader) Metadata() ReaderMetadata {
	return ReaderMetadata{
		Name:                "CSVReader",
		SupportedExtensions: []string{".csv", ".tsv"},
		Description:         "Reads CSV and TSV files",
	}
}

func (r *CSVReader) getFiles() ([]string, error) {
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
		if ext == ".csv" || ext == ".tsv" {
			files = append(files, path)
		}
		return nil
	}

	if err := filepath.Walk(r.InputDir, walkFn); err != nil {
		return nil, fmt.Errorf("failed to walk directory %s: %w", r.InputDir, err)
	}

	return files, nil
}

func (r *CSVReader) loadFile(filePath string) ([]schema.Node, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	// Detect delimiter from extension if not explicitly set
	delimiter := r.Delimiter
	if delimiter == ',' && strings.ToLower(filepath.Ext(filePath)) == ".tsv" {
		delimiter = '\t'
	}

	reader := csv.NewReader(file)
	reader.Comma = delimiter
	reader.LazyQuotes = true
	reader.TrimLeadingSpace = true

	// Read all records
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("failed to parse CSV: %w", err)
	}

	if len(records) == 0 {
		return nil, nil
	}

	// Determine headers
	var headers []string
	startRow := 0
	if r.HasHeader {
		headers = records[0]
		startRow = 1
	} else {
		// Generate column indices as headers
		if len(records) > 0 {
			headers = make([]string, len(records[0]))
			for i := range headers {
				headers[i] = fmt.Sprintf("col_%d", i)
			}
		}
	}

	// Determine text and metadata column indices
	textIndices := r.getColumnIndices(headers, r.TextColumns)
	metadataIndices := r.getColumnIndices(headers, r.MetadataColumns)

	// If no text columns specified, use all columns
	if len(textIndices) == 0 {
		textIndices = make([]int, len(headers))
		for i := range headers {
			textIndices[i] = i
		}
	}

	// If no metadata columns specified and text columns are specified,
	// use remaining columns as metadata
	if len(metadataIndices) == 0 && len(r.TextColumns) > 0 {
		textSet := make(map[int]bool)
		for _, idx := range textIndices {
			textSet[idx] = true
		}
		for i := range headers {
			if !textSet[i] {
				metadataIndices = append(metadataIndices, i)
			}
		}
	}

	if r.ConcatRows {
		return r.createConcatenatedDocument(records[startRow:], headers, textIndices, metadataIndices, filePath)
	}

	return r.createRowDocuments(records[startRow:], headers, textIndices, metadataIndices, filePath)
}

func (r *CSVReader) getColumnIndices(headers []string, columns []string) []int {
	if len(columns) == 0 {
		return nil
	}

	headerMap := make(map[string]int)
	for i, h := range headers {
		headerMap[strings.ToLower(strings.TrimSpace(h))] = i
	}

	var indices []int
	for _, col := range columns {
		col = strings.TrimSpace(col)
		// Try as column name first
		if idx, ok := headerMap[strings.ToLower(col)]; ok {
			indices = append(indices, idx)
			continue
		}
		// Try as numeric index
		var idx int
		if _, err := fmt.Sscanf(col, "%d", &idx); err == nil && idx >= 0 && idx < len(headers) {
			indices = append(indices, idx)
		}
	}

	return indices
}

func (r *CSVReader) createRowDocuments(records [][]string, headers []string, textIndices, metadataIndices []int, filePath string) ([]schema.Node, error) {
	var docs []schema.Node

	for rowIdx, record := range records {
		// Build text from text columns
		var textParts []string
		for _, idx := range textIndices {
			if idx < len(record) {
				val := strings.TrimSpace(record[idx])
				if val != "" {
					if len(textIndices) > 1 && idx < len(headers) {
						textParts = append(textParts, fmt.Sprintf("%s: %s", headers[idx], val))
					} else {
						textParts = append(textParts, val)
					}
				}
			}
		}
		text := strings.Join(textParts, "\n")

		// Build metadata
		metadata := make(map[string]interface{})
		metadata["source"] = filePath
		metadata["filename"] = filepath.Base(filePath)
		metadata["row"] = rowIdx + 1

		for _, idx := range metadataIndices {
			if idx < len(record) && idx < len(headers) {
				val := strings.TrimSpace(record[idx])
				if val != "" {
					metadata[headers[idx]] = val
				}
			}
		}

		doc := schema.Node{
			ID:       fmt.Sprintf("%s_row_%d", filePath, rowIdx),
			Text:     text,
			Type:     schema.ObjectTypeDocument,
			Metadata: metadata,
		}
		docs = append(docs, doc)
	}

	return docs, nil
}

func (r *CSVReader) createConcatenatedDocument(records [][]string, headers []string, textIndices, metadataIndices []int, filePath string) ([]schema.Node, error) {
	var allRows []string

	for _, record := range records {
		var textParts []string
		for _, idx := range textIndices {
			if idx < len(record) {
				val := strings.TrimSpace(record[idx])
				if val != "" {
					if len(textIndices) > 1 && idx < len(headers) {
						textParts = append(textParts, fmt.Sprintf("%s: %s", headers[idx], val))
					} else {
						textParts = append(textParts, val)
					}
				}
			}
		}
		if len(textParts) > 0 {
			allRows = append(allRows, strings.Join(textParts, " | "))
		}
	}

	text := strings.Join(allRows, r.RowSeparator)

	metadata := make(map[string]interface{})
	metadata["source"] = filePath
	metadata["filename"] = filepath.Base(filePath)
	metadata["total_rows"] = len(records)
	metadata["headers"] = headers

	doc := schema.Node{
		ID:       filePath,
		Text:     text,
		Type:     schema.ObjectTypeDocument,
		Metadata: metadata,
	}

	return []schema.Node{doc}, nil
}

// CSVStreamReader provides streaming CSV reading for large files.
type CSVStreamReader struct {
	*CSVReader
	file   *os.File
	reader *csv.Reader
	headers []string
	rowIdx  int
}

// NewCSVStreamReader creates a streaming CSV reader.
func NewCSVStreamReader(filePath string) (*CSVStreamReader, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}

	csvReader := &CSVReader{
		InputFiles:   []string{filePath},
		Delimiter:    ',',
		HasHeader:    true,
		RowSeparator: "\n",
	}

	// Detect delimiter from extension
	delimiter := csvReader.Delimiter
	if strings.ToLower(filepath.Ext(filePath)) == ".tsv" {
		delimiter = '\t'
	}

	reader := csv.NewReader(file)
	reader.Comma = delimiter
	reader.LazyQuotes = true
	reader.TrimLeadingSpace = true

	return &CSVStreamReader{
		CSVReader: csvReader,
		file:      file,
		reader:    reader,
		rowIdx:    0,
	}, nil
}

// ReadHeaders reads and returns the header row.
func (r *CSVStreamReader) ReadHeaders() ([]string, error) {
	if r.headers != nil {
		return r.headers, nil
	}

	record, err := r.reader.Read()
	if err != nil {
		return nil, fmt.Errorf("failed to read headers: %w", err)
	}

	r.headers = record
	return r.headers, nil
}

// ReadNext reads the next row and returns it as a document.
func (r *CSVStreamReader) ReadNext() (*schema.Node, error) {
	// Ensure headers are read first
	if r.headers == nil && r.HasHeader {
		if _, err := r.ReadHeaders(); err != nil {
			return nil, err
		}
	}

	record, err := r.reader.Read()
	if err == io.EOF {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("failed to read row: %w", err)
	}

	r.rowIdx++

	// Build text from all columns
	var textParts []string
	for i, val := range record {
		val = strings.TrimSpace(val)
		if val != "" {
			if r.headers != nil && i < len(r.headers) {
				textParts = append(textParts, fmt.Sprintf("%s: %s", r.headers[i], val))
			} else {
				textParts = append(textParts, val)
			}
		}
	}
	text := strings.Join(textParts, "\n")

	metadata := make(map[string]interface{})
	metadata["source"] = r.InputFiles[0]
	metadata["filename"] = filepath.Base(r.InputFiles[0])
	metadata["row"] = r.rowIdx

	doc := schema.Node{
		ID:       fmt.Sprintf("%s_row_%d", r.InputFiles[0], r.rowIdx),
		Text:     text,
		Type:     schema.ObjectTypeDocument,
		Metadata: metadata,
	}

	return &doc, nil
}

// Close closes the underlying file.
func (r *CSVStreamReader) Close() error {
	if r.file != nil {
		return r.file.Close()
	}
	return nil
}

// LazyLoadData returns a channel that yields documents one at a time.
func (r *CSVStreamReader) LazyLoadData() (<-chan schema.Node, <-chan error) {
	docChan := make(chan schema.Node)
	errChan := make(chan error, 1)

	go func() {
		defer close(docChan)
		defer close(errChan)
		defer r.Close()

		for {
			doc, err := r.ReadNext()
			if err != nil {
				errChan <- err
				return
			}
			if doc == nil {
				return // EOF
			}
			docChan <- *doc
		}
	}()

	return docChan, errChan
}
