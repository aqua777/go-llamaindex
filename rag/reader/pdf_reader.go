package reader

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/aqua777/go-llamaindex/schema"
	"github.com/ledongthuc/pdf"
)

// PDFReader reads PDF files and converts them to documents.
// It uses the ledongthuc/pdf library for text extraction.
type PDFReader struct {
	// InputFiles is a list of PDF file paths to read
	InputFiles []string
	// InputDir is a directory containing PDF files
	InputDir string
	// Recursive determines if subdirectories should be searched
	Recursive bool
	// SplitByPage creates separate nodes for each page
	SplitByPage bool
	// ExtraMetadata is additional metadata to add to all documents
	ExtraMetadata map[string]interface{}
	// PasswordFunc is a function that returns the password for a PDF file
	// The function receives the file path and should return the password
	PasswordFunc func(filePath string) string
}

// PDFReaderOption configures PDFReader.
type PDFReaderOption func(*PDFReader)

// WithPDFInputFiles sets the input files.
func WithPDFInputFiles(files ...string) PDFReaderOption {
	return func(r *PDFReader) {
		r.InputFiles = files
	}
}

// WithPDFInputDir sets the input directory.
func WithPDFInputDir(dir string) PDFReaderOption {
	return func(r *PDFReader) {
		r.InputDir = dir
	}
}

// WithPDFRecursive enables recursive directory scanning.
func WithPDFRecursive(recursive bool) PDFReaderOption {
	return func(r *PDFReader) {
		r.Recursive = recursive
	}
}

// WithPDFSplitByPage enables splitting by page.
func WithPDFSplitByPage(split bool) PDFReaderOption {
	return func(r *PDFReader) {
		r.SplitByPage = split
	}
}

// WithPDFExtraMetadata sets extra metadata.
func WithPDFExtraMetadata(metadata map[string]interface{}) PDFReaderOption {
	return func(r *PDFReader) {
		r.ExtraMetadata = metadata
	}
}

// WithPDFPasswordFunc sets the password function.
func WithPDFPasswordFunc(fn func(filePath string) string) PDFReaderOption {
	return func(r *PDFReader) {
		r.PasswordFunc = fn
	}
}

// NewPDFReader creates a new PDFReader for specific files.
func NewPDFReader(inputFiles ...string) *PDFReader {
	return &PDFReader{
		InputFiles: inputFiles,
		Recursive:  false,
	}
}

// NewPDFReaderFromDir creates a new PDFReader for a directory.
func NewPDFReaderFromDir(inputDir string, recursive bool) *PDFReader {
	return &PDFReader{
		InputDir:  inputDir,
		Recursive: recursive,
	}
}

// NewPDFReaderWithOptions creates a new PDFReader with options.
func NewPDFReaderWithOptions(opts ...PDFReaderOption) *PDFReader {
	r := &PDFReader{
		Recursive: false,
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

// WithSplitByPage enables splitting by page (fluent API).
func (r *PDFReader) WithSplitByPage(split bool) *PDFReader {
	r.SplitByPage = split
	return r
}

// WithExtraMetadata sets extra metadata (fluent API).
func (r *PDFReader) WithExtraMetadata(metadata map[string]interface{}) *PDFReader {
	r.ExtraMetadata = metadata
	return r
}

// LoadData loads PDF files and returns documents.
func (r *PDFReader) LoadData() ([]schema.Node, error) {
	files, err := r.getFiles()
	if err != nil {
		return nil, err
	}

	var docs []schema.Node
	for _, file := range files {
		fileDocs, err := r.loadFile(file)
		if err != nil {
			return nil, NewReaderError(file, "failed to load PDF file", err)
		}
		docs = append(docs, fileDocs...)
	}

	return docs, nil
}

// LoadDataWithContext loads PDF files with context support.
func (r *PDFReader) LoadDataWithContext(ctx context.Context) ([]schema.Node, error) {
	files, err := r.getFiles()
	if err != nil {
		return nil, err
	}

	var docs []schema.Node
	for _, file := range files {
		select {
		case <-ctx.Done():
			return docs, ctx.Err()
		default:
			fileDocs, err := r.loadFile(file)
			if err != nil {
				return nil, NewReaderError(file, "failed to load PDF file", err)
			}
			docs = append(docs, fileDocs...)
		}
	}

	return docs, nil
}

// LoadFromFile loads a single PDF file.
func (r *PDFReader) LoadFromFile(filePath string) ([]schema.Node, error) {
	return r.loadFile(filePath)
}

// Metadata returns reader metadata.
func (r *PDFReader) Metadata() ReaderMetadata {
	return ReaderMetadata{
		Name:                "PDFReader",
		SupportedExtensions: []string{".pdf"},
		Description:         "Reads PDF files and extracts text content",
	}
}

// getFiles returns all PDF files to process.
func (r *PDFReader) getFiles() ([]string, error) {
	if len(r.InputFiles) > 0 {
		return r.InputFiles, nil
	}

	if r.InputDir == "" {
		return nil, fmt.Errorf("no input files or directory specified")
	}

	var files []string
	err := filepath.Walk(r.InputDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip directories unless we're at the root
		if info.IsDir() {
			if path != r.InputDir && !r.Recursive {
				return filepath.SkipDir
			}
			return nil
		}

		// Check if it's a PDF file
		if strings.ToLower(filepath.Ext(path)) == ".pdf" {
			files = append(files, path)
		}

		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to walk directory: %w", err)
	}

	return files, nil
}

// loadFile loads a single PDF file and returns nodes.
func (r *PDFReader) loadFile(filePath string) ([]schema.Node, error) {
	// Open the PDF file
	f, pdfReader, err := pdf.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open PDF: %w", err)
	}
	defer f.Close()

	numPages := pdfReader.NumPage()
	if numPages == 0 {
		return nil, fmt.Errorf("PDF has no pages")
	}

	// Get absolute path for metadata
	absPath, err := filepath.Abs(filePath)
	if err != nil {
		absPath = filePath
	}

	// Base metadata for all nodes
	baseMetadata := map[string]interface{}{
		"file_path":  absPath,
		"file_name":  filepath.Base(filePath),
		"file_type":  "pdf",
		"total_pages": numPages,
	}

	// Add extra metadata if provided
	if r.ExtraMetadata != nil {
		for k, v := range r.ExtraMetadata {
			baseMetadata[k] = v
		}
	}

	if r.SplitByPage {
		return r.loadByPage(pdfReader, numPages, absPath, baseMetadata)
	}

	return r.loadEntireDocument(pdfReader, numPages, absPath, baseMetadata)
}

// loadByPage loads each page as a separate node.
func (r *PDFReader) loadByPage(pdfReader *pdf.Reader, numPages int, filePath string, baseMetadata map[string]interface{}) ([]schema.Node, error) {
	var nodes []schema.Node

	for pageNum := 1; pageNum <= numPages; pageNum++ {
		page := pdfReader.Page(pageNum)
		if page.V.IsNull() {
			continue
		}

		text, err := page.GetPlainText(nil)
		if err != nil {
			// Try to continue with other pages
			continue
		}

		text = strings.TrimSpace(text)
		if text == "" {
			continue
		}

		// Create metadata for this page
		metadata := make(map[string]interface{})
		for k, v := range baseMetadata {
			metadata[k] = v
		}
		metadata["page_number"] = pageNum

		node := schema.NewTextNode(text)
		node.Metadata = metadata

		nodes = append(nodes, *node)
	}

	if len(nodes) == 0 {
		return nil, fmt.Errorf("no text content found in PDF")
	}

	return nodes, nil
}

// loadEntireDocument loads the entire PDF as a single node.
func (r *PDFReader) loadEntireDocument(pdfReader *pdf.Reader, numPages int, filePath string, baseMetadata map[string]interface{}) ([]schema.Node, error) {
	var textBuilder strings.Builder

	for pageNum := 1; pageNum <= numPages; pageNum++ {
		page := pdfReader.Page(pageNum)
		if page.V.IsNull() {
			continue
		}

		text, err := page.GetPlainText(nil)
		if err != nil {
			// Try to continue with other pages
			continue
		}

		text = strings.TrimSpace(text)
		if text != "" {
			if textBuilder.Len() > 0 {
				textBuilder.WriteString("\n\n")
			}
			textBuilder.WriteString(text)
		}
	}

	fullText := strings.TrimSpace(textBuilder.String())
	if fullText == "" {
		return nil, fmt.Errorf("no text content found in PDF")
	}

	node := schema.NewTextNode(fullText)
	node.Metadata = baseMetadata

	return []schema.Node{*node}, nil
}

// LazyLoadData returns a channel that yields documents one at a time.
func (r *PDFReader) LazyLoadData() (<-chan schema.Node, <-chan error) {
	nodeChan := make(chan schema.Node)
	errChan := make(chan error, 1)

	go func() {
		defer close(nodeChan)
		defer close(errChan)

		files, err := r.getFiles()
		if err != nil {
			errChan <- err
			return
		}

		for _, file := range files {
			nodes, err := r.loadFile(file)
			if err != nil {
				errChan <- NewReaderError(file, "failed to load PDF file", err)
				return
			}

			for _, node := range nodes {
				nodeChan <- node
			}
		}
	}()

	return nodeChan, errChan
}

// ExtractTextFromPDF is a utility function to extract text from a PDF file.
func ExtractTextFromPDF(filePath string) (string, error) {
	reader := NewPDFReader(filePath)
	nodes, err := reader.LoadData()
	if err != nil {
		return "", err
	}

	if len(nodes) == 0 {
		return "", fmt.Errorf("no content extracted from PDF")
	}

	var texts []string
	for _, node := range nodes {
		texts = append(texts, node.GetContent(schema.MetadataModeNone))
	}

	return strings.Join(texts, "\n\n"), nil
}

// ExtractTextFromPDFByPage extracts text from a PDF file, returning text per page.
func ExtractTextFromPDFByPage(filePath string) ([]string, error) {
	reader := NewPDFReader(filePath).WithSplitByPage(true)
	nodes, err := reader.LoadData()
	if err != nil {
		return nil, err
	}

	var pages []string
	for _, node := range nodes {
		pages = append(pages, node.GetContent(schema.MetadataModeNone))
	}

	return pages, nil
}

// GetPDFPageCount returns the number of pages in a PDF file.
func GetPDFPageCount(filePath string) (int, error) {
	f, pdfReader, err := pdf.Open(filePath)
	if err != nil {
		return 0, fmt.Errorf("failed to open PDF: %w", err)
	}
	defer f.Close()

	return pdfReader.NumPage(), nil
}

// GetPDFMetadata extracts metadata from a PDF file.
func GetPDFMetadata(filePath string) (map[string]string, error) {
	f, pdfReader, err := pdf.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open PDF: %w", err)
	}
	defer f.Close()

	metadata := make(map[string]string)

	// Try to get trailer info
	trailer := pdfReader.Trailer()
	if !trailer.IsNull() {
		// Extract common metadata fields if available
		if info := trailer.Key("Info"); !info.IsNull() {
			for _, key := range []string{"Title", "Author", "Subject", "Keywords", "Creator", "Producer", "CreationDate", "ModDate"} {
				if val := info.Key(key); !val.IsNull() {
					if str := val.Text(); str != "" {
						metadata[key] = str
					}
				}
			}
		}
	}

	metadata["PageCount"] = fmt.Sprintf("%d", pdfReader.NumPage())

	return metadata, nil
}

// Ensure PDFReader implements the interfaces.
var _ Reader = (*PDFReader)(nil)
var _ FileReader = (*PDFReader)(nil)
var _ ReaderWithMetadata = (*PDFReader)(nil)
var _ ReaderWithContext = (*PDFReader)(nil)
var _ LazyReader = (*PDFReader)(nil)
