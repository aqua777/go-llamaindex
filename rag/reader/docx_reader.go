package reader

import (
	"archive/zip"
	"bytes"
	"encoding/xml"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/aqua777/go-llamaindex/schema"
)

// DocxReader reads Microsoft Word (.docx) files and converts them to documents.
type DocxReader struct {
	// InputFiles is a list of DOCX file paths to read
	InputFiles []string
	// InputDir is a directory containing DOCX files
	InputDir string
	// Recursive determines if subdirectories should be searched
	Recursive bool
	// ExtractImages determines if images should be extracted as separate nodes
	ExtractImages bool
	// PreserveParagraphs keeps paragraph breaks in the output
	PreserveParagraphs bool
	// ExtractMetadata extracts document properties (author, title, etc.)
	ExtractMetadata bool
	// ExtractTables extracts table content
	ExtractTables bool
}

// NewDocxReader creates a new DocxReader for specific files.
func NewDocxReader(inputFiles ...string) *DocxReader {
	return &DocxReader{
		InputFiles:         inputFiles,
		PreserveParagraphs: true,
		ExtractMetadata:    true,
		ExtractTables:      true,
	}
}

// NewDocxReaderFromDir creates a new DocxReader for a directory.
func NewDocxReaderFromDir(inputDir string, recursive bool) *DocxReader {
	return &DocxReader{
		InputDir:           inputDir,
		Recursive:          recursive,
		PreserveParagraphs: true,
		ExtractMetadata:    true,
		ExtractTables:      true,
	}
}

// WithExtractImages enables image extraction.
func (r *DocxReader) WithExtractImages(extract bool) *DocxReader {
	r.ExtractImages = extract
	return r
}

// WithPreserveParagraphs sets whether to preserve paragraph breaks.
func (r *DocxReader) WithPreserveParagraphs(preserve bool) *DocxReader {
	r.PreserveParagraphs = preserve
	return r
}

// WithExtractMetadata sets whether to extract document properties.
func (r *DocxReader) WithExtractMetadata(extract bool) *DocxReader {
	r.ExtractMetadata = extract
	return r
}

// WithExtractTables sets whether to extract table content.
func (r *DocxReader) WithExtractTables(extract bool) *DocxReader {
	r.ExtractTables = extract
	return r
}

// LoadData loads DOCX files and returns documents.
func (r *DocxReader) LoadData() ([]schema.Node, error) {
	files, err := r.getFiles()
	if err != nil {
		return nil, err
	}

	var docs []schema.Node
	for _, file := range files {
		fileDocs, err := r.loadFile(file)
		if err != nil {
			return nil, NewReaderError(file, "failed to load DOCX file", err)
		}
		docs = append(docs, fileDocs...)
	}

	return docs, nil
}

// LoadFromFile loads a single DOCX file.
func (r *DocxReader) LoadFromFile(filePath string) ([]schema.Node, error) {
	return r.loadFile(filePath)
}

// Metadata returns reader metadata.
func (r *DocxReader) Metadata() ReaderMetadata {
	return ReaderMetadata{
		Name:                "DocxReader",
		SupportedExtensions: []string{".docx"},
		Description:         "Reads Microsoft Word (.docx) files",
	}
}

func (r *DocxReader) getFiles() ([]string, error) {
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
		if info.IsDir() {
			if !r.Recursive && path != r.InputDir {
				return filepath.SkipDir
			}
			return nil
		}
		ext := strings.ToLower(filepath.Ext(path))
		if ext == ".docx" {
			files = append(files, path)
		}
		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to walk directory %s: %w", r.InputDir, err)
	}

	return files, nil
}

func (r *DocxReader) loadFile(filePath string) ([]schema.Node, error) {
	// Open the DOCX file (which is a ZIP archive)
	zipReader, err := zip.OpenReader(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open DOCX file: %w", err)
	}
	defer zipReader.Close()

	var docs []schema.Node

	// Extract metadata
	metadata := make(map[string]interface{})
	metadata["source"] = filePath
	metadata["filename"] = filepath.Base(filePath)

	if r.ExtractMetadata {
		props, err := r.extractCoreProperties(&zipReader.Reader)
		if err == nil {
			for k, v := range props {
				metadata[k] = v
			}
		}
	}

	// Extract main document text
	text, err := r.extractDocumentText(&zipReader.Reader)
	if err != nil {
		return nil, fmt.Errorf("failed to extract text: %w", err)
	}

	// Create main document node
	doc := schema.Node{
		ID:       filePath,
		Text:     text,
		Type:     schema.ObjectTypeDocument,
		Metadata: metadata,
	}
	docs = append(docs, doc)

	// Extract images if requested
	if r.ExtractImages {
		images, err := r.extractImages(&zipReader.Reader, filePath)
		if err == nil {
			docs = append(docs, images...)
		}
	}

	return docs, nil
}

// extractDocumentText extracts text from word/document.xml
func (r *DocxReader) extractDocumentText(zipReader *zip.Reader) (string, error) {
	for _, file := range zipReader.File {
		if file.Name == "word/document.xml" {
			rc, err := file.Open()
			if err != nil {
				return "", err
			}
			defer rc.Close()

			content, err := io.ReadAll(rc)
			if err != nil {
				return "", err
			}

			return r.parseDocumentXML(content)
		}
	}

	return "", fmt.Errorf("document.xml not found in DOCX")
}

// docxDocument represents the document structure
type docxDocument struct {
	XMLName xml.Name `xml:"document"`
	Body    docxBody `xml:"body"`
}

type docxBody struct {
	Paragraphs []docxParagraph `xml:"p"`
	Tables     []docxTable     `xml:"tbl"`
	Content    []docxContent   `xml:",any"`
}

type docxContent struct {
	XMLName    xml.Name
	Paragraphs []docxParagraph `xml:"p"`
	Tables     []docxTable     `xml:"tbl"`
}

type docxParagraph struct {
	Runs       []docxRun       `xml:"r"`
	Properties *docxParaProps  `xml:"pPr"`
	Hyperlinks []docxHyperlink `xml:"hyperlink"`
}

type docxParaProps struct {
	Style *docxStyle `xml:"pStyle"`
}

type docxStyle struct {
	Val string `xml:"val,attr"`
}

type docxRun struct {
	Text  []docxText `xml:"t"`
	Tab   []struct{} `xml:"tab"`
	Break []struct{} `xml:"br"`
}

type docxText struct {
	Content string `xml:",chardata"`
	Space   string `xml:"space,attr"`
}

type docxHyperlink struct {
	Runs []docxRun `xml:"r"`
}

type docxTable struct {
	Rows []docxTableRow `xml:"tr"`
}

type docxTableRow struct {
	Cells []docxTableCell `xml:"tc"`
}

type docxTableCell struct {
	Paragraphs []docxParagraph `xml:"p"`
}

func (r *DocxReader) parseDocumentXML(content []byte) (string, error) {
	var doc docxDocument
	if err := xml.Unmarshal(content, &doc); err != nil {
		// Fall back to regex-based extraction if XML parsing fails
		return r.extractTextFallback(content), nil
	}

	var textParts []string

	// Process body content in order
	for _, content := range doc.Body.Content {
		switch content.XMLName.Local {
		case "p":
			// Process paragraphs within content
			for _, para := range content.Paragraphs {
				text := r.extractParagraphText(&para)
				if text != "" {
					textParts = append(textParts, text)
				}
			}
		case "tbl":
			if r.ExtractTables {
				for _, tbl := range content.Tables {
					tableText := r.extractTableText(&tbl)
					if tableText != "" {
						textParts = append(textParts, tableText)
					}
				}
			}
		}
	}

	// Also process direct paragraphs and tables
	for _, para := range doc.Body.Paragraphs {
		text := r.extractParagraphText(&para)
		if text != "" {
			textParts = append(textParts, text)
		}
	}

	if r.ExtractTables {
		for _, tbl := range doc.Body.Tables {
			tableText := r.extractTableText(&tbl)
			if tableText != "" {
				textParts = append(textParts, tableText)
			}
		}
	}

	separator := " "
	if r.PreserveParagraphs {
		separator = "\n\n"
	}

	return strings.Join(textParts, separator), nil
}

func (r *DocxReader) extractParagraphText(para *docxParagraph) string {
	var parts []string

	// Extract text from runs
	for _, run := range para.Runs {
		for _, text := range run.Text {
			if text.Content != "" {
				parts = append(parts, text.Content)
			}
		}
		// Handle tabs
		for range run.Tab {
			parts = append(parts, "\t")
		}
	}

	// Extract text from hyperlinks
	for _, link := range para.Hyperlinks {
		for _, run := range link.Runs {
			for _, text := range run.Text {
				if text.Content != "" {
					parts = append(parts, text.Content)
				}
			}
		}
	}

	return strings.TrimSpace(strings.Join(parts, ""))
}

func (r *DocxReader) extractTableText(tbl *docxTable) string {
	var rows []string

	for _, row := range tbl.Rows {
		var cells []string
		for _, cell := range row.Cells {
			var cellText []string
			for _, para := range cell.Paragraphs {
				text := r.extractParagraphText(&para)
				if text != "" {
					cellText = append(cellText, text)
				}
			}
			cells = append(cells, strings.Join(cellText, " "))
		}
		if len(cells) > 0 {
			rows = append(rows, strings.Join(cells, " | "))
		}
	}

	if len(rows) == 0 {
		return ""
	}

	return strings.Join(rows, "\n")
}

// extractTextFallback uses regex when XML parsing fails
func (r *DocxReader) extractTextFallback(content []byte) string {
	// Remove XML tags and extract text content
	textRegex := regexp.MustCompile(`<w:t[^>]*>([^<]*)</w:t>`)
	matches := textRegex.FindAllSubmatch(content, -1)

	var parts []string
	for _, match := range matches {
		if len(match) > 1 {
			text := string(match[1])
			if text != "" {
				parts = append(parts, text)
			}
		}
	}

	return strings.Join(parts, " ")
}

// coreProperties represents docProps/core.xml
type coreProperties struct {
	XMLName     xml.Name `xml:"coreProperties"`
	Title       string   `xml:"title"`
	Subject     string   `xml:"subject"`
	Creator     string   `xml:"creator"`
	Keywords    string   `xml:"keywords"`
	Description string   `xml:"description"`
	LastModBy   string   `xml:"lastModifiedBy"`
	Revision    string   `xml:"revision"`
	Created     string   `xml:"created"`
	Modified    string   `xml:"modified"`
}

func (r *DocxReader) extractCoreProperties(zipReader *zip.Reader) (map[string]interface{}, error) {
	props := make(map[string]interface{})

	for _, file := range zipReader.File {
		if file.Name == "docProps/core.xml" {
			rc, err := file.Open()
			if err != nil {
				return nil, err
			}
			defer rc.Close()

			content, err := io.ReadAll(rc)
			if err != nil {
				return nil, err
			}

			var core coreProperties
			if err := xml.Unmarshal(content, &core); err != nil {
				return nil, err
			}

			if core.Title != "" {
				props["title"] = core.Title
			}
			if core.Subject != "" {
				props["subject"] = core.Subject
			}
			if core.Creator != "" {
				props["author"] = core.Creator
			}
			if core.Keywords != "" {
				props["keywords"] = core.Keywords
			}
			if core.Description != "" {
				props["description"] = core.Description
			}
			if core.LastModBy != "" {
				props["last_modified_by"] = core.LastModBy
			}
			if core.Created != "" {
				props["created"] = core.Created
			}
			if core.Modified != "" {
				props["modified"] = core.Modified
			}

			return props, nil
		}
	}

	return props, nil
}

func (r *DocxReader) extractImages(zipReader *zip.Reader, filePath string) ([]schema.Node, error) {
	var images []schema.Node
	imageIndex := 0

	for _, file := range zipReader.File {
		if strings.HasPrefix(file.Name, "word/media/") {
			ext := strings.ToLower(filepath.Ext(file.Name))
			if ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".gif" || ext == ".bmp" {
				rc, err := file.Open()
				if err != nil {
					continue
				}

				data, err := io.ReadAll(rc)
				rc.Close()
				if err != nil {
					continue
				}

				// Create image node
				imageNode := schema.Node{
					ID:   fmt.Sprintf("%s_image_%d", filePath, imageIndex),
					Text: fmt.Sprintf("[Image: %s]", filepath.Base(file.Name)),
					Type: schema.ObjectTypeImage,
					Metadata: map[string]interface{}{
						"source":       filePath,
						"image_name":   filepath.Base(file.Name),
						"image_path":   file.Name,
						"image_size":   len(data),
						"content_type": getMimeType(ext),
					},
				}

				images = append(images, imageNode)
				imageIndex++
			}
		}
	}

	return images, nil
}

func getMimeType(ext string) string {
	switch ext {
	case ".png":
		return "image/png"
	case ".jpg", ".jpeg":
		return "image/jpeg"
	case ".gif":
		return "image/gif"
	case ".bmp":
		return "image/bmp"
	default:
		return "application/octet-stream"
	}
}

// LoadFromBytes loads a DOCX from byte content.
func (r *DocxReader) LoadFromBytes(content []byte, sourceName string) ([]schema.Node, error) {
	reader := bytes.NewReader(content)
	zipReader, err := zip.NewReader(reader, int64(len(content)))
	if err != nil {
		return nil, fmt.Errorf("failed to read DOCX from bytes: %w", err)
	}

	var docs []schema.Node

	// Extract metadata
	metadata := make(map[string]interface{})
	metadata["source"] = sourceName

	if r.ExtractMetadata {
		props, err := r.extractCoreProperties(zipReader)
		if err == nil {
			for k, v := range props {
				metadata[k] = v
			}
		}
	}

	// Extract main document text
	text, err := r.extractDocumentText(zipReader)
	if err != nil {
		return nil, fmt.Errorf("failed to extract text: %w", err)
	}

	doc := schema.Node{
		ID:       sourceName,
		Text:     text,
		Type:     schema.ObjectTypeDocument,
		Metadata: metadata,
	}
	docs = append(docs, doc)

	if r.ExtractImages {
		images, err := r.extractImages(zipReader, sourceName)
		if err == nil {
			docs = append(docs, images...)
		}
	}

	return docs, nil
}
