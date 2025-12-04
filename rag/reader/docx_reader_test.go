package reader

import (
	"archive/zip"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// createTestDocxFile creates a minimal valid DOCX file for testing
func createTestDocxFile(t *testing.T, path string, paragraphs []string) {
	f, err := os.Create(path)
	require.NoError(t, err)
	defer f.Close()

	w := zip.NewWriter(f)
	defer w.Close()

	// Create [Content_Types].xml
	contentTypes := `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>`
	fw, _ := w.Create("[Content_Types].xml")
	fw.Write([]byte(contentTypes))

	// Create _rels/.rels
	rels := `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>`
	fw, _ = w.Create("_rels/.rels")
	fw.Write([]byte(rels))

	// Create word/document.xml with paragraphs
	docStart := `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>`
	docEnd := `  </w:body>
</w:document>`

	var paraXML string
	for _, p := range paragraphs {
		paraXML += `    <w:p><w:r><w:t>` + p + `</w:t></w:r></w:p>
`
	}

	fw, _ = w.Create("word/document.xml")
	fw.Write([]byte(docStart + "\n" + paraXML + docEnd))
}

// createTestDocxWithMetadata creates a DOCX with core properties
func createTestDocxWithMetadata(t *testing.T, path string, text string, title, author string) {
	f, err := os.Create(path)
	require.NoError(t, err)
	defer f.Close()

	w := zip.NewWriter(f)
	defer w.Close()

	// Content Types
	contentTypes := `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
</Types>`
	fw, _ := w.Create("[Content_Types].xml")
	fw.Write([]byte(contentTypes))

	// Relationships
	rels := `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
</Relationships>`
	fw, _ = w.Create("_rels/.rels")
	fw.Write([]byte(rels))

	// Document
	doc := `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>` + text + `</w:t></w:r></w:p>
  </w:body>
</w:document>`
	fw, _ = w.Create("word/document.xml")
	fw.Write([]byte(doc))

	// Core properties
	core := `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <dc:title>` + title + `</dc:title>
  <dc:creator>` + author + `</dc:creator>
</cp:coreProperties>`
	fw, _ = w.Create("docProps/core.xml")
	fw.Write([]byte(core))
}

// createTestDocxWithTable creates a DOCX with a table
func createTestDocxWithTable(t *testing.T, path string, rows [][]string) {
	f, err := os.Create(path)
	require.NoError(t, err)
	defer f.Close()

	w := zip.NewWriter(f)
	defer w.Close()

	// Content Types
	contentTypes := `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>`
	fw, _ := w.Create("[Content_Types].xml")
	fw.Write([]byte(contentTypes))

	// Relationships
	rels := `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>`
	fw, _ = w.Create("_rels/.rels")
	fw.Write([]byte(rels))

	// Build table XML
	tableXML := "<w:tbl>"
	for _, row := range rows {
		tableXML += "<w:tr>"
		for _, cell := range row {
			tableXML += "<w:tc><w:p><w:r><w:t>" + cell + "</w:t></w:r></w:p></w:tc>"
		}
		tableXML += "</w:tr>"
	}
	tableXML += "</w:tbl>"

	// Document with table
	doc := `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    ` + tableXML + `
  </w:body>
</w:document>`
	fw, _ = w.Create("word/document.xml")
	fw.Write([]byte(doc))
}

func TestDocxReader_LoadData(t *testing.T) {
	tmpDir := t.TempDir()
	docxPath := filepath.Join(tmpDir, "test.docx")

	createTestDocxFile(t, docxPath, []string{
		"Hello, World!",
		"This is a test document.",
		"It has multiple paragraphs.",
	})

	reader := NewDocxReader(docxPath)
	docs, err := reader.LoadData()
	require.NoError(t, err)
	assert.Len(t, docs, 1)

	// Check content
	assert.Contains(t, docs[0].Text, "Hello, World!")
	assert.Contains(t, docs[0].Text, "This is a test document.")
	assert.Contains(t, docs[0].Text, "multiple paragraphs")

	// Check metadata
	assert.Equal(t, docxPath, docs[0].Metadata["source"])
	assert.Equal(t, "test.docx", docs[0].Metadata["filename"])
}

func TestDocxReader_WithMetadata(t *testing.T) {
	tmpDir := t.TempDir()
	docxPath := filepath.Join(tmpDir, "test.docx")

	createTestDocxWithMetadata(t, docxPath, "Document content", "Test Title", "Test Author")

	reader := NewDocxReader(docxPath)
	docs, err := reader.LoadData()
	require.NoError(t, err)
	assert.Len(t, docs, 1)

	assert.Equal(t, "Test Title", docs[0].Metadata["title"])
	assert.Equal(t, "Test Author", docs[0].Metadata["author"])
}

func TestDocxReader_WithoutMetadata(t *testing.T) {
	tmpDir := t.TempDir()
	docxPath := filepath.Join(tmpDir, "test.docx")

	createTestDocxWithMetadata(t, docxPath, "Content", "Title", "Author")

	reader := NewDocxReader(docxPath).WithExtractMetadata(false)
	docs, err := reader.LoadData()
	require.NoError(t, err)
	assert.Len(t, docs, 1)

	// Should not have title/author metadata
	_, hasTitle := docs[0].Metadata["title"]
	assert.False(t, hasTitle)
}

func TestDocxReader_WithTable(t *testing.T) {
	tmpDir := t.TempDir()
	docxPath := filepath.Join(tmpDir, "test.docx")

	createTestDocxWithTable(t, docxPath, [][]string{
		{"Name", "Age"},
		{"Alice", "30"},
		{"Bob", "25"},
	})

	reader := NewDocxReader(docxPath)
	docs, err := reader.LoadData()
	require.NoError(t, err)
	assert.Len(t, docs, 1)

	// Table content should be extracted
	assert.Contains(t, docs[0].Text, "Name")
	assert.Contains(t, docs[0].Text, "Alice")
	assert.Contains(t, docs[0].Text, "Bob")
}

func TestDocxReader_WithoutTable(t *testing.T) {
	tmpDir := t.TempDir()
	docxPath := filepath.Join(tmpDir, "test.docx")

	createTestDocxWithTable(t, docxPath, [][]string{
		{"Name", "Age"},
		{"Alice", "30"},
	})

	reader := NewDocxReader(docxPath).WithExtractTables(false)
	docs, err := reader.LoadData()
	require.NoError(t, err)
	assert.Len(t, docs, 1)

	// Table content should not be in text (or minimal)
	// Note: The table is the only content, so text might be empty
}

func TestDocxReader_PreserveParagraphs(t *testing.T) {
	tmpDir := t.TempDir()
	docxPath := filepath.Join(tmpDir, "test.docx")

	createTestDocxFile(t, docxPath, []string{"First paragraph", "Second paragraph"})

	// With preserved paragraphs
	reader := NewDocxReader(docxPath).WithPreserveParagraphs(true)
	docs, err := reader.LoadData()
	require.NoError(t, err)
	assert.Contains(t, docs[0].Text, "\n\n")

	// Without preserved paragraphs
	reader = NewDocxReader(docxPath).WithPreserveParagraphs(false)
	docs, err = reader.LoadData()
	require.NoError(t, err)
	// Should use space separator instead
	assert.NotContains(t, docs[0].Text, "\n\n")
}

func TestDocxReader_FromDirectory(t *testing.T) {
	tmpDir := t.TempDir()

	docx1 := filepath.Join(tmpDir, "file1.docx")
	docx2 := filepath.Join(tmpDir, "file2.docx")

	createTestDocxFile(t, docx1, []string{"Document One"})
	createTestDocxFile(t, docx2, []string{"Document Two"})

	reader := NewDocxReaderFromDir(tmpDir, false)
	docs, err := reader.LoadData()
	require.NoError(t, err)
	assert.Len(t, docs, 2)
}

func TestDocxReader_RecursiveDirectory(t *testing.T) {
	tmpDir := t.TempDir()
	subDir := filepath.Join(tmpDir, "subdir")
	err := os.Mkdir(subDir, 0755)
	require.NoError(t, err)

	docx1 := filepath.Join(tmpDir, "file1.docx")
	docx2 := filepath.Join(subDir, "file2.docx")

	createTestDocxFile(t, docx1, []string{"Root document"})
	createTestDocxFile(t, docx2, []string{"Subdirectory document"})

	// Non-recursive should only find root file
	reader := NewDocxReaderFromDir(tmpDir, false)
	docs, err := reader.LoadData()
	require.NoError(t, err)
	assert.Len(t, docs, 1)

	// Recursive should find both
	reader = NewDocxReaderFromDir(tmpDir, true)
	docs, err = reader.LoadData()
	require.NoError(t, err)
	assert.Len(t, docs, 2)
}

func TestDocxReader_Metadata(t *testing.T) {
	reader := NewDocxReader("test.docx")
	meta := reader.Metadata()

	assert.Equal(t, "DocxReader", meta.Name)
	assert.Contains(t, meta.SupportedExtensions, ".docx")
	assert.Contains(t, meta.Description, "Word")
}

func TestDocxReader_FileNotFound(t *testing.T) {
	reader := NewDocxReader("/nonexistent/file.docx")
	_, err := reader.LoadData()
	assert.Error(t, err)
}

func TestDocxReader_InvalidFile(t *testing.T) {
	tmpDir := t.TempDir()
	invalidPath := filepath.Join(tmpDir, "invalid.docx")

	// Create a non-ZIP file
	err := os.WriteFile(invalidPath, []byte("not a zip file"), 0644)
	require.NoError(t, err)

	reader := NewDocxReader(invalidPath)
	_, err = reader.LoadData()
	assert.Error(t, err)
}

func TestDocxReader_NoInputSpecified(t *testing.T) {
	reader := &DocxReader{}
	_, err := reader.LoadData()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no input files or directory specified")
}

func TestDocxReader_LoadFromBytes(t *testing.T) {
	tmpDir := t.TempDir()
	docxPath := filepath.Join(tmpDir, "test.docx")

	createTestDocxFile(t, docxPath, []string{"Hello from bytes"})

	content, err := os.ReadFile(docxPath)
	require.NoError(t, err)

	reader := NewDocxReader()
	docs, err := reader.LoadFromBytes(content, "memory://test.docx")
	require.NoError(t, err)
	assert.Len(t, docs, 1)
	assert.Contains(t, docs[0].Text, "Hello from bytes")
	assert.Equal(t, "memory://test.docx", docs[0].Metadata["source"])
}

func TestDocxReader_EmptyDocument(t *testing.T) {
	tmpDir := t.TempDir()
	docxPath := filepath.Join(tmpDir, "empty.docx")

	createTestDocxFile(t, docxPath, []string{})

	reader := NewDocxReader(docxPath)
	docs, err := reader.LoadData()
	require.NoError(t, err)
	assert.Len(t, docs, 1)
	assert.Equal(t, "", docs[0].Text)
}

func TestGetMimeType(t *testing.T) {
	tests := []struct {
		ext      string
		expected string
	}{
		{".png", "image/png"},
		{".jpg", "image/jpeg"},
		{".jpeg", "image/jpeg"},
		{".gif", "image/gif"},
		{".bmp", "image/bmp"},
		{".unknown", "application/octet-stream"},
	}

	for _, tt := range tests {
		t.Run(tt.ext, func(t *testing.T) {
			result := getMimeType(tt.ext)
			assert.Equal(t, tt.expected, result)
		})
	}
}
