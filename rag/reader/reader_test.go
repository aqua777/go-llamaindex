package reader

import (
	"os"
	"path/filepath"
	"testing"
)

func TestJSONReader(t *testing.T) {
	// Create temp directory
	tmpDir, err := os.MkdirTemp("", "json_reader_test")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	t.Run("single JSON object", func(t *testing.T) {
		jsonFile := filepath.Join(tmpDir, "single.json")
		content := `{"title": "Test Doc", "content": "This is test content", "author": "Test Author"}`
		if err := os.WriteFile(jsonFile, []byte(content), 0644); err != nil {
			t.Fatalf("failed to write test file: %v", err)
		}

		reader := NewJSONReader(jsonFile).WithTextContentKey("content")
		docs, err := reader.LoadData()
		if err != nil {
			t.Fatalf("LoadData() error = %v", err)
		}

		if len(docs) != 1 {
			t.Errorf("expected 1 doc, got %d", len(docs))
		}

		if docs[0].Text != "This is test content" {
			t.Errorf("expected text 'This is test content', got '%s'", docs[0].Text)
		}

		if docs[0].Metadata["title"] != "Test Doc" {
			t.Errorf("expected title 'Test Doc' in metadata")
		}
	})

	t.Run("JSON array", func(t *testing.T) {
		jsonFile := filepath.Join(tmpDir, "array.json")
		content := `[{"text": "Doc 1"}, {"text": "Doc 2"}, {"text": "Doc 3"}]`
		if err := os.WriteFile(jsonFile, []byte(content), 0644); err != nil {
			t.Fatalf("failed to write test file: %v", err)
		}

		reader := NewJSONReader(jsonFile).WithTextContentKey("text")
		docs, err := reader.LoadData()
		if err != nil {
			t.Fatalf("LoadData() error = %v", err)
		}

		if len(docs) != 3 {
			t.Errorf("expected 3 docs, got %d", len(docs))
		}
	})

	t.Run("JSONL format", func(t *testing.T) {
		jsonlFile := filepath.Join(tmpDir, "data.jsonl")
		content := `{"text": "Line 1"}
{"text": "Line 2"}
{"text": "Line 3"}`
		if err := os.WriteFile(jsonlFile, []byte(content), 0644); err != nil {
			t.Fatalf("failed to write test file: %v", err)
		}

		reader := NewJSONReader(jsonlFile).WithTextContentKey("text").WithJSONL(true)
		docs, err := reader.LoadData()
		if err != nil {
			t.Fatalf("LoadData() error = %v", err)
		}

		if len(docs) != 3 {
			t.Errorf("expected 3 docs, got %d", len(docs))
		}
	})
}

func TestMarkdownReader(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "md_reader_test")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	t.Run("basic markdown", func(t *testing.T) {
		mdFile := filepath.Join(tmpDir, "test.md")
		content := `# Title

This is a paragraph.

## Section 1

Content of section 1.

## Section 2

Content of section 2.
`
		if err := os.WriteFile(mdFile, []byte(content), 0644); err != nil {
			t.Fatalf("failed to write test file: %v", err)
		}

		reader := NewMarkdownReader(mdFile)
		docs, err := reader.LoadData()
		if err != nil {
			t.Fatalf("LoadData() error = %v", err)
		}

		if len(docs) != 1 {
			t.Errorf("expected 1 doc, got %d", len(docs))
		}
	})

	t.Run("split by headers", func(t *testing.T) {
		mdFile := filepath.Join(tmpDir, "split.md")
		content := `# Title

Intro text.

## Section 1

Content 1.

## Section 2

Content 2.
`
		if err := os.WriteFile(mdFile, []byte(content), 0644); err != nil {
			t.Fatalf("failed to write test file: %v", err)
		}

		reader := NewMarkdownReader(mdFile).WithSplitByHeaders(true, 1, 2)
		docs, err := reader.LoadData()
		if err != nil {
			t.Fatalf("LoadData() error = %v", err)
		}

		// Should split into: Title section, Section 1, Section 2
		if len(docs) < 2 {
			t.Errorf("expected at least 2 docs from header splitting, got %d", len(docs))
		}
	})

	t.Run("frontmatter extraction", func(t *testing.T) {
		mdFile := filepath.Join(tmpDir, "frontmatter.md")
		content := `---
title: My Document
author: Test Author
date: 2024-01-01
---

# Content

This is the main content.
`
		if err := os.WriteFile(mdFile, []byte(content), 0644); err != nil {
			t.Fatalf("failed to write test file: %v", err)
		}

		reader := NewMarkdownReader(mdFile)
		docs, err := reader.LoadData()
		if err != nil {
			t.Fatalf("LoadData() error = %v", err)
		}

		if len(docs) != 1 {
			t.Fatalf("expected 1 doc, got %d", len(docs))
		}

		if docs[0].Metadata["title"] != "My Document" {
			t.Errorf("expected title 'My Document' in metadata, got %v", docs[0].Metadata["title"])
		}
		if docs[0].Metadata["author"] != "Test Author" {
			t.Errorf("expected author 'Test Author' in metadata, got %v", docs[0].Metadata["author"])
		}
	})

	t.Run("remove hyperlinks", func(t *testing.T) {
		mdFile := filepath.Join(tmpDir, "links.md")
		content := `Check out [this link](https://example.com) for more info.`
		if err := os.WriteFile(mdFile, []byte(content), 0644); err != nil {
			t.Fatalf("failed to write test file: %v", err)
		}

		reader := NewMarkdownReader(mdFile).WithRemoveHyperlinks(true)
		docs, err := reader.LoadData()
		if err != nil {
			t.Fatalf("LoadData() error = %v", err)
		}

		expected := "Check out this link for more info."
		if docs[0].Text != expected {
			t.Errorf("expected '%s', got '%s'", expected, docs[0].Text)
		}
	})
}

func TestHTMLReader(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "html_reader_test")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	t.Run("basic HTML", func(t *testing.T) {
		htmlFile := filepath.Join(tmpDir, "test.html")
		content := `<!DOCTYPE html>
<html lang="en">
<head>
    <title>Test Page</title>
    <meta name="description" content="A test page">
</head>
<body>
    <h1>Welcome</h1>
    <p>This is a paragraph.</p>
    <script>console.log("removed");</script>
</body>
</html>`
		if err := os.WriteFile(htmlFile, []byte(content), 0644); err != nil {
			t.Fatalf("failed to write test file: %v", err)
		}

		reader := NewHTMLReader(htmlFile)
		docs, err := reader.LoadData()
		if err != nil {
			t.Fatalf("LoadData() error = %v", err)
		}

		if len(docs) != 1 {
			t.Fatalf("expected 1 doc, got %d", len(docs))
		}

		// Check metadata extraction
		if docs[0].Metadata["title"] != "Test Page" {
			t.Errorf("expected title 'Test Page', got %v", docs[0].Metadata["title"])
		}
		if docs[0].Metadata["description"] != "A test page" {
			t.Errorf("expected description 'A test page', got %v", docs[0].Metadata["description"])
		}
		if docs[0].Metadata["language"] != "en" {
			t.Errorf("expected language 'en', got %v", docs[0].Metadata["language"])
		}

		// Check that script content is removed
		if containsString(docs[0].Text, "console.log") {
			t.Error("script content should be removed")
		}

		// Check that main content is present
		if !containsString(docs[0].Text, "Welcome") {
			t.Error("expected 'Welcome' in text")
		}
		if !containsString(docs[0].Text, "This is a paragraph") {
			t.Error("expected 'This is a paragraph' in text")
		}
	})

	t.Run("extract specific tags", func(t *testing.T) {
		htmlFile := filepath.Join(tmpDir, "tags.html")
		content := `<html>
<body>
    <header>Header content</header>
    <article>Article content</article>
    <footer>Footer content</footer>
</body>
</html>`
		if err := os.WriteFile(htmlFile, []byte(content), 0644); err != nil {
			t.Fatalf("failed to write test file: %v", err)
		}

		reader := NewHTMLReader(htmlFile).WithTagsToExtract("article")
		docs, err := reader.LoadData()
		if err != nil {
			t.Fatalf("LoadData() error = %v", err)
		}

		if !containsString(docs[0].Text, "Article content") {
			t.Error("expected 'Article content' in text")
		}
		if containsString(docs[0].Text, "Header content") {
			t.Error("should not contain 'Header content'")
		}
	})

	t.Run("HTML entities", func(t *testing.T) {
		htmlFile := filepath.Join(tmpDir, "entities.html")
		content := `<html><body><p>Tom &amp; Jerry &mdash; &copy; 2024</p></body></html>`
		if err := os.WriteFile(htmlFile, []byte(content), 0644); err != nil {
			t.Fatalf("failed to write test file: %v", err)
		}

		reader := NewHTMLReader(htmlFile)
		docs, err := reader.LoadData()
		if err != nil {
			t.Fatalf("LoadData() error = %v", err)
		}

		if !containsString(docs[0].Text, "Tom & Jerry") {
			t.Errorf("expected decoded entities, got '%s'", docs[0].Text)
		}
	})
}

func TestSimpleDirectoryReader(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "dir_reader_test")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create test files
	files := map[string]string{
		"doc1.txt": "Content of doc 1",
		"doc2.txt": "Content of doc 2",
		"doc3.md":  "# Markdown\n\nContent",
		"other.py": "print('ignored')",
	}

	for name, content := range files {
		if err := os.WriteFile(filepath.Join(tmpDir, name), []byte(content), 0644); err != nil {
			t.Fatalf("failed to write %s: %v", name, err)
		}
	}

	t.Run("default extensions", func(t *testing.T) {
		reader := NewSimpleDirectoryReader(tmpDir)
		docs, err := reader.LoadData()
		if err != nil {
			t.Fatalf("LoadData() error = %v", err)
		}

		// Should load .txt and .md files (3 total)
		if len(docs) != 3 {
			t.Errorf("expected 3 docs, got %d", len(docs))
		}
	})

	t.Run("specific extensions", func(t *testing.T) {
		reader := NewSimpleDirectoryReader(tmpDir, ".txt")
		docs, err := reader.LoadData()
		if err != nil {
			t.Fatalf("LoadData() error = %v", err)
		}

		// Should load only .txt files (2 total)
		if len(docs) != 2 {
			t.Errorf("expected 2 docs, got %d", len(docs))
		}
	})
}

func containsString(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsStringHelper(s, substr))
}

func containsStringHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func TestPDFReader(t *testing.T) {
	t.Run("constructor with files", func(t *testing.T) {
		reader := NewPDFReader("test1.pdf", "test2.pdf")
		if len(reader.InputFiles) != 2 {
			t.Errorf("expected 2 input files, got %d", len(reader.InputFiles))
		}
		if reader.Recursive {
			t.Error("expected Recursive to be false by default")
		}
	})

	t.Run("constructor from directory", func(t *testing.T) {
		reader := NewPDFReaderFromDir("/some/dir", true)
		if reader.InputDir != "/some/dir" {
			t.Errorf("expected InputDir '/some/dir', got '%s'", reader.InputDir)
		}
		if !reader.Recursive {
			t.Error("expected Recursive to be true")
		}
	})

	t.Run("constructor with options", func(t *testing.T) {
		extraMeta := map[string]interface{}{"key": "value"}
		reader := NewPDFReaderWithOptions(
			WithPDFInputFiles("file1.pdf"),
			WithPDFInputDir("/dir"),
			WithPDFRecursive(true),
			WithPDFSplitByPage(true),
			WithPDFExtraMetadata(extraMeta),
		)

		if len(reader.InputFiles) != 1 || reader.InputFiles[0] != "file1.pdf" {
			t.Error("expected input file 'file1.pdf'")
		}
		if reader.InputDir != "/dir" {
			t.Errorf("expected InputDir '/dir', got '%s'", reader.InputDir)
		}
		if !reader.Recursive {
			t.Error("expected Recursive to be true")
		}
		if !reader.SplitByPage {
			t.Error("expected SplitByPage to be true")
		}
		if reader.ExtraMetadata["key"] != "value" {
			t.Error("expected extra metadata to be set")
		}
	})

	t.Run("fluent API", func(t *testing.T) {
		extraMeta := map[string]interface{}{"source": "test"}
		reader := NewPDFReader("test.pdf").
			WithSplitByPage(true).
			WithExtraMetadata(extraMeta)

		if !reader.SplitByPage {
			t.Error("expected SplitByPage to be true")
		}
		if reader.ExtraMetadata["source"] != "test" {
			t.Error("expected extra metadata to be set")
		}
	})

	t.Run("metadata", func(t *testing.T) {
		reader := NewPDFReader()
		meta := reader.Metadata()

		if meta.Name != "PDFReader" {
			t.Errorf("expected Name 'PDFReader', got '%s'", meta.Name)
		}
		if len(meta.SupportedExtensions) != 1 || meta.SupportedExtensions[0] != ".pdf" {
			t.Error("expected SupportedExtensions to contain '.pdf'")
		}
		if meta.Description == "" {
			t.Error("expected Description to be non-empty")
		}
	})

	t.Run("error on no input", func(t *testing.T) {
		reader := NewPDFReader()
		_, err := reader.LoadData()
		if err == nil {
			t.Error("expected error when no input files or directory specified")
		}
	})

	t.Run("error on non-existent file", func(t *testing.T) {
		reader := NewPDFReader("/non/existent/file.pdf")
		_, err := reader.LoadData()
		if err == nil {
			t.Error("expected error for non-existent file")
		}
	})

	t.Run("error on non-existent directory", func(t *testing.T) {
		reader := NewPDFReaderFromDir("/non/existent/directory", false)
		_, err := reader.LoadData()
		if err == nil {
			t.Error("expected error for non-existent directory")
		}
	})

	t.Run("directory scanning", func(t *testing.T) {
		tmpDir, err := os.MkdirTemp("", "pdf_reader_test")
		if err != nil {
			t.Fatalf("failed to create temp dir: %v", err)
		}
		defer os.RemoveAll(tmpDir)

		// Create subdirectory
		subDir := filepath.Join(tmpDir, "subdir")
		if err := os.MkdirAll(subDir, 0755); err != nil {
			t.Fatalf("failed to create subdir: %v", err)
		}

		// Create dummy files (not valid PDFs, but tests file discovery)
		files := []string{
			filepath.Join(tmpDir, "doc1.pdf"),
			filepath.Join(tmpDir, "doc2.pdf"),
			filepath.Join(tmpDir, "doc.txt"),
			filepath.Join(subDir, "doc3.pdf"),
		}
		for _, f := range files {
			if err := os.WriteFile(f, []byte("dummy"), 0644); err != nil {
				t.Fatalf("failed to create file %s: %v", f, err)
			}
		}

		// Test non-recursive
		reader := NewPDFReaderFromDir(tmpDir, false)
		foundFiles, err := reader.getFiles()
		if err != nil {
			t.Fatalf("getFiles() error: %v", err)
		}
		if len(foundFiles) != 2 {
			t.Errorf("expected 2 PDF files (non-recursive), got %d", len(foundFiles))
		}

		// Test recursive
		reader = NewPDFReaderFromDir(tmpDir, true)
		foundFiles, err = reader.getFiles()
		if err != nil {
			t.Fatalf("getFiles() error: %v", err)
		}
		if len(foundFiles) != 3 {
			t.Errorf("expected 3 PDF files (recursive), got %d", len(foundFiles))
		}
	})

	t.Run("password function", func(t *testing.T) {
		passwordCalled := false
		passwordFunc := func(filePath string) string {
			passwordCalled = true
			return "secret"
		}

		reader := NewPDFReaderWithOptions(
			WithPDFInputFiles("test.pdf"),
			WithPDFPasswordFunc(passwordFunc),
		)

		if reader.PasswordFunc == nil {
			t.Error("expected PasswordFunc to be set")
		}

		// Call the function to verify it works
		pwd := reader.PasswordFunc("test.pdf")
		if !passwordCalled {
			t.Error("expected password function to be called")
		}
		if pwd != "secret" {
			t.Errorf("expected password 'secret', got '%s'", pwd)
		}
	})
}

func TestPDFReaderInterfaces(t *testing.T) {
	// Verify interface compliance at compile time
	var _ Reader = (*PDFReader)(nil)
	var _ FileReader = (*PDFReader)(nil)
	var _ ReaderWithMetadata = (*PDFReader)(nil)
	var _ ReaderWithContext = (*PDFReader)(nil)
	var _ LazyReader = (*PDFReader)(nil)

	t.Run("Reader interface", func(t *testing.T) {
		var r Reader = NewPDFReader("test.pdf")
		if r == nil {
			t.Error("expected non-nil Reader")
		}
	})

	t.Run("FileReader interface", func(t *testing.T) {
		var r FileReader = NewPDFReader("test.pdf")
		if r == nil {
			t.Error("expected non-nil FileReader")
		}
	})

	t.Run("ReaderWithMetadata interface", func(t *testing.T) {
		var r ReaderWithMetadata = NewPDFReader("test.pdf")
		meta := r.Metadata()
		if meta.Name != "PDFReader" {
			t.Errorf("expected Name 'PDFReader', got '%s'", meta.Name)
		}
	})
}

func TestPDFReaderUtilityFunctions(t *testing.T) {
	t.Run("ExtractTextFromPDF error", func(t *testing.T) {
		_, err := ExtractTextFromPDF("/non/existent/file.pdf")
		if err == nil {
			t.Error("expected error for non-existent file")
		}
	})

	t.Run("ExtractTextFromPDFByPage error", func(t *testing.T) {
		_, err := ExtractTextFromPDFByPage("/non/existent/file.pdf")
		if err == nil {
			t.Error("expected error for non-existent file")
		}
	})

	t.Run("GetPDFPageCount error", func(t *testing.T) {
		_, err := GetPDFPageCount("/non/existent/file.pdf")
		if err == nil {
			t.Error("expected error for non-existent file")
		}
	})

	t.Run("GetPDFMetadata error", func(t *testing.T) {
		_, err := GetPDFMetadata("/non/existent/file.pdf")
		if err == nil {
			t.Error("expected error for non-existent file")
		}
	})
}

func TestPDFReaderLazyLoad(t *testing.T) {
	t.Run("lazy load with no input", func(t *testing.T) {
		reader := NewPDFReader()
		nodeChan, errChan := reader.LazyLoadData()

		// Should receive an error
		select {
		case <-nodeChan:
			t.Error("expected no nodes")
		case err := <-errChan:
			if err == nil {
				t.Error("expected error for no input")
			}
		}
	})

	t.Run("lazy load with non-existent file", func(t *testing.T) {
		reader := NewPDFReader("/non/existent/file.pdf")
		nodeChan, errChan := reader.LazyLoadData()

		// Drain the channels
		var gotError bool
		for {
			select {
			case _, ok := <-nodeChan:
				if !ok {
					nodeChan = nil
				}
			case err, ok := <-errChan:
				if !ok {
					errChan = nil
				} else if err != nil {
					gotError = true
				}
			}
			if nodeChan == nil && errChan == nil {
				break
			}
		}

		if !gotError {
			t.Error("expected error for non-existent file")
		}
	})
}
