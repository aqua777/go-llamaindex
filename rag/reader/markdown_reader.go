package reader

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/aqua777/go-llamaindex/schema"
)

// MarkdownReader reads Markdown files and converts them to documents.
type MarkdownReader struct {
	// InputFiles is a list of Markdown file paths to read
	InputFiles []string
	// InputDir is a directory containing Markdown files
	InputDir string
	// Recursive determines if subdirectories should be searched
	Recursive bool
	// RemoveHyperlinks removes hyperlinks from the text
	RemoveHyperlinks bool
	// RemoveImages removes image references from the text
	RemoveImages bool
	// SplitByHeaders splits document into multiple nodes by headers
	SplitByHeaders bool
	// HeadersToSplitOn specifies which header levels to split on (e.g., []int{1, 2})
	HeadersToSplitOn []int
}

// NewMarkdownReader creates a new MarkdownReader for specific files.
func NewMarkdownReader(inputFiles ...string) *MarkdownReader {
	return &MarkdownReader{
		InputFiles: inputFiles,
		Recursive:  false,
	}
}

// NewMarkdownReaderFromDir creates a new MarkdownReader for a directory.
func NewMarkdownReaderFromDir(inputDir string, recursive bool) *MarkdownReader {
	return &MarkdownReader{
		InputDir:  inputDir,
		Recursive: recursive,
	}
}

// WithRemoveHyperlinks enables hyperlink removal.
func (r *MarkdownReader) WithRemoveHyperlinks(remove bool) *MarkdownReader {
	r.RemoveHyperlinks = remove
	return r
}

// WithRemoveImages enables image reference removal.
func (r *MarkdownReader) WithRemoveImages(remove bool) *MarkdownReader {
	r.RemoveImages = remove
	return r
}

// WithSplitByHeaders enables splitting by headers.
func (r *MarkdownReader) WithSplitByHeaders(split bool, levels ...int) *MarkdownReader {
	r.SplitByHeaders = split
	if len(levels) > 0 {
		r.HeadersToSplitOn = levels
	} else {
		r.HeadersToSplitOn = []int{1, 2} // Default to H1 and H2
	}
	return r
}

// LoadData loads Markdown files and returns documents.
func (r *MarkdownReader) LoadData() ([]schema.Node, error) {
	files, err := r.getFiles()
	if err != nil {
		return nil, err
	}

	var docs []schema.Node
	for _, file := range files {
		fileDocs, err := r.loadFile(file)
		if err != nil {
			return nil, NewReaderError(file, "failed to load Markdown file", err)
		}
		docs = append(docs, fileDocs...)
	}

	return docs, nil
}

// LoadFromFile loads a single Markdown file.
func (r *MarkdownReader) LoadFromFile(filePath string) ([]schema.Node, error) {
	return r.loadFile(filePath)
}

// Metadata returns reader metadata.
func (r *MarkdownReader) Metadata() ReaderMetadata {
	return ReaderMetadata{
		Name:                "MarkdownReader",
		SupportedExtensions: []string{".md", ".markdown", ".mdown", ".mkd"},
		Description:         "Reads Markdown files with optional preprocessing",
	}
}

func (r *MarkdownReader) getFiles() ([]string, error) {
	if len(r.InputFiles) > 0 {
		return r.InputFiles, nil
	}

	if r.InputDir == "" {
		return nil, fmt.Errorf("no input files or directory specified")
	}

	var files []string
	mdExtensions := map[string]bool{
		".md":       true,
		".markdown": true,
		".mdown":    true,
		".mkd":      true,
	}

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
		if mdExtensions[ext] {
			files = append(files, path)
		}
		return nil
	}

	if err := filepath.Walk(r.InputDir, walkFn); err != nil {
		return nil, fmt.Errorf("failed to walk directory %s: %w", r.InputDir, err)
	}

	return files, nil
}

func (r *MarkdownReader) loadFile(filePath string) ([]schema.Node, error) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	text := string(content)

	// Apply preprocessing
	text = r.preprocess(text)

	// Extract frontmatter metadata if present
	metadata, text := r.extractFrontmatter(text)
	metadata["source"] = filePath
	metadata["filename"] = filepath.Base(filePath)

	if r.SplitByHeaders {
		return r.splitByHeaders(text, filePath, metadata)
	}

	doc := schema.Node{
		ID:       filePath,
		Text:     strings.TrimSpace(text),
		Type:     schema.ObjectTypeDocument,
		Metadata: metadata,
	}

	return []schema.Node{doc}, nil
}

func (r *MarkdownReader) preprocess(text string) string {
	if r.RemoveHyperlinks {
		// Replace [text](url) with text
		linkRegex := regexp.MustCompile(`\[([^\]]+)\]\([^)]+\)`)
		text = linkRegex.ReplaceAllString(text, "$1")
	}

	if r.RemoveImages {
		// Remove ![alt](url) patterns
		imageRegex := regexp.MustCompile(`!\[[^\]]*\]\([^)]+\)`)
		text = imageRegex.ReplaceAllString(text, "")
	}

	return text
}

func (r *MarkdownReader) extractFrontmatter(text string) (map[string]interface{}, string) {
	metadata := make(map[string]interface{})

	// Check for YAML frontmatter (--- ... ---)
	if !strings.HasPrefix(text, "---") {
		return metadata, text
	}

	lines := strings.Split(text, "\n")
	if len(lines) < 3 {
		return metadata, text
	}

	// Find closing ---
	endIdx := -1
	for i := 1; i < len(lines); i++ {
		if strings.TrimSpace(lines[i]) == "---" {
			endIdx = i
			break
		}
	}

	if endIdx == -1 {
		return metadata, text
	}

	// Parse simple key: value pairs from frontmatter
	for i := 1; i < endIdx; i++ {
		line := strings.TrimSpace(lines[i])
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, ":", 2)
		if len(parts) == 2 {
			key := strings.TrimSpace(parts[0])
			value := strings.TrimSpace(parts[1])
			// Remove quotes if present
			value = strings.Trim(value, `"'`)
			metadata[key] = value
		}
	}

	// Return text without frontmatter
	remainingText := strings.Join(lines[endIdx+1:], "\n")
	return metadata, remainingText
}

func (r *MarkdownReader) splitByHeaders(text, filePath string, baseMetadata map[string]interface{}) ([]schema.Node, error) {
	var docs []schema.Node

	// Build header regex based on levels to split on
	levels := r.HeadersToSplitOn
	if len(levels) == 0 {
		levels = []int{1, 2}
	}

	// Create pattern like ^(#{1,2})\s+(.+)$ for levels 1 and 2
	maxLevel := 0
	for _, l := range levels {
		if l > maxLevel {
			maxLevel = l
		}
	}
	headerPattern := fmt.Sprintf(`(?m)^(#{1,%d})\s+(.+)$`, maxLevel)
	headerRegex := regexp.MustCompile(headerPattern)

	scanner := bufio.NewScanner(strings.NewReader(text))
	var currentSection strings.Builder
	var currentHeader string
	var currentLevel int
	sectionIndex := 0

	flushSection := func() {
		content := strings.TrimSpace(currentSection.String())
		if content == "" && currentHeader == "" {
			return
		}

		metadata := make(map[string]interface{})
		for k, v := range baseMetadata {
			metadata[k] = v
		}
		if currentHeader != "" {
			metadata["header"] = currentHeader
			metadata["header_level"] = currentLevel
		}
		metadata["section_index"] = sectionIndex

		doc := schema.Node{
			ID:       fmt.Sprintf("%s_section_%d", filePath, sectionIndex),
			Text:     content,
			Type:     schema.ObjectTypeDocument,
			Metadata: metadata,
		}
		docs = append(docs, doc)
		sectionIndex++
		currentSection.Reset()
	}

	for scanner.Scan() {
		line := scanner.Text()

		matches := headerRegex.FindStringSubmatch(line)
		if matches != nil {
			headerMarks := matches[1]
			headerText := matches[2]
			level := len(headerMarks)

			// Check if this level should trigger a split
			shouldSplit := false
			for _, l := range levels {
				if level == l {
					shouldSplit = true
					break
				}
			}

			if shouldSplit {
				flushSection()
				currentHeader = headerText
				currentLevel = level
				currentSection.WriteString(line)
				currentSection.WriteString("\n")
				continue
			}
		}

		currentSection.WriteString(line)
		currentSection.WriteString("\n")
	}

	// Flush remaining content
	flushSection()

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error scanning file: %w", err)
	}

	return docs, nil
}
