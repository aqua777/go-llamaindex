package reader

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/aqua777/go-llamaindex/schema"
)

// HTMLReader reads HTML files and extracts text content.
type HTMLReader struct {
	// InputFiles is a list of HTML file paths to read
	InputFiles []string
	// InputDir is a directory containing HTML files
	InputDir string
	// Recursive determines if subdirectories should be searched
	Recursive bool
	// TagsToExtract specifies which HTML tags to extract text from.
	// If empty, extracts from body. Common values: "p", "div", "article", "main"
	TagsToExtract []string
	// TagsToRemove specifies which HTML tags to remove entirely (e.g., "script", "style")
	TagsToRemove []string
	// PreserveWhitespace keeps original whitespace formatting
	PreserveWhitespace bool
}

// NewHTMLReader creates a new HTMLReader for specific files.
func NewHTMLReader(inputFiles ...string) *HTMLReader {
	return &HTMLReader{
		InputFiles:   inputFiles,
		Recursive:    false,
		TagsToRemove: []string{"script", "style", "noscript", "iframe", "svg"},
	}
}

// NewHTMLReaderFromDir creates a new HTMLReader for a directory.
func NewHTMLReaderFromDir(inputDir string, recursive bool) *HTMLReader {
	return &HTMLReader{
		InputDir:     inputDir,
		Recursive:    recursive,
		TagsToRemove: []string{"script", "style", "noscript", "iframe", "svg"},
	}
}

// WithTagsToExtract sets which tags to extract text from.
func (r *HTMLReader) WithTagsToExtract(tags ...string) *HTMLReader {
	r.TagsToExtract = tags
	return r
}

// WithTagsToRemove sets which tags to remove entirely.
func (r *HTMLReader) WithTagsToRemove(tags ...string) *HTMLReader {
	r.TagsToRemove = tags
	return r
}

// WithPreserveWhitespace enables whitespace preservation.
func (r *HTMLReader) WithPreserveWhitespace(preserve bool) *HTMLReader {
	r.PreserveWhitespace = preserve
	return r
}

// LoadData loads HTML files and returns documents.
func (r *HTMLReader) LoadData() ([]schema.Node, error) {
	files, err := r.getFiles()
	if err != nil {
		return nil, err
	}

	var docs []schema.Node
	for _, file := range files {
		fileDocs, err := r.loadFile(file)
		if err != nil {
			return nil, NewReaderError(file, "failed to load HTML file", err)
		}
		docs = append(docs, fileDocs...)
	}

	return docs, nil
}

// LoadFromFile loads a single HTML file.
func (r *HTMLReader) LoadFromFile(filePath string) ([]schema.Node, error) {
	return r.loadFile(filePath)
}

// Metadata returns reader metadata.
func (r *HTMLReader) Metadata() ReaderMetadata {
	return ReaderMetadata{
		Name:                "HTMLReader",
		SupportedExtensions: []string{".html", ".htm", ".xhtml"},
		Description:         "Reads HTML files and extracts text content",
	}
}

func (r *HTMLReader) getFiles() ([]string, error) {
	if len(r.InputFiles) > 0 {
		return r.InputFiles, nil
	}

	if r.InputDir == "" {
		return nil, fmt.Errorf("no input files or directory specified")
	}

	var files []string
	htmlExtensions := map[string]bool{
		".html":  true,
		".htm":   true,
		".xhtml": true,
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
		if htmlExtensions[ext] {
			files = append(files, path)
		}
		return nil
	}

	if err := filepath.Walk(r.InputDir, walkFn); err != nil {
		return nil, fmt.Errorf("failed to walk directory %s: %w", r.InputDir, err)
	}

	return files, nil
}

func (r *HTMLReader) loadFile(filePath string) ([]schema.Node, error) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	html := string(content)

	// Extract metadata from HTML
	metadata := r.extractMetadata(html)
	metadata["source"] = filePath
	metadata["filename"] = filepath.Base(filePath)

	// Extract text content
	text := r.extractText(html)

	doc := schema.Node{
		ID:       filePath,
		Text:     text,
		Type:     schema.ObjectTypeDocument,
		Metadata: metadata,
		MimeType: "text/html",
	}

	return []schema.Node{doc}, nil
}

func (r *HTMLReader) extractMetadata(html string) map[string]interface{} {
	metadata := make(map[string]interface{})

	// Extract title
	titleRegex := regexp.MustCompile(`(?i)<title[^>]*>([^<]+)</title>`)
	if matches := titleRegex.FindStringSubmatch(html); len(matches) > 1 {
		metadata["title"] = strings.TrimSpace(matches[1])
	}

	// Extract meta description
	descRegex := regexp.MustCompile(`(?i)<meta[^>]+name=["']description["'][^>]+content=["']([^"']+)["']`)
	if matches := descRegex.FindStringSubmatch(html); len(matches) > 1 {
		metadata["description"] = strings.TrimSpace(matches[1])
	}
	// Also try reversed attribute order
	descRegex2 := regexp.MustCompile(`(?i)<meta[^>]+content=["']([^"']+)["'][^>]+name=["']description["']`)
	if matches := descRegex2.FindStringSubmatch(html); len(matches) > 1 {
		if _, exists := metadata["description"]; !exists {
			metadata["description"] = strings.TrimSpace(matches[1])
		}
	}

	// Extract meta keywords
	keywordsRegex := regexp.MustCompile(`(?i)<meta[^>]+name=["']keywords["'][^>]+content=["']([^"']+)["']`)
	if matches := keywordsRegex.FindStringSubmatch(html); len(matches) > 1 {
		metadata["keywords"] = strings.TrimSpace(matches[1])
	}

	// Extract language
	langRegex := regexp.MustCompile(`(?i)<html[^>]+lang=["']([^"']+)["']`)
	if matches := langRegex.FindStringSubmatch(html); len(matches) > 1 {
		metadata["language"] = strings.TrimSpace(matches[1])
	}

	return metadata
}

func (r *HTMLReader) extractText(html string) string {
	text := html

	// Remove tags that should be completely removed
	for _, tag := range r.TagsToRemove {
		// Remove tag and its content
		pattern := fmt.Sprintf(`(?is)<%s[^>]*>.*?</%s>`, tag, tag)
		regex := regexp.MustCompile(pattern)
		text = regex.ReplaceAllString(text, "")

		// Also remove self-closing variants
		selfClosing := fmt.Sprintf(`(?i)<%s[^>]*/?>`, tag)
		regex = regexp.MustCompile(selfClosing)
		text = regex.ReplaceAllString(text, "")
	}

	// Remove HTML comments
	commentRegex := regexp.MustCompile(`<!--[\s\S]*?-->`)
	text = commentRegex.ReplaceAllString(text, "")

	// If specific tags to extract are specified, extract only from those
	if len(r.TagsToExtract) > 0 {
		var extracted strings.Builder
		for _, tag := range r.TagsToExtract {
			pattern := fmt.Sprintf(`(?is)<%s[^>]*>(.*?)</%s>`, tag, tag)
			regex := regexp.MustCompile(pattern)
			matches := regex.FindAllStringSubmatch(text, -1)
			for _, match := range matches {
				if len(match) > 1 {
					extracted.WriteString(match[1])
					extracted.WriteString("\n")
				}
			}
		}
		text = extracted.String()
	} else {
		// Extract body content if present
		bodyRegex := regexp.MustCompile(`(?is)<body[^>]*>(.*?)</body>`)
		if matches := bodyRegex.FindStringSubmatch(text); len(matches) > 1 {
			text = matches[1]
		}
	}

	// Replace common block elements with newlines
	blockTags := []string{"div", "p", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre"}
	for _, tag := range blockTags {
		// Opening tags
		openPattern := fmt.Sprintf(`(?i)<%s[^>]*>`, tag)
		openRegex := regexp.MustCompile(openPattern)
		text = openRegex.ReplaceAllString(text, "\n")

		// Closing tags
		closePattern := fmt.Sprintf(`(?i)</%s>`, tag)
		closeRegex := regexp.MustCompile(closePattern)
		text = closeRegex.ReplaceAllString(text, "\n")
	}

	// Remove all remaining HTML tags
	tagRegex := regexp.MustCompile(`<[^>]+>`)
	text = tagRegex.ReplaceAllString(text, "")

	// Decode common HTML entities
	text = decodeHTMLEntities(text)

	// Clean up whitespace
	if !r.PreserveWhitespace {
		// Replace multiple spaces with single space
		spaceRegex := regexp.MustCompile(`[ \t]+`)
		text = spaceRegex.ReplaceAllString(text, " ")

		// Replace multiple newlines with double newline
		newlineRegex := regexp.MustCompile(`\n\s*\n+`)
		text = newlineRegex.ReplaceAllString(text, "\n\n")
	}

	return strings.TrimSpace(text)
}

// decodeHTMLEntities decodes common HTML entities.
func decodeHTMLEntities(text string) string {
	entities := map[string]string{
		"&nbsp;":   " ",
		"&amp;":    "&",
		"&lt;":     "<",
		"&gt;":     ">",
		"&quot;":   `"`,
		"&apos;":   "'",
		"&#39;":    "'",
		"&mdash;":  "—",
		"&ndash;":  "–",
		"&copy;":   "©",
		"&reg;":    "®",
		"&trade;":  "™",
		"&hellip;": "…",
		"&lsquo;":  "'",
		"&rsquo;":  "'",
		"&ldquo;":  "\u201C",
		"&rdquo;":  "\u201D",
		"&bull;":   "•",
	}

	for entity, replacement := range entities {
		text = strings.ReplaceAll(text, entity, replacement)
	}

	// Handle numeric entities (&#NNN;)
	numericRegex := regexp.MustCompile(`&#(\d+);`)
	text = numericRegex.ReplaceAllStringFunc(text, func(match string) string {
		var num int
		fmt.Sscanf(match, "&#%d;", &num)
		if num > 0 && num < 0x10FFFF {
			return string(rune(num))
		}
		return match
	})

	return text
}
