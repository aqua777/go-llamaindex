package textsplitter

import (
	"go/parser"
	"go/token"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"
)

const (
	defaultCodeChunkLines        = 40
	defaultCodeChunkLinesOverlap = 15
	defaultCodeMaxChars          = 1500
)

// CodeSplitter splits code using language-specific rules: Go uses the Go parser
// (top-level declaration boundaries); Python uses top-level def/class/async def
// line detection; other languages use line windows with overlap and max size.
type CodeSplitter struct {
	Language          string
	ChunkLines        int
	ChunkLinesOverlap int
	MaxChars          int
}

// NewCodeSplitter creates a new CodeSplitter.
//
// Args:
//
//	language: The programming language of the code being split.
//	chunkLines: The number of lines to include in each chunk.
//	chunkLinesOverlap: How many lines of code each chunk overlaps with.
//	maxChars: Maximum number of characters (Unicode code points) per chunk.
//
// Returns:
//
//	A pointer to the newly created CodeSplitter.
//
// Non-positive chunkLines or maxChars are replaced with defaults (40 lines, 1500 code points).
// Negative chunkLinesOverlap is replaced with the default (15). Zero overlap means no overlap.
// Positive overlap is clamped to be less than the effective chunk line count.
func NewCodeSplitter(language string, chunkLines int, chunkLinesOverlap int, maxChars int) *CodeSplitter {
	cl, co, mc := normalizeCodeSplitterParams(chunkLines, chunkLinesOverlap, maxChars)
	return &CodeSplitter{
		Language:          language,
		ChunkLines:        cl,
		ChunkLinesOverlap: co,
		MaxChars:          mc,
	}
}

// SplitText splits the provided code string into chunks.
//
// Args:
//
//	text: The code string to split.
//
// Returns:
//
//	A slice of code chunks.
func (s *CodeSplitter) SplitText(text string) []string {
	if text == "" {
		return []string{}
	}
	lines := splitCodeLines(text)
	lang := normalizeCodeLang(s.Language)
	declStarts, declEnds := languageBoundaries(text, lines, lang)
	return buildCodeChunks(lines, s.ChunkLines, s.ChunkLinesOverlap, s.MaxChars, declStarts, declEnds)
}

func normalizeCodeLang(language string) string {
	lang := strings.ToLower(strings.TrimSpace(language))
	switch lang {
	case "golang":
		return "go"
	case "py":
		return "python"
	default:
		return lang
	}
}

// languageBoundaries returns declaration start indices (for snapping chunk starts after overlap)
// and exclusive declaration end indices (for preferring chunk ends that align with complete declarations).
// Either may be nil for generic line-only chunking.
func languageBoundaries(fullText string, lines []string, lang string) (declStarts []int, declEnds []int) {
	switch lang {
	case "go":
		return goDeclBoundaries(fullText)
	case "python":
		return pythonDeclStartLines(lines), nil
	default:
		return nil, nil
	}
}

// goDeclBoundaries uses the Go parser: starts from declaration positions, ends from End()
// (1-based line of end token equals exclusive 0-based line index after the declaration).
func goDeclBoundaries(src string) ([]int, []int) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", src, parser.ParseComments|parser.SkipObjectResolution)
	if err != nil || f == nil {
		return nil, nil
	}
	seen := map[int]struct{}{0: {}}
	var ends []int
	for _, d := range f.Decls {
		sl := fset.Position(d.Pos()).Line - 1
		if sl >= 0 {
			seen[sl] = struct{}{}
		}
		el := fset.Position(d.End()).Line
		if el > 0 {
			ends = append(ends, el)
		}
	}
	starts := make([]int, 0, len(seen))
	for l := range seen {
		starts = append(starts, l)
	}
	sort.Ints(starts)
	sort.Ints(ends)
	return starts, ends
}

var pythonTopLevelDecl = regexp.MustCompile(`^(class\s+\w|def\s+\w|async\s+def\s+\w)`)

func pythonDeclStartLines(lines []string) []int {
	seen := map[int]struct{}{0: {}}
	for i, line := range lines {
		if len(line) > 0 && (line[0] == ' ' || line[0] == '\t') {
			continue
		}
		if pythonTopLevelDecl.MatchString(strings.TrimRight(line, " \t\r")) {
			seen[i] = struct{}{}
		}
	}
	out := make([]int, 0, len(seen))
	for l := range seen {
		out = append(out, l)
	}
	sort.Ints(out)
	return out
}

func normalizeCodeSplitterParams(chunkLines, chunkLinesOverlap, maxChars int) (int, int, int) {
	if chunkLines <= 0 {
		chunkLines = defaultCodeChunkLines
	}
	if chunkLinesOverlap < 0 {
		chunkLinesOverlap = defaultCodeChunkLinesOverlap
	}
	if maxChars <= 0 {
		maxChars = defaultCodeMaxChars
	}
	if chunkLinesOverlap > 0 && chunkLinesOverlap >= chunkLines {
		chunkLinesOverlap = chunkLines - 1
		if chunkLinesOverlap < 0 {
			chunkLinesOverlap = 0
		}
	}
	return chunkLines, chunkLinesOverlap, maxChars
}

func splitCodeLines(text string) []string {
	return strings.Split(text, "\n")
}

func joinLineRange(lines []string, start, end int) string {
	if start >= end {
		return ""
	}
	return strings.Join(lines[start:end], "\n")
}

func nextChunkStart(start, end, overlap, lineCount int) int {
	if end >= lineCount {
		return lineCount
	}
	if overlap <= 0 {
		return end
	}
	next := end - overlap
	if next <= start {
		return start + 1
	}
	return next
}

// firstDeclStartAtOrAfter returns the smallest break >= i, or -1 if none.
func firstDeclStartAtOrAfter(breaks []int, i int) int {
	for _, b := range breaks {
		if b >= i {
			return b
		}
	}
	return -1
}

// snapChunkStart aligns i to a declaration start when language rules apply.
func snapChunkStart(i int, breaks []int) int {
	if len(breaks) == 0 {
		return i
	}
	if j := firstDeclStartAtOrAfter(breaks, i); j >= 0 {
		return j
	}
	return i
}

// splitStringByMaxRunes splits s into segments each with at most maxRunes Unicode code points.
func splitStringByMaxRunes(s string, maxRunes int) []string {
	if maxRunes <= 0 {
		return []string{s}
	}
	runes := []rune(s)
	if len(runes) <= maxRunes {
		return []string{s}
	}
	var out []string
	for len(runes) > 0 {
		n := maxRunes
		if n > len(runes) {
			n = len(runes)
		}
		out = append(out, string(runes[:n]))
		runes = runes[n:]
	}
	return out
}

func buildCodeChunks(lines []string, chunkLines, overlap, maxChars int, declStarts []int, declEnds []int) []string {
	n := len(lines)
	if n == 0 {
		return nil
	}
	var out []string
	i := 0
	if len(declStarts) > 0 {
		i = snapChunkStart(0, declStarts)
	}
	for i < n {
		if utf8.RuneCountInString(lines[i]) > maxChars {
			out = append(out, splitStringByMaxRunes(lines[i], maxChars)...)
			i++
			if len(declStarts) > 0 {
				i = snapChunkStart(i, declStarts)
			}
			continue
		}
		maxEnd := i + chunkLines
		if maxEnd > n {
			maxEnd = n
		}
		end := chooseChunkEnd(lines, i, maxEnd, maxChars, declEnds)
		if end == i {
			out = append(out, splitStringByMaxRunes(lines[i], maxChars)...)
			i++
			if len(declStarts) > 0 {
				i = snapChunkStart(i, declStarts)
			}
			continue
		}
		out = append(out, joinLineRange(lines, i, end))
		if end >= n {
			break
		}
		next := nextChunkStart(i, end, overlap, n)
		if len(declStarts) > 0 {
			next = snapChunkStart(next, declStarts)
		}
		i = next
	}
	return out
}

// chooseChunkEnd picks an exclusive end index in (i, maxEnd] using maxChars and optional AST decl end lines.
func chooseChunkEnd(lines []string, i, maxEnd, maxChars int, declEnds []int) int {
	end := maxEnd
	if len(declEnds) > 0 {
		best := -1
		for _, pe := range declEnds {
			if pe > i && pe <= maxEnd {
				if utf8.RuneCountInString(joinLineRange(lines, i, pe)) <= maxChars {
					if best < 0 || pe < best {
						best = pe
					}
				}
			}
		}
		if best > i {
			end = best
		}
	}
	for end > i && utf8.RuneCountInString(joinLineRange(lines, i, end)) > maxChars {
		end--
	}
	return end
}
