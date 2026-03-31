package textsplitter

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type CodeSplitterTestSuite struct {
	suite.Suite
}

func TestCodeSplitterTestSuite(t *testing.T) {
	suite.Run(t, new(CodeSplitterTestSuite))
}

func (s *CodeSplitterTestSuite) TestSplitText_RespectsMaxCharsPerChunk() {
	text := strings.Repeat("a\n", 50)
	text = strings.TrimSuffix(text, "\n")
	splitter := NewCodeSplitter("go", 10, 2, 20)
	chunks := splitter.SplitText(text)
	s.Require().NotEmpty(chunks)
	for _, c := range chunks {
		s.LessOrEqual(len([]rune(c)), 20, "chunk: %q", c)
	}
}

func (s *CodeSplitterTestSuite) TestSplitText_OverlapBetweenChunks() {
	lines := []string{
		"line0", "line1", "line2", "line3", "line4", "line5",
	}
	text := strings.Join(lines, "\n")
	splitter := NewCodeSplitter("python", 3, 1, 500)
	chunks := splitter.SplitText(text)
	s.Require().GreaterOrEqual(len(chunks), 2)
	s.True(strings.Contains(chunks[0], "line2"))
	s.True(strings.Contains(chunks[1], "line2"), "overlap should repeat boundary line")
}

func (s *CodeSplitterTestSuite) TestSplitText_SmallerThanChunkSize() {
	text := "func main() {}"
	splitter := NewCodeSplitter("go", 40, 15, 1500)
	chunks := splitter.SplitText(text)
	s.Require().Len(chunks, 1)
	s.Equal(text, chunks[0])
}

func (s *CodeSplitterTestSuite) TestSplitText_Empty() {
	splitter := NewCodeSplitter("go", 10, 2, 100)
	s.Empty(splitter.SplitText(""))
}

func (s *CodeSplitterTestSuite) TestSplitText_GoUsesDeclBoundaries() {
	src := `package p

func A() {
}

func B() {
}
`
	splitter := NewCodeSplitter("go", 100, 0, 5000)
	chunks := splitter.SplitText(src)
	s.Require().GreaterOrEqual(len(chunks), 2)
	s.Contains(chunks[0], "func A")
	s.NotContains(chunks[0], "func B")
	s.Contains(chunks[1], "func B")
}

func (s *CodeSplitterTestSuite) TestSplitText_LongSingleLineSplit() {
	line := strings.Repeat("x", 100)
	splitter := NewCodeSplitter("go", 5, 1, 30)
	chunks := splitter.SplitText(line)
	s.Require().NotEmpty(chunks)
	for _, c := range chunks {
		s.LessOrEqual(len([]rune(c)), 30)
	}
	s.Equal(line, strings.Join(chunks, ""))
}

func TestNormalizeCodeSplitterParams_OverlapClamp(t *testing.T) {
	cl, co, mc := normalizeCodeSplitterParams(5, 10, 100)
	require.Equal(t, 5, cl)
	require.Equal(t, 4, co)
	require.Equal(t, 100, mc)
}

func TestNormalizeCodeSplitterParams_ZeroOverlapPreserved(t *testing.T) {
	_, co, _ := normalizeCodeSplitterParams(40, 0, 1500)
	require.Equal(t, 0, co)
}

func TestSplitStringByMaxRunes(t *testing.T) {
	parts := splitStringByMaxRunes("abcdef", 2)
	assert.Equal(t, []string{"ab", "cd", "ef"}, parts)
}

func TestNextChunkStart(t *testing.T) {
	assert.Equal(t, 9, nextChunkStart(0, 10, 1, 20))
	assert.Equal(t, 3, nextChunkStart(0, 4, 1, 20))
	assert.Equal(t, 4, nextChunkStart(0, 4, 0, 20))
}

func TestJoinLineRange(t *testing.T) {
	lines := []string{"a", "b", "c"}
	assert.Equal(t, "a\nb", joinLineRange(lines, 0, 2))
	assert.Equal(t, "", joinLineRange(lines, 1, 1))
}
