package textsplitter

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/suite"
)

type fixedSentenceSplitter struct {
	out []string
}

func (f *fixedSentenceSplitter) Split(text string) []string {
	return f.out
}

type seqEmbeddingMock struct {
	vecs [][]float32
	i    int
	err  error
}

func (m *seqEmbeddingMock) GetTextEmbedding(_ context.Context, _ string) ([]float32, error) {
	if m.err != nil {
		return nil, m.err
	}
	if m.i >= len(m.vecs) {
		return []float32{0, 0, 1}, nil
	}
	v := m.vecs[m.i]
	m.i++
	return v, nil
}

func (m *seqEmbeddingMock) GetQueryEmbedding(ctx context.Context, text string) ([]float32, error) {
	return m.GetTextEmbedding(ctx, text)
}

type SemanticSplitterTestSuite struct {
	suite.Suite
}

func TestSemanticSplitterTestSuite(t *testing.T) {
	suite.Run(t, new(SemanticSplitterTestSuite))
}

func (s *SemanticSplitterTestSuite) TestBuildCombinedSentenceGroups_Buffer() {
	sent := []string{"a", "b", "c", "d"}
	got := buildCombinedSentenceGroups(sent, 1)
	s.Require().Len(got, 4)
	s.Equal("ab", got[0])
	s.Equal("abc", got[1])
	s.Equal("bcd", got[2])
	s.Equal("cd", got[3])
}

func (s *SemanticSplitterTestSuite) TestPairwiseDissimilarities() {
	embs := [][]float32{
		{1, 0, 0},
		{1, 0, 0},
		{0, 1, 0},
	}
	d, err := pairwiseDissimilarities(embs)
	s.NoError(err)
	s.Require().Len(d, 2)
	s.InDelta(0, d[0], 1e-6)
	s.InDelta(1, d[1], 1e-6)
}

func (s *SemanticSplitterTestSuite) TestPairwiseDissimilarities_CosineError() {
	embs := [][]float32{{1}, {1, 0}}
	d, err := pairwiseDissimilarities(embs)
	s.Error(err)
	s.Nil(d)
}

func (s *SemanticSplitterTestSuite) TestPercentileLinear() {
	s.InDelta(0.5, percentileLinear([]float64{0.1, 0.9}, 50), 1e-9)
	s.Equal(0.1, percentileLinear([]float64{0.1, 0.9}, 0))
	s.Equal(0.9, percentileLinear([]float64{0.1, 0.9}, 100))
}

func (s *SemanticSplitterTestSuite) TestBuildSemanticChunksFromDistances_SingleSentenceFallback() {
	ch := buildSemanticChunksFromDistances([]string{"only"}, nil, 95)
	s.Require().Len(ch, 1)
	s.Equal("only", ch[0])
}

func (s *SemanticSplitterTestSuite) TestSplitText_GroupsByMockDistances() {
	sent := []string{"x", "y", "z"}
	mock := &seqEmbeddingMock{
		vecs: [][]float32{
			{1, 0, 0},
			{1, 0, 0},
			{0, 1, 0},
		},
	}
	sp := NewSemanticSplitterNodeParser(mock, 1, 50, &fixedSentenceSplitter{out: sent})
	chunks := sp.SplitText("ignored")
	s.Require().Len(chunks, 2)
	s.Equal("xy", chunks[0])
	s.Equal("z", chunks[1])
}

func (s *SemanticSplitterTestSuite) TestBuildSemanticChunks_PercentileAffectsBreakpoints() {
	ch100 := buildSemanticChunksFromDistances([]string{"a", "b", "c"}, []float64{0.01, 0.99}, 100)
	s.Require().Len(ch100, 1)
	s.Equal("abc", ch100[0])
	ch50 := buildSemanticChunksFromDistances([]string{"a", "b", "c"}, []float64{0.01, 0.99}, 50)
	s.Require().Len(ch50, 2)
	s.Equal("ab", ch50[0])
	s.Equal("c", ch50[1])
}

func (s *SemanticSplitterTestSuite) TestSplitText_Empty() {
	sp := NewSemanticSplitterNodeParser(&seqEmbeddingMock{}, 1, 95, &fixedSentenceSplitter{out: []string{}})
	s.Empty(sp.SplitText(""))
}

func (s *SemanticSplitterTestSuite) TestSplitText_SingleSentence() {
	sp := NewSemanticSplitterNodeParser(&seqEmbeddingMock{}, 1, 95, &fixedSentenceSplitter{out: []string{"Hello."}})
	ch := sp.SplitText("Hello.")
	s.Require().Len(ch, 1)
	s.Contains(ch[0], "Hello")
}

func (s *SemanticSplitterTestSuite) TestSplitText_EmbeddingError() {
	sp := NewSemanticSplitterNodeParser(&seqEmbeddingMock{err: errors.New("embed failed")}, 1, 95, &fixedSentenceSplitter{out: []string{"a", "b"}})
	s.Nil(sp.SplitText("x"))
}

func (s *SemanticSplitterTestSuite) TestCosineSimilarityFloat32_Orthogonal() {
	s0, err := cosineSimilarityFloat32([]float32{1, 0}, []float32{0, 1})
	s.NoError(err)
	s.InDelta(0, s0, 1e-6)
	s1, err := cosineSimilarityFloat32([]float32{3, 4}, []float32{6, 8})
	s.NoError(err)
	s.InDelta(1, s1, 1e-6)
}

func (s *SemanticSplitterTestSuite) TestSplitText_PairwiseSimilarityFailure() {
	mock := &seqEmbeddingMock{
		vecs: [][]float32{
			{1, 0},
			{1},
			{0, 1},
		},
	}
	sp := NewSemanticSplitterNodeParser(mock, 1, 50, &fixedSentenceSplitter{out: []string{"a", "b", "c"}})
	s.Nil(sp.SplitText("ignored"))
}
