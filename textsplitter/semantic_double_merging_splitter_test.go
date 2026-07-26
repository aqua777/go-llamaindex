package textsplitter

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"
)

// Package-level similarity helpers for createInitialChunks / mergeInitialChunks tests (not closures).

func simCreateInitialChunksAlphaBetaGamma(a, b string) float64 {
	if strings.Contains(a, "alpha") && strings.Contains(b, "beta") {
		return 0.9
	}
	if strings.Contains(a, "beta") && strings.Contains(b, "gamma") {
		return 0.85
	}
	return 0.1
}

func simAlwaysLowSimilarity(_, _ string) float64 { return 0.1 }

func simMergeAdjacentAB(a, b string) float64 {
	if a == "A" && b == "B" {
		return 0.9
	}
	return 0.1
}

func simMergeFirstThirdAC(a, b string) float64 {
	if a == "A" && b == "C" {
		return 0.9
	}
	return 0.1
}

func simMergeCurrentWithD(a, b string) float64 {
	if a == "A" && b == "D" {
		return 0.9
	}
	return 0.1
}

func simAlwaysOne(_, _ string) float64 { return 1.0 }

type SemanticDoubleMergingTestSuite struct {
	suite.Suite
}

func TestSemanticDoubleMergingTestSuite(t *testing.T) {
	suite.Run(t, new(SemanticDoubleMergingTestSuite))
}

func (s *SemanticDoubleMergingTestSuite) TestValidateLanguageConfig_OK() {
	s.NoError(ValidateLanguageConfig(LanguageConfig{Language: "english", SpacyModel: "en_core_web_md"}))
	s.NoError(ValidateLanguageConfig(LanguageConfig{Language: "german", SpacyModel: "de_core_news_lg"}))
}

func (s *SemanticDoubleMergingTestSuite) TestValidateLanguageConfig_Errors() {
	s.Error(ValidateLanguageConfig(LanguageConfig{Language: "polish", SpacyModel: "en_core_web_md"}))
	s.Error(ValidateLanguageConfig(LanguageConfig{Language: "english", SpacyModel: "de_core_news_md"}))
}

func (s *SemanticDoubleMergingTestSuite) TestCreateInitialChunks_InitializeAndAppend() {
	sep := " "
	sent := []string{"alpha one", "beta two", "gamma three"}
	out := createInitialChunks(sent, 0.6, 0.8, 500, sep, simCreateInitialChunksAlphaBetaGamma)
	s.Require().NotEmpty(out)
	s.True(len(out) < len(sent) || strings.Contains(out[0], "beta"))
}

func (s *SemanticDoubleMergingTestSuite) TestCreateInitialChunks_SplitWhenLowSimilarity() {
	sep := " "
	sent := []string{"a", "b", "c", "d"}
	out := createInitialChunks(sent, 0.6, 0.8, 1000, sep, simAlwaysLowSimilarity)
	s.Len(out, 4)
}

func (s *SemanticDoubleMergingTestSuite) TestMergeInitialChunks_AdjacentMerge() {
	sep := " "
	init := []string{"A", "B", "C"}
	out := mergeInitialChunks(init, 0.7, 1000, 1, sep, simMergeAdjacentAB)
	s.Require().Len(out, 2)
	s.Contains(out[0], "A")
	s.Contains(out[0], "B")
}

func (s *SemanticDoubleMergingTestSuite) TestMergeInitialChunks_MergeFirstAndThirdSkipsSecond() {
	sep := " "
	init := []string{"A", "B", "C"}
	out := mergeInitialChunks(init, 0.7, 1000, 1, sep, simMergeFirstThirdAC)
	s.Require().Len(out, 1)
	s.Equal("A B C", out[0])
}

func (s *SemanticDoubleMergingTestSuite) TestMergeInitialChunks_RangeTwoMergesFour() {
	sep := " "
	init := []string{"A", "B", "C", "D"}
	out := mergeInitialChunks(init, 0.7, 1000, 2, sep, simMergeCurrentWithD)
	s.Require().Len(out, 1)
	s.Equal("A B C D", out[0])
}

func (s *SemanticDoubleMergingTestSuite) TestMergeInitialChunks_RangeOneSkipsFourth() {
	sep := " "
	init := []string{"A", "B", "C", "D"}
	out := mergeInitialChunks(init, 0.7, 1000, 1, sep, simMergeCurrentWithD)
	s.GreaterOrEqual(len(out), 2)
}

func (s *SemanticDoubleMergingTestSuite) TestMergeInitialChunks_MaxChunkSize() {
	sep := " "
	init := []string{"xx", "yy", "zz"}
	out := mergeInitialChunks(init, 0.9, 3, 1, sep, simAlwaysOne)
	for _, c := range out {
		s.LessOrEqual(len(c), 3)
	}
}

func (s *SemanticDoubleMergingTestSuite) TestSplitText_Empty() {
	sp := NewSemanticDoubleMergingSplitter(
		LanguageConfig{Language: "english", SpacyModel: "en_core_web_md"},
		0.6, 0.8, 0.8, 1000, 1, " ", NewRegexSplitterStrategy(DefaultChunkingRegex),
	)
	s.Empty(sp.SplitText("   "))
}

func (s *SemanticDoubleMergingTestSuite) TestSplitText_InvalidConfig() {
	sp := NewSemanticDoubleMergingSplitter(
		LanguageConfig{Language: "french", SpacyModel: "en_core_web_md"},
		0.6, 0.8, 0.8, 1000, 1, " ", NewRegexSplitterStrategy(DefaultChunkingRegex),
	)
	s.Empty(sp.SplitText("Hello. World."))
}

func (s *SemanticDoubleMergingTestSuite) TestCleanTextAdvanced_RemovesURLAndStopwords() {
	sw := englishStopwordSet()
	got := cleanTextAdvanced("The quick http://x.com brown fox", sw)
	s.NotContains(got, "http")
	s.NotContains(got, "the")
	s.Contains(got, "quick")
}

func (s *SemanticDoubleMergingTestSuite) TestJaccardSimilarityOnTokens() {
	s.InDelta(1.0, jaccardSimilarityOnTokens("a b c", "a b c"), 1e-9)
	s.InDelta(0.0, jaccardSimilarityOnTokens("a", "z"), 1e-9)
}

func TestSemanticDoubleMergingSplitter_TextSplitterInterface(t *testing.T) {
	var _ TextSplitter = &SemanticDoubleMergingSplitter{}
}
