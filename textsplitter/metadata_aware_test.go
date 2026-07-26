package textsplitter

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"
)

type MetadataAwareTestSuite struct {
	suite.Suite
}

func TestMetadataAwareTestSuite(t *testing.T) {
	suite.Run(t, new(MetadataAwareTestSuite))
}

func (s *MetadataAwareTestSuite) TestCompileTime_SentenceSplitterIsMetadataAware() {
	var _ MetadataAwareTextSplitter = (*SentenceSplitter)(nil)
}

func (s *MetadataAwareTestSuite) TestCompileTime_TokenTextSplitterIsMetadataAware() {
	var _ MetadataAwareTextSplitter = (*TokenTextSplitter)(nil)
}

func (s *MetadataAwareTestSuite) TestCompileTime_MarkdownSplitterIsMetadataAware() {
	var _ MetadataAwareTextSplitter = (*MarkdownSplitter)(nil)
}

func (s *MetadataAwareTestSuite) TestMetadataTokenCount_NilTokenizer() {
	s.Equal(0, MetadataTokenCount(nil, "hello world"))
}

func (s *MetadataAwareTestSuite) TestMetadataTokenCount_SimpleTokenizer() {
	tok := NewSimpleTokenizer()
	meta := "one two three"
	s.Equal(3, MetadataTokenCount(tok, meta))
}

func (s *MetadataAwareTestSuite) TestEffectiveChunkSizeAfterMetadata_OK() {
	effective, err := EffectiveChunkSizeAfterMetadata(100, 40)
	s.NoError(err)
	s.Equal(60, effective)
}

func (s *MetadataAwareTestSuite) TestEffectiveChunkSizeAfterMetadata_TooLargeMetadata() {
	_, err := EffectiveChunkSizeAfterMetadata(100, 60)
	s.Error(err)
	s.Contains(err.Error(), "metadata length")
}

func (s *MetadataAwareTestSuite) TestEffectiveChunkSizeAfterMetadata_NegativeMetadataCountClamped() {
	effective, err := EffectiveChunkSizeAfterMetadata(100, -1)
	s.NoError(err)
	s.Equal(100, effective)
}

func (s *MetadataAwareTestSuite) TestSplitTextMetadataAware_AtLeastAsManyChunksAsPlainSplit() {
	// Word tokens; long text so smaller effective chunk yields more chunks.
	splitter := NewSentenceSplitter(100, 0, nil, nil)
	meta := strings.Repeat("word ", 30) // 30 tokens; effective 70
	text := strings.Repeat("Hello world. ", 80)

	plain := splitter.SplitText(text)
	aware, err := splitter.SplitTextMetadataAware(text, meta)
	s.NoError(err)
	s.GreaterOrEqual(len(aware), len(plain))
}

func (s *MetadataAwareTestSuite) TestSplitTextMetadataAware_MetadataTooLarge() {
	splitter := NewSentenceSplitter(60, 0, nil, nil)
	meta := strings.Repeat("x ", 25) // 25 tokens -> effective 35 < 50
	_, err := splitter.SplitTextMetadataAware("short text here", meta)
	s.Error(err)
}

func (s *MetadataAwareTestSuite) TestEffectiveChunkSizeForMetadataAwareSplit_OK() {
	tokenizer, _ := NewTikTokenTokenizer("gpt-3.5-turbo")
	// "hello world" is 2 tokens
	effective, err := EffectiveChunkSizeForMetadataAwareSplit(100, tokenizer, "hello world")
	s.NoError(err)
	s.Equal(98, effective)
}

func (s *MetadataAwareTestSuite) TestEffectiveChunkSizeForMetadataAwareSplit_TooLarge() {
	tokenizer, _ := NewTikTokenTokenizer("gpt-3.5-turbo")
	// "hello world" is 2 tokens
	_, err := EffectiveChunkSizeForMetadataAwareSplit(50, tokenizer, "hello world")
	s.Error(err)
	s.Contains(err.Error(), "metadata length")
}
