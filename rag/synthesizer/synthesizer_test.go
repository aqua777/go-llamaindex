package synthesizer

import (
	"context"
	"testing"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func createTestNodes() []schema.NodeWithScore {
	node1 := schema.NewTextNode("The capital of France is Paris.")
	node1.ID = "node1"
	node2 := schema.NewTextNode("Paris is known for the Eiffel Tower.")
	node2.ID = "node2"

	return []schema.NodeWithScore{
		{Node: *node1, Score: 0.9},
		{Node: *node2, Score: 0.8},
	}
}

func TestResponseMode(t *testing.T) {
	assert.Equal(t, "refine", ResponseModeRefine.String())
	assert.Equal(t, "compact", ResponseModeCompact.String())
	assert.Equal(t, "tree_summarize", ResponseModeTreeSummarize.String())
	assert.Equal(t, "simple_summarize", ResponseModeSimpleSummarize.String())
	assert.Equal(t, "accumulate", ResponseModeAccumulate.String())

	assert.True(t, ResponseModeRefine.IsValid())
	assert.True(t, ResponseModeCompact.IsValid())
	assert.False(t, ResponseMode("invalid").IsValid())
}

func TestResponse(t *testing.T) {
	nodes := createTestNodes()
	resp := NewResponse("Test response", nodes)

	assert.Equal(t, "Test response", resp.String())
	assert.Len(t, resp.SourceNodes, 2)

	formatted := resp.GetFormattedSources(50)
	assert.Contains(t, formatted, "node1")
	assert.Contains(t, formatted, "node2")
}

func TestResponseWithMetadata(t *testing.T) {
	nodes := createTestNodes()
	metadata := map[string]interface{}{"key": "value"}
	resp := NewResponseWithMetadata("Test", nodes, metadata)

	assert.Equal(t, "value", resp.Metadata["key"])
}

func TestEmptyResponse(t *testing.T) {
	resp := NewResponse("", nil)
	assert.Equal(t, "None", resp.String())
}

func TestStreamingResponse(t *testing.T) {
	ch := make(chan string, 3)
	ch <- "Hello"
	ch <- " "
	ch <- "World"
	close(ch)

	nodes := createTestNodes()
	sr := NewStreamingResponse(ch, nodes)

	assert.Equal(t, "Hello World", sr.String())

	// Second call should return cached value
	assert.Equal(t, "Hello World", sr.String())
}

func TestStreamingResponseGetResponse(t *testing.T) {
	ch := make(chan string, 2)
	ch <- "Test"
	ch <- " Response"
	close(ch)

	sr := NewStreamingResponse(ch, nil)
	resp := sr.GetResponse()

	assert.Equal(t, "Test Response", resp.Response)
}

func TestCompactTextChunks(t *testing.T) {
	chunks := []string{"chunk1", "chunk2", "chunk3", "chunk4"}

	// With large max size, should combine all
	compacted := CompactTextChunks(chunks, 1000, "\n")
	assert.Len(t, compacted, 1)
	assert.Contains(t, compacted[0], "chunk1")
	assert.Contains(t, compacted[0], "chunk4")

	// With small max size, should keep separate
	compacted = CompactTextChunks(chunks, 10, "\n")
	assert.Len(t, compacted, 4)
}

func TestCompactTextChunksEmpty(t *testing.T) {
	compacted := CompactTextChunks([]string{}, 100, "\n")
	assert.Len(t, compacted, 0)
}

func TestGetTextChunksFromNodes(t *testing.T) {
	nodes := createTestNodes()
	chunks := GetTextChunksFromNodes(nodes, schema.MetadataModeLLM)

	assert.Len(t, chunks, 2)
	assert.Contains(t, chunks[0], "France")
	assert.Contains(t, chunks[1], "Eiffel")
}

func TestSimpleSynthesizer(t *testing.T) {
	mockLLM := llm.NewMockLLM("Paris is the capital of France.")
	ss := NewSimpleSynthesizer(mockLLM)

	ctx := context.Background()
	nodes := createTestNodes()

	resp, err := ss.Synthesize(ctx, "What is the capital of France?", nodes)
	require.NoError(t, err)

	assert.Equal(t, "Paris is the capital of France.", resp.Response)
	assert.Len(t, resp.SourceNodes, 2)
}

func TestSimpleSynthesizerEmptyNodes(t *testing.T) {
	mockLLM := llm.NewMockLLM("")
	ss := NewSimpleSynthesizer(mockLLM)

	ctx := context.Background()
	resp, err := ss.Synthesize(ctx, "test", nil)
	require.NoError(t, err)

	assert.Equal(t, "Empty Response", resp.Response)
}

func TestRefineSynthesizer(t *testing.T) {
	mockLLM := llm.NewMockLLM("Refined answer about Paris.")
	rs := NewRefineSynthesizer(mockLLM)

	ctx := context.Background()
	nodes := createTestNodes()

	resp, err := rs.Synthesize(ctx, "Tell me about Paris", nodes)
	require.NoError(t, err)

	assert.NotEmpty(t, resp.Response)
	assert.Len(t, resp.SourceNodes, 2)
}

func TestRefineSynthesizerGetResponse(t *testing.T) {
	mockLLM := llm.NewMockLLM("Answer")
	rs := NewRefineSynthesizer(mockLLM)

	ctx := context.Background()
	chunks := []string{"chunk1", "chunk2", "chunk3"}

	resp, err := rs.GetResponse(ctx, "query", chunks)
	require.NoError(t, err)

	assert.Equal(t, "Answer", resp)
}

func TestCompactAndRefineSynthesizer(t *testing.T) {
	mockLLM := llm.NewMockLLM("Compacted and refined answer.")
	cs := NewCompactAndRefineSynthesizer(mockLLM, WithMaxChunkSize(100))

	ctx := context.Background()
	nodes := createTestNodes()

	resp, err := cs.Synthesize(ctx, "test query", nodes)
	require.NoError(t, err)

	assert.NotEmpty(t, resp.Response)
}

func TestTreeSummarizeSynthesizer(t *testing.T) {
	mockLLM := llm.NewMockLLM("Tree summarized answer.")
	ts := NewTreeSummarizeSynthesizer(mockLLM)

	ctx := context.Background()
	nodes := createTestNodes()

	resp, err := ts.Synthesize(ctx, "Summarize", nodes)
	require.NoError(t, err)

	assert.NotEmpty(t, resp.Response)
}

func TestTreeSummarizeSynthesizerRecursive(t *testing.T) {
	mockLLM := llm.NewMockLLM("Summary")
	ts := NewTreeSummarizeSynthesizer(mockLLM, WithTreeMaxChunkSize(20))

	ctx := context.Background()
	chunks := []string{
		"First long chunk of text",
		"Second long chunk of text",
		"Third long chunk of text",
	}

	resp, err := ts.GetResponse(ctx, "query", chunks)
	require.NoError(t, err)

	assert.Equal(t, "Summary", resp)
}

func TestAccumulateSynthesizer(t *testing.T) {
	mockLLM := llm.NewMockLLM("Accumulated answer.")
	as := NewAccumulateSynthesizer(mockLLM)

	ctx := context.Background()
	nodes := createTestNodes()

	resp, err := as.Synthesize(ctx, "test", nodes)
	require.NoError(t, err)

	// Should have two accumulated responses
	assert.Contains(t, resp.Response, "Accumulated answer.")
}

func TestCompactAccumulateSynthesizer(t *testing.T) {
	mockLLM := llm.NewMockLLM("Compact accumulated.")
	cas := NewCompactAccumulateSynthesizer(mockLLM)

	ctx := context.Background()
	nodes := createTestNodes()

	resp, err := cas.Synthesize(ctx, "test", nodes)
	require.NoError(t, err)

	assert.NotEmpty(t, resp.Response)
}

func TestGetSynthesizer(t *testing.T) {
	mockLLM := llm.NewMockLLM("")

	tests := []struct {
		mode     ResponseMode
		wantType string
	}{
		{ResponseModeSimpleSummarize, "*synthesizer.SimpleSynthesizer"},
		{ResponseModeRefine, "*synthesizer.RefineSynthesizer"},
		{ResponseModeCompact, "*synthesizer.CompactAndRefineSynthesizer"},
		{ResponseModeTreeSummarize, "*synthesizer.TreeSummarizeSynthesizer"},
		{ResponseModeAccumulate, "*synthesizer.AccumulateSynthesizer"},
		{ResponseModeCompactAccumulate, "*synthesizer.CompactAccumulateSynthesizer"},
	}

	for _, tt := range tests {
		synth, err := GetSynthesizer(tt.mode, mockLLM)
		require.NoError(t, err)
		assert.NotNil(t, synth)
	}
}

func TestGetSynthesizerInvalid(t *testing.T) {
	mockLLM := llm.NewMockLLM("")
	_, err := GetSynthesizer(ResponseMode("invalid"), mockLLM)
	assert.Error(t, err)
}

func TestBaseSynthesizerOptions(t *testing.T) {
	mockLLM := llm.NewMockLLM("")
	bs := NewBaseSynthesizerWithOptions(mockLLM,
		WithStreaming(true),
		WithSynthesizerVerbose(true),
	)

	assert.True(t, bs.Streaming)
	assert.True(t, bs.Verbose)
}

func TestBaseSynthesizerGetMetadata(t *testing.T) {
	mockLLM := llm.NewMockLLM("")
	bs := NewBaseSynthesizer(mockLLM)

	nodes := createTestNodes()
	metadata := bs.GetMetadataForResponse(nodes)

	assert.Contains(t, metadata, "node1")
	assert.Contains(t, metadata, "node2")
}
