package queryengine

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/rag/synthesizer"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// MockRetriever for testing.
type MockRetriever struct {
	Nodes []schema.NodeWithScore
	Err   error
}

func (m *MockRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	if m.Err != nil {
		return nil, m.Err
	}
	return m.Nodes, nil
}

// MockQueryEngine for testing.
type MockQueryEngine struct {
	Response  *synthesizer.Response
	Err       error
	CallCount int
}

func (m *MockQueryEngine) Query(ctx context.Context, query string) (*synthesizer.Response, error) {
	m.CallCount++
	if m.Err != nil {
		return nil, m.Err
	}
	return m.Response, nil
}

func createTestNodes() []schema.NodeWithScore {
	node1 := schema.NewTextNode("Test content 1")
	node1.ID = "node1"
	node2 := schema.NewTextNode("Test content 2")
	node2.ID = "node2"

	return []schema.NodeWithScore{
		{Node: *node1, Score: 0.9},
		{Node: *node2, Score: 0.8},
	}
}

func TestBaseQueryEngine(t *testing.T) {
	bqe := NewBaseQueryEngine()
	assert.NotNil(t, bqe.BasePromptMixin)
	assert.False(t, bqe.Verbose)
}

func TestBaseQueryEngineWithOptions(t *testing.T) {
	bqe := NewBaseQueryEngineWithOptions(WithQueryEngineVerbose(true))
	assert.True(t, bqe.Verbose)
}

func TestRetrieverQueryEngine(t *testing.T) {
	ctx := context.Background()

	mockRetriever := &MockRetriever{Nodes: createTestNodes()}
	mockLLM := llm.NewMockLLM("Test response")
	mockSynth := synthesizer.NewSimpleSynthesizer(mockLLM)

	rqe := NewRetrieverQueryEngine(mockRetriever, mockSynth)

	resp, err := rqe.Query(ctx, "test query")
	require.NoError(t, err)
	assert.NotEmpty(t, resp.Response)
}

func TestRetrieverQueryEngineRetrieve(t *testing.T) {
	ctx := context.Background()

	nodes := createTestNodes()
	mockRetriever := &MockRetriever{Nodes: nodes}
	mockLLM := llm.NewMockLLM("")
	mockSynth := synthesizer.NewSimpleSynthesizer(mockLLM)

	rqe := NewRetrieverQueryEngine(mockRetriever, mockSynth)

	retrieved, err := rqe.Retrieve(ctx, schema.QueryBundle{QueryString: "test"})
	require.NoError(t, err)
	assert.Len(t, retrieved, 2)
}

func TestRetrieverQueryEngineSynthesize(t *testing.T) {
	ctx := context.Background()

	mockRetriever := &MockRetriever{}
	mockLLM := llm.NewMockLLM("Synthesized response")
	mockSynth := synthesizer.NewSimpleSynthesizer(mockLLM)

	rqe := NewRetrieverQueryEngine(mockRetriever, mockSynth)

	nodes := createTestNodes()
	resp, err := rqe.Synthesize(ctx, "test query", nodes)
	require.NoError(t, err)
	assert.Equal(t, "Synthesized response", resp.Response)
}

func TestQueryEngineTool(t *testing.T) {
	mockEngine := &MockQueryEngine{Response: &synthesizer.Response{Response: "test"}}
	tool := NewQueryEngineTool(mockEngine, "test_tool", "A test tool")

	assert.Equal(t, "test_tool", tool.Name)
	assert.Equal(t, "A test tool", tool.Description)

	metadata := tool.ToolMetadata()
	assert.Equal(t, "test_tool", metadata.Name)
}

func TestRouterQueryEngineSingle(t *testing.T) {
	ctx := context.Background()

	mockEngine := &MockQueryEngine{
		Response: &synthesizer.Response{Response: "Engine 1 response"},
	}
	tools := []*QueryEngineTool{
		NewQueryEngineTool(mockEngine, "engine1", "First engine"),
	}

	rqe := NewRouterQueryEngine(tools)

	resp, err := rqe.Query(ctx, "test query")
	require.NoError(t, err)
	assert.Equal(t, "Engine 1 response", resp.Response)
}

func TestRouterQueryEngineMulti(t *testing.T) {
	ctx := context.Background()

	mockEngine1 := &MockQueryEngine{
		Response: &synthesizer.Response{Response: "Response 1"},
	}
	mockEngine2 := &MockQueryEngine{
		Response: &synthesizer.Response{Response: "Response 2"},
	}
	tools := []*QueryEngineTool{
		NewQueryEngineTool(mockEngine1, "engine1", "First engine"),
		NewQueryEngineTool(mockEngine2, "engine2", "Second engine"),
	}

	rqe := NewRouterQueryEngine(tools, WithRouterSelector(&MultiSelector{}))

	resp, err := rqe.Query(ctx, "test query")
	require.NoError(t, err)
	// Should contain both responses
	assert.Contains(t, resp.Response, "Response 1")
	assert.Contains(t, resp.Response, "Response 2")
}

func TestRouterQueryEngineNoEngines(t *testing.T) {
	ctx := context.Background()

	rqe := NewRouterQueryEngine([]*QueryEngineTool{})

	_, err := rqe.Query(ctx, "test")
	assert.Error(t, err)
}

func TestTransformQueryEngine(t *testing.T) {
	ctx := context.Background()

	mockEngine := &MockQueryEngine{
		Response: &synthesizer.Response{Response: "Transformed response"},
	}

	tqe := NewTransformQueryEngine(mockEngine, &IdentityTransform{})

	resp, err := tqe.Query(ctx, "test query")
	require.NoError(t, err)
	assert.Equal(t, "Transformed response", resp.Response)
}

func TestIdentityTransform(t *testing.T) {
	ctx := context.Background()
	transform := &IdentityTransform{}

	query := schema.QueryBundle{QueryString: "test query"}
	result, err := transform.Transform(ctx, query)
	require.NoError(t, err)
	assert.Equal(t, "test query", result.QueryString)
}

func TestHyDETransform(t *testing.T) {
	ctx := context.Background()
	mockLLM := llm.NewMockLLM("Hypothetical document about the topic")

	transform := NewHyDETransform(mockLLM)

	query := schema.QueryBundle{QueryString: "What is AI?"}
	result, err := transform.Transform(ctx, query)
	require.NoError(t, err)
	assert.Equal(t, "Hypothetical document about the topic", result.QueryString)
}

func TestRetryQueryEngineSuccess(t *testing.T) {
	ctx := context.Background()

	mockEngine := &MockQueryEngine{
		Response: &synthesizer.Response{Response: "Success"},
	}

	rqe := NewRetryQueryEngine(mockEngine, WithMaxRetries(3))

	resp, err := rqe.Query(ctx, "test")
	require.NoError(t, err)
	assert.Equal(t, "Success", resp.Response)
	assert.Equal(t, 1, mockEngine.CallCount)
}

func TestRetryQueryEngineRetry(t *testing.T) {
	ctx := context.Background()

	callCount := 0
	rqe := NewRetryQueryEngine(&retryTestEngine{callCount: &callCount},
		WithMaxRetries(3),
		WithRetryDelay(10*time.Millisecond),
	)

	resp, err := rqe.Query(ctx, "test")
	require.NoError(t, err)
	assert.Equal(t, "Success after retry", resp.Response)
	assert.Equal(t, 3, callCount)
}

type retryTestEngine struct {
	callCount *int
}

func (e *retryTestEngine) Query(ctx context.Context, query string) (*synthesizer.Response, error) {
	*e.callCount++
	if *e.callCount < 3 {
		return nil, errors.New("temporary error")
	}
	return &synthesizer.Response{Response: "Success after retry"}, nil
}

func TestRetryQueryEngineMaxRetries(t *testing.T) {
	ctx := context.Background()

	mockEngine := &MockQueryEngine{
		Err: errors.New("persistent error"),
	}

	rqe := NewRetryQueryEngine(mockEngine,
		WithMaxRetries(2),
		WithRetryDelay(10*time.Millisecond),
	)

	_, err := rqe.Query(ctx, "test")
	assert.Error(t, err)
	assert.Equal(t, 3, mockEngine.CallCount) // Initial + 2 retries
}

func TestSubQuestionQueryEngine(t *testing.T) {
	ctx := context.Background()

	// Create mock engines
	mockEngine1 := &MockQueryEngine{
		Response: &synthesizer.Response{Response: "Answer from engine 1"},
	}
	mockEngine2 := &MockQueryEngine{
		Response: &synthesizer.Response{Response: "Answer from engine 2"},
	}

	tools := []*QueryEngineTool{
		NewQueryEngineTool(mockEngine1, "engine1", "First engine"),
		NewQueryEngineTool(mockEngine2, "engine2", "Second engine"),
	}

	// Create mock question generator
	mockLLM := llm.NewMockLLM("[engine1] What is the first part?\n[engine2] What is the second part?")
	questionGen := NewLLMQuestionGenerator(mockLLM)

	// Create synthesizer
	synthLLM := llm.NewMockLLM("Combined answer")
	synth := synthesizer.NewSimpleSynthesizer(synthLLM)

	sqe := NewSubQuestionQueryEngine(questionGen, synth, tools)

	resp, err := sqe.Query(ctx, "Complex question")
	require.NoError(t, err)
	assert.NotEmpty(t, resp.Response)
}

func TestLLMQuestionGenerator(t *testing.T) {
	ctx := context.Background()

	mockLLM := llm.NewMockLLM("[tool1] Sub question 1\n[tool2] Sub question 2")
	qg := NewLLMQuestionGenerator(mockLLM)

	tools := []*QueryEngineTool{
		{Name: "tool1", Description: "First tool"},
		{Name: "tool2", Description: "Second tool"},
	}

	subQuestions, err := qg.Generate(ctx, tools, "Main question")
	require.NoError(t, err)
	assert.Len(t, subQuestions, 2)
	assert.Equal(t, "tool1", subQuestions[0].ToolName)
	assert.Equal(t, "tool2", subQuestions[1].ToolName)
}

func TestSelectorResult(t *testing.T) {
	result := &SelectorResult{
		Indices: []int{0, 2},
		Reasons: []string{"reason1", "reason2"},
	}

	assert.Len(t, result.Indices, 2)
	assert.Len(t, result.Reasons, 2)
}

func TestSingleSelector(t *testing.T) {
	ctx := context.Background()
	selector := &SingleSelector{}

	tools := []*QueryEngineTool{
		{Name: "tool1"},
		{Name: "tool2"},
	}

	result, err := selector.Select(ctx, tools, schema.QueryBundle{})
	require.NoError(t, err)
	assert.Len(t, result.Indices, 1)
	assert.Equal(t, 0, result.Indices[0])
}

func TestMultiSelector(t *testing.T) {
	ctx := context.Background()
	selector := &MultiSelector{}

	tools := []*QueryEngineTool{
		{Name: "tool1"},
		{Name: "tool2"},
		{Name: "tool3"},
	}

	result, err := selector.Select(ctx, tools, schema.QueryBundle{})
	require.NoError(t, err)
	assert.Len(t, result.Indices, 3)
}

func TestSelectorEmpty(t *testing.T) {
	ctx := context.Background()

	single := &SingleSelector{}
	_, err := single.Select(ctx, []*QueryEngineTool{}, schema.QueryBundle{})
	assert.Error(t, err)

	multi := &MultiSelector{}
	_, err = multi.Select(ctx, []*QueryEngineTool{}, schema.QueryBundle{})
	assert.Error(t, err)
}
