package evaluation

import (
	"context"
	"fmt"
	"testing"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

// MockLLM implements llm.LLM for testing.
type MockLLM struct {
	responses []string
	callCount int
}

func NewMockLLM(responses ...string) *MockLLM {
	return &MockLLM{responses: responses}
}

func (m *MockLLM) Complete(ctx context.Context, prompt string) (string, error) {
	return m.getNextResponse(), nil
}

func (m *MockLLM) Chat(ctx context.Context, messages []llm.ChatMessage) (string, error) {
	return m.getNextResponse(), nil
}

func (m *MockLLM) Stream(ctx context.Context, prompt string) (<-chan string, error) {
	ch := make(chan string, 1)
	ch <- m.getNextResponse()
	close(ch)
	return ch, nil
}

func (m *MockLLM) getNextResponse() string {
	if m.callCount >= len(m.responses) {
		return "YES"
	}
	response := m.responses[m.callCount]
	m.callCount++
	return response
}

// MockEmbeddingModel implements embedding.EmbeddingModel for testing.
type MockEmbeddingModel struct {
	embeddings map[string][]float64
}

func NewMockEmbeddingModel() *MockEmbeddingModel {
	return &MockEmbeddingModel{
		embeddings: make(map[string][]float64),
	}
}

func (m *MockEmbeddingModel) SetEmbedding(text string, embedding []float64) {
	m.embeddings[text] = embedding
}

func (m *MockEmbeddingModel) GetTextEmbedding(ctx context.Context, text string) ([]float64, error) {
	if emb, ok := m.embeddings[text]; ok {
		return emb, nil
	}
	// Return a default embedding
	return []float64{0.1, 0.2, 0.3, 0.4, 0.5}, nil
}

func (m *MockEmbeddingModel) GetQueryEmbedding(ctx context.Context, query string) ([]float64, error) {
	return m.GetTextEmbedding(ctx, query)
}

// EvaluationTestSuite tests the evaluation package.
type EvaluationTestSuite struct {
	suite.Suite
}

func TestEvaluationSuite(t *testing.T) {
	suite.Run(t, new(EvaluationTestSuite))
}

// Test EvaluationResult

func (s *EvaluationTestSuite) TestEvaluationResultCreation() {
	result := NewEvaluationResult()
	s.NotNil(result)
	s.NotNil(result.Metadata)
}

func (s *EvaluationTestSuite) TestEvaluationResultBuilders() {
	result := NewEvaluationResult().
		WithQuery("test query").
		WithResponse("test response").
		WithContexts([]string{"context1", "context2"}).
		WithReference("reference answer").
		WithPassing(true).
		WithScore(0.95).
		WithFeedback("Good answer")

	s.Equal("test query", result.Query)
	s.Equal("test response", result.Response)
	s.Len(result.Contexts, 2)
	s.Equal("reference answer", result.Reference)
	s.True(result.IsPassing())
	s.Equal(0.95, result.GetScore())
	s.Equal("Good answer", result.Feedback)
}

func (s *EvaluationTestSuite) TestEvaluationResultInvalid() {
	result := NewEvaluationResult().WithInvalid("missing data")

	s.True(result.InvalidResult)
	s.Equal("missing data", result.InvalidReason)
}

func (s *EvaluationTestSuite) TestEvaluationResultNilValues() {
	result := NewEvaluationResult()

	s.False(result.IsPassing())
	s.Equal(0.0, result.GetScore())
}

// Test EvaluateInput

func (s *EvaluationTestSuite) TestEvaluateInputCreation() {
	input := NewEvaluateInput().
		WithQuery("What is AI?").
		WithResponse("AI is artificial intelligence").
		WithContexts([]string{"AI context"}).
		WithReference("AI stands for artificial intelligence")

	s.Equal("What is AI?", input.Query)
	s.Equal("AI is artificial intelligence", input.Response)
	s.Len(input.Contexts, 1)
	s.Equal("AI stands for artificial intelligence", input.Reference)
}

// Test FaithfulnessEvaluator

func (s *EvaluationTestSuite) TestFaithfulnessEvaluatorCreation() {
	evaluator := NewFaithfulnessEvaluator()
	s.Equal("faithfulness", evaluator.Name())
}

func (s *EvaluationTestSuite) TestFaithfulnessEvaluatorPassing() {
	mockLLM := NewMockLLM("YES")
	evaluator := NewFaithfulnessEvaluator(WithFaithfulnessLLM(mockLLM))

	ctx := context.Background()
	input := NewEvaluateInput().
		WithResponse("The sky is blue").
		WithContexts([]string{"The sky appears blue due to Rayleigh scattering"})

	result, err := evaluator.Evaluate(ctx, input)
	s.NoError(err)
	s.True(result.IsPassing())
	s.Equal(1.0, result.GetScore())
}

func (s *EvaluationTestSuite) TestFaithfulnessEvaluatorFailing() {
	mockLLM := NewMockLLM("NO")
	evaluator := NewFaithfulnessEvaluator(WithFaithfulnessLLM(mockLLM))

	ctx := context.Background()
	input := NewEvaluateInput().
		WithResponse("The sky is green").
		WithContexts([]string{"The sky appears blue due to Rayleigh scattering"})

	result, err := evaluator.Evaluate(ctx, input)
	s.NoError(err)
	s.False(result.IsPassing())
	s.Equal(0.0, result.GetScore())
}

func (s *EvaluationTestSuite) TestFaithfulnessEvaluatorMissingContexts() {
	mockLLM := NewMockLLM("YES")
	evaluator := NewFaithfulnessEvaluator(WithFaithfulnessLLM(mockLLM))

	ctx := context.Background()
	input := NewEvaluateInput().WithResponse("test")

	result, err := evaluator.Evaluate(ctx, input)
	s.NoError(err)
	s.True(result.InvalidResult)
}

func (s *EvaluationTestSuite) TestFaithfulnessEvaluatorMissingLLM() {
	evaluator := NewFaithfulnessEvaluator()

	ctx := context.Background()
	input := NewEvaluateInput().
		WithResponse("test").
		WithContexts([]string{"context"})

	_, err := evaluator.Evaluate(ctx, input)
	s.Error(err)
}

// Test RelevancyEvaluator

func (s *EvaluationTestSuite) TestRelevancyEvaluatorCreation() {
	evaluator := NewRelevancyEvaluator()
	s.Equal("relevancy", evaluator.Name())
}

func (s *EvaluationTestSuite) TestRelevancyEvaluatorPassing() {
	mockLLM := NewMockLLM("YES")
	evaluator := NewRelevancyEvaluator(WithRelevancyLLM(mockLLM))

	ctx := context.Background()
	input := NewEvaluateInput().
		WithQuery("What color is the sky?").
		WithResponse("The sky is blue").
		WithContexts([]string{"The sky appears blue"})

	result, err := evaluator.Evaluate(ctx, input)
	s.NoError(err)
	s.True(result.IsPassing())
}

func (s *EvaluationTestSuite) TestRelevancyEvaluatorMissingQuery() {
	mockLLM := NewMockLLM("YES")
	evaluator := NewRelevancyEvaluator(WithRelevancyLLM(mockLLM))

	ctx := context.Background()
	input := NewEvaluateInput().
		WithResponse("test").
		WithContexts([]string{"context"})

	result, err := evaluator.Evaluate(ctx, input)
	s.NoError(err)
	s.True(result.InvalidResult)
}

// Test ContextRelevancyEvaluator

func (s *EvaluationTestSuite) TestContextRelevancyEvaluatorCreation() {
	evaluator := NewContextRelevancyEvaluator()
	s.Equal("context_relevancy", evaluator.Name())
}

func (s *EvaluationTestSuite) TestContextRelevancyEvaluatorAllRelevant() {
	mockLLM := NewMockLLM("YES", "YES", "YES")
	evaluator := NewContextRelevancyEvaluator(WithContextRelevancyLLM(mockLLM))

	ctx := context.Background()
	input := NewEvaluateInput().
		WithQuery("What is AI?").
		WithContexts([]string{"AI is artificial intelligence", "AI is used in many fields", "AI can learn"})

	result, err := evaluator.Evaluate(ctx, input)
	s.NoError(err)
	s.Equal(1.0, result.GetScore())
	s.True(result.IsPassing())
}

func (s *EvaluationTestSuite) TestContextRelevancyEvaluatorPartialRelevant() {
	mockLLM := NewMockLLM("YES", "NO", "YES")
	evaluator := NewContextRelevancyEvaluator(WithContextRelevancyLLM(mockLLM))

	ctx := context.Background()
	input := NewEvaluateInput().
		WithQuery("What is AI?").
		WithContexts([]string{"AI is artificial intelligence", "Unrelated context", "AI can learn"})

	result, err := evaluator.Evaluate(ctx, input)
	s.NoError(err)
	s.InDelta(0.666, result.GetScore(), 0.01)
	s.True(result.IsPassing()) // > 0.5
}

// Test AnswerRelevancyEvaluator

func (s *EvaluationTestSuite) TestAnswerRelevancyEvaluatorCreation() {
	evaluator := NewAnswerRelevancyEvaluator()
	s.Equal("answer_relevancy", evaluator.Name())
}

func (s *EvaluationTestSuite) TestAnswerRelevancyEvaluatorPassing() {
	mockLLM := NewMockLLM("YES")
	evaluator := NewAnswerRelevancyEvaluator(WithAnswerRelevancyLLM(mockLLM))

	ctx := context.Background()
	input := NewEvaluateInput().
		WithQuery("What is 2+2?").
		WithResponse("4")

	result, err := evaluator.Evaluate(ctx, input)
	s.NoError(err)
	s.True(result.IsPassing())
}

// Test CorrectnessEvaluator

func (s *EvaluationTestSuite) TestCorrectnessEvaluatorCreation() {
	evaluator := NewCorrectnessEvaluator()
	s.Equal("correctness", evaluator.Name())
}

func (s *EvaluationTestSuite) TestCorrectnessEvaluatorPassing() {
	mockLLM := NewMockLLM("4.5\nThe answer is correct and well-formatted.")
	evaluator := NewCorrectnessEvaluator(WithCorrectnessLLM(mockLLM))

	ctx := context.Background()
	input := NewEvaluateInput().
		WithQuery("What is 2+2?").
		WithResponse("4").
		WithReference("4")

	result, err := evaluator.Evaluate(ctx, input)
	s.NoError(err)
	s.True(result.IsPassing())
	s.Equal(4.5, result.GetScore())
}

func (s *EvaluationTestSuite) TestCorrectnessEvaluatorFailing() {
	mockLLM := NewMockLLM("2.0\nThe answer is incorrect.")
	evaluator := NewCorrectnessEvaluator(WithCorrectnessLLM(mockLLM))

	ctx := context.Background()
	input := NewEvaluateInput().
		WithQuery("What is 2+2?").
		WithResponse("5").
		WithReference("4")

	result, err := evaluator.Evaluate(ctx, input)
	s.NoError(err)
	s.False(result.IsPassing())
	s.Equal(2.0, result.GetScore())
}

func (s *EvaluationTestSuite) TestCorrectnessEvaluatorCustomThreshold() {
	mockLLM := NewMockLLM("3.5\nPartially correct.")
	evaluator := NewCorrectnessEvaluator(
		WithCorrectnessLLM(mockLLM),
		WithCorrectnessScoreThreshold(3.0),
	)

	ctx := context.Background()
	input := NewEvaluateInput().
		WithQuery("test").
		WithResponse("test")

	result, err := evaluator.Evaluate(ctx, input)
	s.NoError(err)
	s.True(result.IsPassing()) // 3.5 >= 3.0
}

// Test SemanticSimilarityEvaluator

func (s *EvaluationTestSuite) TestSemanticSimilarityEvaluatorCreation() {
	evaluator := NewSemanticSimilarityEvaluator()
	s.Equal("semantic_similarity", evaluator.Name())
}

func (s *EvaluationTestSuite) TestSemanticSimilarityEvaluatorIdentical() {
	mockEmbed := NewMockEmbeddingModel()
	mockEmbed.SetEmbedding("The answer is 4", []float64{1.0, 0.0, 0.0})
	mockEmbed.SetEmbedding("The answer is four", []float64{1.0, 0.0, 0.0})

	evaluator := NewSemanticSimilarityEvaluator(
		WithSemanticSimilarityEmbedModel(mockEmbed),
	)

	ctx := context.Background()
	input := NewEvaluateInput().
		WithResponse("The answer is 4").
		WithReference("The answer is four")

	result, err := evaluator.Evaluate(ctx, input)
	s.NoError(err)
	s.Equal(1.0, result.GetScore())
	s.True(result.IsPassing())
}

func (s *EvaluationTestSuite) TestSemanticSimilarityEvaluatorDifferent() {
	mockEmbed := NewMockEmbeddingModel()
	mockEmbed.SetEmbedding("The answer is 4", []float64{1.0, 0.0, 0.0})
	mockEmbed.SetEmbedding("Completely different", []float64{0.0, 1.0, 0.0})

	evaluator := NewSemanticSimilarityEvaluator(
		WithSemanticSimilarityEmbedModel(mockEmbed),
	)

	ctx := context.Background()
	input := NewEvaluateInput().
		WithResponse("The answer is 4").
		WithReference("Completely different")

	result, err := evaluator.Evaluate(ctx, input)
	s.NoError(err)
	s.Equal(0.0, result.GetScore())
	s.False(result.IsPassing())
}

func (s *EvaluationTestSuite) TestSemanticSimilarityEvaluatorMissingEmbedModel() {
	evaluator := NewSemanticSimilarityEvaluator()

	ctx := context.Background()
	input := NewEvaluateInput().
		WithResponse("test").
		WithReference("test")

	_, err := evaluator.Evaluate(ctx, input)
	s.Error(err)
}

// Test BatchEvalRunner

func (s *EvaluationTestSuite) TestBatchEvalRunnerCreation() {
	evaluators := map[string]Evaluator{
		"faithfulness": NewFaithfulnessEvaluator(WithFaithfulnessLLM(NewMockLLM("YES"))),
	}
	runner := NewBatchEvalRunner(evaluators)
	s.NotNil(runner)
	s.Len(runner.Evaluators(), 1)
}

func (s *EvaluationTestSuite) TestBatchEvalRunnerSingleEvaluation() {
	mockLLM := NewMockLLM("YES")
	evaluators := map[string]Evaluator{
		"faithfulness": NewFaithfulnessEvaluator(WithFaithfulnessLLM(mockLLM)),
	}
	runner := NewBatchEvalRunner(evaluators)

	ctx := context.Background()
	input := NewEvaluateInput().
		WithResponse("test").
		WithContexts([]string{"context"})

	results, err := runner.RunSingle(ctx, input)
	s.NoError(err)
	s.Len(results, 1)
	s.True(results["faithfulness"].IsPassing())
}

func (s *EvaluationTestSuite) TestBatchEvalRunnerBatchEvaluation() {
	mockLLM := NewMockLLM("YES", "YES", "NO")
	evaluators := map[string]Evaluator{
		"faithfulness": NewFaithfulnessEvaluator(WithFaithfulnessLLM(mockLLM)),
	}
	runner := NewBatchEvalRunner(evaluators, WithBatchWorkers(2))

	ctx := context.Background()
	queries := []string{"q1", "q2", "q3"}
	responses := []string{"r1", "r2", "r3"}
	contextsList := [][]string{{"c1"}, {"c2"}, {"c3"}}

	result, err := runner.EvaluateResponseStrs(ctx, queries, responses, contextsList, nil)
	s.NoError(err)
	s.Len(result.Results["faithfulness"], 3)
}

func (s *EvaluationTestSuite) TestBatchEvalRunnerWithReferences() {
	mockLLM := NewMockLLM("4.5\nCorrect", "3.0\nPartially correct")
	evaluators := map[string]Evaluator{
		"correctness": NewCorrectnessEvaluator(WithCorrectnessLLM(mockLLM)),
	}
	runner := NewBatchEvalRunner(evaluators)

	ctx := context.Background()
	queries := []string{"q1", "q2"}
	responses := []string{"r1", "r2"}
	references := []string{"ref1", "ref2"}

	result, err := runner.EvaluateWithReferences(ctx, queries, responses, references)
	s.NoError(err)
	s.Len(result.Results["correctness"], 2)
}

func (s *EvaluationTestSuite) TestBatchEvalResultSummary() {
	result := NewBatchEvalResult([]string{"eval1", "eval2"})

	// Add some results
	result.Results["eval1"] = []*EvaluationResult{
		NewEvaluationResult().WithScore(0.8).WithPassing(true),
		NewEvaluationResult().WithScore(0.6).WithPassing(true),
		NewEvaluationResult().WithScore(0.4).WithPassing(false),
	}
	result.Results["eval2"] = []*EvaluationResult{
		NewEvaluationResult().WithScore(1.0).WithPassing(true),
		NewEvaluationResult().WithScore(0.5).WithPassing(false),
	}

	summary := result.Summary()

	s.InDelta(0.6, summary["eval1"]["average_score"], 0.01)
	s.InDelta(0.666, summary["eval1"]["passing_rate"], 0.01)
	s.Equal(3.0, summary["eval1"]["total_count"])

	s.InDelta(0.75, summary["eval2"]["average_score"], 0.01)
	s.Equal(0.5, summary["eval2"]["passing_rate"])
}

// Test Similarity Functions

func (s *EvaluationTestSuite) TestCosineSimilarity() {
	vec1 := []float64{1.0, 0.0, 0.0}
	vec2 := []float64{1.0, 0.0, 0.0}
	s.Equal(1.0, CosineSimilarity(vec1, vec2))

	vec3 := []float64{0.0, 1.0, 0.0}
	s.Equal(0.0, CosineSimilarity(vec1, vec3))

	vec4 := []float64{0.707, 0.707, 0.0}
	s.InDelta(0.707, CosineSimilarity(vec1, vec4), 0.01)
}

func (s *EvaluationTestSuite) TestDotProduct() {
	vec1 := []float64{1.0, 2.0, 3.0}
	vec2 := []float64{4.0, 5.0, 6.0}
	s.Equal(32.0, DotProduct(vec1, vec2)) // 1*4 + 2*5 + 3*6 = 32
}

func (s *EvaluationTestSuite) TestEuclideanDistance() {
	vec1 := []float64{0.0, 0.0, 0.0}
	vec2 := []float64{3.0, 4.0, 0.0}
	s.Equal(5.0, EuclideanDistance(vec1, vec2)) // sqrt(9 + 16) = 5
}

// Test Correctness Parser

func TestDefaultCorrectnessParser(t *testing.T) {
	tests := []struct {
		name          string
		input         string
		expectedScore float64
		expectError   bool
	}{
		{
			name:          "simple score",
			input:         "4.0\nThe answer is correct.",
			expectedScore: 4.0,
			expectError:   false,
		},
		{
			name:          "score with text",
			input:         "Score: 3.5\nPartially correct.",
			expectedScore: 3.5,
			expectError:   false,
		},
		{
			name:          "integer score",
			input:         "5\nPerfect answer.",
			expectedScore: 5.0,
			expectError:   false,
		},
		{
			name:        "no score",
			input:       "The answer is correct.",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			score, _, err := defaultCorrectnessParser(tt.input)
			if tt.expectError {
				assert.Error(t, err)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tt.expectedScore, score)
			}
		})
	}
}

// Test Score Normalization

func TestNormalizeScore(t *testing.T) {
	assert.Equal(t, 0.0, NormalizeScore(1.0))
	assert.Equal(t, 0.5, NormalizeScore(3.0))
	assert.Equal(t, 1.0, NormalizeScore(5.0))
}

func TestDenormalizeScore(t *testing.T) {
	assert.Equal(t, 1.0, DenormalizeScore(0.0))
	assert.Equal(t, 3.0, DenormalizeScore(0.5))
	assert.Equal(t, 5.0, DenormalizeScore(1.0))
}

// Test EvaluatorRegistry

func (s *EvaluationTestSuite) TestEvaluatorRegistry() {
	registry := NewEvaluatorRegistry()

	eval1 := NewFaithfulnessEvaluator()
	eval2 := NewRelevancyEvaluator()

	registry.Register(eval1)
	registry.Register(eval2)

	s.Len(registry.List(), 2)

	retrieved, ok := registry.Get("faithfulness")
	s.True(ok)
	s.Equal("faithfulness", retrieved.Name())

	_, ok = registry.Get("nonexistent")
	s.False(ok)
}

// Test Interface Compliance

func (s *EvaluationTestSuite) TestEvaluatorInterfaceCompliance() {
	var _ Evaluator = (*FaithfulnessEvaluator)(nil)
	var _ Evaluator = (*RelevancyEvaluator)(nil)
	var _ Evaluator = (*ContextRelevancyEvaluator)(nil)
	var _ Evaluator = (*AnswerRelevancyEvaluator)(nil)
	var _ Evaluator = (*CorrectnessEvaluator)(nil)
	var _ Evaluator = (*SemanticSimilarityEvaluator)(nil)
}

// Test AggregateScore

func TestAggregateScore(t *testing.T) {
	results := []*EvaluationResult{
		NewEvaluationResult().WithScore(0.8),
		NewEvaluationResult().WithScore(0.6),
		NewEvaluationResult().WithScore(1.0),
	}

	avg := AggregateScore(results)
	assert.InDelta(t, 0.8, avg, 0.01)

	// Empty results
	assert.Equal(t, 0.0, AggregateScore([]*EvaluationResult{}))
}

// Test CompareResults

func TestCompareResults(t *testing.T) {
	result1 := NewBatchEvalResult([]string{"eval1"})
	result1.Results["eval1"] = []*EvaluationResult{
		NewEvaluationResult().WithScore(0.8).WithPassing(true),
		NewEvaluationResult().WithScore(0.6).WithPassing(false),
	}

	result2 := NewBatchEvalResult([]string{"eval1"})
	result2.Results["eval1"] = []*EvaluationResult{
		NewEvaluationResult().WithScore(0.5).WithPassing(false),
		NewEvaluationResult().WithScore(0.5).WithPassing(false),
	}

	comparison := CompareResults(result1, result2)

	assert.InDelta(t, 0.2, comparison["eval1"]["score_diff"], 0.01)        // 0.7 - 0.5
	assert.InDelta(t, 0.5, comparison["eval1"]["passing_rate_diff"], 0.01) // 0.5 - 0.0
}

// Benchmark Tests

func BenchmarkCosineSimilarity(b *testing.B) {
	vec1 := make([]float64, 1536)
	vec2 := make([]float64, 1536)
	for i := range vec1 {
		vec1[i] = float64(i) / 1536.0
		vec2[i] = float64(1536-i) / 1536.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CosineSimilarity(vec1, vec2)
	}
}

func BenchmarkFaithfulnessEvaluator(b *testing.B) {
	mockLLM := NewMockLLM()
	for i := 0; i < b.N*2; i++ {
		mockLLM.responses = append(mockLLM.responses, "YES")
	}

	evaluator := NewFaithfulnessEvaluator(WithFaithfulnessLLM(mockLLM))
	ctx := context.Background()
	input := NewEvaluateInput().
		WithResponse("test response").
		WithContexts([]string{"test context"})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = evaluator.Evaluate(ctx, input)
	}
}

// Test edge cases

func (s *EvaluationTestSuite) TestEmptyVectors() {
	s.Equal(0.0, CosineSimilarity([]float64{}, []float64{}))
	s.Equal(0.0, DotProduct([]float64{}, []float64{}))
}

func (s *EvaluationTestSuite) TestMismatchedVectorLengths() {
	vec1 := []float64{1.0, 2.0}
	vec2 := []float64{1.0, 2.0, 3.0}
	s.Equal(0.0, CosineSimilarity(vec1, vec2))
	s.Equal(0.0, DotProduct(vec1, vec2))
}

func (s *EvaluationTestSuite) TestZeroVectors() {
	vec1 := []float64{0.0, 0.0, 0.0}
	vec2 := []float64{1.0, 2.0, 3.0}
	s.Equal(0.0, CosineSimilarity(vec1, vec2))
}

// Test custom templates

func (s *EvaluationTestSuite) TestCustomFaithfulnessTemplate() {
	customTemplate := "Is this faithful? {response} Context: {context} Answer:"
	mockLLM := NewMockLLM("YES")

	evaluator := NewFaithfulnessEvaluator(
		WithFaithfulnessLLM(mockLLM),
		WithFaithfulnessTemplate(customTemplate),
	)

	ctx := context.Background()
	input := NewEvaluateInput().
		WithResponse("test").
		WithContexts([]string{"context"})

	result, err := evaluator.Evaluate(ctx, input)
	s.NoError(err)
	s.True(result.IsPassing())
}

// Test error handling

func (s *EvaluationTestSuite) TestBatchRunnerValidationError() {
	evaluators := map[string]Evaluator{
		"faithfulness": NewFaithfulnessEvaluator(WithFaithfulnessLLM(NewMockLLM("YES"))),
	}
	runner := NewBatchEvalRunner(evaluators)

	ctx := context.Background()

	// Mismatched lengths
	queries := []string{"q1", "q2"}
	responses := []string{"r1"}

	_, err := runner.EvaluateResponseStrs(ctx, queries, responses, nil, nil)
	s.Error(err)
}

func (s *EvaluationTestSuite) TestBatchRunnerNoInputs() {
	evaluators := map[string]Evaluator{
		"faithfulness": NewFaithfulnessEvaluator(WithFaithfulnessLLM(NewMockLLM("YES"))),
	}
	runner := NewBatchEvalRunner(evaluators)

	ctx := context.Background()

	_, err := runner.EvaluateResponseStrs(ctx, nil, nil, nil, nil)
	s.Error(err)
}

// Test multiple evaluators

func (s *EvaluationTestSuite) TestMultipleEvaluators() {
	mockLLM := NewMockLLM("YES", "YES") // One for each evaluator
	evaluators := map[string]Evaluator{
		"faithfulness": NewFaithfulnessEvaluator(WithFaithfulnessLLM(mockLLM)),
		"relevancy":    NewRelevancyEvaluator(WithRelevancyLLM(mockLLM)),
	}
	runner := NewBatchEvalRunner(evaluators)

	ctx := context.Background()
	input := NewEvaluateInput().
		WithQuery("test query").
		WithResponse("test response").
		WithContexts([]string{"test context"})

	results, err := runner.RunSingle(ctx, input)
	s.NoError(err)
	s.Len(results, 2)
}

// Test EvaluateStatements

func (s *EvaluationTestSuite) TestEvaluateStatements() {
	mockLLM := NewMockLLM("YES", "NO", "YES")
	evaluator := NewFaithfulnessEvaluator(WithFaithfulnessLLM(mockLLM))

	ctx := context.Background()
	statements := []string{"statement1", "statement2", "statement3"}
	contexts := []string{"context"}

	results, err := evaluator.EvaluateStatements(ctx, statements, contexts)
	s.NoError(err)
	s.Len(results, 3)
	s.True(results[0].IsPassing())
	s.False(results[1].IsPassing())
	s.True(results[2].IsPassing())
}

// Test similarity modes

func (s *EvaluationTestSuite) TestSemanticSimilarityModes() {
	mockEmbed := NewMockEmbeddingModel()
	mockEmbed.SetEmbedding("text1", []float64{1.0, 2.0, 3.0})
	mockEmbed.SetEmbedding("text2", []float64{1.0, 2.0, 3.0})

	ctx := context.Background()
	input := NewEvaluateInput().
		WithResponse("text1").
		WithReference("text2")

	// Test cosine mode
	evaluator := NewSemanticSimilarityEvaluator(
		WithSemanticSimilarityEmbedModel(mockEmbed),
		WithSemanticSimilarityMode(SimilarityModeCosine),
	)
	result, err := evaluator.Evaluate(ctx, input)
	s.NoError(err)
	s.Equal(1.0, result.GetScore())

	// Test dot product mode
	evaluator = NewSemanticSimilarityEvaluator(
		WithSemanticSimilarityEmbedModel(mockEmbed),
		WithSemanticSimilarityMode(SimilarityModeDotProduct),
	)
	result, err = evaluator.Evaluate(ctx, input)
	s.NoError(err)
	s.Equal(14.0, result.GetScore()) // 1*1 + 2*2 + 3*3 = 14
}

// Test AddEvaluator and RemoveEvaluator

func (s *EvaluationTestSuite) TestBatchRunnerAddRemoveEvaluator() {
	runner := NewBatchEvalRunner(map[string]Evaluator{})

	s.Len(runner.Evaluators(), 0)

	runner.AddEvaluator("faithfulness", NewFaithfulnessEvaluator())
	s.Len(runner.Evaluators(), 1)

	runner.AddEvaluator("relevancy", NewRelevancyEvaluator())
	s.Len(runner.Evaluators(), 2)

	runner.RemoveEvaluator("faithfulness")
	s.Len(runner.Evaluators(), 1)

	_, ok := runner.Evaluators()["faithfulness"]
	s.False(ok)
}

// Test raiseError option

func (s *EvaluationTestSuite) TestFaithfulnessRaiseError() {
	mockLLM := NewMockLLM("NO")
	evaluator := NewFaithfulnessEvaluator(
		WithFaithfulnessLLM(mockLLM),
		WithFaithfulnessRaiseError(true),
	)

	ctx := context.Background()
	input := NewEvaluateInput().
		WithResponse("test").
		WithContexts([]string{"context"})

	_, err := evaluator.Evaluate(ctx, input)
	s.Error(err)
	s.Contains(err.Error(), "faithfulness evaluation failed")
}

func (s *EvaluationTestSuite) TestRelevancyRaiseError() {
	mockLLM := NewMockLLM("NO")
	evaluator := NewRelevancyEvaluator(
		WithRelevancyLLM(mockLLM),
		WithRelevancyRaiseError(true),
	)

	ctx := context.Background()
	input := NewEvaluateInput().
		WithQuery("test").
		WithResponse("test").
		WithContexts([]string{"context"})

	_, err := evaluator.Evaluate(ctx, input)
	s.Error(err)
	s.Contains(err.Error(), "relevancy evaluation failed")
}

// Test BaseEvaluator

func (s *EvaluationTestSuite) TestBaseEvaluator() {
	base := NewBaseEvaluator(WithEvaluatorName("custom"))
	s.Equal("custom", base.Name())
}

// Test with nil extra kwargs

func (s *EvaluationTestSuite) TestBatchRunnerNilExtraKwargs() {
	mockLLM := NewMockLLM("YES")
	evaluators := map[string]Evaluator{
		"faithfulness": NewFaithfulnessEvaluator(WithFaithfulnessLLM(mockLLM)),
	}
	runner := NewBatchEvalRunner(evaluators)

	ctx := context.Background()
	queries := []string{"q1"}
	responses := []string{"r1"}
	contextsList := [][]string{{"c1"}}

	result, err := runner.EvaluateResponseStrs(ctx, queries, responses, contextsList, nil)
	s.NoError(err)
	s.Len(result.Results["faithfulness"], 1)
}

// Test correctness with no reference

func (s *EvaluationTestSuite) TestCorrectnessNoReference() {
	mockLLM := NewMockLLM("4.0\nGood answer without reference.")
	evaluator := NewCorrectnessEvaluator(WithCorrectnessLLM(mockLLM))

	ctx := context.Background()
	input := NewEvaluateInput().
		WithQuery("What is AI?").
		WithResponse("AI is artificial intelligence")
	// No reference provided

	result, err := evaluator.Evaluate(ctx, input)
	s.NoError(err)
	s.True(result.IsPassing())
}

// Test similarity threshold

func (s *EvaluationTestSuite) TestSemanticSimilarityThreshold() {
	mockEmbed := NewMockEmbeddingModel()
	mockEmbed.SetEmbedding("text1", []float64{1.0, 0.0, 0.0})
	mockEmbed.SetEmbedding("text2", []float64{0.8, 0.6, 0.0}) // cosine ~ 0.8

	evaluator := NewSemanticSimilarityEvaluator(
		WithSemanticSimilarityEmbedModel(mockEmbed),
		WithSemanticSimilarityThreshold(0.9), // Higher threshold
	)

	ctx := context.Background()
	input := NewEvaluateInput().
		WithResponse("text1").
		WithReference("text2")

	result, err := evaluator.Evaluate(ctx, input)
	s.NoError(err)
	s.False(result.IsPassing()) // 0.8 < 0.9
}

func main() {
	fmt.Println("Run tests with: go test ./evaluation/...")
}
