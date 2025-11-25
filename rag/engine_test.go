package rag

import (
	"context"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/rag/store"
	"github.com/aqua777/go-llamaindex/rag/store/chromem"
	"github.com/aqua777/go-llamaindex/schema"
)

// MockRetriever is a mock implementation of the Retriever interface.
type MockRetriever struct {
	Nodes []schema.NodeWithScore
	Err   error
}

func (m *MockRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	return m.Nodes, m.Err
}

// MockSynthesizer is a mock implementation of the Synthesizer interface.
type MockSynthesizer struct {
	Response schema.EngineResponse
	Err      error
}

func (m *MockSynthesizer) Synthesize(ctx context.Context, query schema.QueryBundle, nodes []schema.NodeWithScore) (schema.EngineResponse, error) {
	return m.Response, m.Err
}

func (m *MockSynthesizer) SynthesizeStream(ctx context.Context, query schema.QueryBundle, nodes []schema.NodeWithScore) (schema.StreamingEngineResponse, error) {
	return schema.StreamingEngineResponse{}, nil
}

type EngineTestSuite struct {
	suite.Suite
}

func TestEngineTestSuite(t *testing.T) {
	suite.Run(t, new(EngineTestSuite))
}

func (s *EngineTestSuite) TestRetrieverQueryEngine_Query() {
	ctx := context.Background()
	query := schema.QueryBundle{QueryString: "test query"}

	expectedNodes := []schema.NodeWithScore{
		{
			Node:  schema.Node{ID: "1", Text: "node 1"},
			Score: 1.0,
		},
	}
	expectedResponse := schema.EngineResponse{
		Response:    "test response",
		SourceNodes: expectedNodes,
	}

	retriever := &MockRetriever{Nodes: expectedNodes}
	synthesizer := &MockSynthesizer{Response: expectedResponse}

	engine := NewRetrieverQueryEngine(retriever, synthesizer)

	response, err := engine.Query(ctx, query)

	s.NoError(err)
	s.Equal(expectedResponse, response)
	s.Equal(expectedResponse.SourceNodes, response.SourceNodes)
}

func (s *EngineTestSuite) TestRetrieverQueryEngine_RetrieveFail() {
	ctx := context.Background()
	query := schema.QueryBundle{QueryString: "test query"}

	retriever := &MockRetriever{Err: context.DeadlineExceeded} // Example error
	synthesizer := &MockSynthesizer{}

	engine := NewRetrieverQueryEngine(retriever, synthesizer)

	_, err := engine.Query(ctx, query)

	s.Error(err)
	s.Contains(err.Error(), "retrieve failed")
}

func (s *EngineTestSuite) TestRetrieverQueryEngine_SynthesizeFail() {
	ctx := context.Background()
	query := schema.QueryBundle{QueryString: "test query"}
	nodes := []schema.NodeWithScore{{Node: schema.Node{Text: "foo"}, Score: 0.9}}

	retriever := &MockRetriever{Nodes: nodes}
	synthesizer := &MockSynthesizer{Err: context.DeadlineExceeded}

	engine := NewRetrieverQueryEngine(retriever, synthesizer)

	_, err := engine.Query(ctx, query)

	s.Error(err)
	s.Contains(err.Error(), "synthesize failed")
}

func (s *EngineTestSuite) TestFullRAGFlow() {
	ctx := context.Background()

	// 1. Setup Components
	// Mock Embedding Model
	mockEmbedding := &embedding.MockEmbeddingModel{
		Embedding: []float64{0.1, 0.2, 0.3}, // Simple mock embedding
	}
	// Mock LLM
	mockLLM := &llm.MockLLM{
		Response: "This is a generated answer based on context.",
	}
	// In-memory Vector Store
	vectorStore := store.NewSimpleVectorStore()

	// 2. Add Documents
	nodes := []schema.Node{
		{ID: "1", Text: "The capital of France is Paris.", Type: schema.ObjectTypeText, Embedding: []float64{0.1, 0.2, 0.3}},
		{ID: "2", Text: "The capital of Germany is Berlin.", Type: schema.ObjectTypeText, Embedding: []float64{0.9, 0.8, 0.7}},
	}
	_, err := vectorStore.Add(ctx, nodes)
	s.NoError(err)

	// 3. Create Retriever & Synthesizer
	retriever := NewVectorRetriever(vectorStore, mockEmbedding, 1) // TopK=1
	synthesizer := NewSimpleSynthesizer(mockLLM)

	// 4. Create Engine
	engine := NewRetrieverQueryEngine(retriever, synthesizer)

	// 5. Execute Query
	query := schema.QueryBundle{QueryString: "What is the capital of France?"}
	response, err := engine.Query(ctx, query)

	s.NoError(err)
	s.Equal("This is a generated answer based on context.", response.Response)
	s.Len(response.SourceNodes, 1)
	s.Equal("The capital of France is Paris.", response.SourceNodes[0].Node.Text)
}

func (s *EngineTestSuite) TestFullRAGFlow_Chromem() {
	ctx := context.Background()

	// 1. Setup Components
	// Mock Embedding Model (same as before)
	mockEmbedding := &embedding.MockEmbeddingModel{
		Embedding: []float64{0.1, 0.2, 0.3},
	}
	// Mock LLM (same as before)
	mockLLM := &llm.MockLLM{
		Response: "This is a generated answer based on context.",
	}

	// Chromem Store (In-Memory)
	// Empty path = in-memory only
	chromemStore, err := chromem.NewChromemStore("", "test-engine-collection")
	s.NoError(err)

	// 2. Add Documents
	nodes := []schema.Node{
		{ID: "1", Text: "The capital of France is Paris.", Type: schema.ObjectTypeText, Embedding: []float64{0.1, 0.2, 0.3}},
		{ID: "2", Text: "The capital of Germany is Berlin.", Type: schema.ObjectTypeText, Embedding: []float64{0.9, 0.8, 0.7}},
	}
	_, err = chromemStore.Add(ctx, nodes)
	s.NoError(err)

	// 3. Create Retriever & Synthesizer
	// chromemStore satisfies store.VectorStore
	retriever := NewVectorRetriever(chromemStore, mockEmbedding, 1) // TopK=1
	synthesizer := NewSimpleSynthesizer(mockLLM)

	// 4. Create Engine
	engine := NewRetrieverQueryEngine(retriever, synthesizer)

	// 5. Execute Query
	query := schema.QueryBundle{QueryString: "What is the capital of France?"}
	response, err := engine.Query(ctx, query)

	s.NoError(err)
	s.Equal("This is a generated answer based on context.", response.Response)
	s.Len(response.SourceNodes, 1)
	// chromem-go result order/score is based on cosine similarity.
	// 1st node has exact same vector as query (mock), so it should be first.
	s.Equal("The capital of France is Paris.", response.SourceNodes[0].Node.Text)
}
