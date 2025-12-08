package index

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"github.com/aqua777/go-llamaindex/graphstore"
	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/prompts"
	"github.com/aqua777/go-llamaindex/rag/queryengine"
	"github.com/aqua777/go-llamaindex/rag/retriever"
	"github.com/aqua777/go-llamaindex/rag/synthesizer"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/storage"
	"github.com/aqua777/go-llamaindex/storage/indexstore"
)

// KGRetrieverMode specifies how the KG index retrieves nodes.
type KGRetrieverMode string

const (
	// KGRetrieverModeKeyword uses keywords to find triplets.
	KGRetrieverModeKeyword KGRetrieverMode = "keyword"
	// KGRetrieverModeEmbedding uses embeddings to find similar triplets.
	KGRetrieverModeEmbedding KGRetrieverMode = "embedding"
	// KGRetrieverModeHybrid combines keywords and embeddings.
	KGRetrieverModeHybrid KGRetrieverMode = "hybrid"
)

// Default prompts for KG index.
const (
	DefaultKGTripletExtractPrompt = `Some text is provided below. Given the text, extract up to {max_knowledge_triplets} knowledge triplets in the form of (subject, predicate, object). Avoid stopwords.

---------------------
Text: {text}
---------------------
Triplets:
`

	DefaultQueryKeywordExtractPrompt = `A question is provided below. Given the question, extract up to {max_keywords} keywords from the text. Focus on extracting the keywords that we can use to best lookup answers to the question. Avoid stopwords.

---------------------
Question: {question}
---------------------
Provide keywords in the following comma-separated format: 'KEYWORDS: <keywords>'
`
)

// KnowledgeGraphIndex builds a knowledge graph by extracting triplets from documents.
// During query time, it leverages the KG to find relevant information.
type KnowledgeGraphIndex struct {
	*BaseIndex
	llm                       llm.LLM
	graphStore                graphstore.GraphStore
	tripletExtractTemplate    *prompts.PromptTemplate
	maxTripletsPerChunk       int
	includeEmbeddings         bool
	maxObjectLength           int
	tripletExtractFn          func(string) ([]graphstore.Triplet, error)
	keywordExtractTemplate    *prompts.PromptTemplate
	maxKeywordsPerQuery       int
	graphStoreQueryDepth      int
}

// KGIndexOption configures KnowledgeGraphIndex creation.
type KGIndexOption func(*KnowledgeGraphIndex)

// WithKGIndexStorageContext sets the storage context.
func WithKGIndexStorageContext(sc *storage.StorageContext) KGIndexOption {
	return func(kg *KnowledgeGraphIndex) {
		kg.storageContext = sc
	}
}

// WithKGIndexEmbedModel sets the embedding model.
func WithKGIndexEmbedModel(model EmbeddingModel) KGIndexOption {
	return func(kg *KnowledgeGraphIndex) {
		kg.embedModel = model
	}
}

// WithKGIndexLLM sets the LLM for triplet extraction.
func WithKGIndexLLM(l llm.LLM) KGIndexOption {
	return func(kg *KnowledgeGraphIndex) {
		kg.llm = l
	}
}

// WithKGIndexGraphStore sets the graph store.
func WithKGIndexGraphStore(gs graphstore.GraphStore) KGIndexOption {
	return func(kg *KnowledgeGraphIndex) {
		kg.graphStore = gs
	}
}

// WithKGIndexMaxTripletsPerChunk sets the max triplets to extract per chunk.
func WithKGIndexMaxTripletsPerChunk(n int) KGIndexOption {
	return func(kg *KnowledgeGraphIndex) {
		if n > 0 {
			kg.maxTripletsPerChunk = n
		}
	}
}

// WithKGIndexIncludeEmbeddings enables embedding storage for triplets.
func WithKGIndexIncludeEmbeddings(include bool) KGIndexOption {
	return func(kg *KnowledgeGraphIndex) {
		kg.includeEmbeddings = include
	}
}

// WithKGIndexMaxObjectLength sets the max length for triplet objects.
func WithKGIndexMaxObjectLength(n int) KGIndexOption {
	return func(kg *KnowledgeGraphIndex) {
		if n > 0 {
			kg.maxObjectLength = n
		}
	}
}

// WithKGIndexTripletExtractFn sets a custom triplet extraction function.
func WithKGIndexTripletExtractFn(fn func(string) ([]graphstore.Triplet, error)) KGIndexOption {
	return func(kg *KnowledgeGraphIndex) {
		kg.tripletExtractFn = fn
	}
}

// WithKGIndexTripletExtractTemplate sets the triplet extraction prompt.
func WithKGIndexTripletExtractTemplate(tmpl *prompts.PromptTemplate) KGIndexOption {
	return func(kg *KnowledgeGraphIndex) {
		kg.tripletExtractTemplate = tmpl
	}
}

// WithKGIndexGraphStoreQueryDepth sets the depth for graph traversal.
func WithKGIndexGraphStoreQueryDepth(depth int) KGIndexOption {
	return func(kg *KnowledgeGraphIndex) {
		if depth > 0 {
			kg.graphStoreQueryDepth = depth
		}
	}
}

// NewKnowledgeGraphIndex creates a new KnowledgeGraphIndex.
func NewKnowledgeGraphIndex(ctx context.Context, nodes []schema.Node, opts ...KGIndexOption) (*KnowledgeGraphIndex, error) {
	indexStruct := indexstore.NewIndexStruct(indexstore.IndexStructTypeKG)
	indexStruct.Table = make(map[string][]string)

	kg := &KnowledgeGraphIndex{
		BaseIndex:                 NewBaseIndex(indexStruct),
		graphStore:               graphstore.NewSimpleGraphStore(),
		maxTripletsPerChunk:      10,
		includeEmbeddings:        false,
		maxObjectLength:          128,
		maxKeywordsPerQuery:      10,
		graphStoreQueryDepth:     2,
		tripletExtractTemplate:   prompts.NewPromptTemplate(DefaultKGTripletExtractPrompt, prompts.PromptTypeKnowledgeTripletExtract),
		keywordExtractTemplate:   prompts.NewPromptTemplate(DefaultQueryKeywordExtractPrompt, prompts.PromptTypeQueryKeywordExtract),
	}

	for _, opt := range opts {
		opt(kg)
	}

	// Build index from nodes
	if len(nodes) > 0 {
		if err := kg.buildIndexFromNodes(ctx, nodes); err != nil {
			return nil, err
		}
	}

	// Add index struct to store
	if err := kg.storageContext.IndexStore.AddIndexStruct(ctx, indexStruct); err != nil {
		return nil, err
	}

	return kg, nil
}

// NewKnowledgeGraphIndexFromDocuments creates a KnowledgeGraphIndex from documents.
func NewKnowledgeGraphIndexFromDocuments(
	ctx context.Context,
	documents []schema.Document,
	opts ...KGIndexOption,
) (*KnowledgeGraphIndex, error) {
	var nodes []schema.Node
	for _, doc := range documents {
		node := schema.NewTextNode(doc.Text)
		node.Metadata = doc.Metadata
		if doc.ID != "" {
			node.ID = doc.ID
		}
		nodes = append(nodes, *node)
	}

	return NewKnowledgeGraphIndex(ctx, nodes, opts...)
}

// buildIndexFromNodes builds the index from nodes by extracting triplets.
func (kg *KnowledgeGraphIndex) buildIndexFromNodes(ctx context.Context, nodes []schema.Node) error {
	for _, node := range nodes {
		triplets, err := kg.extractTriplets(ctx, node.GetContent(schema.MetadataModeLLM))
		if err != nil {
			return err
		}

		for _, triplet := range triplets {
			// Add triplet to graph store
			if err := kg.graphStore.UpsertTriplet(ctx, triplet.Subject, triplet.Relation, triplet.Object); err != nil {
				return err
			}

			// Add node to index struct (keyed by subject and object)
			kg.addNodeToIndex(triplet.Subject, node.ID)
			kg.addNodeToIndex(triplet.Object, node.ID)

			// Add embeddings if enabled
			if kg.includeEmbeddings && kg.embedModel != nil {
				tripletStr := triplet.String()
				embedding, err := kg.embedModel.GetTextEmbedding(ctx, tripletStr)
				if err == nil {
					kg.indexStruct.EmbeddingDict[tripletStr] = embedding
				}
			}
		}

		// Add node to docstore
		if err := kg.storageContext.DocStore.AddDocuments(ctx, []schema.BaseNode{&node}, true); err != nil {
			return err
		}
	}

	return nil
}

// addNodeToIndex adds a node ID to the index under the given keyword.
func (kg *KnowledgeGraphIndex) addNodeToIndex(keyword, nodeID string) {
	if kg.indexStruct.Table == nil {
		kg.indexStruct.Table = make(map[string][]string)
	}
	if _, ok := kg.indexStruct.Table[keyword]; !ok {
		kg.indexStruct.Table[keyword] = make([]string, 0)
	}
	// Check if already exists
	for _, id := range kg.indexStruct.Table[keyword] {
		if id == nodeID {
			return
		}
	}
	kg.indexStruct.Table[keyword] = append(kg.indexStruct.Table[keyword], nodeID)
}

// extractTriplets extracts triplets from text.
func (kg *KnowledgeGraphIndex) extractTriplets(ctx context.Context, text string) ([]graphstore.Triplet, error) {
	if kg.tripletExtractFn != nil {
		return kg.tripletExtractFn(text)
	}
	return kg.llmExtractTriplets(ctx, text)
}

// llmExtractTriplets uses the LLM to extract triplets.
func (kg *KnowledgeGraphIndex) llmExtractTriplets(ctx context.Context, text string) ([]graphstore.Triplet, error) {
	if kg.llm == nil {
		return nil, fmt.Errorf("LLM not configured for triplet extraction")
	}

	prompt := kg.tripletExtractTemplate.Format(map[string]string{
		"max_knowledge_triplets": fmt.Sprintf("%d", kg.maxTripletsPerChunk),
		"text":                   text,
	})

	response, err := kg.llm.Complete(ctx, prompt)
	if err != nil {
		return nil, err
	}

	return kg.parseTripletResponse(response)
}

// parseTripletResponse parses the LLM response to extract triplets.
func (kg *KnowledgeGraphIndex) parseTripletResponse(response string) ([]graphstore.Triplet, error) {
	var triplets []graphstore.Triplet

	// Match patterns like (subject, predicate, object)
	re := regexp.MustCompile(`\(([^,]+),\s*([^,]+),\s*([^)]+)\)`)
	matches := re.FindAllStringSubmatch(response, -1)

	for _, match := range matches {
		if len(match) != 4 {
			continue
		}

		subj := strings.TrimSpace(match[1])
		pred := strings.TrimSpace(match[2])
		obj := strings.TrimSpace(match[3])

		// Skip if any part is empty
		if subj == "" || pred == "" || obj == "" {
			continue
		}

		// Skip if any part is too long
		if len(subj) > kg.maxObjectLength || len(pred) > kg.maxObjectLength || len(obj) > kg.maxObjectLength {
			continue
		}

		// Clean up quotes and capitalize
		subj = strings.Trim(subj, `"'`)
		pred = strings.Trim(pred, `"'`)
		obj = strings.Trim(obj, `"'`)

		if len(subj) > 0 {
			subj = strings.ToUpper(subj[:1]) + subj[1:]
		}
		if len(obj) > 0 {
			obj = strings.ToUpper(obj[:1]) + obj[1:]
		}

		triplets = append(triplets, graphstore.Triplet{
			Subject:  subj,
			Relation: pred,
			Object:   obj,
		})

		if len(triplets) >= kg.maxTripletsPerChunk {
			break
		}
	}

	return triplets, nil
}

// GraphStore returns the graph store.
func (kg *KnowledgeGraphIndex) GraphStore() graphstore.GraphStore {
	return kg.graphStore
}

// AsRetriever returns a retriever for this index.
func (kg *KnowledgeGraphIndex) AsRetriever(opts ...RetrieverOption) retriever.Retriever {
	config := &RetrieverConfig{
		SimilarityTopK: 2,
		EmbedModel:     kg.embedModel,
	}

	for _, opt := range opts {
		opt(config)
	}

	return NewKGTableRetriever(kg,
		WithKGRetrieverMode(KGRetrieverModeKeyword),
		WithKGRetrieverSimilarityTopK(config.SimilarityTopK),
	)
}

// AsRetrieverWithMode returns a retriever with the specified mode.
func (kg *KnowledgeGraphIndex) AsRetrieverWithMode(mode KGRetrieverMode, opts ...KGRetrieverOption) (retriever.Retriever, error) {
	// Validate mode
	if mode == KGRetrieverModeEmbedding || mode == KGRetrieverModeHybrid {
		if len(kg.indexStruct.EmbeddingDict) == 0 {
			return nil, fmt.Errorf("index was not constructed with embeddings, cannot use %s mode", mode)
		}
	}

	opts = append([]KGRetrieverOption{WithKGRetrieverMode(mode)}, opts...)
	return NewKGTableRetriever(kg, opts...), nil
}

// AsQueryEngine returns a query engine for this index.
func (kg *KnowledgeGraphIndex) AsQueryEngine(opts ...QueryEngineOption) queryengine.QueryEngine {
	config := &QueryEngineConfig{
		ResponseMode: synthesizer.ResponseModeCompact,
	}

	for _, opt := range opts {
		opt(config)
	}

	ret := kg.AsRetriever()

	var synth synthesizer.Synthesizer
	if config.Synthesizer != nil {
		synth = config.Synthesizer
	} else if config.LLM != nil {
		synth, _ = synthesizer.GetSynthesizer(config.ResponseMode, config.LLM)
	} else if kg.llm != nil {
		synth, _ = synthesizer.GetSynthesizer(config.ResponseMode, kg.llm)
	} else {
		synth = synthesizer.NewSimpleSynthesizer(llm.NewMockLLM(""))
	}

	return queryengine.NewRetrieverQueryEngine(ret, synth)
}

// InsertNodes inserts nodes into the index.
func (kg *KnowledgeGraphIndex) InsertNodes(ctx context.Context, nodes []schema.Node) error {
	for _, node := range nodes {
		triplets, err := kg.extractTriplets(ctx, node.GetContent(schema.MetadataModeLLM))
		if err != nil {
			return err
		}

		for _, triplet := range triplets {
			if err := kg.graphStore.UpsertTriplet(ctx, triplet.Subject, triplet.Relation, triplet.Object); err != nil {
				return err
			}

			kg.addNodeToIndex(triplet.Subject, node.ID)
			kg.addNodeToIndex(triplet.Object, node.ID)

			if kg.includeEmbeddings && kg.embedModel != nil {
				tripletStr := triplet.String()
				if _, exists := kg.indexStruct.EmbeddingDict[tripletStr]; !exists {
					embedding, err := kg.embedModel.GetTextEmbedding(ctx, tripletStr)
					if err == nil {
						kg.indexStruct.EmbeddingDict[tripletStr] = embedding
					}
				}
			}
		}

		// Add node to docstore
		if err := kg.storageContext.DocStore.AddDocuments(ctx, []schema.BaseNode{&node}, true); err != nil {
			return err
		}
	}

	// Update index store
	return kg.storageContext.IndexStore.AddIndexStruct(ctx, kg.indexStruct)
}

// DeleteNodes removes nodes from the index.
// Note: Delete is not fully implemented for KG index.
func (kg *KnowledgeGraphIndex) DeleteNodes(ctx context.Context, nodeIDs []string) error {
	return fmt.Errorf("delete not implemented for knowledge graph index")
}

// RefreshDocuments refreshes the index with updated documents.
func (kg *KnowledgeGraphIndex) RefreshDocuments(ctx context.Context, documents []schema.Document) ([]bool, error) {
	refreshed := make([]bool, len(documents))

	for i, doc := range documents {
		existingHash, err := kg.storageContext.DocStore.GetDocumentHash(ctx, doc.ID)
		if err != nil || existingHash == "" {
			// Document doesn't exist, insert it
			node := schema.NewTextNode(doc.Text)
			node.Metadata = doc.Metadata
			if doc.ID != "" {
				node.ID = doc.ID
			}
			if err := kg.InsertNodes(ctx, []schema.Node{*node}); err != nil {
				return refreshed, err
			}
			refreshed[i] = true
		}
	}

	return refreshed, nil
}

// UpsertTriplet manually inserts a triplet into the index.
func (kg *KnowledgeGraphIndex) UpsertTriplet(ctx context.Context, triplet graphstore.Triplet, includeEmbeddings bool) error {
	if err := kg.graphStore.UpsertTriplet(ctx, triplet.Subject, triplet.Relation, triplet.Object); err != nil {
		return err
	}

	if includeEmbeddings && kg.embedModel != nil {
		tripletStr := triplet.String()
		embedding, err := kg.embedModel.GetTextEmbedding(ctx, tripletStr)
		if err == nil {
			kg.indexStruct.EmbeddingDict[tripletStr] = embedding
		}
	}

	return kg.storageContext.IndexStore.AddIndexStruct(ctx, kg.indexStruct)
}

// AddNode manually adds a node to the index under the given keywords.
func (kg *KnowledgeGraphIndex) AddNode(ctx context.Context, keywords []string, node schema.Node) error {
	for _, keyword := range keywords {
		kg.addNodeToIndex(keyword, node.ID)
	}

	if err := kg.storageContext.DocStore.AddDocuments(ctx, []schema.BaseNode{&node}, true); err != nil {
		return err
	}

	return kg.storageContext.IndexStore.AddIndexStruct(ctx, kg.indexStruct)
}

// UpsertTripletAndNode inserts both a triplet and its associated node.
func (kg *KnowledgeGraphIndex) UpsertTripletAndNode(ctx context.Context, triplet graphstore.Triplet, node schema.Node, includeEmbeddings bool) error {
	if err := kg.UpsertTriplet(ctx, triplet, includeEmbeddings); err != nil {
		return err
	}
	return kg.AddNode(ctx, []string{triplet.Subject, triplet.Object}, node)
}

// SearchNodeByKeyword returns node IDs associated with a keyword.
func (kg *KnowledgeGraphIndex) SearchNodeByKeyword(keyword string) []string {
	if kg.indexStruct.Table == nil {
		return nil
	}
	return kg.indexStruct.Table[keyword]
}

// GetAllKeywords returns all keywords in the index.
func (kg *KnowledgeGraphIndex) GetAllKeywords() []string {
	if kg.indexStruct.Table == nil {
		return nil
	}
	keywords := make([]string, 0, len(kg.indexStruct.Table))
	for k := range kg.indexStruct.Table {
		keywords = append(keywords, k)
	}
	return keywords
}

// LLM returns the LLM used by this index.
func (kg *KnowledgeGraphIndex) LLM() llm.LLM {
	return kg.llm
}

// KeywordExtractTemplate returns the keyword extraction template.
func (kg *KnowledgeGraphIndex) KeywordExtractTemplate() *prompts.PromptTemplate {
	return kg.keywordExtractTemplate
}

// MaxKeywordsPerQuery returns the max keywords per query.
func (kg *KnowledgeGraphIndex) MaxKeywordsPerQuery() int {
	return kg.maxKeywordsPerQuery
}

// GraphStoreQueryDepth returns the graph store query depth.
func (kg *KnowledgeGraphIndex) GraphStoreQueryDepth() int {
	return kg.graphStoreQueryDepth
}

// Ensure KnowledgeGraphIndex implements Index.
var _ Index = (*KnowledgeGraphIndex)(nil)
