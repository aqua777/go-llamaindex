package ingestion

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	"github.com/aqua777/go-llamaindex/schema"
)

// DocstoreStrategy defines document deduplication strategies.
type DocstoreStrategy string

const (
	// DocstoreStrategyUpserts uses upserts to handle duplicates.
	DocstoreStrategyUpserts DocstoreStrategy = "upserts"
	// DocstoreStrategyDuplicatesOnly only handles duplicates.
	DocstoreStrategyDuplicatesOnly DocstoreStrategy = "duplicates_only"
	// DocstoreStrategyUpsertsAndDelete uses upserts and deletes.
	DocstoreStrategyUpsertsAndDelete DocstoreStrategy = "upserts_and_delete"
)

// TransformComponent is an interface for transformation components.
type TransformComponent interface {
	// Transform transforms a list of nodes.
	Transform(ctx context.Context, nodes []schema.Node) ([]schema.Node, error)
	// Name returns the name of the transformation.
	Name() string
}

// VectorStoreInterface is an interface for vector stores used in the pipeline.
type VectorStoreInterface interface {
	Add(ctx context.Context, nodes []schema.Node) error
	Delete(ctx context.Context, refDocID string) error
}

// DocStoreInterface is an interface for document stores used in the pipeline.
type DocStoreInterface interface {
	GetDocumentHash(docID string) (string, bool)
	SetDocumentHash(docID string, hash string)
	GetAllDocumentHashes() map[string]string
	AddDocuments(nodes []schema.Node) error
	DeleteDocument(docID string) error
	DeleteRefDoc(refDocID string) error
}

// IngestionPipeline is a document processing pipeline.
type IngestionPipeline struct {
	name             string
	transformations  []TransformComponent
	cache            *IngestionCache
	disableCache     bool
	docstore         DocStoreInterface
	vectorStore      VectorStoreInterface
	docstoreStrategy DocstoreStrategy
}

// IngestionPipelineOption configures an IngestionPipeline.
type IngestionPipelineOption func(*IngestionPipeline)

// WithPipelineName sets the pipeline name.
func WithPipelineName(name string) IngestionPipelineOption {
	return func(p *IngestionPipeline) {
		p.name = name
	}
}

// WithTransformations sets the transformations.
func WithTransformations(transformations []TransformComponent) IngestionPipelineOption {
	return func(p *IngestionPipeline) {
		p.transformations = transformations
	}
}

// WithPipelineCache sets the cache.
func WithPipelineCache(cache *IngestionCache) IngestionPipelineOption {
	return func(p *IngestionPipeline) {
		p.cache = cache
	}
}

// WithDisableCache disables the cache.
func WithDisableCache(disable bool) IngestionPipelineOption {
	return func(p *IngestionPipeline) {
		p.disableCache = disable
	}
}

// WithDocstore sets the document store.
func WithDocstore(docstore DocStoreInterface) IngestionPipelineOption {
	return func(p *IngestionPipeline) {
		p.docstore = docstore
	}
}

// WithVectorStore sets the vector store.
func WithVectorStore(vectorStore VectorStoreInterface) IngestionPipelineOption {
	return func(p *IngestionPipeline) {
		p.vectorStore = vectorStore
	}
}

// WithDocstoreStrategy sets the docstore strategy.
func WithDocstoreStrategy(strategy DocstoreStrategy) IngestionPipelineOption {
	return func(p *IngestionPipeline) {
		p.docstoreStrategy = strategy
	}
}

// NewIngestionPipeline creates a new IngestionPipeline.
func NewIngestionPipeline(opts ...IngestionPipelineOption) *IngestionPipeline {
	p := &IngestionPipeline{
		name:             "default",
		transformations:  []TransformComponent{},
		cache:            NewIngestionCache(),
		disableCache:     false,
		docstoreStrategy: DocstoreStrategyUpserts,
	}

	for _, opt := range opts {
		opt(p)
	}

	return p
}

// getTransformationHash computes a hash for the transformation and nodes.
func getTransformationHash(nodes []schema.Node, transform TransformComponent) string {
	var content string
	for _, node := range nodes {
		content += node.GetContent(schema.MetadataModeAll)
	}
	content += transform.Name()

	hash := sha256.Sum256([]byte(content))
	return hex.EncodeToString(hash[:])
}

// Run runs the ingestion pipeline on the given documents/nodes.
func (p *IngestionPipeline) Run(ctx context.Context, documents []schema.Document, nodes []schema.Node) ([]schema.Node, error) {
	// Prepare input nodes
	inputNodes := p.prepareInputs(documents, nodes)

	// Handle deduplication if docstore is set
	nodesToRun := inputNodes
	if p.docstore != nil {
		var err error
		nodesToRun, err = p.handleDeduplication(inputNodes)
		if err != nil {
			return nil, err
		}
	}

	// Run transformations
	resultNodes, err := p.runTransformations(ctx, nodesToRun)
	if err != nil {
		return nil, err
	}

	// Add to vector store if set
	if p.vectorStore != nil {
		nodesWithEmbeddings := filterNodesWithEmbeddings(resultNodes)
		if len(nodesWithEmbeddings) > 0 {
			if err := p.vectorStore.Add(ctx, nodesWithEmbeddings); err != nil {
				return nil, fmt.Errorf("failed to add nodes to vector store: %w", err)
			}
		}
	}

	// Update docstore if set
	if p.docstore != nil {
		if err := p.updateDocstore(nodesToRun); err != nil {
			return nil, fmt.Errorf("failed to update docstore: %w", err)
		}
	}

	return resultNodes, nil
}

// prepareInputs prepares input nodes from documents and nodes.
func (p *IngestionPipeline) prepareInputs(documents []schema.Document, nodes []schema.Node) []schema.Node {
	inputNodes := make([]schema.Node, 0, len(documents)+len(nodes))

	// Convert documents to nodes
	for _, doc := range documents {
		inputNodes = append(inputNodes, schema.Node{
			ID:       doc.ID,
			Text:     doc.Text,
			Metadata: doc.Metadata,
		})
	}

	// Add existing nodes
	inputNodes = append(inputNodes, nodes...)

	return inputNodes
}

// handleDeduplication handles document deduplication based on strategy.
func (p *IngestionPipeline) handleDeduplication(nodes []schema.Node) ([]schema.Node, error) {
	switch p.docstoreStrategy {
	case DocstoreStrategyUpserts, DocstoreStrategyUpsertsAndDelete:
		return p.handleUpserts(nodes)
	case DocstoreStrategyDuplicatesOnly:
		return p.handleDuplicates(nodes)
	default:
		return nil, fmt.Errorf("invalid docstore strategy: %s", p.docstoreStrategy)
	}
}

// handleDuplicates handles duplicates by checking all hashes.
func (p *IngestionPipeline) handleDuplicates(nodes []schema.Node) ([]schema.Node, error) {
	existingHashes := p.docstore.GetAllDocumentHashes()
	// Build a set of existing hash values
	existingHashSet := make(map[string]bool)
	for _, hash := range existingHashes {
		existingHashSet[hash] = true
	}

	currentHashes := make(map[string]bool)
	nodesToRun := make([]schema.Node, 0)

	for _, node := range nodes {
		hash := node.GetHash()
		if !existingHashSet[hash] && !currentHashes[hash] {
			p.docstore.SetDocumentHash(node.ID, hash)
			nodesToRun = append(nodesToRun, node)
			currentHashes[hash] = true
		}
	}

	return nodesToRun, nil
}

// handleUpserts handles upserts by checking hashes and IDs.
func (p *IngestionPipeline) handleUpserts(nodes []schema.Node) ([]schema.Node, error) {
	docIDsFromNodes := make(map[string]bool)
	dedupedNodesToRun := make(map[string]schema.Node)

	for _, node := range nodes {
		refDocID := node.ID
		// Check for source relationship to get ref doc ID
		if sourceInfo := node.Relationships.GetSource(); sourceInfo != nil {
			refDocID = sourceInfo.NodeID
		}
		docIDsFromNodes[refDocID] = true

		existingHash, exists := p.docstore.GetDocumentHash(refDocID)
		if !exists {
			// Document doesn't exist, add it
			dedupedNodesToRun[refDocID] = node
		} else if existingHash != node.GetHash() {
			// Document exists but hash changed, update it
			p.docstore.DeleteRefDoc(refDocID)
			if p.vectorStore != nil {
				p.vectorStore.Delete(context.Background(), refDocID)
			}
			dedupedNodesToRun[refDocID] = node
		}
		// Otherwise, document exists and is unchanged, skip it
	}

	// Handle delete strategy
	if p.docstoreStrategy == DocstoreStrategyUpsertsAndDelete {
		existingHashes := p.docstore.GetAllDocumentHashes()
		for _, docID := range existingHashes {
			if !docIDsFromNodes[docID] {
				p.docstore.DeleteDocument(docID)
				if p.vectorStore != nil {
					p.vectorStore.Delete(context.Background(), docID)
				}
			}
		}
	}

	// Convert map to slice
	result := make([]schema.Node, 0, len(dedupedNodesToRun))
	for _, node := range dedupedNodesToRun {
		result = append(result, node)
	}

	return result, nil
}

// runTransformations runs all transformations on the nodes.
func (p *IngestionPipeline) runTransformations(ctx context.Context, nodes []schema.Node) ([]schema.Node, error) {
	currentNodes := nodes

	for _, transform := range p.transformations {
		// Check cache if enabled
		if !p.disableCache && p.cache != nil {
			hash := getTransformationHash(currentNodes, transform)
			if cachedNodes, found := p.cache.Get(hash, ""); found {
				currentNodes = cachedNodes
				continue
			}

			// Run transformation
			transformedNodes, err := transform.Transform(ctx, currentNodes)
			if err != nil {
				return nil, fmt.Errorf("transformation %s failed: %w", transform.Name(), err)
			}

			// Cache result
			p.cache.Put(hash, transformedNodes, "")
			currentNodes = transformedNodes
		} else {
			// Run transformation without cache
			transformedNodes, err := transform.Transform(ctx, currentNodes)
			if err != nil {
				return nil, fmt.Errorf("transformation %s failed: %w", transform.Name(), err)
			}
			currentNodes = transformedNodes
		}
	}

	return currentNodes, nil
}

// updateDocstore updates the document store with processed nodes.
func (p *IngestionPipeline) updateDocstore(nodes []schema.Node) error {
	// Set document hashes
	for _, node := range nodes {
		p.docstore.SetDocumentHash(node.ID, node.GetHash())
	}

	// Add documents
	return p.docstore.AddDocuments(nodes)
}

// filterNodesWithEmbeddings filters nodes that have embeddings.
func filterNodesWithEmbeddings(nodes []schema.Node) []schema.Node {
	result := make([]schema.Node, 0)
	for _, node := range nodes {
		if len(node.Embedding) > 0 {
			result = append(result, node)
		}
	}
	return result
}

// AddTransformation adds a transformation to the pipeline.
func (p *IngestionPipeline) AddTransformation(transform TransformComponent) {
	p.transformations = append(p.transformations, transform)
}

// Transformations returns the current transformations.
func (p *IngestionPipeline) Transformations() []TransformComponent {
	return p.transformations
}

// Name returns the pipeline name.
func (p *IngestionPipeline) Name() string {
	return p.name
}

// Cache returns the pipeline cache.
func (p *IngestionPipeline) Cache() *IngestionCache {
	return p.cache
}

// RunTransformations is a standalone function to run transformations.
func RunTransformations(
	ctx context.Context,
	nodes []schema.Node,
	transformations []TransformComponent,
	cache *IngestionCache,
	cacheCollection string,
) ([]schema.Node, error) {
	currentNodes := nodes

	for _, transform := range transformations {
		if cache != nil {
			hash := getTransformationHash(currentNodes, transform)
			if cachedNodes, found := cache.Get(hash, cacheCollection); found {
				currentNodes = cachedNodes
				continue
			}

			transformedNodes, err := transform.Transform(ctx, currentNodes)
			if err != nil {
				return nil, err
			}

			cache.Put(hash, transformedNodes, cacheCollection)
			currentNodes = transformedNodes
		} else {
			transformedNodes, err := transform.Transform(ctx, currentNodes)
			if err != nil {
				return nil, err
			}
			currentNodes = transformedNodes
		}
	}

	return currentNodes, nil
}
