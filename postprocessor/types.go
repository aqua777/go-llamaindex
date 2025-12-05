// Package postprocessor provides node postprocessing functionality.
package postprocessor

import (
	"context"

	"github.com/aqua777/go-llamaindex/schema"
)

// NodePostprocessor is the interface for node postprocessors.
type NodePostprocessor interface {
	// PostprocessNodes processes nodes after retrieval.
	PostprocessNodes(
		ctx context.Context,
		nodes []schema.NodeWithScore,
		queryBundle *schema.QueryBundle,
	) ([]schema.NodeWithScore, error)

	// Name returns the name of the postprocessor.
	Name() string
}

// BaseNodePostprocessor provides a base implementation of NodePostprocessor.
type BaseNodePostprocessor struct {
	name string
}

// BaseNodePostprocessorOption configures a BaseNodePostprocessor.
type BaseNodePostprocessorOption func(*BaseNodePostprocessor)

// WithPostprocessorName sets the postprocessor name.
func WithPostprocessorName(name string) BaseNodePostprocessorOption {
	return func(p *BaseNodePostprocessor) {
		p.name = name
	}
}

// NewBaseNodePostprocessor creates a new BaseNodePostprocessor.
func NewBaseNodePostprocessor(opts ...BaseNodePostprocessorOption) *BaseNodePostprocessor {
	p := &BaseNodePostprocessor{
		name: "BaseNodePostprocessor",
	}

	for _, opt := range opts {
		opt(p)
	}

	return p
}

// Name returns the name of the postprocessor.
func (p *BaseNodePostprocessor) Name() string {
	return p.name
}

// PostprocessNodes is a no-op implementation that returns nodes unchanged.
func (p *BaseNodePostprocessor) PostprocessNodes(
	ctx context.Context,
	nodes []schema.NodeWithScore,
	queryBundle *schema.QueryBundle,
) ([]schema.NodeWithScore, error) {
	return nodes, nil
}

// Ensure BaseNodePostprocessor implements NodePostprocessor.
var _ NodePostprocessor = (*BaseNodePostprocessor)(nil)

// PostprocessorChain chains multiple postprocessors together.
type PostprocessorChain struct {
	postprocessors []NodePostprocessor
}

// NewPostprocessorChain creates a new PostprocessorChain.
func NewPostprocessorChain(postprocessors ...NodePostprocessor) *PostprocessorChain {
	return &PostprocessorChain{
		postprocessors: postprocessors,
	}
}

// PostprocessNodes runs all postprocessors in sequence.
func (c *PostprocessorChain) PostprocessNodes(
	ctx context.Context,
	nodes []schema.NodeWithScore,
	queryBundle *schema.QueryBundle,
) ([]schema.NodeWithScore, error) {
	currentNodes := nodes

	for _, pp := range c.postprocessors {
		var err error
		currentNodes, err = pp.PostprocessNodes(ctx, currentNodes, queryBundle)
		if err != nil {
			return nil, err
		}
	}

	return currentNodes, nil
}

// Name returns the name of the chain.
func (c *PostprocessorChain) Name() string {
	return "PostprocessorChain"
}

// Add adds a postprocessor to the chain.
func (c *PostprocessorChain) Add(pp NodePostprocessor) {
	c.postprocessors = append(c.postprocessors, pp)
}

// Postprocessors returns the postprocessors in the chain.
func (c *PostprocessorChain) Postprocessors() []NodePostprocessor {
	return c.postprocessors
}

// Ensure PostprocessorChain implements NodePostprocessor.
var _ NodePostprocessor = (*PostprocessorChain)(nil)
