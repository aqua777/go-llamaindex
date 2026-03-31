package nodeparser

import (
	"strings"
	"testing"

	"github.com/aqua777/go-llamaindex/schema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/net/html"
)

func TestNormalizeHTMLTagName(t *testing.T) {
	assert.Equal(t, "p", normalizeHTMLTagName(" P "))
	assert.Equal(t, "h1", normalizeHTMLTagName("H1"))
	assert.Equal(t, "", normalizeHTMLTagName("   "))
}

func TestHTMLTagSetFromList_DedupesAndSkipsEmpty(t *testing.T) {
	set := htmlTagSetFromList([]string{"P", "p", "", "  div  "})
	_, hasP := set["p"]
	_, hasDiv := set["div"]
	assert.True(t, hasP)
	assert.True(t, hasDiv)
	assert.Len(t, set, 2)
}

func TestHTMLTextContentSkippingUnsafe_SkipsScript(t *testing.T) {
	const input = `<div>visible<script>bad()</script><p>more</p></div>`
	doc, err := html.Parse(strings.NewReader(input))
	require.NoError(t, err)
	body := findFirstHTMLElement(doc, "div")
	require.NotNil(t, body)
	assert.Equal(t, "visiblemore", htmlTextContentSkippingUnsafe(body))
}

func TestHTMLFragmentsFromString_DefaultTags(t *testing.T) {
	const htmlStr = `<!doctype html><html><body>
<p>First para</p>
<h1>Title</h1>
<section><p>Inside</p></section>
</body></html>`
	set := htmlTagSetFromList(defaultHTMLExtractTags())
	frags, err := htmlFragmentsFromString(htmlStr, set)
	require.NoError(t, err)
	require.NotEmpty(t, frags)

	tags := make([]string, len(frags))
	for i, f := range frags {
		tags[i] = f.Tag
	}
	texts := make([]string, len(frags))
	for i, f := range frags {
		texts[i] = f.Text
	}
	assert.Contains(t, texts, "First para")
	assert.Contains(t, texts, "Title")
	assert.Contains(t, texts, "Inside")
	assert.Contains(t, tags, "p")
	assert.Contains(t, tags, "h1")
	assert.Contains(t, tags, "section")
}

func TestHTMLFragmentsFromString_CustomTagsOnly(t *testing.T) {
	const htmlStr = `<div><span class="x">ignored</span><article>Keep me</article></div>`
	set := htmlTagSetFromList([]string{"article"})
	frags, err := htmlFragmentsFromString(htmlStr, set)
	require.NoError(t, err)
	require.Len(t, frags, 1)
	assert.Equal(t, "article", frags[0].Tag)
	assert.Equal(t, "Keep me", frags[0].Text)
}

func TestHTMLNodeParser_GetNodesFromDocuments_DefaultTags(t *testing.T) {
	p := NewHTMLNodeParser()
	docs := []schema.Document{
		{
			ID:   "d1",
			Text: `<p>Alpha</p><h2>Beta</h2>`,
		},
	}
	nodes := p.GetNodesFromDocuments(docs)
	require.Len(t, nodes, 2)
	assert.Equal(t, "Alpha", nodes[0].Text)
	assert.Equal(t, "p", nodes[0].Metadata[MetadataKeyHTMLTag])
	assert.Equal(t, "Beta", nodes[1].Text)
	assert.Equal(t, "h2", nodes[1].Metadata[MetadataKeyHTMLTag])
	assert.Equal(t, "d1", nodes[0].Metadata["source_doc_id"])
}

func TestHTMLNodeParser_GetNodesFromDocuments_CustomTags(t *testing.T) {
	p := NewHTMLNodeParserWithTags([]string{"blockquote", "code"})
	docs := []schema.Document{
		{ID: "doc", Text: `<p>skip</p><blockquote cite="x">Quote</blockquote><code>fn()</code>`},
	}
	nodes := p.GetNodesFromDocuments(docs)
	require.Len(t, nodes, 2)
	assert.Equal(t, "Quote", nodes[0].Text)
	assert.Equal(t, "blockquote", nodes[0].Metadata[MetadataKeyHTMLTag])
	assert.Equal(t, "fn()", nodes[1].Text)
	assert.Equal(t, "code", nodes[1].Metadata[MetadataKeyHTMLTag])
}

func TestHTMLNodeParser_ParseNodes(t *testing.T) {
	p := NewHTMLNodeParser()
	parent := schema.NewTextNode(`<h3>Sub</h3>`)
	parent.ID = "parent-1"
	nodes := p.ParseNodes([]*schema.Node{parent})
	require.Len(t, nodes, 1)
	assert.Equal(t, "Sub", nodes[0].Text)
	assert.Equal(t, "h3", nodes[0].Metadata[MetadataKeyHTMLTag])
	assert.Equal(t, "parent-1", nodes[0].Metadata["source_node_id"])
}

func findFirstHTMLElement(root *html.Node, tag string) *html.Node {
	if root == nil {
		return nil
	}
	if root.Type == html.ElementNode && strings.EqualFold(root.Data, tag) {
		return root
	}
	for c := root.FirstChild; c != nil; c = c.NextSibling {
		if r := findFirstHTMLElement(c, tag); r != nil {
			return r
		}
	}
	return nil
}
