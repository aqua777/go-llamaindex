package nodeparser

import (
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/aqua777/go-llamaindex/schema"
)

// MetadataKeyJSONPath is the metadata key for the JSON path of each leaf value.
const MetadataKeyJSONPath = "json_path"

// JSONNodeParser parses JSON documents into one node per leaf scalar value.
type JSONNodeParser struct {
	*BaseNodeParser
}

var _ NodeParser = (*JSONNodeParser)(nil)

// NewJSONNodeParser creates a new JSONNodeParser.
func NewJSONNodeParser() *JSONNodeParser {
	return &JSONNodeParser{BaseNodeParser: NewBaseNodeParser()}
}

// WithIncludeMetadata sets whether to include parent metadata in child nodes.
func (p *JSONNodeParser) WithIncludeMetadata(include bool) NodeParserWithOptions {
	p.BaseNodeParser.WithIncludeMetadata(include)
	return p
}

// WithIncludePrevNextRel sets whether to establish PREVIOUS/NEXT relationships.
func (p *JSONNodeParser) WithIncludePrevNextRel(include bool) NodeParserWithOptions {
	p.BaseNodeParser.WithIncludePrevNextRel(include)
	return p
}

// GetNodesFromDocuments parses documents into nodes.
func (p *JSONNodeParser) GetNodesFromDocuments(documents []schema.Document) []*schema.Node {
	var allNodes []*schema.Node
	for _, doc := range documents {
		p.appendParsedJSONForSource(&allNodes, doc.ID, doc.Text, nil, &doc, "source_doc_id", doc.ID)
	}
	return allNodes
}

// ParseNodes parses existing nodes into smaller nodes.
func (p *JSONNodeParser) ParseNodes(nodes []*schema.Node) []*schema.Node {
	var allNodes []*schema.Node
	for _, node := range nodes {
		p.appendParsedJSONForSource(&allNodes, node.ID, node.Text, node, nil, "source_node_id", node.ID)
	}
	return allNodes
}

func (p *JSONNodeParser) appendParsedJSONForSource(
	allNodes *[]*schema.Node,
	id string,
	text string,
	parentNode *schema.Node,
	parentDoc *schema.Document,
	sourceMetaKey string,
	sourceMetaVal string,
) {
	p.EmitStart(id)

	parts, err := jsonLeavesFromString(text)
	if err != nil {
		p.EmitError(id, err)
		return
	}

	nodes := buildNodesFromTextParts(p.BaseNodeParser, parts, parentNode, parentDoc)
	for _, n := range nodes {
		n.Metadata[sourceMetaKey] = sourceMetaVal
	}

	*allNodes = append(*allNodes, nodes...)
	p.EmitComplete(id, len(nodes))
}

func jsonLeavesFromString(s string) ([]textPart, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, fmt.Errorf("empty JSON input")
	}
	var v interface{}
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		return nil, err
	}
	var out []textPart
	walkJSONLeaves(v, "", &out)
	return out, nil
}

func jsonPathAppendKey(prefix, key string) string {
	if prefix == "" {
		return key
	}
	return prefix + "." + key
}

func jsonPathAppendIndex(prefix string, i int) string {
	seg := "[" + strconv.Itoa(i) + "]"
	if prefix == "" {
		return seg
	}
	return prefix + seg
}

func jsonValueToLeafText(v interface{}) string {
	switch t := v.(type) {
	case string:
		return t
	case float64:
		return strconv.FormatFloat(t, 'g', -1, 64)
	case bool:
		if t {
			return "true"
		}
		return "false"
	case nil:
		return "null"
	default:
		b, err := json.Marshal(t)
		if err != nil {
			return fmt.Sprintf("%v", t)
		}
		return string(b)
	}
}

func walkJSONLeaves(v interface{}, path string, out *[]textPart) {
	switch val := v.(type) {
	case map[string]interface{}:
		keys := make([]string, 0, len(val))
		for k := range val {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			walkJSONLeaves(val[k], jsonPathAppendKey(path, k), out)
		}
	case []interface{}:
		for i, el := range val {
			walkJSONLeaves(el, jsonPathAppendIndex(path, i), out)
		}
	default:
		p := path
		if p == "" {
			p = "$"
		}
		*out = append(*out, textPart{
			Text: jsonValueToLeafText(val),
			Meta: map[string]interface{}{MetadataKeyJSONPath: p},
		})
	}
}
