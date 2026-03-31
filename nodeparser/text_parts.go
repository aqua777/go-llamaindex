package nodeparser

import "github.com/aqua777/go-llamaindex/schema"

// textPart is one logical chunk of text plus per-node metadata (e.g. html_tag, json_path).
type textPart struct {
	Text string
	Meta map[string]interface{}
}

func buildNodesFromTextParts(
	base *BaseNodeParser,
	parts []textPart,
	parentNode *schema.Node,
	parentDoc *schema.Document,
) []*schema.Node {
	if len(parts) == 0 {
		return nil
	}
	nodes := make([]*schema.Node, len(parts))
	for i, part := range parts {
		node := schema.NewNode()
		node.ID = base.GenerateID()
		node.Text = part.Text
		node.Type = schema.ObjectTypeText
		for k, v := range part.Meta {
			node.Metadata[k] = v
		}
		node.Metadata["chunk_index"] = i
		node.Metadata["chunk_count"] = len(parts)
		node.Hash = node.GenerateHash()
		nodes[i] = node
	}
	return base.PostProcessNodes(nodes, parentNode, parentDoc)
}
