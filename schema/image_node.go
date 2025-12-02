package schema

import (
	"crypto/sha256"
	"encoding/hex"

	"github.com/google/uuid"
)

// ImageNode represents a node with image content.
// It extends the base Node with image-specific fields.
type ImageNode struct {
	Node
	// Image is the base64 encoded image data.
	Image string `json:"image,omitempty"`
	// ImagePath is the local filesystem path to the image.
	ImagePath string `json:"image_path,omitempty"`
	// ImageURL is the URL to the image.
	ImageURL string `json:"image_url,omitempty"`
	// ImageMimeType is the MIME type of the image.
	ImageMimeType string `json:"image_mimetype,omitempty"`
	// TextEmbedding is the embedding of the text field if filled out.
	TextEmbedding []float64 `json:"text_embedding,omitempty"`
}

// NewImageNode creates a new ImageNode with default values.
func NewImageNode() *ImageNode {
	return &ImageNode{
		Node: Node{
			ID:                uuid.New().String(),
			Type:              ObjectTypeImage,
			Metadata:          make(map[string]interface{}),
			Relationships:     make(NodeRelationships),
			MetadataTemplate:  DefaultMetadataTemplate,
			MetadataSeparator: DefaultMetadataSeparator,
			TextTemplate:      DefaultTextNodeTemplate,
		},
	}
}

// NewImageNodeFromPath creates an ImageNode from a file path.
func NewImageNodeFromPath(path string, mimeType string) *ImageNode {
	node := NewImageNode()
	node.ImagePath = path
	node.ImageMimeType = mimeType
	node.Hash = node.GenerateHash()
	return node
}

// NewImageNodeFromURL creates an ImageNode from a URL.
func NewImageNodeFromURL(url string, mimeType string) *ImageNode {
	node := NewImageNode()
	node.ImageURL = url
	node.ImageMimeType = mimeType
	node.Hash = node.GenerateHash()
	return node
}

// NewImageNodeFromBase64 creates an ImageNode from base64 encoded data.
func NewImageNodeFromBase64(data string, mimeType string) *ImageNode {
	node := NewImageNode()
	node.Image = data
	node.ImageMimeType = mimeType
	node.Hash = node.GenerateHash()
	return node
}

// ClassName returns the class name for serialization.
func (n *ImageNode) ClassName() string {
	return "ImageNode"
}

// GetType returns the node type.
func (n *ImageNode) GetType() NodeType {
	return ObjectTypeImage
}

// GenerateHash generates a SHA256 hash of the image node content.
func (n *ImageNode) GenerateHash() string {
	// Image identity depends on which image source is set
	imageStr := n.Image
	if imageStr == "" {
		imageStr = "None"
	}
	imagePathStr := n.ImagePath
	if imagePathStr == "" {
		imagePathStr = "None"
	}
	imageURLStr := n.ImageURL
	if imageURLStr == "" {
		imageURLStr = "None"
	}
	imageText := n.Text
	if imageText == "" {
		imageText = "None"
	}

	docIdentity := imageStr + "-" + imagePathStr + "-" + imageURLStr + "-" + imageText
	h := sha256.Sum256([]byte(docIdentity))
	return hex.EncodeToString(h[:])
}

// ToDict converts the image node to a map representation.
func (n *ImageNode) ToDict() map[string]interface{} {
	result := n.Node.ToDict()
	result["class_name"] = n.ClassName()
	result["type"] = string(ObjectTypeImage)
	if n.Image != "" {
		result["image"] = n.Image
	}
	if n.ImagePath != "" {
		result["image_path"] = n.ImagePath
	}
	if n.ImageURL != "" {
		result["image_url"] = n.ImageURL
	}
	if n.ImageMimeType != "" {
		result["image_mimetype"] = n.ImageMimeType
	}
	if len(n.TextEmbedding) > 0 {
		result["text_embedding"] = n.TextEmbedding
	}
	return result
}

// HasImage returns true if the node has image content.
func (n *ImageNode) HasImage() bool {
	return n.Image != "" || n.ImagePath != "" || n.ImageURL != ""
}

// GetImageSource returns the image source (base64, path, or URL).
func (n *ImageNode) GetImageSource() string {
	if n.Image != "" {
		return n.Image
	}
	if n.ImagePath != "" {
		return n.ImagePath
	}
	return n.ImageURL
}
