// Package graphstore provides graph store interfaces and implementations for knowledge graphs.
package graphstore

import (
	"context"
	"encoding/json"
	"fmt"
)

// Triplet represents a knowledge graph triplet (subject, relation, object).
type Triplet struct {
	Subject  string `json:"subject"`
	Relation string `json:"relation"`
	Object   string `json:"object"`
}

// String returns the string representation of the triplet.
func (t Triplet) String() string {
	return fmt.Sprintf("(%s, %s, %s)", t.Subject, t.Relation, t.Object)
}

// MarshalJSON implements json.Marshaler.
func (t Triplet) MarshalJSON() ([]byte, error) {
	return json.Marshal([]string{t.Subject, t.Relation, t.Object})
}

// UnmarshalJSON implements json.Unmarshaler.
func (t *Triplet) UnmarshalJSON(data []byte) error {
	var arr []string
	if err := json.Unmarshal(data, &arr); err != nil {
		return err
	}
	if len(arr) != 3 {
		return fmt.Errorf("triplet must have exactly 3 elements")
	}
	t.Subject = arr[0]
	t.Relation = arr[1]
	t.Object = arr[2]
	return nil
}

// EntityNode represents an entity in the knowledge graph.
type EntityNode struct {
	Name       string                 `json:"name"`
	Label      string                 `json:"label"`
	Properties map[string]interface{} `json:"properties,omitempty"`
	Embedding  []float64              `json:"embedding,omitempty"`
}

// NewEntityNode creates a new entity node.
func NewEntityNode(name string) *EntityNode {
	return &EntityNode{
		Name:       name,
		Label:      "entity",
		Properties: make(map[string]interface{}),
	}
}

// ID returns the entity ID (the name).
func (e *EntityNode) ID() string {
	return e.Name
}

// String returns the string representation of the entity.
func (e *EntityNode) String() string {
	if len(e.Properties) > 0 {
		return fmt.Sprintf("%s (%v)", e.Name, e.Properties)
	}
	return e.Name
}

// Relation represents a relation between two entities.
type Relation struct {
	Label      string                 `json:"label"`
	SourceID   string                 `json:"source_id"`
	TargetID   string                 `json:"target_id"`
	Properties map[string]interface{} `json:"properties,omitempty"`
}

// NewRelation creates a new relation.
func NewRelation(label, sourceID, targetID string) *Relation {
	return &Relation{
		Label:      label,
		SourceID:   sourceID,
		TargetID:   targetID,
		Properties: make(map[string]interface{}),
	}
}

// ID returns the relation ID (the label).
func (r *Relation) ID() string {
	return r.Label
}

// String returns the string representation of the relation.
func (r *Relation) String() string {
	if len(r.Properties) > 0 {
		return fmt.Sprintf("%s (%v)", r.Label, r.Properties)
	}
	return r.Label
}

// GraphStore is the interface for knowledge graph stores.
// It provides methods for storing and retrieving triplets.
type GraphStore interface {
	// Get returns all triplets for a given subject.
	// Returns a list of [relation, object] pairs.
	Get(ctx context.Context, subj string) ([][]string, error)

	// GetRelMap returns a depth-aware relation map for the given subjects.
	// If subjs is nil, returns relations for all subjects.
	// The map keys are subjects, values are lists of [subject, relation, object] triplets.
	GetRelMap(ctx context.Context, subjs []string, depth int, limit int) (map[string][][]string, error)

	// UpsertTriplet adds or updates a triplet.
	UpsertTriplet(ctx context.Context, subj, rel, obj string) error

	// Delete removes a triplet.
	Delete(ctx context.Context, subj, rel, obj string) error

	// GetSchema returns the schema of the graph store.
	GetSchema(ctx context.Context, refresh bool) (string, error)

	// Query executes a query against the graph store.
	Query(ctx context.Context, query string, params map[string]interface{}) (interface{}, error)

	// Persist saves the graph store to the given path.
	Persist(ctx context.Context, path string) error

	// GetAllSubjects returns all subjects in the graph.
	GetAllSubjects(ctx context.Context) ([]string, error)
}

// PropertyGraphStore is an extended interface for property graph stores.
// It supports labeled nodes and relations with properties.
type PropertyGraphStore interface {
	GraphStore

	// GetNodes returns nodes matching the given criteria.
	GetNodes(ctx context.Context, properties map[string]interface{}, ids []string) ([]*EntityNode, error)

	// GetTriplets returns triplets matching the given criteria.
	GetTriplets(ctx context.Context, entityNames, relationNames []string, properties map[string]interface{}, ids []string) ([]Triplet, error)

	// UpsertNodes adds or updates nodes.
	UpsertNodes(ctx context.Context, nodes []*EntityNode) error

	// UpsertRelations adds or updates relations.
	UpsertRelations(ctx context.Context, relations []*Relation) error

	// DeleteNodes removes nodes matching the given criteria.
	DeleteNodes(ctx context.Context, entityNames, relationNames []string, properties map[string]interface{}, ids []string) error
}

// GraphStoreData holds the data for a simple graph store.
type GraphStoreData struct {
	// GraphDict maps subject to list of [relation, object] pairs.
	GraphDict map[string][][]string `json:"graph_dict"`
}

// NewGraphStoreData creates a new GraphStoreData.
func NewGraphStoreData() *GraphStoreData {
	return &GraphStoreData{
		GraphDict: make(map[string][][]string),
	}
}

// GetRelMap returns a depth-aware relation map.
func (d *GraphStoreData) GetRelMap(subjs []string, depth, limit int) map[string][][]string {
	if subjs == nil {
		subjs = make([]string, 0, len(d.GraphDict))
		for subj := range d.GraphDict {
			subjs = append(subjs, subj)
		}
	}

	relMap := make(map[string][][]string)
	for _, subj := range subjs {
		relMap[subj] = d.getRelMapForSubject(subj, depth, limit)
	}

	// Truncate to limit
	relCount := 0
	returnMap := make(map[string][][]string)
	for subj, rels := range relMap {
		if relCount+len(rels) > limit {
			returnMap[subj] = rels[:limit-relCount]
			break
		}
		returnMap[subj] = rels
		relCount += len(rels)
	}

	return returnMap
}

// getRelMapForSubject returns the relation map for a single subject.
func (d *GraphStoreData) getRelMapForSubject(subj string, depth, limit int) [][]string {
	if depth == 0 {
		return nil
	}

	var relMap [][]string
	relCount := 0

	if rels, ok := d.GraphDict[subj]; ok {
		for _, rel := range rels {
			if relCount >= limit {
				break
			}
			if len(rel) >= 2 {
				relMap = append(relMap, []string{subj, rel[0], rel[1]})
				// Recursively get relations for the object
				childRels := d.getRelMapForSubject(rel[1], depth-1, limit-relCount-1)
				relMap = append(relMap, childRels...)
				relCount++
			}
		}
	}

	return relMap
}

// ToJSON converts the data to JSON.
func (d *GraphStoreData) ToJSON() ([]byte, error) {
	return json.Marshal(d)
}

// FromJSON creates GraphStoreData from JSON.
func FromJSON(data []byte) (*GraphStoreData, error) {
	var d GraphStoreData
	if err := json.Unmarshal(data, &d); err != nil {
		return nil, err
	}
	if d.GraphDict == nil {
		d.GraphDict = make(map[string][][]string)
	}
	return &d, nil
}
