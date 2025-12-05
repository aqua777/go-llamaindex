package graphstore

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

const (
	// DefaultPersistDir is the default directory for persisting graph stores.
	DefaultPersistDir = "./storage"
	// DefaultPersistFilename is the default filename for graph store persistence.
	DefaultPersistFilename = "graph_store.json"
)

// SimpleGraphStore is a simple in-memory graph store.
// Triplets are stored in a dictionary mapping subjects to [relation, object] pairs.
type SimpleGraphStore struct {
	data *GraphStoreData
}

// SimpleGraphStoreOption configures SimpleGraphStore.
type SimpleGraphStoreOption func(*SimpleGraphStore)

// WithGraphStoreData sets the initial data.
func WithGraphStoreData(data *GraphStoreData) SimpleGraphStoreOption {
	return func(s *SimpleGraphStore) {
		s.data = data
	}
}

// NewSimpleGraphStore creates a new SimpleGraphStore.
func NewSimpleGraphStore(opts ...SimpleGraphStoreOption) *SimpleGraphStore {
	s := &SimpleGraphStore{
		data: NewGraphStoreData(),
	}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

// NewSimpleGraphStoreFromFile loads a SimpleGraphStore from a file.
func NewSimpleGraphStoreFromFile(path string) (*SimpleGraphStore, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return NewSimpleGraphStore(), nil
		}
		return nil, err
	}

	graphData, err := FromJSON(data)
	if err != nil {
		return nil, err
	}

	return NewSimpleGraphStore(WithGraphStoreData(graphData)), nil
}

// Get returns all triplets for a given subject.
func (s *SimpleGraphStore) Get(ctx context.Context, subj string) ([][]string, error) {
	if rels, ok := s.data.GraphDict[subj]; ok {
		return rels, nil
	}
	return nil, nil
}

// GetRelMap returns a depth-aware relation map.
func (s *SimpleGraphStore) GetRelMap(ctx context.Context, subjs []string, depth, limit int) (map[string][][]string, error) {
	return s.data.GetRelMap(subjs, depth, limit), nil
}

// UpsertTriplet adds or updates a triplet.
func (s *SimpleGraphStore) UpsertTriplet(ctx context.Context, subj, rel, obj string) error {
	if _, ok := s.data.GraphDict[subj]; !ok {
		s.data.GraphDict[subj] = make([][]string, 0)
	}

	// Check if triplet already exists
	for _, existing := range s.data.GraphDict[subj] {
		if len(existing) >= 2 && existing[0] == rel && existing[1] == obj {
			return nil // Already exists
		}
	}

	s.data.GraphDict[subj] = append(s.data.GraphDict[subj], []string{rel, obj})
	return nil
}

// Delete removes a triplet.
func (s *SimpleGraphStore) Delete(ctx context.Context, subj, rel, obj string) error {
	if rels, ok := s.data.GraphDict[subj]; ok {
		newRels := make([][]string, 0, len(rels))
		for _, r := range rels {
			if len(r) >= 2 && r[0] == rel && r[1] == obj {
				continue // Skip this one
			}
			newRels = append(newRels, r)
		}
		if len(newRels) == 0 {
			delete(s.data.GraphDict, subj)
		} else {
			s.data.GraphDict[subj] = newRels
		}
	}
	return nil
}

// GetSchema returns the schema of the graph store.
func (s *SimpleGraphStore) GetSchema(ctx context.Context, refresh bool) (string, error) {
	return "", fmt.Errorf("SimpleGraphStore does not support get_schema")
}

// Query executes a query against the graph store.
func (s *SimpleGraphStore) Query(ctx context.Context, query string, params map[string]interface{}) (interface{}, error) {
	return nil, fmt.Errorf("SimpleGraphStore does not support query")
}

// Persist saves the graph store to the given path.
func (s *SimpleGraphStore) Persist(ctx context.Context, path string) error {
	if path == "" {
		path = filepath.Join(DefaultPersistDir, DefaultPersistFilename)
	}

	// Create directory if needed
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	data, err := s.data.ToJSON()
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)
}

// GetAllSubjects returns all subjects in the graph.
func (s *SimpleGraphStore) GetAllSubjects(ctx context.Context) ([]string, error) {
	subjects := make([]string, 0, len(s.data.GraphDict))
	for subj := range s.data.GraphDict {
		subjects = append(subjects, subj)
	}
	return subjects, nil
}

// ToDict returns the graph store data as a map.
func (s *SimpleGraphStore) ToDict() map[string]interface{} {
	return map[string]interface{}{
		"graph_dict": s.data.GraphDict,
	}
}

// FromDict creates a SimpleGraphStore from a map.
func FromDict(data map[string]interface{}) (*SimpleGraphStore, error) {
	graphDict := make(map[string][][]string)

	gdRaw := data["graph_dict"]
	switch gd := gdRaw.(type) {
	case map[string][][]string:
		// Direct type (from ToDict)
		graphDict = gd
	case map[string]interface{}:
		// From JSON unmarshaling
		for subj, rels := range gd {
			switch relList := rels.(type) {
			case [][]string:
				graphDict[subj] = relList
			case []interface{}:
				graphDict[subj] = make([][]string, 0, len(relList))
				for _, rel := range relList {
					switch relArr := rel.(type) {
					case []string:
						graphDict[subj] = append(graphDict[subj], relArr)
					case []interface{}:
						strRel := make([]string, len(relArr))
						for i, v := range relArr {
							if str, ok := v.(string); ok {
								strRel[i] = str
							}
						}
						graphDict[subj] = append(graphDict[subj], strRel)
					}
				}
			}
		}
	}

	return NewSimpleGraphStore(WithGraphStoreData(&GraphStoreData{GraphDict: graphDict})), nil
}

// GetTriplets returns all triplets in the graph store.
func (s *SimpleGraphStore) GetTriplets(ctx context.Context) ([]Triplet, error) {
	var triplets []Triplet
	for subj, rels := range s.data.GraphDict {
		for _, rel := range rels {
			if len(rel) >= 2 {
				triplets = append(triplets, Triplet{
					Subject:  subj,
					Relation: rel[0],
					Object:   rel[1],
				})
			}
		}
	}
	return triplets, nil
}

// Size returns the number of subjects in the graph.
func (s *SimpleGraphStore) Size() int {
	return len(s.data.GraphDict)
}

// TripletCount returns the total number of triplets in the graph.
func (s *SimpleGraphStore) TripletCount() int {
	count := 0
	for _, rels := range s.data.GraphDict {
		count += len(rels)
	}
	return count
}

// Clear removes all data from the graph store.
func (s *SimpleGraphStore) Clear() {
	s.data.GraphDict = make(map[string][][]string)
}

// MarshalJSON implements json.Marshaler.
func (s *SimpleGraphStore) MarshalJSON() ([]byte, error) {
	return json.Marshal(s.data)
}

// UnmarshalJSON implements json.Unmarshaler.
func (s *SimpleGraphStore) UnmarshalJSON(data []byte) error {
	graphData, err := FromJSON(data)
	if err != nil {
		return err
	}
	s.data = graphData
	return nil
}

// Ensure SimpleGraphStore implements GraphStore.
var _ GraphStore = (*SimpleGraphStore)(nil)
