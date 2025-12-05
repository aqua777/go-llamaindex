package postprocessor

import (
	"context"
	"sort"
	"time"

	"github.com/aqua777/go-llamaindex/schema"
)

// TimeWeightMode determines how time affects node scoring.
type TimeWeightMode string

const (
	// TimeWeightModeLinear applies linear decay based on age.
	TimeWeightModeLinear TimeWeightMode = "linear"
	// TimeWeightModeExponential applies exponential decay based on age.
	TimeWeightModeExponential TimeWeightMode = "exponential"
	// TimeWeightModeStep applies step function (recent vs old).
	TimeWeightModeStep TimeWeightMode = "step"
)

// NodeRecencyPostprocessor filters or reweights nodes based on recency.
type NodeRecencyPostprocessor struct {
	*BaseNodePostprocessor
	// DateKey is the metadata key containing the date/timestamp.
	DateKey string
	// DateFormat is the format string for parsing dates (Go time format).
	DateFormat string
	// MaxAge is the maximum age for nodes to be included.
	MaxAge time.Duration
	// TimeWeightMode determines how time affects scoring.
	TimeWeightMode TimeWeightMode
	// DecayRate is the decay rate for exponential mode (0-1).
	DecayRate float64
	// RecentThreshold is the threshold for step mode.
	RecentThreshold time.Duration
	// RecentWeight is the weight for recent nodes in step mode.
	RecentWeight float64
	// OldWeight is the weight for old nodes in step mode.
	OldWeight float64
	// TopK limits the number of nodes returned (0 = no limit).
	TopK int
	// SortByDate sorts nodes by date (newest first).
	SortByDate bool
	// Now is the reference time (defaults to time.Now()).
	Now func() time.Time
}

// NodeRecencyOption configures a NodeRecencyPostprocessor.
type NodeRecencyOption func(*NodeRecencyPostprocessor)

// WithRecencyDateKey sets the metadata key for dates.
func WithRecencyDateKey(key string) NodeRecencyOption {
	return func(p *NodeRecencyPostprocessor) {
		p.DateKey = key
	}
}

// WithRecencyDateFormat sets the date format string.
func WithRecencyDateFormat(format string) NodeRecencyOption {
	return func(p *NodeRecencyPostprocessor) {
		p.DateFormat = format
	}
}

// WithRecencyMaxAge sets the maximum age for nodes.
func WithRecencyMaxAge(maxAge time.Duration) NodeRecencyOption {
	return func(p *NodeRecencyPostprocessor) {
		p.MaxAge = maxAge
	}
}

// WithRecencyTimeWeightMode sets the time weight mode.
func WithRecencyTimeWeightMode(mode TimeWeightMode) NodeRecencyOption {
	return func(p *NodeRecencyPostprocessor) {
		p.TimeWeightMode = mode
	}
}

// WithRecencyDecayRate sets the decay rate for exponential mode.
func WithRecencyDecayRate(rate float64) NodeRecencyOption {
	return func(p *NodeRecencyPostprocessor) {
		p.DecayRate = rate
	}
}

// WithRecencyStepThreshold sets the threshold and weights for step mode.
func WithRecencyStepThreshold(threshold time.Duration, recentWeight, oldWeight float64) NodeRecencyOption {
	return func(p *NodeRecencyPostprocessor) {
		p.RecentThreshold = threshold
		p.RecentWeight = recentWeight
		p.OldWeight = oldWeight
	}
}

// WithRecencyTopK sets the maximum number of nodes to return.
func WithRecencyTopK(k int) NodeRecencyOption {
	return func(p *NodeRecencyPostprocessor) {
		p.TopK = k
	}
}

// WithRecencySortByDate enables sorting by date.
func WithRecencySortByDate(sortByDate bool) NodeRecencyOption {
	return func(p *NodeRecencyPostprocessor) {
		p.SortByDate = sortByDate
	}
}

// WithRecencyNowFunc sets a custom function for getting current time.
func WithRecencyNowFunc(fn func() time.Time) NodeRecencyOption {
	return func(p *NodeRecencyPostprocessor) {
		p.Now = fn
	}
}

// NewNodeRecencyPostprocessor creates a new NodeRecencyPostprocessor.
func NewNodeRecencyPostprocessor(opts ...NodeRecencyOption) *NodeRecencyPostprocessor {
	p := &NodeRecencyPostprocessor{
		BaseNodePostprocessor: NewBaseNodePostprocessor(WithPostprocessorName("NodeRecencyPostprocessor")),
		DateKey:               "date",
		DateFormat:            time.RFC3339,
		TimeWeightMode:        TimeWeightModeLinear,
		DecayRate:             0.5,
		RecentThreshold:       24 * time.Hour,
		RecentWeight:          1.0,
		OldWeight:             0.5,
		Now:                   time.Now,
	}

	for _, opt := range opts {
		opt(p)
	}

	return p
}

// PostprocessNodes processes nodes based on recency.
func (p *NodeRecencyPostprocessor) PostprocessNodes(
	ctx context.Context,
	nodes []schema.NodeWithScore,
	queryBundle *schema.QueryBundle,
) ([]schema.NodeWithScore, error) {
	now := p.Now()

	// Parse dates and filter/weight nodes
	type nodeWithTime struct {
		node schema.NodeWithScore
		time time.Time
	}

	var nodesWithTime []nodeWithTime

	for _, nodeWithScore := range nodes {
		nodeTime, ok := p.getNodeTime(nodeWithScore.Node)
		if !ok {
			// If no date, include with zero time (oldest)
			nodesWithTime = append(nodesWithTime, nodeWithTime{
				node: nodeWithScore,
				time: time.Time{},
			})
			continue
		}

		// Check max age filter
		if p.MaxAge > 0 {
			age := now.Sub(nodeTime)
			if age > p.MaxAge {
				continue // Skip nodes older than max age
			}
		}

		// Apply time-based weight adjustment
		adjustedScore := p.adjustScore(nodeWithScore.Score, nodeTime, now)

		nodesWithTime = append(nodesWithTime, nodeWithTime{
			node: schema.NodeWithScore{
				Node:  nodeWithScore.Node,
				Score: adjustedScore,
			},
			time: nodeTime,
		})
	}

	// Sort by date if requested
	if p.SortByDate {
		sort.Slice(nodesWithTime, func(i, j int) bool {
			return nodesWithTime[i].time.After(nodesWithTime[j].time)
		})
	} else {
		// Sort by adjusted score
		sort.Slice(nodesWithTime, func(i, j int) bool {
			return nodesWithTime[i].node.Score > nodesWithTime[j].node.Score
		})
	}

	// Apply top K limit
	if p.TopK > 0 && len(nodesWithTime) > p.TopK {
		nodesWithTime = nodesWithTime[:p.TopK]
	}

	// Extract nodes
	result := make([]schema.NodeWithScore, len(nodesWithTime))
	for i, nwt := range nodesWithTime {
		result[i] = nwt.node
	}

	return result, nil
}

// getNodeTime extracts the time from a node's metadata.
func (p *NodeRecencyPostprocessor) getNodeTime(node schema.Node) (time.Time, bool) {
	if node.Metadata == nil {
		return time.Time{}, false
	}

	dateVal, exists := node.Metadata[p.DateKey]
	if !exists {
		return time.Time{}, false
	}

	// Try different date formats
	switch v := dateVal.(type) {
	case time.Time:
		return v, true
	case string:
		t, err := time.Parse(p.DateFormat, v)
		if err != nil {
			// Try common formats
			formats := []string{
				time.RFC3339,
				"2006-01-02T15:04:05Z",
				"2006-01-02 15:04:05",
				"2006-01-02",
				"01/02/2006",
				"Jan 2, 2006",
			}
			for _, format := range formats {
				t, err = time.Parse(format, v)
				if err == nil {
					return t, true
				}
			}
			return time.Time{}, false
		}
		return t, true
	case int64:
		return time.Unix(v, 0), true
	case float64:
		return time.Unix(int64(v), 0), true
	}

	return time.Time{}, false
}

// adjustScore adjusts the score based on time.
func (p *NodeRecencyPostprocessor) adjustScore(score float64, nodeTime, now time.Time) float64 {
	age := now.Sub(nodeTime)

	switch p.TimeWeightMode {
	case TimeWeightModeLinear:
		return p.linearWeight(score, age)
	case TimeWeightModeExponential:
		return p.exponentialWeight(score, age)
	case TimeWeightModeStep:
		return p.stepWeight(score, age)
	default:
		return score
	}
}

// linearWeight applies linear decay.
func (p *NodeRecencyPostprocessor) linearWeight(score float64, age time.Duration) float64 {
	if p.MaxAge <= 0 {
		return score
	}

	// Weight decreases linearly from 1.0 to 0.0 over MaxAge
	weight := 1.0 - float64(age)/float64(p.MaxAge)
	if weight < 0 {
		weight = 0
	}

	return score * weight
}

// exponentialWeight applies exponential decay.
func (p *NodeRecencyPostprocessor) exponentialWeight(score float64, age time.Duration) float64 {
	// Half-life based decay
	halfLife := float64(24 * time.Hour) // Default 1 day half-life
	if p.MaxAge > 0 {
		halfLife = float64(p.MaxAge) / 2
	}

	weight := p.DecayRate
	for t := float64(0); t < float64(age); t += halfLife {
		weight *= p.DecayRate
	}

	return score * weight
}

// stepWeight applies step function.
func (p *NodeRecencyPostprocessor) stepWeight(score float64, age time.Duration) float64 {
	if age <= p.RecentThreshold {
		return score * p.RecentWeight
	}
	return score * p.OldWeight
}

// Ensure NodeRecencyPostprocessor implements NodePostprocessor.
var _ NodePostprocessor = (*NodeRecencyPostprocessor)(nil)
