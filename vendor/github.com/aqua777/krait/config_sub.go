package krait

import (
	"strings"

	"github.com/spf13/viper"
)

// viperNamedSubMap returns a scoped view of v at key: settings under key as a
// nested map[string]any with the key prefix stripped from paths, using the same
// per-key resolution order as Viper Get (flags, env, config, defaults).
// Returns nil if key is empty, or there are no descendant keys under key.
func viperNamedSubMap(v *viper.Viper, key string) map[string]any {
	if key == "" {
		return nil
	}
	// krait always uses viper.New() with the default "." delimiter.
	delim := "."
	pfx := strings.ToLower(key) + delim
	var under []string
	rootLower := strings.ToLower(key)
	for _, k := range v.AllKeys() {
		kk := strings.ToLower(k)
		if kk == rootLower {
			continue
		}
		if strings.HasPrefix(kk, pfx) {
			under = append(under, k)
		}
	}
	if len(under) == 0 {
		return nil
	}
	return nestMapFromPrefixedKeys(v, pfx, delim, under)
}

func nestMapFromPrefixedKeys(v *viper.Viper, prefixLowerPlusDelim string, delim string, fullKeys []string) map[string]any {
	root := make(map[string]any)
	for _, fullKey := range fullKeys {
		suffix := strings.TrimPrefix(strings.ToLower(fullKey), prefixLowerPlusDelim)
		if suffix == "" {
			continue
		}
		val := v.Get(fullKey)
		if val == nil {
			continue
		}
		placeNested(root, strings.Split(suffix, delim), val)
	}
	if len(root) == 0 {
		return nil
	}
	return root
}

func placeNested(m map[string]any, segments []string, val any) {
	if len(segments) == 0 {
		return
	}
	last := strings.ToLower(segments[len(segments)-1])
	if len(segments) == 1 {
		m[last] = val
		return
	}
	cur := m
	for _, seg := range segments[:len(segments)-1] {
		s := strings.ToLower(seg)
		next, ok := cur[s].(map[string]any)
		if !ok {
			next = make(map[string]any)
			cur[s] = next
		}
		cur = next
	}
	cur[last] = val
}
