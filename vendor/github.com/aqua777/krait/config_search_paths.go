package krait

import "github.com/spf13/viper"

// readViperConfigFromSearchPaths sets the config base name, registers each search
// directory in declaration order, and loads the first matching file via ReadInConfig.
func readViperConfigFromSearchPaths(v *viper.Viper, name string, paths []string) error {
	v.SetConfigName(name)
	for _, p := range paths {
		v.AddConfigPath(p)
	}
	return v.ReadInConfig()
}
