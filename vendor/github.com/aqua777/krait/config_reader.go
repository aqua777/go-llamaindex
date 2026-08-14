package krait

import (
	"bytes"
	"io"

	"github.com/spf13/viper"
)

// readNamedVipersFromReader drains r once and loads the same bytes into v1 and v2
// via ReadConfig so one-shot readers work for both Viper instances.
func readNamedVipersFromReader(v1, v2 *viper.Viper, r io.Reader, configType string) error {
	data, err := io.ReadAll(r)
	if err != nil {
		return err
	}
	for _, v := range []*viper.Viper{v1, v2} {
		v.SetConfigType(configType)
		if err := v.ReadConfig(bytes.NewReader(data)); err != nil {
			return err
		}
	}
	return nil
}
