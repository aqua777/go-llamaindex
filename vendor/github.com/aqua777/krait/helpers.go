package krait

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/spf13/cobra"
)

func getDefaultValue[T any](defaultValue []T) T {
	var value T
	if len(defaultValue) > 0 {
		value = defaultValue[0]
	}
	return value
}

func isDefault[T any](defaultValue T) bool {
	var v any = defaultValue
	switch value := v.(type) {
	case string:
		return value == ""
	case int:
		return value == 0
	case int8:
		return value == 0
	case int16:
		return value == 0
	case int32:
		return value == 0
	case int64:
		return value == 0
	case uint:
		return value == 0
	case uint8:
		return value == 0
	case uint16:
		return value == 0
	case uint32:
		return value == 0
	case uint64:
		return value == 0
	case float32:
		return value == 0
	case float64:
		return value == 0
	case bool:
		return !value
	case time.Duration:
		return value == 0
	case []string:
		return len(value) == 0
	case map[string]string:
		return len(value) == 0
	}
	return false
}

func getDummyRunner(commandName string) func(*cobra.Command, []string) error {
	return func(cmd *cobra.Command, _ []string) error {
		if cmd.HasSubCommands() {
			fmt.Fprintln(usageOutput, cmd.UsageString())
			return nil
		}
		return fmt.Errorf("command '%s' cannot be executed", commandName)
	}
}

func asJson(v any) string {
	json, _ := json.MarshalIndent(v, "", "  ")
	return string(json)
}
