// Package krait wraps Cobra and Viper into a single fluent API.
// This package is not safe for concurrent use across multiple goroutines;
// it is designed for single-goroutine CLI applications matching Cobra's own
// documented threading model.
package krait

import (
	"time"

	"github.com/spf13/cast"
	"github.com/spf13/viper"
)

var (
	currentViper *viper.Viper
)

var newViper = viper.New

// getViperInstance returns the active command's viper instance, or a bare
// viper instance if no command is currently executing.
func getViperInstance() *viper.Viper {
	if currentViper != nil {
		return currentViper
	}
	return viper.New()
}

// Reset clears all global krait state. Intended for use between tests.
func Reset() {
	currentViper = nil
	currentCommand = nil
	rootCommand = nil
}

// IsSet reports whether the named parameter key was explicitly set in the
// currently-executing command. Delegates to the current command's viper
// instance. Returns false when called before Execute().
func IsSet(key string) bool {
	return getViperInstance().IsSet(key)
}

// Get returns the value associated with the key as an interface{}
func Get(key string) interface{} {
	return getViperInstance().Get(key)
}

// GetBool returns the value associated with the key as a boolean
func GetBool(key string) bool {
	return getViperInstance().GetBool(key)
}

// GetFloat32 returns the value associated with the key as a float32
func GetFloat32(key string) float32 {
	return cast.ToFloat32(getViperInstance().Get(key))
}

// GetFloat64 returns the value associated with the key as a float64
func GetFloat64(key string) float64 {
	return getViperInstance().GetFloat64(key)
}

// GetInt returns the value associated with the key as an integer
func GetInt(key string) int {
	return getViperInstance().GetInt(key)
}

// GetInt8 returns the value associated with the key as an int8
func GetInt8(key string) int8 {
	return cast.ToInt8(getViperInstance().Get(key))
}

// GetInt16 returns the value associated with the key as an int16
func GetInt16(key string) int16 {
	return cast.ToInt16(getViperInstance().Get(key))
}

// GetInt32 returns the value associated with the key as an int32
func GetInt32(key string) int32 {
	return getViperInstance().GetInt32(key)
}

// GetInt64 returns the value associated with the key as an int64
func GetInt64(key string) int64 {
	return getViperInstance().GetInt64(key)
}

// GetString returns the value associated with the key as a string
func GetString(key string) string {
	return getViperInstance().GetString(key)
}

// GetUint returns the value associated with the key as an unsigned integer
func GetUint(key string) uint {
	return getViperInstance().GetUint(key)
}

// GetUint8 returns the value associated with the key as an uint8
func GetUint8(key string) uint8 {
	return cast.ToUint8(getViperInstance().Get(key))
}

// GetUint16 returns the value associated with the key as an uint16
func GetUint16(key string) uint16 {
	return getViperInstance().GetUint16(key)
}

// GetUint32 returns the value associated with the key as an unsigned integer
func GetUint32(key string) uint32 {
	return getViperInstance().GetUint32(key)
}

// GetUint64 returns the value associated with the key as an unsigned integer
func GetUint64(key string) uint64 {
	return getViperInstance().GetUint64(key)
}

// GetDuration returns the value associated with the key as a time.Duration
func GetDuration(key string) time.Duration {
	return getViperInstance().GetDuration(key)
}

// AllSettings returns all settings as a map[string]interface{}
func AllSettings() map[string]interface{} {
	return getViperInstance().AllSettings()
}

// AsJson returns all settings as a JSON string
func AsJson() string {
	return asJson(AllSettings())
}

// GetStringSlice returns the value associated with the key as a slice of strings
func GetStringSlice(key string) []string {
	return getViperInstance().GetStringSlice(key)
}

// GetStringToString returns the value associated with the key as a map of strings
func GetStringToString(key string) map[string]string {
	return getViperInstance().GetStringMapString(key)
}

// GetIntSlice returns the value associated with the key as a slice of ints.
func GetIntSlice(key string) []int {
	return getViperInstance().GetIntSlice(key)
}

// GetTime returns the value associated with the key as a time.Time.
// The value must come from a config file or environment variable; no CLI flag type exists.
func GetTime(key string) time.Time {
	return cast.ToTime(getViperInstance().Get(key))
}

// GetSizeInBytes returns the value associated with the key as a uint, parsing size
// strings such as "10mb" or "1gb".
// The value must come from a config file or environment variable; no CLI flag type exists.
func GetSizeInBytes(key string) uint {
	return getViperInstance().GetSizeInBytes(key)
}

// GetStringMapStringSlice returns the value associated with the key as a map[string][]string.
// The value must come from a config file or environment variable; no CLI flag type exists.
func GetStringMapStringSlice(key string) map[string][]string {
	return getViperInstance().GetStringMapStringSlice(key)
}

// Unmarshal decodes all named-parameter values from the currently-executing command
// into the value pointed to by v using mapstructure. Returns an error if decoding fails
// or if called before Execute() (in which case a bare Viper with no keys is decoded).
// Must be called from within Run, BeforeRun, AfterRun, or SanityCheck.
func Unmarshal(v any) error {
	return getViperInstance().Unmarshal(v)
}

func IsDebug() bool {
	return (currentCommand != nil && currentCommand.IsDebug())
}

func Execute() error {
	if rootCommand == nil {
		panic("root command not set")
	}
	return rootCommand.Execute()
}

func Root() *Command {
	return rootCommand
}

func Current() *Command {
	return currentCommand
}

// Sub returns a scoped view of the named-parameter config tree of the
// currently-executing command rooted at key. Delegates to Current().Sub(key).
// Returns nil if there is no current command or if the key is not a non-empty
// subtree. Must be called from within Run, BeforeRun, AfterRun, or SanityCheck.
func Sub(key string) map[string]any {
	c := Current()
	if c == nil {
		return nil
	}
	return c.Sub(key)
}

// WriteConfig writes the resolved named-parameter configuration of the
// currently-executing command to its configured config file path. Delegates to
// Current().WriteConfig(). Must be called from within Run, BeforeRun, AfterRun,
// or SanityCheck.
func WriteConfig() error {
	return withExecutingCommand((*Command).WriteConfig)
}

// SafeWriteConfig writes the resolved named-parameter configuration of the
// currently-executing command to its config file path, returning an error if
// the file already exists. Delegates to Current().SafeWriteConfig(). Must be
// called from within Run, BeforeRun, AfterRun, or SanityCheck.
func SafeWriteConfig() error {
	return withExecutingCommand((*Command).SafeWriteConfig)
}

// WriteConfigAs writes the resolved named-parameter configuration of the
// currently-executing command to path. Delegates to Current().WriteConfigAs(path).
// Must be called from within Run, BeforeRun, AfterRun, or SanityCheck.
func WriteConfigAs(path string) error {
	return withExecutingCommandPath(path, (*Command).WriteConfigAs)
}

// SafeWriteConfigAs writes the resolved named-parameter configuration of the
// currently-executing command to path, returning an error if the file already
// exists. Delegates to Current().SafeWriteConfigAs(path). Must be called from
// within Run, BeforeRun, AfterRun, or SanityCheck.
func SafeWriteConfigAs(path string) error {
	return withExecutingCommandPath(path, (*Command).SafeWriteConfigAs)
}

func App(name string, description string, longDescription string) *Command {
	if rootCommand != nil {
		panic("root command already exists - cannot create another root command")
	}
	rootCommand = New(name, description, longDescription)
	return rootCommand
}
