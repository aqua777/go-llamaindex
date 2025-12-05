package krait

import (
	"sync"
	"time"

	"github.com/spf13/cast"
	"github.com/spf13/viper"
)

var (
	currentViper *viper.Viper
	viperOnce    sync.Once
	viperLock    sync.Mutex
)

var newViper = viper.New

// getViperInstance returns the appropriate viper instance to use
func getViperInstance() *viper.Viper {
	viperLock.Lock()
	defer viperLock.Unlock()

	viperOnce.Do(func() {
		if currentCommand != nil && currentCommand.viper != nil {
			currentViper = currentCommand.viper
		} else {
			currentViper = viper.New()
		}
	})
	return currentViper
}

func Reset() {
	viperLock.Lock()
	defer viperLock.Unlock()

	currentViper = nil
	currentCommand = nil
	rootCommand = nil
	viperOnce = sync.Once{}
}

// func IsSet(key string) bool {
// 	return getViperInstance().IsSet(key)
// }

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

func App(name string, description string, longDescription string) *Command {
	if rootCommand != nil {
		panic("root command already exists - cannot create another root command")
	}
	rootCommand = New(name, description, longDescription)
	return rootCommand
}
