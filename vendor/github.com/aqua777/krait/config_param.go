package krait

import (
	"fmt"
	"time"

	"github.com/spf13/pflag"
	"github.com/spf13/viper"
)

type ConfigParamType string

// Type constants for configuration parameters
const (
	typeString         ConfigParamType = "string"
	typeInt            ConfigParamType = "int"
	typeInt8           ConfigParamType = "int8"
	typeInt16          ConfigParamType = "int16"
	typeInt32          ConfigParamType = "int32"
	typeInt64          ConfigParamType = "int64"
	typeUint           ConfigParamType = "uint"
	typeUint8          ConfigParamType = "uint8"
	typeUint16         ConfigParamType = "uint16"
	typeUint32         ConfigParamType = "uint32"
	typeUint64         ConfigParamType = "uint64"
	typeBool           ConfigParamType = "bool"
	typeFloat32        ConfigParamType = "float32"
	typeFloat64        ConfigParamType = "float64"
	typeDuration       ConfigParamType = "duration"
	typeStringSlice    ConfigParamType = "stringSlice"
	typeStringToString ConfigParamType = "stringToString"
)

// ConfigParam represents a configuration option for a command
type ConfigParam struct {
	Name               string
	Description        string
	Flag               string
	ShortFlag          string
	EnvironmentVarName string
	DefaultValue       any
	VarPtr             any
}

func (me *ConfigParam) String() string {
	return fmt.Sprintf("Name: %s, Flag: %s, ShortFlag: %s, EnvironmentVarName: %s, DefaultValue: %v", me.Name, me.Flag, me.ShortFlag, me.EnvironmentVarName, me.DefaultValue)
}

func (me *ConfigParam) setVarPtrValue(flag *pflag.Flag, viper *viper.Viper) {
	switch ConfigParamType(flag.Value.Type()) {
	case typeString:
		varPtr := me.VarPtr.(*string)
		*varPtr = viper.GetString(me.Flag)
	case typeInt:
		varPtr := me.VarPtr.(*int)
		*varPtr = viper.GetInt(me.Flag)
	case typeInt8:
		varPtr := me.VarPtr.(*int8)
		*varPtr = int8(viper.GetInt(me.Flag))
	case typeInt16:
		varPtr := me.VarPtr.(*int16)
		*varPtr = int16(viper.GetInt(me.Flag))
	case typeInt32:
		varPtr := me.VarPtr.(*int32)
		*varPtr = int32(viper.GetInt(me.Flag))
	case typeInt64:
		varPtr := me.VarPtr.(*int64)
		*varPtr = viper.GetInt64(me.Flag)
	case typeUint:
		varPtr := me.VarPtr.(*uint)
		*varPtr = uint(viper.GetInt(me.Flag))
	case typeUint8:
		varPtr := me.VarPtr.(*uint8)
		*varPtr = uint8(viper.GetInt(me.Flag))
	case typeUint16:
		varPtr := me.VarPtr.(*uint16)
		*varPtr = uint16(viper.GetInt(me.Flag))
	case typeUint32:
		varPtr := me.VarPtr.(*uint32)
		*varPtr = uint32(viper.GetInt(me.Flag))
	case typeUint64:
		varPtr := me.VarPtr.(*uint64)
		*varPtr = uint64(viper.GetInt(me.Flag))
	case typeBool:
		varPtr := me.VarPtr.(*bool)
		*varPtr = viper.GetBool(me.Flag)
	case typeFloat32:
		varPtr := me.VarPtr.(*float32)
		*varPtr = float32(viper.GetFloat64(me.Flag))
	case typeFloat64:
		varPtr := me.VarPtr.(*float64)
		*varPtr = viper.GetFloat64(me.Flag)
	case typeDuration:
		varPtr := me.VarPtr.(*time.Duration)
		*varPtr = viper.GetDuration(me.Flag)
	case typeStringSlice:
		varPtr := me.VarPtr.(*[]string)
		*varPtr = viper.GetStringSlice(me.Flag)
	case typeStringToString:
		varPtr := me.VarPtr.(*map[string]string)
		if *varPtr == nil {
			*varPtr = make(map[string]string)
		}
		*varPtr = viper.GetStringMapString(me.Flag)
	}
}

type ConfigParams struct {
	Params []*ConfigParam
}

func (me *ConfigParams) With(name, flag, shortFlag, environmentVarName, description string, defaultValue, varPtr any) *ConfigParams {
	me.Params = append(me.Params, &ConfigParam{
		Name:               name,
		Flag:               flag,
		ShortFlag:          shortFlag,
		EnvironmentVarName: environmentVarName,
		Description:        description,
		DefaultValue:       defaultValue,
		VarPtr:             varPtr,
	})
	return me
}

func (me *ConfigParams) List() []*ConfigParam {
	return me.Params
}

func (me *ConfigParams) ForEach(fn func(param *ConfigParam)) {
	if me == nil || fn == nil {
		return
	}
	for _, param := range me.Params {
		fn(param)
	}
}

func NewConfigParams() *ConfigParams {
	return &ConfigParams{
		Params: make([]*ConfigParam, 0),
	}
}
