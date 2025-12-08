package krait

import "time"

func (me *Command) withLongFlag(varPtr any, flag string, description string, defaultValue any) *Command {
	switch v := defaultValue.(type) {
	case string:
		if varPtr != nil {
			me.cmd.Flags().StringVar(varPtr.(*string), flag, v, description)
		} else {
			me.cmd.Flags().String(flag, v, description)
		}
	case bool:
		if varPtr != nil {
			me.cmd.Flags().BoolVar(varPtr.(*bool), flag, v, description)
		} else {
			me.cmd.Flags().Bool(flag, v, description)
		}
	case int:
		if varPtr != nil {
			me.cmd.Flags().IntVar(varPtr.(*int), flag, v, description)
		} else {
			me.cmd.Flags().Int(flag, v, description)
		}
	case int8:
		if varPtr != nil {
			me.cmd.Flags().Int8Var(varPtr.(*int8), flag, v, description)
		} else {
			me.cmd.Flags().Int8(flag, v, description)
		}
	case int16:
		if varPtr != nil {
			me.cmd.Flags().Int16Var(varPtr.(*int16), flag, v, description)
		} else {
			me.cmd.Flags().Int16(flag, v, description)
		}
	case int32:
		if varPtr != nil {
			me.cmd.Flags().Int32Var(varPtr.(*int32), flag, v, description)
		} else {
			me.cmd.Flags().Int32(flag, v, description)
		}
	case int64:
		if varPtr != nil {
			me.cmd.Flags().Int64Var(varPtr.(*int64), flag, v, description)
		} else {
			me.cmd.Flags().Int64(flag, v, description)
		}
	case uint:
		if varPtr != nil {
			me.cmd.Flags().UintVar(varPtr.(*uint), flag, v, description)
		} else {
			me.cmd.Flags().Uint(flag, v, description)
		}
	case uint8:
		if varPtr != nil {
			me.cmd.Flags().Uint8Var(varPtr.(*uint8), flag, v, description)
		} else {
			me.cmd.Flags().Uint8(flag, v, description)
		}
	case uint16:
		if varPtr != nil {
			me.cmd.Flags().Uint16Var(varPtr.(*uint16), flag, v, description)
		} else {
			me.cmd.Flags().Uint16(flag, v, description)
		}
	case uint32:
		if varPtr != nil {
			me.cmd.Flags().Uint32Var(varPtr.(*uint32), flag, v, description)
		} else {
			me.cmd.Flags().Uint32(flag, v, description)
		}
	case uint64:
		if varPtr != nil {
			me.cmd.Flags().Uint64Var(varPtr.(*uint64), flag, v, description)
		} else {
			me.cmd.Flags().Uint64(flag, v, description)
		}
	case float32:
		if varPtr != nil {
			me.cmd.Flags().Float32Var(varPtr.(*float32), flag, v, description)
		} else {
			me.cmd.Flags().Float32(flag, v, description)
		}
	case float64:
		if varPtr != nil {
			me.cmd.Flags().Float64Var(varPtr.(*float64), flag, v, description)
		} else {
			me.cmd.Flags().Float64(flag, v, description)
		}
	case time.Duration:
		if varPtr != nil {
			me.cmd.Flags().DurationVar(varPtr.(*time.Duration), flag, v, description)
		} else {
			me.cmd.Flags().Duration(flag, v, description)
		}
	case []string:
		if varPtr != nil {
			me.cmd.Flags().StringSliceVar(varPtr.(*[]string), flag, v, description)
		} else {
			me.cmd.Flags().StringSlice(flag, v, description)
		}
	case map[string]string:
		if varPtr != nil {
			me.cmd.Flags().StringToStringVar(varPtr.(*map[string]string), flag, v, description)
		} else {
			me.cmd.Flags().StringToString(flag, v, description)
		}
	}
	return me
}

func (me *Command) withShortFlag(varPtr any, flag, shortFlag, description string, defaultValue any) *Command {
	switch v := defaultValue.(type) {
	case string:
		if varPtr != nil {
			me.cmd.Flags().StringVarP(varPtr.(*string), flag, shortFlag, v, description)
		} else {
			me.cmd.Flags().StringP(flag, shortFlag, v, description)
		}
	case bool:
		if varPtr != nil {
			me.cmd.Flags().BoolVarP(varPtr.(*bool), flag, shortFlag, v, description)
		} else {
			me.cmd.Flags().BoolP(flag, shortFlag, v, description)
		}
	case int:
		if varPtr != nil {
			me.cmd.Flags().IntVarP(varPtr.(*int), flag, shortFlag, v, description)
		} else {
			me.cmd.Flags().IntP(flag, shortFlag, v, description)
		}
	case int8:
		if varPtr != nil {
			me.cmd.Flags().Int8VarP(varPtr.(*int8), flag, shortFlag, v, description)
		} else {
			me.cmd.Flags().Int8P(flag, shortFlag, v, description)
		}
	case int16:
		if varPtr != nil {
			me.cmd.Flags().Int16VarP(varPtr.(*int16), flag, shortFlag, v, description)
		} else {
			me.cmd.Flags().Int16P(flag, shortFlag, v, description)
		}
	case int32:
		if varPtr != nil {
			me.cmd.Flags().Int32VarP(varPtr.(*int32), flag, shortFlag, v, description)
		} else {
			me.cmd.Flags().Int32P(flag, shortFlag, v, description)
		}
	case int64:
		if varPtr != nil {
			me.cmd.Flags().Int64VarP(varPtr.(*int64), flag, shortFlag, v, description)
		} else {
			me.cmd.Flags().Int64P(flag, shortFlag, v, description)
		}
	case uint:
		if varPtr != nil {
			me.cmd.Flags().UintVarP(varPtr.(*uint), flag, shortFlag, v, description)
		} else {
			me.cmd.Flags().UintP(flag, shortFlag, v, description)
		}
	case uint8:
		if varPtr != nil {
			me.cmd.Flags().Uint8VarP(varPtr.(*uint8), flag, shortFlag, v, description)
		} else {
			me.cmd.Flags().Uint8P(flag, shortFlag, v, description)
		}
	case uint16:
		if varPtr != nil {
			me.cmd.Flags().Uint16VarP(varPtr.(*uint16), flag, shortFlag, v, description)
		} else {
			me.cmd.Flags().Uint16P(flag, shortFlag, v, description)
		}
	case uint32:
		if varPtr != nil {
			me.cmd.Flags().Uint32VarP(varPtr.(*uint32), flag, shortFlag, v, description)
		} else {
			me.cmd.Flags().Uint32P(flag, shortFlag, v, description)
		}
	case uint64:
		if varPtr != nil {
			me.cmd.Flags().Uint64VarP(varPtr.(*uint64), flag, shortFlag, v, description)
		} else {
			me.cmd.Flags().Uint64P(flag, shortFlag, v, description)
		}
	case float32:
		if varPtr != nil {
			me.cmd.Flags().Float32VarP(varPtr.(*float32), flag, shortFlag, v, description)
		} else {
			me.cmd.Flags().Float32P(flag, shortFlag, v, description)
		}
	case float64:
		if varPtr != nil {
			me.cmd.Flags().Float64VarP(varPtr.(*float64), flag, shortFlag, v, description)
		} else {
			me.cmd.Flags().Float64P(flag, shortFlag, v, description)
		}
	case time.Duration:
		if varPtr != nil {
			me.cmd.Flags().DurationVarP(varPtr.(*time.Duration), flag, shortFlag, v, description)
		} else {
			me.cmd.Flags().DurationP(flag, shortFlag, v, description)
		}
	case []string:
		if varPtr != nil {
			me.cmd.Flags().StringSliceVarP(varPtr.(*[]string), flag, shortFlag, v, description)
		} else {
			me.cmd.Flags().StringSliceP(flag, shortFlag, v, description)
		}
	case map[string]string:
		if varPtr != nil {
			me.cmd.Flags().StringToStringVarP(varPtr.(*map[string]string), flag, shortFlag, v, description)
		} else {
			me.cmd.Flags().StringToStringP(flag, shortFlag, v, description)
		}
	}
	return me
}

func (me *Command) withFlag(varPtr any, flag, shortFlag, description string, defaultValue any) *Command {
	if len(flag) > 0 {
		if len(shortFlag) > 0 {
			return me.withShortFlag(varPtr, flag, shortFlag, description, defaultValue)
		} else {
			return me.withLongFlag(varPtr, flag, description, defaultValue)
		}
	}
	return me
}
