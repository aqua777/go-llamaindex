package krait

import (
	"fmt"
	"io"
	"os"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

const (
	emptyStr         = ""
	debugModeMessage = "Enable debug mode"
)

type Command struct {
	cmd             *cobra.Command
	viper           *viper.Viper
	argsViper       *viper.Viper
	Name            string
	Description     string
	LongDescription string
	SubCommands     []*Command
	Params          *ConfigParams
	BeforeRun       func(args []string) error
	AfterRun        func(args []string) error
	Run             func(args []string) error
	SanityCheck     func() error
	ConfigFile      string
	debug           bool
}

var (
	rootCommand    *Command = nil
	currentCommand *Command = nil

	usageOutput io.Writer = os.Stdout
)

// func (me *Command) isSet(key string) bool {
// 	return me.viper.IsSet(key)
// }

func (me *Command) internalProcessParams() error {
	for _, option := range me.Params.List() {
		if option.Name != emptyStr {
			me.viper.SetDefault(option.Name, option.DefaultValue)
			me.viper.BindEnv(option.Name, option.EnvironmentVarName)
		} else {
			me.argsViper.BindEnv(option.Flag, option.EnvironmentVarName)
			me.argsViper.BindPFlag(option.Flag, me.cmd.Flags().Lookup(option.Flag))
		}
	}

	if len(me.ConfigFile) > 0 {
		for _, viperInstance := range []*viper.Viper{me.viper, me.argsViper} {
			viperInstance.SetConfigFile(me.ConfigFile)
			if err := viperInstance.ReadInConfig(); err != nil {
				return err
			}
		}
	}

	me.viper.AutomaticEnv()
	for _, option := range me.Params.List() {
		if option.Name != emptyStr {
			me.viper.BindPFlag(option.Name, me.cmd.Flags().Lookup(option.Flag))
		}
	}

	me.argsViper.AutomaticEnv()
	for _, option := range me.Params.List() {
		if flag := me.cmd.Flags().Lookup(option.Flag); flag != nil && !flag.Changed && me.argsViper.IsSet(option.Flag) {
			option.setVarPtrValue(flag, me.argsViper)
		}
	}
	return nil
}

func (me *Command) internalBeforeRun(args []string) error {
	if err := me.internalProcessParams(); err != nil {
		return err
	}
	if me.SanityCheck != nil {
		if err := me.SanityCheck(); err != nil {
			return err
		}
	}
	if me.BeforeRun != nil {
		return me.BeforeRun(args)
	}
	return nil
}

func (me *Command) internalAfterRun(args []string) error {
	if me.AfterRun != nil {
		return me.AfterRun(args)
	}
	return nil
}

func (me *Command) getRunWrapper(run func(args []string) error) func(cmd *cobra.Command, args []string) error {
	return func(cmd *cobra.Command, args []string) error {
		currentCommand = me
		if err := me.internalBeforeRun(args); err != nil {
			return err
		}
		if run == nil {
			return fmt.Errorf("run function is nil")
		}
		if err := run(args); err != nil {
			return err
		}
		if err := me.internalAfterRun(args); err != nil {
			return err
		}
		return nil
	}
}

func (me *Command) withParam(name string, varPtr any, flag, shortFlag, environmentVarName, description string, defaultValue any) *Command {
	me.Params.With(name, flag, shortFlag, environmentVarName, description, defaultValue, varPtr)
	return me.withFlag(varPtr, flag, shortFlag, description, defaultValue)
}

func (me *Command) WithConfig(configFile string, flag string, shortFlag string, environmentVarName string) *Command {
	return me.withParam(emptyStr, &me.ConfigFile, flag, shortFlag, environmentVarName, "Path to config file", configFile)
}

func (me *Command) WithDebug(flag string, shortFlag string, environmentVarName string) *Command {
	return me.withParam(emptyStr, &me.debug, flag, shortFlag, environmentVarName, debugModeMessage, false)
}

func (me *Command) WithParams(params *ConfigParams) *Command {
	params.ForEach(func(p *ConfigParam) {
		me.withParam(p.Name, p.VarPtr, p.Flag, p.ShortFlag, p.EnvironmentVarName, p.Description, p.DefaultValue)
	})
	return me
}

// *<Type> and *<Type>P functions acting on Viper config  ------------------------------------------------------------

// WithString adds a string parameter to the command with support for flag and environment variable configuration.
//
// This method configures a string parameter that can be set via command-line flag or environment variable.
// The parameter value can be accessed using the provided name through the configuration system.
//
// Parameters:
//   - name: The configuration key used to access this parameter's value
//   - description: Help text describing the parameter's purpose
//   - flag: The long form flag name used on the command line (without --)
//   - environmentVarName: The environment variable name that can set this parameter
//   - defaultValue: Optional default value for the parameter (empty string if not provided)
//
// Returns:
//   - *Command: Returns the command instance for method chaining
//
// Example:
//
//	cmd.WithString("config", "Path to config file", "config", "APP_CONFIG", "/etc/app/config.yaml")
func (me *Command) WithString(name string, description string, flag string, environmentVarName string, defaultValue ...string) *Command {
	return me.withParam(name, nil, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

// WithStringP adds a string parameter to the command with support for both long and short flags, and environment variable configuration.
//
// This method is similar to WithString but adds support for a short flag option (single letter flag with single hyphen).
// The parameter value can be accessed using the provided name through the configuration system.
//
// Parameters:
//   - name: The configuration key used to access this parameter's value
//   - description: Help text describing the parameter's purpose
//   - flag: The long form flag name used on the command line (without --)
//   - shortFlag: Single character flag name (without -)
//   - environmentVarName: The environment variable name that can set this parameter
//   - defaultValue: Optional default value for the parameter (empty string if not provided)
//
// Returns:
//   - *Command: Returns the command instance for method chaining
//
// Example:
//
//	cmd.WithStringP("config", "Path to config file", "config", "c", "APP_CONFIG", "/etc/app/config.yaml")
func (me *Command) WithStringP(name string, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...string) *Command {
	return me.withParam(name, nil, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithBool(name string, description string, flag string, environmentVarName string, defaultValue ...bool) *Command {
	return me.withParam(name, nil, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithBoolP(name string, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...bool) *Command {
	return me.withParam(name, nil, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithInt(name string, description string, flag string, environmentVarName string, defaultValue ...int) *Command {
	return me.withParam(name, nil, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithIntP(name string, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...int) *Command {
	return me.withParam(name, nil, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithInt8(name string, description string, flag string, environmentVarName string, defaultValue ...int8) *Command {
	return me.withParam(name, nil, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithInt8P(name string, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...int8) *Command {
	return me.withParam(name, nil, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithInt16(name string, description string, flag string, environmentVarName string, defaultValue ...int16) *Command {
	return me.withParam(name, nil, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithInt16P(name string, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...int16) *Command {
	return me.withParam(name, nil, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithInt32(name string, description string, flag string, environmentVarName string, defaultValue ...int32) *Command {
	return me.withParam(name, nil, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithInt32P(name string, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...int32) *Command {
	return me.withParam(name, nil, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithInt64(name string, description string, flag string, environmentVarName string, defaultValue ...int64) *Command {
	return me.withParam(name, nil, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithInt64P(name string, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...int64) *Command {
	return me.withParam(name, nil, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithUint(name string, description string, flag string, environmentVarName string, defaultValue ...uint) *Command {
	return me.withParam(name, nil, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithUintP(name string, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...uint) *Command {
	return me.withParam(name, nil, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithUint8(name string, description string, flag string, environmentVarName string, defaultValue ...uint8) *Command {
	return me.withParam(name, nil, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithUint8P(name string, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...uint8) *Command {
	return me.withParam(name, nil, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithUint16(name string, description string, flag string, environmentVarName string, defaultValue ...uint16) *Command {
	return me.withParam(name, nil, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithUint16P(name string, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...uint16) *Command {
	return me.withParam(name, nil, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithUint32(name string, description string, flag string, environmentVarName string, defaultValue ...uint32) *Command {
	return me.withParam(name, nil, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithUint32P(name string, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...uint32) *Command {
	return me.withParam(name, nil, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithUint64(name string, description string, flag string, environmentVarName string, defaultValue ...uint64) *Command {
	return me.withParam(name, nil, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithUint64P(name string, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...uint64) *Command {
	return me.withParam(name, nil, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithFloat32(name string, description string, flag string, environmentVarName string, defaultValue ...float32) *Command {
	return me.withParam(name, nil, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithFloat32P(name string, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...float32) *Command {
	return me.withParam(name, nil, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithFloat64(name string, description string, flag string, environmentVarName string, defaultValue ...float64) *Command {
	return me.withParam(name, nil, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithFloat64P(name string, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...float64) *Command {
	return me.withParam(name, nil, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithDuration(name string, description string, flag string, environmentVarName string, defaultValue ...time.Duration) *Command {
	return me.withParam(name, nil, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithDurationP(name string, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...time.Duration) *Command {
	return me.withParam(name, nil, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithStringSlice(name string, description string, flag string, environmentVarName string, defaultValue ...[]string) *Command {
	return me.withParam(name, nil, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithStringSliceP(name string, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...[]string) *Command {
	return me.withParam(name, nil, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithStringToString(name string, description string, flag string, environmentVarName string, defaultValue ...map[string]string) *Command {
	return me.withParam(name, nil, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithStringToStringP(name string, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...map[string]string) *Command {
	return me.withParam(name, nil, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

// *Var and *VarP ------------------------------------------------------------
func (me *Command) WithStringVar(p *string, description string, flag string, environmentVarName string, defaultValue ...string) *Command {
	return me.withParam(emptyStr, p, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithStringVarP(p *string, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...string) *Command {
	return me.withParam(emptyStr, p, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithBoolVar(p *bool, description string, flag string, environmentVarName string, defaultValue ...bool) *Command {
	return me.withParam(emptyStr, p, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithBoolVarP(p *bool, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...bool) *Command {
	return me.withParam(emptyStr, p, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithFloat32Var(p *float32, description string, flag string, environmentVarName string, defaultValue ...float32) *Command {
	return me.withParam(emptyStr, p, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithFloat32VarP(p *float32, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...float32) *Command {
	return me.withParam(emptyStr, p, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithFloat64Var(p *float64, description string, flag string, environmentVarName string, defaultValue ...float64) *Command {
	return me.withParam(emptyStr, p, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithFloat64VarP(p *float64, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...float64) *Command {
	return me.withParam(emptyStr, p, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithIntVar(p *int, description string, flag string, environmentVarName string, defaultValue ...int) *Command {
	return me.withParam(emptyStr, p, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithIntVarP(p *int, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...int) *Command {
	return me.withParam(emptyStr, p, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithInt8Var(p *int8, description string, flag string, environmentVarName string, defaultValue ...int8) *Command {
	return me.withParam(emptyStr, p, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithInt8VarP(p *int8, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...int8) *Command {
	return me.withParam(emptyStr, p, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithInt16Var(p *int16, description string, flag string, environmentVarName string, defaultValue ...int16) *Command {
	return me.withParam(emptyStr, p, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithInt16VarP(p *int16, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...int16) *Command {
	return me.withParam(emptyStr, p, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithInt32Var(p *int32, description string, flag string, environmentVarName string, defaultValue ...int32) *Command {
	return me.withParam(emptyStr, p, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithInt32VarP(p *int32, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...int32) *Command {
	return me.withParam(emptyStr, p, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithInt64Var(p *int64, description string, flag string, environmentVarName string, defaultValue ...int64) *Command {
	return me.withParam(emptyStr, p, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithInt64VarP(p *int64, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...int64) *Command {
	return me.withParam(emptyStr, p, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithUintVar(p *uint, description string, flag string, environmentVarName string, defaultValue ...uint) *Command {
	return me.withParam(emptyStr, p, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithUintVarP(p *uint, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...uint) *Command {
	return me.withParam(emptyStr, p, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithUint8Var(p *uint8, description string, flag string, environmentVarName string, defaultValue ...uint8) *Command {
	return me.withParam(emptyStr, p, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithUint8VarP(p *uint8, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...uint8) *Command {
	return me.withParam(emptyStr, p, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithUint16Var(p *uint16, description string, flag string, environmentVarName string, defaultValue ...uint16) *Command {
	return me.withParam(emptyStr, p, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithUint16VarP(p *uint16, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...uint16) *Command {
	return me.withParam(emptyStr, p, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithUint32Var(p *uint32, description string, flag string, environmentVarName string, defaultValue ...uint32) *Command {
	return me.withParam(emptyStr, p, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithUint32VarP(p *uint32, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...uint32) *Command {
	return me.withParam(emptyStr, p, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithUint64Var(p *uint64, description string, flag string, environmentVarName string, defaultValue ...uint64) *Command {
	return me.withParam(emptyStr, p, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithUint64VarP(p *uint64, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...uint64) *Command {
	return me.withParam(emptyStr, p, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithStringSliceVar(p *[]string, description string, flag string, environmentVarName string, defaultValue ...[]string) *Command {
	return me.withParam(emptyStr, p, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithStringSliceVarP(p *[]string, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...[]string) *Command {
	return me.withParam(emptyStr, p, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithStringToStringVar(p *map[string]string, description string, flag string, environmentVarName string, defaultValue ...map[string]string) *Command {
	return me.withParam(emptyStr, p, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithStringToStringVarP(p *map[string]string, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...map[string]string) *Command {
	return me.withParam(emptyStr, p, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithDurationVar(p *time.Duration, description string, flag string, environmentVarName string, defaultValue ...time.Duration) *Command {
	return me.withParam(emptyStr, p, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithDurationVarP(p *time.Duration, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...time.Duration) *Command {
	return me.withParam(emptyStr, p, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

// --------------------------------------------------------------------------

func (me *Command) WithNoArgs() *Command {
	me.cmd.Args = cobra.NoArgs
	return me
}

func (me *Command) WithExactArgs(n int) *Command {
	me.cmd.Args = cobra.ExactArgs(n)
	return me
}

func (me *Command) WithArbitraryArgs() *Command {
	me.cmd.Args = cobra.ArbitraryArgs
	return me
}

func (me *Command) WithMinimumNArgs(n int) *Command {
	me.cmd.Args = cobra.MinimumNArgs(n)
	return me
}

func (me *Command) WithMaximumNArgs(n int) *Command {
	me.cmd.Args = cobra.MaximumNArgs(n)
	return me
}

func (me *Command) WithRangeArgs(min int, max int) *Command {
	me.cmd.Args = cobra.RangeArgs(min, max)
	return me
}

func (me *Command) WithCustomArgs(argFunc func([]string) error) *Command {
	me.cmd.Args = func(cmd *cobra.Command, args []string) error {
		return argFunc(args)
	}
	return me
}

// --------------------------------------------------------------------------

func (me *Command) WithCommand(command *Command) *Command {
	me.SubCommands = append(me.SubCommands, command)
	me.cmd.AddCommand(command.cmd)
	return me
}

// WithRun sets the main execution function for the command.
//
// The provided run function will be wrapped with pre and post-execution hooks that handle:
// - Parameter processing and environment variable binding
// - Sanity checks (if configured)
// - BeforeRun hooks (if configured)
// - AfterRun hooks (if configured)
//
// Parameters:
//   - run: A function that takes a string slice of command arguments and returns an error.
//     This is the main logic that will be executed when the command runs.
//
// Returns:
//   - *Command: Returns the command instance for method chaining.
//
// Example:
//
//	cmd.WithRun(func(args []string) error {
//	    // Command implementation here
//	    return nil
//	})
func (me *Command) WithRun(run func(args []string) error) *Command {
	me.cmd.RunE = me.getRunWrapper(run)
	return me
}

func (me *Command) WithBeforeRun(beforeRun func(args []string) error) *Command {
	me.BeforeRun = beforeRun
	return me
}

func (me *Command) WithAfterRun(afterRun func(args []string) error) *Command {
	me.AfterRun = afterRun
	return me
}

func (me *Command) WithSanityCheck(sanityCheck func() error) *Command {
	me.SanityCheck = sanityCheck
	return me
}

func (me *Command) WithUsageOnError() *Command {
	me.cmd.SilenceUsage = false
	return me
}

func (me *Command) AllSettings() map[string]any {
	return me.viper.AllSettings()
}

func (me *Command) AllSettingsAsJson() string {
	return asJson(me.AllSettings())
}

func (me *Command) ArgsSettings() map[string]any {
	return me.argsViper.AllSettings()
}

func (me *Command) ArgsSettingsAsJson() string {
	return asJson(me.argsViper.AllSettings())
}

func (me *Command) Usage() {
	fmt.Fprint(usageOutput, me.UsageString())
}

func (me *Command) UsageString() string {
	return me.cmd.UsageString()
}

func (me *Command) IsDebug() bool {
	return me.debug
}

func (me *Command) Execute() error {
	// currentCommand = me
	return me.cmd.Execute()
}

func New(name string, description string, longDescription string) *Command {
	return &Command{
		cmd: &cobra.Command{
			Use:          name,
			Short:        description,
			Long:         longDescription,
			SilenceUsage: true,
			RunE:         getDummyRunner(name),
		},
		viper:           viper.New(),
		argsViper:       viper.New(),
		Name:            name,
		Description:     description,
		LongDescription: longDescription,
		Params:          NewConfigParams(),
		BeforeRun:       nil,
		AfterRun:        nil,
		Run:             nil,
	}
}
