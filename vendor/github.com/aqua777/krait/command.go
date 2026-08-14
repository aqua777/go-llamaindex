package krait

import (
	"fmt"
	"io"
	"os"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	"github.com/spf13/viper"
)

const (
	emptyStr         = ""
	debugModeMessage = "Enable debug mode"
)

// CompletionFunc is the function signature for dynamic shell completion callbacks.
// args is the list of positional arguments already provided; toComplete is the
// partial word the user is completing. Returns a list of completion candidates
// and a directive controlling shell behaviour.
type CompletionFunc func(args []string, toComplete string) ([]string, cobra.ShellCompDirective)

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
	configName       string
	configPaths      []string
	configReader     io.Reader
	configReaderType string
	configWatcher    func()
	debug            bool
	validArgs         []string
	validArgsFunction CompletionFunc
	flagCompletions   map[string]CompletionFunc
	groups            []*cobra.Group
	groupID           string
	disableFlagParsing bool
}

var (
	rootCommand    *Command = nil
	currentCommand *Command = nil

	usageOutput io.Writer = os.Stdout
)

// IsSet reports whether the named parameter key was explicitly set from any
// configuration source (CLI flag, environment variable, or config file), as
// opposed to only having a default value. Returns false if key is unknown.
// Must only be called from within Run, BeforeRun, AfterRun, or SanityCheck.
func (me *Command) IsSet(key string) bool {
	return me.viper.IsSet(key)
}

// lookupFlag returns the pflag.Flag for the given flag name, checking both
// Flags() and PersistentFlags() so that persistent flags inherited from a
// parent command are found when executing a subcommand.
func (me *Command) lookupFlag(flagName string) *pflag.Flag {
	if f := me.cmd.Flags().Lookup(flagName); f != nil {
		return f
	}
	return me.cmd.PersistentFlags().Lookup(flagName)
}

func (me *Command) internalProcessParams() error {
	for _, option := range me.Params.List() {
		if option.Name != emptyStr {
			me.viper.SetDefault(option.Name, option.DefaultValue)
			if option.EnvironmentVarName != emptyStr {
				me.viper.BindEnv(option.Name, option.EnvironmentVarName)
			}
		} else {
			me.argsViper.BindEnv(option.Flag, option.EnvironmentVarName)
			if flag := me.lookupFlag(option.Flag); flag != nil {
				if err := me.argsViper.BindPFlag(option.Flag, flag); err != nil {
					return err
				}
			}
		}
	}

	// Resolve the config file path from the environment variable before reading
	// the config, so that an env-provided path is honored even when no CLI flag
	// was given. We do this after the first loop so argsViper already has the
	// env binding in place.
	me.argsViper.AutomaticEnv()
	for _, option := range me.Params.List() {
		if option.Name == emptyStr && option.VarPtr != nil {
			if flag := me.lookupFlag(option.Flag); flag != nil && !flag.Changed && me.argsViper.IsSet(option.Flag) {
				if err := option.setVarPtrValue(flag, me.argsViper); err != nil {
					return err
				}
			}
		}
	}

	if len(me.ConfigFile) > 0 {
		for _, viperInstance := range []*viper.Viper{me.viper, me.argsViper} {
			viperInstance.SetConfigFile(me.ConfigFile)
			if err := viperInstance.ReadInConfig(); err != nil {
				return err
			}
			if me.configWatcher != nil {
				fn := me.configWatcher
				viperInstance.OnConfigChange(func(e fsnotify.Event) { fn() })
				viperInstance.WatchConfig()
			}
		}
	} else if me.configName != emptyStr {
		for _, viperInstance := range []*viper.Viper{me.viper, me.argsViper} {
			if err := readViperConfigFromSearchPaths(viperInstance, me.configName, me.configPaths); err != nil {
				return err
			}
			if me.configWatcher != nil {
				fn := me.configWatcher
				viperInstance.OnConfigChange(func(e fsnotify.Event) { fn() })
				viperInstance.WatchConfig()
			}
		}
	} else if me.configReader != nil {
		if err := readNamedVipersFromReader(me.viper, me.argsViper, me.configReader, me.configReaderType); err != nil {
			return err
		}
	}

	me.viper.AutomaticEnv()
	for _, option := range me.Params.List() {
		if option.Name != emptyStr {
			if flag := me.lookupFlag(option.Flag); flag != nil {
				if err := me.viper.BindPFlag(option.Name, flag); err != nil {
					return err
				}
			}
		}
	}

	for _, option := range me.Params.List() {
		if option.Name == emptyStr && option.VarPtr != nil {
			if flag := me.lookupFlag(option.Flag); flag != nil && !flag.Changed && me.argsViper.IsSet(option.Flag) {
				if err := option.setVarPtrValue(flag, me.argsViper); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func (me *Command) registerFlagCompletionPair(flagName string, fn CompletionFunc) error {
	capturedFn := fn
	return me.cmd.RegisterFlagCompletionFunc(flagName, func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
		return capturedFn(args, toComplete)
	})
}

func (me *Command) registerFlagCompletions() error {
	for flagName, fn := range me.flagCompletions {
		if _, already := me.cmd.GetFlagCompletionFunc(flagName); already {
			continue
		}
		if err := me.registerFlagCompletionPair(flagName, fn); err != nil {
			return fmt.Errorf("WithFlagCompletion: %w", err)
		}
	}
	return nil
}

func (me *Command) internalBeforeRun(args []string) error {
	if err := me.registerFlagCompletions(); err != nil {
		return err
	}
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
		currentViper = me.viper
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
	if flag == emptyStr && varPtr != nil {
		panic(fmt.Sprintf("config-only parameters (empty flag) are not supported for var-bound params; use a non-empty flag for var-bound parameter %q", name))
	}
	me.Params.With(name, flag, shortFlag, environmentVarName, description, defaultValue, varPtr)
	return me.withFlag(varPtr, flag, shortFlag, description, defaultValue)
}

func (me *Command) withPersistentParam(name string, varPtr any, flag, shortFlag, environmentVarName, description string, defaultValue any) *Command {
	if flag == emptyStr && varPtr != nil {
		panic(fmt.Sprintf("config-only parameters (empty flag) are not supported for var-bound params; use a non-empty flag for var-bound parameter %q", name))
	}
	me.Params.WithPersistent(name, flag, shortFlag, environmentVarName, description, defaultValue, varPtr)
	return me.withPersistentFlag(varPtr, flag, shortFlag, description, defaultValue)
}

func (me *Command) WithConfig(configFile string, flag string, shortFlag string, environmentVarName string) *Command {
	return me.withParam(emptyStr, &me.ConfigFile, flag, shortFlag, environmentVarName, "Path to config file", configFile)
}

// WithConfigName sets the config file base name (without extension) for
// auto-discovery. Combined with one or more WithConfigPath calls, Viper will
// search each path in declaration order for a file named name.<ext> where
// <ext> is any format Viper supports (yaml, toml, json, etc.).
//
// Mutually exclusive with WithConfig by convention. If both are used and the
// resolved config file path is non-empty, WithConfig takes precedence. Must be
// called before Execute().
func (me *Command) WithConfigName(name string) *Command {
	me.configName = name
	return me
}

// WithConfigPath adds a directory to the config file search list used when
// WithConfigName is set. Paths are searched in the order they are declared.
// Has no effect if WithConfigName has not been called. Must be called before Execute().
func (me *Command) WithConfigPath(path string) *Command {
	me.configPaths = append(me.configPaths, path)
	return me
}

// WithConfigReader registers an io.Reader as the configuration source for this
// command. During internalProcessParams, the reader's content is merged into
// both Viper instances via ReadConfig, applying the same source-priority rules
// as a file-based config. configType must be a format Viper recognises:
// "yaml", "toml", "json", "hcl", "ini", "dotenv", or "properties".
//
// Mutually exclusive with WithConfig and WithConfigName by convention. If
// multiple config sources are registered, their precedence follows Viper's
// own merge order (last ReadConfig wins over defaults, loses to flags and env).
//
// Must be called before Execute().
func (me *Command) WithConfigReader(r io.Reader, configType string) *Command {
	me.configReader = r
	me.configReaderType = configType
	return me
}

// namedViperWrite runs fn after validating that me is the executing command
// (named-param Viper is loaded). Shared by WriteConfig and related methods.
func (me *Command) namedViperWrite(fn func() error) error {
	if err := validateCommandWriteContext(me); err != nil {
		return err
	}
	return fn()
}

// WriteConfig writes the current resolved named-parameter configuration to the
// file that was loaded via WithConfig or discovered via WithConfigName /
// WithConfigPath. Returns an error if no config file path is known to Viper or
// if the write fails. Must be called from within Run, BeforeRun, AfterRun, or
// SanityCheck on the executing command.
func (me *Command) WriteConfig() error {
	return me.namedViperWrite(me.viper.WriteConfig)
}

// SafeWriteConfig writes the current resolved named-parameter configuration to
// the configured config file path. Returns an error if the file already exists
// or if no config file path is known to Viper. Must be called from within Run,
// BeforeRun, AfterRun, or SanityCheck on the executing command.
func (me *Command) SafeWriteConfig() error {
	return me.namedViperWrite(me.viper.SafeWriteConfig)
}

// WriteConfigAs writes the current resolved named-parameter configuration to
// path, creating or overwriting the file. The format is inferred from the file
// extension. Must be called from within Run, BeforeRun, AfterRun, or
// SanityCheck on the executing command.
func (me *Command) WriteConfigAs(path string) error {
	if err := validateCommandWriteContext(me); err != nil {
		return err
	}
	return me.viper.WriteConfigAs(path)
}

// SafeWriteConfigAs writes the current resolved named-parameter configuration
// to path. Returns an error if the file already exists. The format is inferred
// from the file extension. Must be called from within Run, BeforeRun, AfterRun,
// or SanityCheck on the executing command.
func (me *Command) SafeWriteConfigAs(path string) error {
	if err := validateCommandWriteContext(me); err != nil {
		return err
	}
	return me.viper.SafeWriteConfigAs(path)
}

// Sub returns a scoped view of the named-parameter config tree rooted at key.
// The returned map contains all keys under the prefix, with the prefix stripped.
// Returns nil if key is empty, the key does not exist, or it has no child settings.
// Must be called from within Run, BeforeRun, AfterRun, or SanityCheck for a
// fully resolved view (config, env, flags merged).
//
// Example: if the config contains "db.host" and "db.port", Sub("db") returns
// map[string]any{"host": "localhost", "port": 5432}.
func (me *Command) Sub(key string) map[string]any {
	return viperNamedSubMap(me.viper, key)
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
//	cmd.WithString("path", "Path to file", "path", "FILE_PATH", "/path/to/file.txt")
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
//	cmd.WithStringP("path", "Path to file", "path", "p", "FILE_PATH", "/path/to/file.txt")
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

func (me *Command) WithIntSlice(name string, description string, flag string, environmentVarName string, defaultValue ...[]int) *Command {
	return me.withParam(name, nil, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithIntSliceP(name string, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...[]int) *Command {
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

func (me *Command) WithIntSliceVar(p *[]int, description string, flag string, environmentVarName string, defaultValue ...[]int) *Command {
	return me.withParam(emptyStr, p, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithIntSliceVarP(p *[]int, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...[]int) *Command {
	return me.withParam(emptyStr, p, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithDurationVar(p *time.Duration, description string, flag string, environmentVarName string, defaultValue ...time.Duration) *Command {
	return me.withParam(emptyStr, p, flag, emptyStr, environmentVarName, description, getDefaultValue(defaultValue))
}

func (me *Command) WithDurationVarP(p *time.Duration, description string, flag string, shortFlag string, environmentVarName string, defaultValue ...time.Duration) *Command {
	return me.withParam(emptyStr, p, flag, shortFlag, environmentVarName, description, getDefaultValue(defaultValue))
}

// WithPersistent* — named params -----------------------------------------------

// WithPersistentString adds a string named-parameter as a persistent flag. The flag
// is inherited by all subcommands: when a subcommand runs, the value is accessible via
// krait.GetString(name) from within the subcommand's run lifecycle.
func (me *Command) WithPersistentString(name, description, flag, envVar string, defaultValue ...string) *Command {
	return me.withPersistentParam(name, nil, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentStringP(name, description, flag, shortFlag, envVar string, defaultValue ...string) *Command {
	return me.withPersistentParam(name, nil, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentBool(name, description, flag, envVar string, defaultValue ...bool) *Command {
	return me.withPersistentParam(name, nil, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentBoolP(name, description, flag, shortFlag, envVar string, defaultValue ...bool) *Command {
	return me.withPersistentParam(name, nil, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentInt(name, description, flag, envVar string, defaultValue ...int) *Command {
	return me.withPersistentParam(name, nil, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentIntP(name, description, flag, shortFlag, envVar string, defaultValue ...int) *Command {
	return me.withPersistentParam(name, nil, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentInt8(name, description, flag, envVar string, defaultValue ...int8) *Command {
	return me.withPersistentParam(name, nil, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentInt8P(name, description, flag, shortFlag, envVar string, defaultValue ...int8) *Command {
	return me.withPersistentParam(name, nil, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentInt16(name, description, flag, envVar string, defaultValue ...int16) *Command {
	return me.withPersistentParam(name, nil, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentInt16P(name, description, flag, shortFlag, envVar string, defaultValue ...int16) *Command {
	return me.withPersistentParam(name, nil, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentInt32(name, description, flag, envVar string, defaultValue ...int32) *Command {
	return me.withPersistentParam(name, nil, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentInt32P(name, description, flag, shortFlag, envVar string, defaultValue ...int32) *Command {
	return me.withPersistentParam(name, nil, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentInt64(name, description, flag, envVar string, defaultValue ...int64) *Command {
	return me.withPersistentParam(name, nil, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentInt64P(name, description, flag, shortFlag, envVar string, defaultValue ...int64) *Command {
	return me.withPersistentParam(name, nil, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentUint(name, description, flag, envVar string, defaultValue ...uint) *Command {
	return me.withPersistentParam(name, nil, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentUintP(name, description, flag, shortFlag, envVar string, defaultValue ...uint) *Command {
	return me.withPersistentParam(name, nil, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentUint8(name, description, flag, envVar string, defaultValue ...uint8) *Command {
	return me.withPersistentParam(name, nil, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentUint8P(name, description, flag, shortFlag, envVar string, defaultValue ...uint8) *Command {
	return me.withPersistentParam(name, nil, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentUint16(name, description, flag, envVar string, defaultValue ...uint16) *Command {
	return me.withPersistentParam(name, nil, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentUint16P(name, description, flag, shortFlag, envVar string, defaultValue ...uint16) *Command {
	return me.withPersistentParam(name, nil, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentUint32(name, description, flag, envVar string, defaultValue ...uint32) *Command {
	return me.withPersistentParam(name, nil, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentUint32P(name, description, flag, shortFlag, envVar string, defaultValue ...uint32) *Command {
	return me.withPersistentParam(name, nil, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentUint64(name, description, flag, envVar string, defaultValue ...uint64) *Command {
	return me.withPersistentParam(name, nil, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentUint64P(name, description, flag, shortFlag, envVar string, defaultValue ...uint64) *Command {
	return me.withPersistentParam(name, nil, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentFloat32(name, description, flag, envVar string, defaultValue ...float32) *Command {
	return me.withPersistentParam(name, nil, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentFloat32P(name, description, flag, shortFlag, envVar string, defaultValue ...float32) *Command {
	return me.withPersistentParam(name, nil, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentFloat64(name, description, flag, envVar string, defaultValue ...float64) *Command {
	return me.withPersistentParam(name, nil, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentFloat64P(name, description, flag, shortFlag, envVar string, defaultValue ...float64) *Command {
	return me.withPersistentParam(name, nil, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentDuration(name, description, flag, envVar string, defaultValue ...time.Duration) *Command {
	return me.withPersistentParam(name, nil, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentDurationP(name, description, flag, shortFlag, envVar string, defaultValue ...time.Duration) *Command {
	return me.withPersistentParam(name, nil, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentStringSlice(name, description, flag, envVar string, defaultValue ...[]string) *Command {
	return me.withPersistentParam(name, nil, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentStringSliceP(name, description, flag, shortFlag, envVar string, defaultValue ...[]string) *Command {
	return me.withPersistentParam(name, nil, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentStringToString(name, description, flag, envVar string, defaultValue ...map[string]string) *Command {
	return me.withPersistentParam(name, nil, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentStringToStringP(name, description, flag, shortFlag, envVar string, defaultValue ...map[string]string) *Command {
	return me.withPersistentParam(name, nil, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

// WithPersistent*Var — var-bound params -----------------------------------------

// WithPersistentStringVar adds a string var-bound parameter as a persistent flag.
// The pointer is populated before Run in any subcommand that inherits the flag.
func (me *Command) WithPersistentStringVar(p *string, description, flag, envVar string, defaultValue ...string) *Command {
	return me.withPersistentParam(emptyStr, p, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentStringVarP(p *string, description, flag, shortFlag, envVar string, defaultValue ...string) *Command {
	return me.withPersistentParam(emptyStr, p, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentBoolVar(p *bool, description, flag, envVar string, defaultValue ...bool) *Command {
	return me.withPersistentParam(emptyStr, p, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentBoolVarP(p *bool, description, flag, shortFlag, envVar string, defaultValue ...bool) *Command {
	return me.withPersistentParam(emptyStr, p, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentIntVar(p *int, description, flag, envVar string, defaultValue ...int) *Command {
	return me.withPersistentParam(emptyStr, p, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentIntVarP(p *int, description, flag, shortFlag, envVar string, defaultValue ...int) *Command {
	return me.withPersistentParam(emptyStr, p, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentInt8Var(p *int8, description, flag, envVar string, defaultValue ...int8) *Command {
	return me.withPersistentParam(emptyStr, p, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentInt8VarP(p *int8, description, flag, shortFlag, envVar string, defaultValue ...int8) *Command {
	return me.withPersistentParam(emptyStr, p, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentInt16Var(p *int16, description, flag, envVar string, defaultValue ...int16) *Command {
	return me.withPersistentParam(emptyStr, p, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentInt16VarP(p *int16, description, flag, shortFlag, envVar string, defaultValue ...int16) *Command {
	return me.withPersistentParam(emptyStr, p, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentInt32Var(p *int32, description, flag, envVar string, defaultValue ...int32) *Command {
	return me.withPersistentParam(emptyStr, p, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentInt32VarP(p *int32, description, flag, shortFlag, envVar string, defaultValue ...int32) *Command {
	return me.withPersistentParam(emptyStr, p, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentInt64Var(p *int64, description, flag, envVar string, defaultValue ...int64) *Command {
	return me.withPersistentParam(emptyStr, p, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentInt64VarP(p *int64, description, flag, shortFlag, envVar string, defaultValue ...int64) *Command {
	return me.withPersistentParam(emptyStr, p, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentUintVar(p *uint, description, flag, envVar string, defaultValue ...uint) *Command {
	return me.withPersistentParam(emptyStr, p, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentUintVarP(p *uint, description, flag, shortFlag, envVar string, defaultValue ...uint) *Command {
	return me.withPersistentParam(emptyStr, p, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentUint8Var(p *uint8, description, flag, envVar string, defaultValue ...uint8) *Command {
	return me.withPersistentParam(emptyStr, p, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentUint8VarP(p *uint8, description, flag, shortFlag, envVar string, defaultValue ...uint8) *Command {
	return me.withPersistentParam(emptyStr, p, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentUint16Var(p *uint16, description, flag, envVar string, defaultValue ...uint16) *Command {
	return me.withPersistentParam(emptyStr, p, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentUint16VarP(p *uint16, description, flag, shortFlag, envVar string, defaultValue ...uint16) *Command {
	return me.withPersistentParam(emptyStr, p, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentUint32Var(p *uint32, description, flag, envVar string, defaultValue ...uint32) *Command {
	return me.withPersistentParam(emptyStr, p, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentUint32VarP(p *uint32, description, flag, shortFlag, envVar string, defaultValue ...uint32) *Command {
	return me.withPersistentParam(emptyStr, p, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentUint64Var(p *uint64, description, flag, envVar string, defaultValue ...uint64) *Command {
	return me.withPersistentParam(emptyStr, p, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentUint64VarP(p *uint64, description, flag, shortFlag, envVar string, defaultValue ...uint64) *Command {
	return me.withPersistentParam(emptyStr, p, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentFloat32Var(p *float32, description, flag, envVar string, defaultValue ...float32) *Command {
	return me.withPersistentParam(emptyStr, p, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentFloat32VarP(p *float32, description, flag, shortFlag, envVar string, defaultValue ...float32) *Command {
	return me.withPersistentParam(emptyStr, p, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentFloat64Var(p *float64, description, flag, envVar string, defaultValue ...float64) *Command {
	return me.withPersistentParam(emptyStr, p, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentFloat64VarP(p *float64, description, flag, shortFlag, envVar string, defaultValue ...float64) *Command {
	return me.withPersistentParam(emptyStr, p, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentDurationVar(p *time.Duration, description, flag, envVar string, defaultValue ...time.Duration) *Command {
	return me.withPersistentParam(emptyStr, p, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentDurationVarP(p *time.Duration, description, flag, shortFlag, envVar string, defaultValue ...time.Duration) *Command {
	return me.withPersistentParam(emptyStr, p, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentStringSliceVar(p *[]string, description, flag, envVar string, defaultValue ...[]string) *Command {
	return me.withPersistentParam(emptyStr, p, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentStringSliceVarP(p *[]string, description, flag, shortFlag, envVar string, defaultValue ...[]string) *Command {
	return me.withPersistentParam(emptyStr, p, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentStringToStringVar(p *map[string]string, description, flag, envVar string, defaultValue ...map[string]string) *Command {
	return me.withPersistentParam(emptyStr, p, flag, emptyStr, envVar, description, getDefaultValue(defaultValue))
}

func (me *Command) WithPersistentStringToStringVarP(p *map[string]string, description, flag, shortFlag, envVar string, defaultValue ...map[string]string) *Command {
	return me.withPersistentParam(emptyStr, p, flag, shortFlag, envVar, description, getDefaultValue(defaultValue))
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
	// Propagate persistent params from this command into the subcommand's entire
	// subtree so that internalProcessParams on any descendant can bind them to
	// its own Viper. Cobra merges PersistentFlags down at execution time, so the
	// pflag lookup works; we just need the Params list to be populated.
	me.propagatePersistentParams(command)
	return me
}

func (me *Command) propagatePersistentParams(target *Command) {
	for _, p := range me.Params.List() {
		if p.Persistent {
			target.Params.WithPersistent(p.Name, p.Flag, p.ShortFlag, p.EnvironmentVarName, p.Description, p.DefaultValue, p.VarPtr)
		}
	}
	for _, sub := range target.SubCommands {
		me.propagatePersistentParams(sub)
	}
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

// WithSilenceErrors suppresses Cobra from printing the error message to
// stderr when a command returns an error. The error is still returned to
// the caller of Execute(). By default Cobra prints errors; this opts out.
func (me *Command) WithSilenceErrors() *Command {
	me.cmd.SilenceErrors = true
	return me
}

// WithHidden marks the command as hidden. Hidden commands do not appear in
// help output or completion lists but remain fully functional when invoked
// directly by name.
func (me *Command) WithHidden() *Command {
	me.cmd.Hidden = true
	return me
}

// WithDeprecated marks the command as deprecated. When invoked, Cobra prints
// msg to stderr before executing the command. The command still runs normally.
// msg should describe what callers should use instead.
func (me *Command) WithDeprecated(msg string) *Command {
	me.cmd.Deprecated = msg
	return me
}

// WithFlagHidden hides the named flag from help output. The flag remains
// functional. Panics if flag is not registered on this command.
func (me *Command) WithFlagHidden(flag string) *Command {
	if err := me.cmd.Flags().MarkHidden(flag); err != nil {
		panic(fmt.Sprintf("WithFlagHidden: flag %q not registered on command %q", flag, me.Name))
	}
	return me
}

// WithFlagDeprecated marks the named flag as deprecated. When used, Cobra
// prints msg to stderr. Panics if flag is not registered on this command.
func (me *Command) WithFlagDeprecated(flag string, msg string) *Command {
	if err := me.cmd.Flags().MarkDeprecated(flag, msg); err != nil {
		panic(fmt.Sprintf("WithFlagDeprecated: flag %q not registered on command %q", flag, me.Name))
	}
	return me
}

// WithAliases registers alternate names for this command. Each alias in
// aliases responds exactly as the primary command name. Aliases must not
// conflict with sibling command names.
func (me *Command) WithAliases(aliases ...string) *Command {
	me.cmd.Aliases = aliases
	return me
}

// WithValidArgs registers a static list of valid positional argument completions
// for this command. Used by Cobra's shell completion engine for bash, zsh, fish,
// and PowerShell. Mutually exclusive with WithValidArgsFunction; if both are
// called, WithValidArgsFunction takes precedence (Cobra behaviour).
func (me *Command) WithValidArgs(args ...string) *Command {
	me.validArgs = args
	me.cmd.ValidArgs = args
	return me
}

// WithValidArgsFunction registers a dynamic completion callback for positional
// arguments. The function receives the args seen so far and the current partial
// word, and returns a list of completion candidates and a shell directive.
func (me *Command) WithValidArgsFunction(fn CompletionFunc) *Command {
	me.validArgsFunction = fn
	me.cmd.ValidArgsFunction = func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
		return fn(args, toComplete)
	}
	return me
}

// WithFlagCompletion registers a dynamic completion callback for the named flag.
// Returns an error from Execute() if flagName is not registered on this command.
// When the flag already exists, registration is applied immediately so Cobra's
// hidden __complete command can resolve completions (it does not run BeforeRun).
func (me *Command) WithFlagCompletion(flagName string, fn CompletionFunc) *Command {
	if me.flagCompletions == nil {
		me.flagCompletions = make(map[string]CompletionFunc)
	}
	me.flagCompletions[flagName] = fn
	if me.lookupFlag(flagName) != nil {
		if _, already := me.cmd.GetFlagCompletionFunc(flagName); !already {
			_ = me.registerFlagCompletionPair(flagName, fn)
		}
	}
	return me
}

// WithGroup registers a labelled command group on this command. Subcommands can
// then be associated with the group by calling InGroup with the same id. Groups
// appear as labelled sections in --help output (Cobra v1.6+ behaviour). Panics
// if a group with the same id is registered twice on the same command.
func (me *Command) WithGroup(id, title string) *Command {
	for _, g := range me.groups {
		if g.ID == id {
			panic(fmt.Sprintf("WithGroup: group id %q already registered on command %q", id, me.Name))
		}
	}
	group := &cobra.Group{ID: id, Title: title}
	me.groups = append(me.groups, group)
	me.cmd.AddGroup(group)
	return me
}

// InGroup assigns this command to the group with the given id on its parent.
// Must be called before the command is attached via WithCommand. Panics if id
// is empty.
func (me *Command) InGroup(id string) *Command {
	if id == emptyStr {
		panic(fmt.Sprintf("InGroup: group id must not be empty on command %q", me.Name))
	}
	me.groupID = id
	me.cmd.GroupID = id
	return me
}

// WithDisableFlagParsing disables Cobra's flag parsing for this command. All
// tokens after the command name — including those that look like flags — are
// delivered verbatim as positional arguments to Run. Intended for pass-through
// commands (e.g., "app exec -- docker run ..."). Any With* flag declarations on
// the same command are ignored at runtime when DisableFlagParsing is true.
func (me *Command) WithDisableFlagParsing() *Command {
	me.disableFlagParsing = true
	me.cmd.DisableFlagParsing = true
	return me
}

// WithConfigWatcher registers a callback that is invoked whenever the config
// file changes on disk. Internally wires viper.OnConfigChange and
// viper.WatchConfig on both Viper instances after ReadInConfig succeeds during
// internalProcessParams. Has no effect if no config file is loaded (i.e. neither
// WithConfig nor WithConfigName was called).
//
// The callback runs in a background goroutine spawned by fsnotify. Callers must
// use krait.GetString and friends to read updated values; the callback must not
// call Execute() or Reset(). Callers are responsible for synchronising any
// application-level state mutated inside the callback.
//
// Must be called before Execute().
func (me *Command) WithConfigWatcher(fn func()) *Command {
	me.configWatcher = fn
	return me
}

// WithVersion wires a --version flag (and -v short form) to the command.
// When the flag is passed, Cobra prints version and exits. version is the
// version string to display. Should be called on the root command.
func (me *Command) WithVersion(version string) *Command {
	me.cmd.Version = version
	return me
}

// WithRequired marks the named flag as required. Execute returns an error if the flag
// is not provided by the caller. Panics if flag is not registered on this command.
func (me *Command) WithRequired(flag string) *Command {
	if err := me.cmd.MarkFlagRequired(flag); err != nil {
		panic(fmt.Sprintf("WithRequired: flag %q not registered on command %q", flag, me.Name))
	}
	return me
}

// WithFlagsRequiredTogether marks the named flags as required together: all must be
// provided or none. Panics if any flag is not registered on this command.
func (me *Command) WithFlagsRequiredTogether(flags ...string) *Command {
	for _, flag := range flags {
		if me.cmd.Flags().Lookup(flag) == nil {
			panic(fmt.Sprintf("WithFlagsRequiredTogether: flag %q not registered on command %q", flag, me.Name))
		}
	}
	me.cmd.MarkFlagsRequiredTogether(flags...)
	return me
}

// WithFlagsOneRequired marks the named flags as "one required": at least one of the
// named flags must be provided. Panics if any flag is not registered on this command.
func (me *Command) WithFlagsOneRequired(flags ...string) *Command {
	for _, flag := range flags {
		if me.cmd.Flags().Lookup(flag) == nil {
			panic(fmt.Sprintf("WithFlagsOneRequired: flag %q not registered on command %q", flag, me.Name))
		}
	}
	me.cmd.MarkFlagsOneRequired(flags...)
	return me
}

// WithFlagsMutuallyExclusive marks the named flags as mutually exclusive: at most one
// may be provided. Panics if any flag is not registered on this command.
func (me *Command) WithFlagsMutuallyExclusive(flags ...string) *Command {
	for _, flag := range flags {
		if me.cmd.Flags().Lookup(flag) == nil {
			panic(fmt.Sprintf("WithFlagsMutuallyExclusive: flag %q not registered on command %q", flag, me.Name))
		}
	}
	me.cmd.MarkFlagsMutuallyExclusive(flags...)
	return me
}

// WithEnvPrefix sets an environment variable prefix for AutomaticEnv on both Viper
// instances. When set, AutomaticEnv maps each Viper key to PREFIX_KEY (uppercased).
// Explicit env var names passed to With* methods are not affected. Must be called
// before Execute().
func (me *Command) WithEnvPrefix(prefix string) *Command {
	me.viper.SetEnvPrefix(prefix)
	me.argsViper.SetEnvPrefix(prefix)
	return me
}

// Unmarshal decodes all named-parameter values for this command into the value pointed
// to by v using mapstructure. Only named params (viper-backed) are decoded; var-bound
// params are not included. Returns an error if decoding fails. Must be called from
// within Run, BeforeRun, AfterRun, or SanityCheck.
func (me *Command) Unmarshal(v any) error {
	return me.viper.Unmarshal(v)
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
