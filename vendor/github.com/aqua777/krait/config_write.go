package krait

import "errors"

// errWriteConfigNotInLifecycle is returned when write helpers are used outside
// the command run lifecycle or on a non-executing command receiver.
var errWriteConfigNotInLifecycle = errors.New(
	"krait: WriteConfig and related methods must be called from within Run, BeforeRun, AfterRun, or SanityCheck on the executing command",
)

// validateCommandWriteContext ensures cmd is the command currently executing
// (so configuration has been loaded into its Viper instances).
func validateCommandWriteContext(cmd *Command) error {
	if cmd == nil || currentCommand != cmd {
		return errWriteConfigNotInLifecycle
	}
	return nil
}

// withExecutingCommand runs op on Current(); if there is no current command,
// returns errWriteConfigNotInLifecycle.
func withExecutingCommand(op func(*Command) error) error {
	c := Current()
	if c == nil {
		return errWriteConfigNotInLifecycle
	}
	return op(c)
}

// withExecutingCommandPath runs op(Current(), path); if there is no current
// command, returns errWriteConfigNotInLifecycle.
func withExecutingCommandPath(path string, op func(*Command, string) error) error {
	c := Current()
	if c == nil {
		return errWriteConfigNotInLifecycle
	}
	return op(c, path)
}
