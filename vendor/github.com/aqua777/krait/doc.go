/*
Package krait provides a unified interface for building command-line applications by combining
the functionality of Cobra (command-line interface) and Viper (configuration management) into
a single, cohesive package.

Krait simplifies the process of creating CLI applications with rich configuration support by
offering a fluent API that handles both command definitions and configuration management.

Key Features:

  - Unified command and configuration management
  - Environment variable binding
  - Configuration file support (multiple formats)
  - Command-line flag integration
  - Nested command support
  - Pre and post-execution hooks
  - Sanity checks for commands
  - Type-safe configuration access

Basic Usage:

Create a new command with configuration:

	var app = krait.App("myapp", "Short description", "Long description").
	    WithConfig("", "config", "c", "APP_CONFIG").
	    WithStringP("app.string", "String parameter", "str", "s", "APP_STRING", "default").
	    WithIntP("app.int", "Integer parameter", "int", "i", "APP_INT", 123).
	    WithBoolP("app.debug", "Debug mode", "debug", "d", "APP_DEBUG", false).
	    WithRun(func(args []string) error {
	        // Command implementation
	        return nil
	    })

Configuration can be provided through:

  - Command-line flags
  - Environment variables
  - Configuration files (YAML, JSON, TOML, etc.)
  - Default values

Configuration Access:

Access configuration values using type-safe getters:

	str := krait.GetString("app.string")
	num := krait.GetInt("app.int")
	debug := krait.GetBool("app.debug")

Command Lifecycle:

Commands support various lifecycle hooks:

  - BeforeRun: Executed before the main command
  - Run: Main command execution
  - AfterRun: Executed after successful command completion
  - SanityCheck: Validates command configuration

Example:

	package main

	import (
	    "fmt"
	    "os"
	    "time"

	    "github.com/aqua777/krait"
	)

	func main() {
	    app := krait.App("myapp", "My Application", "A detailed description").
	        WithConfig("", "config", "c", "APP_CONFIG").
	        WithStringP("app.name", "Application name", "name", "n", "APP_NAME", "MyApp").
	        WithDurationP("app.timeout", "Timeout duration", "timeout", "t", "APP_TIMEOUT", time.Second*30).
	        WithRun(func(args []string) error {
	            fmt.Printf("Name: %s, Timeout: %v\n", krait.GetString("app.name"), krait.GetDuration("app.timeout"))
	            return nil
	        })

	    if err := app.Execute(); err != nil {
	        os.Exit(1)
	    }
	}

Sub-commands Example:

Here's how to create a CLI application with nested sub-commands:

	package main

	import (
	    "fmt"
	    "os"

	    "github.com/aqua777/krait"
	)

	func main() {
	    // Create version sub-command
	    versionCmd := krait.New("version", "Print version info", "Display detailed version information").
	        WithBoolP("json", "Output in JSON format", "json", "j", "VERSION_JSON", false).
	        WithRun(func(args []string) error {
	            if krait.GetBool("json") {
	                fmt.Println(`{"version": "1.0.0"}`)
	            } else {
	                fmt.Println("Version 1.0.0")
	            }
	            return nil
	        })

	    // Create config sub-command with its own sub-command
	    configShowCmd := krait.New("show", "Show configuration", "Display current configuration").
	        WithRun(func(args []string) error {
	            fmt.Println("Current configuration:", krait.AsJson())
	            return nil
	        })

	    configCmd := krait.New("config", "Configuration commands", "Manage application configuration").
	        WithCommand(configShowCmd)

	    // Create root command and add sub-commands
	    app := krait.App("myapp", "My Application", "A CLI application with sub-commands").
	        WithConfig("", "config", "c", "APP_CONFIG").
	        WithCommand(versionCmd).
	        WithCommand(configCmd).
	        WithRun(func(args []string) error {
	            fmt.Println("Run 'myapp --help' for usage")
	            return nil
	        })

	    if err := app.Execute(); err != nil {
	        os.Exit(1)
	    }
	}

This creates a CLI with the following command structure:

	myapp
	├── version    # Print version info
	│   └── --json # Output in JSON format
	└── config     # Configuration commands
	    └── show   # Show current configuration

For more examples and detailed usage, see the examples directory in the repository.
*/
package krait
