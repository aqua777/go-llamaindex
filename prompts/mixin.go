package prompts

import (
	"strings"
)

// PromptDictType is a map of prompt names to prompt templates.
type PromptDictType map[string]BasePromptTemplate

// PromptMixinType is a map of module names to PromptMixin implementations.
type PromptMixinType map[string]PromptMixin

// PromptMixin is an interface for components that manage prompts.
// This allows getting and setting prompts for a component and its sub-modules.
type PromptMixin interface {
	// GetPrompts returns all prompts for this component and its sub-modules.
	// Sub-module prompts are prefixed with "module_name:".
	GetPrompts() PromptDictType

	// UpdatePrompts updates prompts for this component and its sub-modules.
	// Use "module_name:prompt_name" to update sub-module prompts.
	UpdatePrompts(prompts PromptDictType)

	// getPrompts returns prompts for this component only (internal).
	getPrompts() PromptDictType

	// getPromptModules returns sub-modules that also implement PromptMixin.
	getPromptModules() PromptMixinType

	// updatePrompts updates prompts for this component only (internal).
	updatePrompts(prompts PromptDictType)
}

// BasePromptMixin provides a base implementation of PromptMixin.
// Embed this in structs that need prompt management.
type BasePromptMixin struct {
	prompts PromptDictType
	modules PromptMixinType
}

// NewBasePromptMixin creates a new BasePromptMixin.
func NewBasePromptMixin() *BasePromptMixin {
	return &BasePromptMixin{
		prompts: make(PromptDictType),
		modules: make(PromptMixinType),
	}
}

// GetPrompts returns all prompts for this component and its sub-modules.
func (bpm *BasePromptMixin) GetPrompts() PromptDictType {
	allPrompts := make(PromptDictType)

	// Add this component's prompts
	for k, v := range bpm.prompts {
		allPrompts[k] = v
	}

	// Add sub-module prompts with prefix
	for moduleName, module := range bpm.modules {
		for promptName, prompt := range module.GetPrompts() {
			allPrompts[moduleName+":"+promptName] = prompt
		}
	}

	return allPrompts
}

// UpdatePrompts updates prompts for this component and its sub-modules.
func (bpm *BasePromptMixin) UpdatePrompts(prompts PromptDictType) {
	// Separate prompts for this component vs sub-modules
	localPrompts := make(PromptDictType)
	subModulePrompts := make(map[string]PromptDictType)

	for key, prompt := range prompts {
		if strings.Contains(key, ":") {
			parts := strings.SplitN(key, ":", 2)
			moduleName := parts[0]
			promptName := parts[1]
			if subModulePrompts[moduleName] == nil {
				subModulePrompts[moduleName] = make(PromptDictType)
			}
			subModulePrompts[moduleName][promptName] = prompt
		} else {
			localPrompts[key] = prompt
		}
	}

	// Update local prompts
	for k, v := range localPrompts {
		bpm.prompts[k] = v
	}

	// Update sub-module prompts
	for moduleName, modulePrompts := range subModulePrompts {
		if module, ok := bpm.modules[moduleName]; ok {
			module.UpdatePrompts(modulePrompts)
		}
	}
}

// getPrompts returns prompts for this component only.
func (bpm *BasePromptMixin) getPrompts() PromptDictType {
	return bpm.prompts
}

// getPromptModules returns sub-modules that implement PromptMixin.
func (bpm *BasePromptMixin) getPromptModules() PromptMixinType {
	return bpm.modules
}

// updatePrompts updates prompts for this component only.
func (bpm *BasePromptMixin) updatePrompts(prompts PromptDictType) {
	for k, v := range prompts {
		bpm.prompts[k] = v
	}
}

// SetPrompt sets a single prompt.
func (bpm *BasePromptMixin) SetPrompt(name string, prompt BasePromptTemplate) {
	bpm.prompts[name] = prompt
}

// GetPrompt gets a single prompt by name.
func (bpm *BasePromptMixin) GetPrompt(name string) BasePromptTemplate {
	return bpm.prompts[name]
}

// AddModule adds a sub-module that implements PromptMixin.
func (bpm *BasePromptMixin) AddModule(name string, module PromptMixin) {
	bpm.modules[name] = module
}

// Ensure BasePromptMixin implements PromptMixin.
var _ PromptMixin = (*BasePromptMixin)(nil)
