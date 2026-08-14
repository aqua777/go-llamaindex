# Agent Instructions for Krait

Look, if you're an AI agent working on this repo, pay attention. I'm not going to repeat myself. This is the `krait` package—a wrapper that smashes Cobra and Viper together so we don't have to deal with their boilerplate garbage. 

Here are the ground rules. Stick to them, or you're just going to make a mess we have to clean up later.

## 1. The Golden Rules of Not Being Annoying
- **Do exactly what's asked, nothing more.** Don't go writing code or adding "cool features" the user didn't explicitly ask for. We aren't mind readers, and nobody likes a show-off.
- **Don't assume functionality.** If you aren't sure what a function should do, ask for suggestions. Don't just guess and pray it works.
- **Hands off `./vendor/`.** If that directory exists, pretend it's radioactive. Do not touch any files inside it.

## 2. Code Quality & Refactoring
- **Hunt down duplication.** Always keep an eye out for duplicated code. If you spot some, don't just blindly fix it—point it out to the user and ask if they want it refactored into reusable functions. 
- **Test coverage.** All functionality is expected to be unit tested with coverage no less than 90% per sub-package. The exception from this rule are situations when setting up test is too complex, such cases should be documented inside relavent test files as comments.

## 3. Unit Testing (Pay Attention Here)
When you're told to create a new unit tests file, do it exactly like this—no exceptions:
- Automatically import `"testing"` and `"github.com/stretchr/testify/suite"`.
- Initiate a data structure that inherits from `suite.Suite`.
- Write a top-level `Test<Custom>(t *testing.T)` function to run the suite.
- **CRITICAL:** *Never* write the actual unit tests when initiating a new unit tests file. Just set up the boilerplate and stop. We'll tell you when to write the tests.

## 4. Working with Krait
- This package uses a fluent API. Don't revert to standard Cobra/Viper spaghetti. Chain the methods like `krait.App().WithConfig().WithStringP().WithRun()`.
- Check the `examples/` directory if you forget how the API works. It's literally right there.
- Use the type-safe getters (`krait.GetString`, `krait.GetInt`, etc.) instead of whatever hacky workaround you might think of.

Keep it practical, keep it clean, and don't overcomplicate things. Now get to work.