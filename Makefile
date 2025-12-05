export BUILDKIT_PROGRESS ?= plain

PROJECT      ?= go-llamaindex
WORKDIR      ?= /$(PROJECT)
GO_USER_NAME ?= dev

DOCKER ?= docker
DOCKER_RUN_FLAG ?= $(shell if [ -t 0 ]; then echo -it; else echo -t; fi)
DOCKER_RUN   = $(DOCKER) run $(DOCKER_RUN_FLAG)
DOCKER_BUILD = $(DOCKER) build

DOCKER_RUN_OPTIONS  = -v $(PWD):$(WORKDIR) -w /$(WORKDIR)
DOCKER_RUN_OPTIONS += -u $(GO_USER_NAME)

.PHONY: go go-shell

go:
	$(DOCKER_BUILD) -t $(PROJECT)-builder:dev -f .docker/Dev.dockerfile \
		--build-arg GO_USER_NAME=$(GO_USER_NAME) \
		.

go-shell:
	$(DOCKER_RUN) $(DOCKER_RUN_OPTIONS) $(PROJECT)-builder:dev

tests:
	$(DOCKER_RUN) $(DOCKER_RUN_OPTIONS) $(PROJECT)-builder:dev go test -v -p 1 ./...
