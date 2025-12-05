FROM golang:1.25 AS golang

ARG GO_USER_ID=1001
ARG GO_USER_NAME=dev

ENV GO_USER_ID=${GO_USER_ID}
ENV GO_USER_NAME=${GO_USER_NAME}

ENV GOCACHE=/tmp/.cache/go/build
ENV GOMODCACHE=/tmp/.cache/go/pkg/mod

COPY --chmod=0755 .docker/scripts/go-* /usr/local/bin/

RUN apt-get update && apt-get install -y net-tools sqlite3 sudo zsh && \
    \
    groupadd -g "${GO_USER_ID}" "${GO_USER_NAME}" && \
    useradd -m -u "${GO_USER_ID}" -g "${GO_USER_ID}" "${GO_USER_NAME}" && \
    echo "${GO_USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    echo "User ${GO_USER_NAME} created with ID: ${GO_USER_ID}"

COPY .docker/zshrc /home/${GO_USER_NAME}/.zshrc

CMD ["/bin/zsh"]

# ------------------------------------------------------------------------------
FROM golang AS devcontainer

RUN go env && bash /usr/local/bin/go-install-vscode-tools
