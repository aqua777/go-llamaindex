FROM golang:1.26-trixie AS golang

ARG DEV_USER_ID=1001
ARG DEV_USER_NAME=dev

ENV DEV_USER_ID=${DEV_USER_ID}
ENV DEV_USER_NAME=${DEV_USER_NAME}


COPY --chmod=0755 .docker/scripts/go-* /usr/local/bin/

RUN apt-get update && apt-get install -y net-tools sqlite3 sudo zsh && \
    \
    groupadd -g "${DEV_USER_ID}" "${DEV_USER_NAME}" && \
    useradd -m -u "${DEV_USER_ID}" -g "${DEV_USER_ID}" "${DEV_USER_NAME}" && \
    echo "${DEV_USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    echo "User ${DEV_USER_NAME} created with ID: ${DEV_USER_ID}"

ENV GOCACHE=/home/${DEV_USER_NAME}/.cache/go/build
ENV GOMODCACHE=/home/${DEV_USER_NAME}/.cache/go/pkg/mod
ENV GOPATH=/usr/local

COPY .docker/zshrc /home/${DEV_USER_NAME}/.zshrc

CMD ["/bin/zsh"]

# ------------------------------------------------------------------------------
FROM golang AS devcontainer

RUN go env && bash /usr/local/bin/go-install-vscode-tools
