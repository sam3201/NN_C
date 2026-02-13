# ensure nvm is loaded (normally from your shell rc)
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# install a modern Node (pick v22 or v25; v22 is enough for OpenClaw)
nvm install 22
nvm use 22

# optionally make it default
nvm alias default 22

# verify
which node
node -v

