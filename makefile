REPO_URL_MA := https://github.com/Mosquito-Alert/ma_dashboard.git
REPO_URL_UkoZJ := https://github.com/UkoZJ/ma_dashboard.git
REPO_DEST := $(CURDIR)
REPO_FOLDER := $(notdir $(CURDIR))
ENV_NAME := $(REPO_FOLDER)_env
ENV_DEST := $(REPO_DEST)/.envs/$(ENV_NAME)

PORT := 5006
PYTHON := $(ENV_DEST)/bin/python

# Run application #############################################################
.PHONY: serve_remote, serve_local, ngrok, pipeline
.SILENT: serve_remote, serve_local, ngrok, pipeline

serve_remote:
	make -j 2 serve_local ngrok
serve_local: 
	$(PYTHON) -m panel serve panel_apps/english.py \
	--allow-websocket-origin=* \
	--warm \
	--reuse-sessions \
	--global-loading-spinner \
	--port $(PORT)
serve_local_access:
	$(PYTHON) -m panel serve panel_apps/english.py \
	--allow-websocket-origin=* \
	--warm \
	--reuse-sessions \
	--global-loading-spinner \
	--port $(PORT) \
	--static-dirs assets=./assets \
	--basic-auth .credentials.json \
	--cookie-secret admin_pass \
	--basic-login-template ./panel_apps/login.html 
ngrok:
	/home/uko/Dev/ma_dashboard/lib/ngrok http $(PORT)
pipeline:
	$(PYTHON) src/pipeline.py

# Install app from repo #######################################################
.PHONY: decrypt-repo setup-env clean

setup: setup-env decrypt-repo

setup-env:
	@echo "Setting up environment..."
	@if ! command -v micromamba > /dev/null; then \
		echo "Installing micromamba..."; \
		curl -L micro.mamba.pm/install.sh | bash || (echo "Failed to install micromamba"; exit 1); \
	fi
	@cd $(REPO_DEST) && micromamba env create -f env.yaml -p $(ENV_DEST) -y || (echo "Failed to create micromamba environment"; exit 1)

decrypt-repo: 
	@echo "Decrypting secrets..."
	@cd $(REPO_DEST) && make decrypt || (echo "Failed to decrypt secrets"; exit 1)
	
# Push to remote ##############################################################
.PHONY: git-push

git-push:
	@echo "Encrypting secrets before pushing..."
	@make encrypt
	@echo "Committing encrypted files..."
	@git add .
	@git diff-index --quiet HEAD || git commit -m "Encrypt secrets before push"
	@echo "Pushing to $(REPO_URL_MA)..."
	@git push $(REPO_URL_MA) || (echo "Failed to push to $(REPO_URL_MA)"; exit 1)
	@echo "Pushing to $(REPO_URL_UkoZJ)..."
	@git push $(REPO_URL_UkoZJ) || (echo "Failed to push to $(REPO_URL_UkoZJ)"; exit 1)
	@echo "Push completed successfully to both remotes."
	
# Handle Secrets ##############################################################
.PHONY: encrypt, decrypt
.SILENT: encrypt, decrypt

encrypt:
	$(PYTHON) src/secure_files.py -d ./config -m encrypt -fp ".*"
decrypt:
	$(PYTHON) src/secure_files.py -d ./config -m decrypt -fp ".*.enc"
