SHELL := /bin/bash

# Variables
SECRETS_DIR := .secrets
SOPS_AGE_KEY_FILE := $(shell pwd)/$(SECRETS_DIR)/age-key.txt
SECRET_CONFIG_FILES := $(shell find . -type f \( -name '.*.ini' -o -name '.*.json' \) ! -name '.config.session.ini')
ENCRYPTED_CONFIG_FILES := $(shell find . -type f \( -name '.*.ini.enc' -o -name '.*.json.enc' \))

REPO_URL_MA := git@github.com:Mosquito-Alert/ma_dashboard.git
REPO_URL_UkoZJ := git@github.com:UkoZJ/ma_dashboard.git

REPO_DEST := $(CURDIR)
REPO_FOLDER := $(notdir $(CURDIR))
ENV_NAME := $(REPO_FOLDER)_env
ENV_DEST := $(REPO_DEST)/.envs/$(ENV_NAME)

PORT := 5006
PYTHON := $(ENV_DEST)/bin/python

.PHONY: help setup-env setup-secrets encrypt decrypt decrypt-clean

help: ## Display this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@awk -F ':|##' '/^[^\t].+?:.*?##/ { printf "  %-20s %s\n", $$1, $$NF }' $(MAKEFILE_LIST)

setup: setup-env decrypt
	@echo "✅ Setup completed successfully"

setup-env:
	@echo "Setting up environment..."
	@if ! command -v micromamba > /dev/null; then \
		echo "Installing micromamba..."; \
		curl -L micro.mamba.pm/install.sh | bash || (echo "❌ Failed to install micromamba"; exit 1); \
		echo "✅ Micromamba installed successfully"; \
	fi
	@cd $(REPO_DEST) && micromamba env create -f env.yaml -p $(ENV_DEST) -y || (echo "❌ Failed to create micromamba environment"; exit 1)
	@echo "✅ Environment created successfully"

setup-secrets:
	@mkdir -p $(SECRETS_DIR)
	@if [ ! -f "$(SOPS_AGE_KEY_FILE)" ]; then \
		echo "Generating new age key pair..."; \
		age-keygen -o "$(SOPS_AGE_KEY_FILE)"; \
		echo "⚠️  Make sure to share $(SOPS_AGE_KEY_FILE) securely with your team"; \
	else \
		echo "✅ Age key already exists at $(SOPS_AGE_KEY_FILE)"; \
	fi;
	@if [ ! -f ".sops.yaml" ]; then \
		echo "Creating .sops.yaml..."; \
		echo "creation_rules:" > .sops.yaml; \
		echo "  - path_regex: .*.ini >> .sops.yaml; \
		echo "    age: $$(age-keygen -y "$(SOPS_AGE_KEY_FILE)")" >> .sops.yaml; \
	fi;
	@echo "✅ Secrets setup complete"

encrypt: 
	@for file in $(SECRET_CONFIG_FILES); do \
		echo "Encrypting $$file..."; \
		sops -e $$file > $$file.enc; \
	done
	@echo "✅ All .config.ini files encrypted or overwritten"

decrypt:
	@for file in $(ENCRYPTED_CONFIG_FILES); do \
		echo "Decrypting $$file..."; \
		decrypted_file="$$(echo $$file | sed 's#\.enc$$##')"; \
		case "$$decrypted_file" in \
			*.ini) \
				echo "--> $$decrypted_file"; \
				SOPS_AGE_KEY_FILE=$(SOPS_AGE_KEY_FILE) sops --input-type ini --output-type ini -d $$file > $$decrypted_file ;; \
			*.json) \
				echo "--> $$decrypted_file"; \
				SOPS_AGE_KEY_FILE=$(SOPS_AGE_KEY_FILE) sops --input-type json --output-type json -d $$file > $$decrypted_file ;; \
			*) \
				echo "Error: Unknown file type for $$decrypted_file" >&2; \
				exit 1 ;; \
		esac; \
	done
	@echo "✅ All encrypted files decrypted"

decrypt-clean: decrypt
	@echo "Removing decrypted config files..."
	@for file in $(ENCRYPTED_CONFIG_FILES); do \
		rm -f "$$file"; \
	done
	@echo "✅ Cleaned up decrypted files"

# Encrypt/decrypt using secure_file.py utility
# encrypt:
# 	$(PYTHON) src/secure_files.py -d ./config -m encrypt -fp ".*"
# decrypt:
# 	$(PYTHON) src/secure_files.py -d ./config -m decrypt -fp ".*.enc"

# Push to remote ##############################################################
.PHONY: git-push install-git-hooks

# Git hooks installation
install-git-hooks: ## Install git hooks to prevent committing unencrypted files
	@mkdir -p .git/hooks
	@echo '#!/bin/sh' > .git/hooks/pre-commit
	@echo 'SECRET_CONFIG_FILES="$(SECRET_CONFIG_FILES)"' >> .git/hooks/pre-commit
	@echo 'files=$$(git diff --cached --name-only | grep -E "$${SECRET_CONFIG_FILES// /|}")' >> .git/hooks/pre-commit
	@echo 'unencrypted_files=$$(echo "$$files" | grep -v "\.enc$$")' >> .git/hooks/pre-commit
	@echo 'if [ -n "$$unencrypted_files" ]; then' >> .git/hooks/pre-commit
	@echo '    echo "Error: Attempting to commit unencrypted config files:"' >> .git/hooks/pre-commit
	@echo '    echo "$$unencrypted_files"' >> .git/hooks/pre-commit
	@echo '    echo "Please encrypt these files first with: make encrypt"' >> .git/hooks/pre-commit
	@echo '    exit 1' >> .git/hooks/pre-commit
	@echo 'fi' >> .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "✅ Git hooks installed"
	@echo "⚠️ Remember to run 'git add .git/hooks/pre-commit' to track the hook"

git-push: encrypt ## Usage: make git-push message="Your commit message here"
	@echo "Committing encrypted files..."
	@git add .
	@if [ -n "$(message)" ]; then \
		git diff-index --quiet HEAD || git commit -m "$(message)"; \
	else \
		git diff-index --quiet HEAD || git commit -m "Encrypt secrets before push"; \
	fi
	@echo "Pushing to $(REPO_URL_MA)..."
	@git push $(REPO_URL_MA) || (echo "Failed to push to $(REPO_URL_MA)"; exit 1)
	@echo "Pushing to $(REPO_URL_UkoZJ)..."
	@git push $(REPO_URL_UkoZJ) || (echo "Failed to push to $(REPO_URL_UkoZJ)"; exit 1)
	@echo "Push completed successfully to both remotes."


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