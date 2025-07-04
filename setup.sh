#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# setup.sh – Modern Python environment setup using uv package manager
# -----------------------------------------------------------------------------
# * Installs uv if not present
# * Creates a Python 3.12.10 environment using uv
# * Installs all project dependencies including optional GPU support
# * Sets up Jupyter kernel for the project
# * Works in both interactive (default) and non-interactive modes
# -----------------------------------------------------------------------------

set -euo pipefail

# -------------------------- Helper utilities ----------------------------------
print_step()    { printf "\033[1;34m==> %s\033[0m\n" "$1"; }
print_info()    { printf "\033[0;32m[INFO] %s\033[0m\n" "$1"; }
print_warning() { printf "\033[0;33m[WARN] %s\033[0m\n" "$1"; }
print_error()   { printf "\033[0;31m[ERR ] %s\033[0m\n" "$1" >&2; }
command_exists() { command -v "$1" >/dev/null 2>&1; }

# -------------------------- Configuration -------------------------------------
PYTHON_VERSION="3.12.10"
VENV_NAME=".venv"  # Standard uv convention
PROJECT_NAME="fpd_etl_pipeline"
NON_INTERACTIVE=false
SKIP_PYTORCH=false

# -------------------------- Argument parsing ----------------------------------
for arg in "$@"; do
  case "$arg" in
    --non-interactive) NON_INTERACTIVE=true ;;
    --skip-pytorch) SKIP_PYTORCH=true ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --non-interactive  Run without prompts (recreates venv if exists)"
      echo "  --skip-pytorch     Skip PyTorch installation"
      echo "  --help            Show this help message"
      exit 0
      ;;
  esac
done

# -------------------------- Install uv ----------------------------------------
install_uv() {
  if command_exists uv; then
    print_info "uv is already installed: $(uv --version)"
    return 0
  fi

  print_step "Installing uv package manager"

  # Detect OS and architecture
  OS="$(uname -s)"
  ARCH="$(uname -m)"

  if [[ "$OS" == "Darwin" ]]; then
    # macOS
    if command_exists brew; then
      print_info "Installing uv via Homebrew"
      brew install uv
    else
      print_info "Installing uv via curl"
      curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
  elif [[ "$OS" == "Linux" ]]; then
    # Linux
    print_info "Installing uv via curl"
    curl -LsSf https://astral.sh/uv/install.sh | sh
  else
    print_error "Unsupported OS: $OS"
    print_info "Please install uv manually: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
  fi

  # Add to PATH for current session if installed via curl
  if [[ -f "$HOME/.cargo/bin/uv" ]]; then
    export PATH="$HOME/.cargo/bin:$PATH"
    print_info "Added uv to PATH for current session"
    print_warning "Add 'export PATH=\"\$HOME/.cargo/bin:\$PATH\"' to your shell profile"
  fi

  # Verify installation
  if ! command_exists uv; then
    print_error "uv installation failed"
    exit 1
  fi

  print_info "uv installed successfully: $(uv --version)"
}

# -------------------------- Create/update virtual environment -----------------
setup_venv() {
  if [[ -d "$VENV_NAME" ]]; then
    if $NON_INTERACTIVE; then
      print_info "(non-interactive) Recreating virtual environment"
      rm -rf "$VENV_NAME"
    else
      read -r -p "Virtual environment '$VENV_NAME' exists. Recreate? (y/N): " response
      if [[ "$response" =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_NAME"
      else
        print_info "Using existing virtual environment"
        return 0
      fi
    fi
  fi

  print_step "Creating virtual environment with Python $PYTHON_VERSION"
  uv venv --python "$PYTHON_VERSION" "$VENV_NAME"

  if [[ ! -f "$VENV_NAME/bin/activate" ]]; then
    print_error "Failed to create virtual environment"
    exit 1
  fi

  print_info "Virtual environment created successfully"
}

# -------------------------- Install dependencies ------------------------------
install_dependencies() {
  print_step "Installing project dependencies"

  # Activate venv for subsequent commands
  source "$VENV_NAME/bin/activate"

  # Upgrade pip first
  uv pip install --upgrade pip

  # Install main requirements
  if [[ -f requirements.txt ]]; then
    print_info "Installing from requirements.txt"
    uv pip install -r requirements.txt
  else
    print_warning "requirements.txt not found"
  fi

  # Install development dependencies
  if [[ -f requirements-dev.txt ]]; then
    print_info "Installing from requirements-dev.txt"
    uv pip install -r requirements-dev.txt
  fi
}

# -------------------------- Install PyTorch with GPU support ------------------
install_pytorch() {
  if $SKIP_PYTORCH; then
    print_info "Skipping PyTorch installation (--skip-pytorch flag)"
    return 0
  fi

  # Check if PyTorch is already installed
  if python -c "import torch" 2>/dev/null; then
    print_info "PyTorch is already installed"
    return 0
  fi

  print_step "Installing PyTorch with appropriate backend"

  # Detect system and install appropriate PyTorch version
  if [[ "$(uname)" == "Darwin" ]]; then
    # macOS
    if [[ "$(uname -m)" == "arm64" ]]; then
      print_info "Detected Apple Silicon Mac - installing PyTorch with MPS support"
      uv pip install torch torchvision torchaudio
    else
      print_info "Detected Intel Mac - installing CPU-only PyTorch"
      uv pip install torch torchvision torchaudio
    fi
  elif command_exists nvidia-smi && nvidia-smi &>/dev/null; then
    # Linux/Windows with NVIDIA GPU
    print_info "Detected NVIDIA GPU - installing PyTorch with CUDA support"
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  else
    # CPU only
    print_info "No GPU detected - installing CPU-only PyTorch"
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  fi
}

# -------------------------- Install Jupyter kernel ----------------------------
install_jupyter_kernel() {
  print_step "Installing Jupyter kernel for project"

  # Create a kernel with the project name
  python -m ipykernel install --user \
    --name "${PROJECT_NAME}" \
    --display-name "FPD ETL Pipeline (Python ${PYTHON_VERSION})"

  print_info "Jupyter kernel '${PROJECT_NAME}' installed"
  print_info "You can select it in Jupyter with: Kernel > Change Kernel > FPD ETL Pipeline"
}

# -------------------------- Setup pre-commit hooks ----------------------------
setup_pre_commit() {
  if [[ ! -f .pre-commit-config.yaml ]]; then
    print_step "Creating pre-commit configuration"
    cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-json
      - id: check-merge-conflict

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.7
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--ignore-missing-imports]
EOF
    print_info "Created .pre-commit-config.yaml"
  fi

  # Install pre-commit hooks
  print_info "Installing pre-commit hooks"
  pre-commit install
}

# -------------------------- Create activation script --------------------------
create_activation_script() {
  print_step "Creating activation helper script"

  cat > activate.sh << 'EOF'
#!/usr/bin/env bash
# Convenient activation script for the project environment

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment
if [[ -f "${SCRIPT_DIR}/.venv/bin/activate" ]]; then
  source "${SCRIPT_DIR}/.venv/bin/activate"

  # Load environment variables if .env exists
  if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    set -a
    source "${SCRIPT_DIR}/.env"
    set +a
    echo "Loaded environment variables from .env"
  fi

  echo "FPD ETL Pipeline environment activated (Python $(python --version 2>&1 | awk '{print $2}'))"
  echo "Run 'deactivate' to exit the environment"
else
  echo "Error: Virtual environment not found. Run ./setup.sh first."
  exit 1
fi
EOF

  chmod +x activate.sh
  print_info "Created activate.sh helper script"
}

# -------------------------- Create .gitignore entries -------------------------
update_gitignore() {
  if [[ -f .gitignore ]]; then
    # Check if .venv is already in gitignore
    if ! grep -q "^\.venv" .gitignore; then
      print_info "Adding .venv to .gitignore"
      echo -e "\n# Virtual environment\n.venv/" >> .gitignore
    fi
  fi
}

# -------------------------- Main execution ------------------------------------
main() {
  print_step "FPD ETL Pipeline Setup (non-interactive=$NON_INTERACTIVE)"

  # Check Python version availability
  print_info "Checking Python $PYTHON_VERSION availability..."

  # Install uv if needed
  install_uv

  # Create/update virtual environment
  setup_venv

  # Install dependencies
  install_dependencies

  # Install PyTorch with GPU support
  install_pytorch

  # Install Jupyter kernel
  install_jupyter_kernel

  # Setup pre-commit hooks
  if ! $NON_INTERACTIVE; then
    read -r -p "Set up pre-commit hooks? (Y/n): " response
    if [[ ! "$response" =~ ^[Nn]$ ]]; then
      setup_pre_commit
    fi
  else
    setup_pre_commit
  fi

  # Create activation script
  create_activation_script

  # Update .gitignore
  update_gitignore

  print_step "Setup completed successfully! 🎉"
  print_info ""
  print_info "Next steps:"
  print_info "  1. Activate environment: source activate.sh"
  print_info "  2. Configure environment: cp .env.example .env && edit .env"
  print_info "  3. Run pipeline: python scripts/pipeline/run_pipeline.py --help"
  print_info "  4. Start Jupyter: jupyter lab"
  print_info ""
  print_info "For VS Code users: Select interpreter from .venv/bin/python"
}

# Run main function
main "$@"
