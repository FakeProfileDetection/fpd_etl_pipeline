#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# setup.sh – Portable, non‑intrusive Python 3.12.5 environment bootstrap
# -----------------------------------------------------------------------------
# * Creates a **project‑local** interpreter in ./local-python or via pyenv / Homebrew
# * Never edits the user’s PATH, pyenv global/local, or Homebrew links
# * Works in fully attended mode (default) or unattended mode with --non-interactive
# -----------------------------------------------------------------------------

set -euo pipefail

# -------------------------- helper utilities ----------------------------------
print_step()    { printf "\033[1;34m==> %s\033[0m\n" "$1"; }
print_info()    { printf "\033[0;32m[INFO] %s\033[0m\n" "$1"; }
print_warning() { printf "\033[0;33m[WARN] %s\033[0m\n" "$1"; }
print_error()   { printf "\033[0;31m[ERR ] %s\033[0m\n" "$1" >&2; }
command_exists() { command -v "$1" >/dev/null 2>&1; }

# -------------------------- configuration --------------------------------------
PYTHON_VERSION="3.12.5"
PYTHON_MAJOR_MINOR="${PYTHON_VERSION%.*}"          # 3.12
VENV_NAME="venv-${PYTHON_VERSION}"
LOCAL_PYTHON_DIR="local-python"                    # relative path
PYTHON_CMD_FOUND=""                                # will hold absolute path
NON_INTERACTIVE=false

# -------------------------- arg parsing ----------------------------------------
for arg in "$@"; do
  case "$arg" in
    --non-interactive) NON_INTERACTIVE=true ;;
  esac
done

# -------------------------- discovery helpers ----------------------------------
check_existing_local_python() {
  local abs_dir="${PWD}/${LOCAL_PYTHON_DIR}"
  if [[ -x "${abs_dir}/bin/python3" ]]; then
    local ver
    ver="$("${abs_dir}/bin/python3" --version 2>&1 | awk '{print $2}')"
    [[ $ver == ${PYTHON_VERSION}* ]] || return 1
    PYTHON_CMD_FOUND="${abs_dir}/bin/python3"
    print_info "Found project‑local Python ${ver}"
    return 0
  fi
  return 1
}

check_pyenv_python_local() {
  command_exists pyenv || return 1
  if ! pyenv versions --bare | grep -qx "${PYTHON_VERSION}"; then
    print_info "pyenv missing ${PYTHON_VERSION}; installing (this can take a while)";
    pyenv install --skip-existing "${PYTHON_VERSION}" || return 1
  fi
  export PYENV_VERSION="${PYTHON_VERSION}"                  # session‑only
  PYTHON_CMD_FOUND="$(pyenv which python3)"
  [[ -x "$PYTHON_CMD_FOUND" ]] || return 1
  print_info "Using pyenv Python ${PYTHON_VERSION} at ${PYTHON_CMD_FOUND} (session‑only)"
  return 0
}

check_homebrew_python_explicit() {
  [[ $(uname) == Darwin ]] || return 1
  local brew_py="/opt/homebrew/bin/python${PYTHON_MAJOR_MINOR}"
  if [[ -x "$brew_py" ]]; then
    PYTHON_CMD_FOUND="$brew_py"
    print_info "Using Homebrew Python at ${brew_py}"
    return 0
  fi
  return 1
}

check_system_python() {
  local cmd
  for cmd in python3 "python${PYTHON_MAJOR_MINOR}"; do
    command_exists "$cmd" || continue
    local ver
    ver="$("$cmd" --version 2>&1 | awk '{print $2}')"
    if [[ $ver == ${PYTHON_VERSION}* ]]; then
      PYTHON_CMD_FOUND="$cmd"
      print_info "Using system Python ${ver}"
      return 0
    fi
  done
  return 1
}

# ----------------------- build from source (local) -----------------------------
build_python_from_source_local() {
  print_step "Building Python ${PYTHON_VERSION} from source (local)"
  local abs_dir="${PWD}/${LOCAL_PYTHON_DIR}"
  rm -rf "$abs_dir"; mkdir -p "$abs_dir"

  local url="https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz"
  curl -sSL "$url" -o Python.tgz || { print_error "Download failed"; return 1; }
  tar -xzf Python.tgz && rm Python.tgz
  cd "Python-${PYTHON_VERSION}"

  local config_opts=""
  if [[ $(uname) == Darwin ]]; then
    # Ensure Homebrew libs, but never touch PATH
    for pkg in openssl@3 zlib xz readline sqlite3; do
      brew ls --versions "$pkg" >/dev/null 2>&1 || brew install "$pkg"
    done
    local openssl_prefix
    openssl_prefix="$(brew --prefix openssl@3)"
    export CPPFLAGS="-I${openssl_prefix}/include"
    export LDFLAGS="-L${openssl_prefix}/lib"
    export PKG_CONFIG_PATH="${openssl_prefix}/lib/pkgconfig"
    config_opts="--with-openssl=${openssl_prefix}"
    print_info "Linking against Homebrew OpenSSL at ${openssl_prefix}"
  fi

  ./configure ${config_opts} --prefix="$abs_dir" --enable-optimizations --with-ensurepip=install
  make -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc || echo 2)"
  make install
  cd .. && rm -rf "Python-${PYTHON_VERSION}"

  PYTHON_CMD_FOUND="${abs_dir}/bin/python3"
  [[ -x "$PYTHON_CMD_FOUND" ]]
}

# --------------------------- acquire python ------------------------------------
acquire_python() {
  check_existing_local_python     && return 0
  check_pyenv_python_local        && return 0
  check_homebrew_python_explicit  && return 0
  check_system_python             && return 0
  build_python_from_source_local  && return 0

  print_error "Failed to acquire Python ${PYTHON_VERSION}. Exiting."; exit 1
}

# --------------------------- virtual env ---------------------------------------
create_or_reuse_venv() {
  if [[ -d "$VENV_NAME" ]]; then
    if $NON_INTERACTIVE; then
      print_info "(non‑interactive) Recreating venv ${VENV_NAME}"
      rm -rf "$VENV_NAME"
    else
      read -r -p "Virtual env exists. Recreate? (y/N): " ans
      if [[ $ans =~ ^[Yy]$ ]]; then rm -rf "$VENV_NAME"; fi
    fi
  fi
  if [[ ! -d "$VENV_NAME" ]]; then
    print_step "Creating venv (${VENV_NAME})"
    "$PYTHON_CMD_FOUND" -m venv "$VENV_NAME"
  fi
  # shellcheck disable=SC1090
  source "${PWD}/${VENV_NAME}/bin/activate"
  print_info "Venv activated: $(python --version)"
}

# --------------------------- install deps --------------------------------------
install_core_deps() {
  python -m pip install --upgrade pip
  [[ -f requirements.txt ]] && python -m pip install -r requirements.txt || print_warning "requirements.txt not found"
}

install_pytorch() {
  if python -m pip show torch >/dev/null 2>&1; then return; fi
  if [[ $(uname) == Darwin && $(uname -m) == arm64 ]]; then
    python -m pip install torch torchvision torchaudio
  elif command_exists nvidia-smi; then
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  else
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  fi
}

install_jupyter_kernel() {
  local project="$(basename "$PWD")"
  local kernel_name="${project}_py${PYTHON_MAJOR_MINOR}"
  python -m ipykernel install --user --name "$kernel_name" --display-name "${project} (Py ${PYTHON_VERSION})"
  print_info "Jupyter kernel '${kernel_name}' installed"
}

make_activate_script() {
  cat > activate.sh << 'EOF'
#!/usr/bin/env bash
# Activate the venv and load .env
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROJECT_ROOT}/${VENV_NAME}/bin/activate"
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  source "${PROJECT_ROOT}/.env"
  set +a
fi
echo "Environment activated. Run 'deactivate' to exit."
EOF
  chmod +x activate.sh
}

# --------------------------- main ---------------------------------------------
print_step "Bootstrap starting (non‑interactive=$NON_INTERACTIVE)"
acquire_python
print_info "Using Python at: $PYTHON_CMD_FOUND"
create_or_reuse_venv
install_core_deps
install_pytorch
install_jupyter_kernel
# make_activate_script

print_step "Done!"
print_info "Next: source activate.sh && jupyter lab"
exit 0
