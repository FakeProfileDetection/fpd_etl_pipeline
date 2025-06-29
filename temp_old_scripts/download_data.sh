#!/usr/bin/env bash
# ------------------------------------------------------------------
# download_data.sh – research data bootstrap (headless‑aware)
# ------------------------------------------------------------------
# * Downloads a .tar.gz dataset from Google Cloud Storage
# * Creates/updates .env with DATA_PATH
# * Works on macOS (Homebrew) & Linux without global config changes
# * Flags:
#     --non-interactive   run unattended (skip y/N prompts)
#     --headless          use --no-launch-browser for gcloud logins
# ------------------------------------------------------------------

set -euo pipefail

# ------------------------------------------------- config ---------
SCRIPT_DIR=''
ENV_FILE=".env.public"

# Globalize CURRENT variables for ease in piplining scripts
RAW_DATA_CURRENT=''
DATA_AFFIX_CURRENT=''
WEB_APP_DATA_DIR_CURRENT=''


def setup_environment() {
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  # shellcheck source=utils.sh
  source "${SCRIPT_DIR}/utils.sh"
  # Source environment variables from .env.public
  # This file should contain the PROJECT_ID, BUCKET_DIR, FILE_PATH, WEB_APP_DATA_DIR
  if [[ -f "${SCRIPT_DIR}/.env.public" ]]; then
    # shellcheck source=.env.public
    source ${SCRIPT_DIR}/.env.public
  else
    print_error "${SCRIPT_DIR}/.env.public not found. Please create it with the required variables."
    exit 1
  fi

  # Add recent dataset download/extraction names and paths to .env.current
  # Each dataset ends in timestamp-hostname
  DATA_AFFIX_CURRENT="$(date $TIME_FORMAT)-$(hostname)"
  RAW_DATA_RECENT="$RAW_DATA_DIR-${DATA_AFFIX_CURRENT}"
  WEB_APP_DATA_DIR_CURRENT="${WEB_APP_DATA_DIR}-${DATA_AFFIX_CURRENT}"
  
  # Create .env.current, overwrite whatever exists--append to end of file--should use last line
  # This can be pushed to the repository to ensure team members have the same environment and dataset version
  ENV_FILE_CURRENT="${SCRIPT_DIR}/.env.current"
  echo "DATA_PATH_CURRENT=${RAW_DATA_RECENT}" >> "$ENV_FILE_CURRENT"
  echo "DATA_AFFIX_CURRENT=${DATA_AFFIX_CURRENT}" >> "$ENV_FILE_CURRENT"
  echo "WEB_APP_DATA_DIR_CURRENT=${WEB_APP_DATA_DIR_CURRENT}" >> "$ENV_FILE_CURRENT"

}


# ------------------------------------------------- flags ----------
NON_INTERACTIVE=false
HEADLESS=false
for arg in "$@"; do
  case "$arg" in
    --non-interactive) NON_INTERACTIVE=true ;;
    --headless)        HEADLESS=true        ;;
  esac
done

# ------------------------------------------------- helpers --------
install_gcloud_macos() {
  [[ $(uname) == Darwin ]] || return 1
  command_exists gcloud && return 0
  command_exists brew   || return 1
  print_info "Installing Google Cloud SDK via Homebrew (one‑time)…"
  brew install --quiet --cask google-cloud-sdk
}

ensure_gcloud() {
  command_exists gcloud && return 0
  install_gcloud_macos   || {
      print_error "gcloud CLI not found. Install manually:"
      echo "  • macOS: brew install --cask google-cloud-sdk"
      echo "  • Linux: https://cloud.google.com/sdk/docs/install"
      return 1
  }
}

confirm_or_skip() {            # $1 = prompt, true → proceed
  $NON_INTERACTIVE && return 0
  read -r -p "$1 (y/N): " ans
  [[ "$ans" =~ ^[Yy]$ ]]
}

maybe_gcloud_login() {
  if [[ -z $(gcloud config get-value account 2>/dev/null) ]]; then
      print_info "No gcloud account; starting login (headless=$HEADLESS)…"
      if $HEADLESS; then
          gcloud auth login --no-launch-browser || { print_error "Login failed"; exit 1; }
      else
          gcloud auth login || { print_error "Login failed"; exit 1; }
      fi
  fi
  print_info "Logged in as: $(gcloud config get-value account)"
}

maybe_adc_login() {
  if ! gcloud auth application-default print-access-token &>/dev/null; then
      print_info "Creating application‑default credentials…"
      if $HEADLESS; then
          gcloud auth application-default login --no-launch-browser
      else
          gcloud auth application-default login
      fi
      gcloud auth application-default set-quota-project "$PROJECT_ID" || true
  fi
}

download_web_app_data() {
  # ----- data directory & (re)download? -----
  mkdir -p "$WEB_APP_DATA_DIR"

  print_step "Downloading dataset"
  SRC="gs://$WEB_APP_DATA_SOURCE"
  DEST="$WEB_APP_DATA_DIR_CURRENT"
  gcloud storage cp "$SRC" "$DEST"

  if [[ ! -f "$DEST" ]]; then
      print_error "Failed to download $SRC to $DEST"
      exit 1
  fi
 
}

# ------------------------------------------------ main ------------
main() {
  setup_environment

  print_step "Checking Google Cloud CLI"
  ensure_gcloud || exit 1

  maybe_gcloud_login

  if [[ $(gcloud config get-value project 2>/dev/null) != "$PROJECT_ID" ]]; then
      gcloud config set project "$PROJECT_ID" >/dev/null
  fi
  maybe_adc_login

  download_web_app_data

  # ----- .env update -----
  print_step "Done!"
  print_info "To use the raw  dataset, source .env.current and use  $RAW_DATA_CURRENT""
  
}

main "$@"
