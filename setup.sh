#!/usr/bin/env bash
set -euo pipefail

DO_BUILD=0
DO_RUN=1
USE_CI=0
INSTALL_SYSTEM_DEPS=0
SETUP_WHISPER=1
WHISPER_MODEL="${WHISPER_MODEL:-base}"
WHISPER_LANGUAGE="${WHISPER_LANGUAGE:-en}"
WHISPER_SEGMENT_MS="${WHISPER_SEGMENT_MS:-4000}"
WHISPER_VENV_DIR=".venv-whisper"
WHISPER_MODEL_DIR=".whisper-models"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OS_NAME="unknown"
PLATFORM_BUILD_SCRIPT="build"
PYTHON_BIN="python3"
WHISPER_PIP_PATH=""
WHISPER_COMMAND_PATH=""
MLX_VENV_DIR="$HOME/Library/Application Support/OpenCluely/WhisperMLX"

print_header() {
  echo "========================================"
  echo " OpenCluely Setup"
  echo "========================================"
}

usage() {
  cat <<EOF
Usage: ./setup.sh [options]

This script will:
1. Create .env from env.example when needed
2. Install Node dependencies
3. Optionally set up local Whisper in ${WHISPER_VENV_DIR}
4. Optionally install system audio dependencies
5. Optionally build the app
6. Optionally run OpenCluely

Options:
  --build                 Build a distributable for this OS
  --no-run                Do not start the app after setup
  --run                   Start the app after setup (default)
  --ci                    Use 'npm ci' instead of 'npm install'
  --install-system-deps   Attempt to install sox where possible
  --skip-whisper          Skip local Whisper environment setup
  -h, --help              Show this help

Environment variables:
  GEMINI_API_KEY          If provided, writes into .env
  WHISPER_MODEL           Whisper model for the original CLI setup (default: base)
  WHISPER_LANGUAGE        Whisper language to configure (default: en)
  WHISPER_SEGMENT_MS      Segment size in ms (default: 4000)
  WHISPER_BACKEND         Set openai to use the original Whisper CLI on Apple Silicon
  WHISPER_BOOTSTRAP_PYTHON  Python executable for MLX setup (default: python3)

Example:
  GEMINI_API_KEY=your_key_here ./setup.sh --install-system-deps
EOF
}

for arg in "$@"; do
  case "$arg" in
    --build) DO_BUILD=1 ;;
    --no-run) DO_RUN=0 ;;
    --run) DO_RUN=1 ;;
    --ci) USE_CI=1 ;;
    --install-system-deps) INSTALL_SYSTEM_DEPS=1 ;;
    --skip-install-system-deps) INSTALL_SYSTEM_DEPS=0 ;;
    --skip-whisper) SETUP_WHISPER=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg"; usage; exit 1 ;;
  esac
done

print_header
cd "$SCRIPT_DIR"

detect_os() {
  local uname_out
  uname_out=$(uname -s || echo "unknown")
  case "$uname_out" in
    Linux*) OS_NAME="linux" ;;
    Darwin*) OS_NAME="macos" ;;
    CYGWIN*|MINGW*|MSYS*) OS_NAME="windows" ;;
    *) OS_NAME="unknown" ;;
  esac

  case "$OS_NAME" in
    macos) PLATFORM_BUILD_SCRIPT="build:mac" ;;
    windows) PLATFORM_BUILD_SCRIPT="build:win" ;;
    linux) PLATFORM_BUILD_SCRIPT="build:linux" ;;
    *) PLATFORM_BUILD_SCRIPT="build" ;;
  esac

  case "$OS_NAME" in
    windows)
      PYTHON_BIN="python"
      WHISPER_PIP_PATH="${WHISPER_VENV_DIR}/Scripts/pip.exe"
      WHISPER_COMMAND_PATH="${WHISPER_VENV_DIR}/Scripts/whisper.exe"
      ;;
    *)
      PYTHON_BIN="python3"
      WHISPER_PIP_PATH="${WHISPER_VENV_DIR}/bin/pip"
      WHISPER_COMMAND_PATH="${WHISPER_VENV_DIR}/bin/whisper"
      ;;
  esac
}

require_command() {
  local cmd="$1"
  local message="$2"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: ${message}"
    exit 1
  fi
}

ensure_env_file() {
  if [[ ! -f .env ]]; then
    if [[ -f env.example ]]; then
      echo "Creating .env from env.example"
      (umask 077; cp env.example .env; chmod 600 .env)
    else
      echo "Error: env.example is missing"
      exit 1
    fi
  fi
}

upsert_env() {
  local key="$1"
  local value="$2"

  if grep -q "^${key}=" .env 2>/dev/null; then
    OPENCLUELY_ENV_KEY="$key" OPENCLUELY_ENV_VALUE="$value" \
      perl -pi -e 's/^\Q$ENV{OPENCLUELY_ENV_KEY}\E=[^\r\n]*/$ENV{OPENCLUELY_ENV_KEY}=$ENV{OPENCLUELY_ENV_VALUE}/' .env
  else
    printf "%s=%s\n" "$key" "$value" >> .env
  fi
}

ensure_gemini_key() {
  if [[ -n "${GEMINI_API_KEY:-}" ]]; then
    upsert_env "GEMINI_API_KEY" "$GEMINI_API_KEY"
  fi

  # GEMINI_API_KEY is now OPTIONAL during setup. The app's first-run
  # flow will auto-open the Settings window if the key is missing and
  # guide the user to enter it. Hard-blocking at install time makes
  # CI / packaging / scripted installs harder for no real benefit.

  if ! grep -q '^GEMINI_API_KEY=' .env 2>/dev/null; then
    # Make sure the key exists in .env even if unset — the app reads
    # this on first-run to know whether onboarding is needed.
    echo "GEMINI_API_KEY=your_gemini_api_key_here" >> .env
  fi

  if grep -q 'your_gemini_api_key_here' .env 2>/dev/null; then
    echo ""
    echo "=========================================="
    echo " No Gemini API key detected"
    echo "=========================================="
    echo ""
    echo "The app will start, but AI features won't work until you set"
    echo "GEMINI_API_KEY in .env (or via the Settings window on first launch)."
    echo ""
    echo "Get a free key from: https://aistudio.google.com/"
    echo ""
    echo "Setup will continue without blocking."
    echo ""
  fi
}

install_system_deps() {
  if [[ "$INSTALL_SYSTEM_DEPS" -ne 1 ]]; then
    return
  fi

  echo "Attempting to install system audio dependencies"

  if command -v sox >/dev/null 2>&1; then
    echo "sox already installed"
    return
  fi

  case "$OS_NAME" in
    macos)
      # OpenCluely captures microphone audio via the renderer (Web Audio API) on
      # macOS, so the native sox/arecord recorders are not used. Skip installing
      # sox to avoid an unnecessary Homebrew dependency.
      echo "macOS uses built-in renderer audio capture; skipping sox install."
      ;;
    linux)
      if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -y && sudo apt-get install -y sox || echo "Could not install sox via apt-get"
      elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y sox || echo "Could not install sox via dnf"
      elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -S --noconfirm sox || echo "Could not install sox via pacman"
      else
        echo "Unknown package manager. Install sox manually."
      fi
      ;;
    windows)
      echo "Install sox manually on Windows, for example via Chocolatey: choco install sox"
      ;;
    *)
      echo "Unknown OS. Install sox manually if you want microphone capture."
      ;;
  esac
}

install_node_deps() {
  if [[ -f package-lock.json && "$USE_CI" -eq 1 ]]; then
    echo "Installing Node dependencies with npm ci"
    npm ci
  else
    echo "Installing Node dependencies with npm install"
    npm install
  fi
}

setup_mlx_whisper() {
  local python="${WHISPER_BOOTSTRAP_PYTHON:-python3}"
  require_command "$python" "Install native Python 3 for Apple Silicon."
  if ! "$python" -c 'import platform, sys; sys.exit(not (platform.machine() == "arm64" and sys.version_info >= (3, 10)))'; then
    echo "Error: MLX Whisper needs native arm64 Python 3.10 or newer."
    return 1
  fi

  local venv_python="$MLX_VENV_DIR/bin/python"
  if [[ ! -x "$venv_python" ]]; then
    "$python" -m venv "$MLX_VENV_DIR"
  fi
  if ! "$venv_python" -c 'from importlib.metadata import version; assert version("mlx-whisper") == "0.4.3"' 2>/dev/null; then
    "$venv_python" -m pip install --disable-pip-version-check 'mlx-whisper==0.4.3'
  fi
  "$venv_python" -c 'import mlx.core as mx; from mlx_whisper import transcribe; assert mx.metal.is_available()'
  touch "$MLX_VENV_DIR/.ready"
  echo "MLX Whisper is ready. The base model is downloaded on first use."
}

setup_whisper_env() {
  if [[ "$SETUP_WHISPER" -ne 1 ]]; then
    echo "Skipping local Whisper setup"
    return
  fi

  # Skip whisper setup when only building (build distributions don't need local whisper)
  if [[ "$DO_BUILD" -eq 1 && "$DO_RUN" -eq 0 ]]; then
    echo "Skipping local Whisper setup (build-only mode)"
    return
  fi

  if [[ "$OS_NAME" == "macos" && "$(uname -m)" == "arm64" ]]; then
    if [[ "${WHISPER_BACKEND:-auto}" != "openai" ]] && ! grep -q '^WHISPER_BACKEND=openai$' .env; then
      setup_mlx_whisper
      return
    fi
    upsert_env "WHISPER_BACKEND" "openai"
  fi

  require_command "$PYTHON_BIN" "Python 3 is required for local Whisper setup."

  if [[ ! -d "$WHISPER_VENV_DIR" ]]; then
    echo "Creating Whisper virtual environment at $WHISPER_VENV_DIR"
    "$PYTHON_BIN" -m venv "$WHISPER_VENV_DIR"
  fi

  echo "Installing local Whisper into $WHISPER_VENV_DIR"
  "$WHISPER_PIP_PATH" install --upgrade pip || true
  "$WHISPER_PIP_PATH" install openai-whisper || {
    echo "WARNING: pip install openai-whisper failed. Whisper may be unavailable."
    echo "Common causes: insufficient disk space (needs ~3-5 GB), missing Python headers, or network issues."
  }

  mkdir -p "$WHISPER_MODEL_DIR"

  # Verify the Whisper CLI actually exists before claiming it's configured
  local whisper_found=0
  if [[ -f "$WHISPER_COMMAND_PATH" ]]; then
    echo "Whisper CLI found at: $WHISPER_COMMAND_PATH"
    whisper_found=1
  else
    # Fallback: try python -m whisper inside the venv
    local venv_python
    if [[ "$OS_NAME" == "windows" ]]; then
      venv_python="${WHISPER_VENV_DIR}/Scripts/python.exe"
    else
      venv_python="${WHISPER_VENV_DIR}/bin/python"
    fi
    if [[ -f "$venv_python" ]] && "$venv_python" -m whisper --help >/dev/null 2>&1; then
      echo "Whisper CLI not found as standalone script, but 'python -m whisper' works."
      echo "Adjusting WHISPER_COMMAND to use venv Python module."
      WHISPER_COMMAND_PATH="$venv_python -m whisper"
      whisper_found=1
    else
      echo "WARNING: Whisper CLI not found at $WHISPER_COMMAND_PATH"
      echo "Speech recognition will be unavailable until Whisper is properly installed."
      echo "You can skip this with: ./setup.sh --skip-whisper"
    fi
  fi

  upsert_env "SPEECH_PROVIDER" "whisper"
  upsert_env "WHISPER_COMMAND" "${WHISPER_COMMAND_PATH}"
  upsert_env "WHISPER_MODEL_DIR" "${WHISPER_MODEL_DIR}"
  upsert_env "WHISPER_MODEL" "${WHISPER_MODEL}"
  upsert_env "WHISPER_LANGUAGE" "${WHISPER_LANGUAGE}"
  upsert_env "WHISPER_SEGMENT_MS" "${WHISPER_SEGMENT_MS}"

  if [[ "$whisper_found" -eq 1 ]]; then
    echo "Running Whisper smoke test"
    npm run test-speech
  else
    echo "Skipping Whisper smoke test (CLI not found)"
  fi
}

build_app() {
  if [[ "$DO_BUILD" -eq 1 ]]; then
    echo "Building app for $OS_NAME with npm run $PLATFORM_BUILD_SCRIPT"
    if [[ "$OS_NAME" == "macos" && "$(uname -m)" == "arm64" ]]; then
      npm run build:mac:arm64
    else
      npm run "$PLATFORM_BUILD_SCRIPT"
    fi
  fi
}

run_app() {
  if [[ "$DO_RUN" -eq 1 ]]; then
    echo "Starting app"
    npm start
  else
    echo "Setup complete. Skipping run."
  fi
}

detect_os
echo "Detected OS: $OS_NAME"
require_command node "Node.js 22.12+ is required."
require_command npm "npm is required."
if ! node -e 'const [major, minor] = process.versions.node.split(".").map(Number); process.exit(major > 22 || (major === 22 && minor >= 12) ? 0 : 1)'; then
  echo "Error: Node.js 22.12 or newer is required."
  exit 1
fi
if [[ "$OS_NAME" == "macos" && "$(sysctl -in hw.optional.arm64 2>/dev/null)" == "1" ]] &&
   [[ "$(node -p process.arch)" != "arm64" || "$(uname -m)" != "arm64" ]]; then
  echo "Error: Run setup in a native arm64 terminal with arm64 Node.js."
  exit 1
fi
echo "Node: $(node -v)"
echo "npm:  $(npm -v)"

ensure_env_file
ensure_gemini_key
install_system_deps
install_node_deps
setup_whisper_env
build_app
run_app
