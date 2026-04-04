# SPDX-License-Identifier: MIT
#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
SKIP_TESTS=0
SKIP_INSTALL=0
CURRENT_STEP="initialization"
PYTHON_BIN=""
VENV_PYTHON=""

STEP_PYTHON="pending"
STEP_VENV="pending"
STEP_INSTALL="pending"
STEP_TESTS="pending"
STEP_DEBUG="pending"

print_header() {
    printf '\n[%s/5] %s\n' "$1" "$2"
}

print_summary() {
    printf '\nSummary:\n'
    printf '  Python check: %s\n' "$STEP_PYTHON"
    printf '  Virtualenv:   %s\n' "$STEP_VENV"
    printf '  Install:      %s\n' "$STEP_INSTALL"
    printf '  Tests:        %s\n' "$STEP_TESTS"
    printf '  Debug run:    %s\n' "$STEP_DEBUG"
}

fail() {
    local message="$1"
    print_summary
    printf '\nERROR: %s\n' "$message" >&2
    exit 1
}

on_error() {
    fail "Step failed: $CURRENT_STEP"
}

trap on_error ERR

for arg in "$@"; do
    case "$arg" in
        --skip-tests)
            SKIP_TESTS=1
            ;;
        --skip-install)
            SKIP_INSTALL=1
            ;;
        -h|--help)
            cat <<'EOF'
Usage: bash reproduce.sh [--skip-tests] [--skip-install]

Options:
  --skip-tests    Skip the pytest step for a faster run
  --skip-install  Skip dependency installation inside .venv
EOF
            exit 0
            ;;
        *)
            fail "Unknown argument: $arg"
            ;;
    esac
done

CURRENT_STEP="checking Python 3.10 availability"
print_header 1 "Checking Python version..."
if command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.10)"
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)' >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    fail "Python 3.10 is required but was not found. Install Python 3.10 and rerun."
fi
printf 'Using Python: %s\n' "$PYTHON_BIN"
STEP_PYTHON="passed"

CURRENT_STEP="creating virtual environment"
print_header 2 "Creating virtual environment..."
if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    printf 'Created virtual environment in %s\n' "$VENV_DIR"
else
    printf 'Virtual environment already exists in %s\n' "$VENV_DIR"
fi
VENV_PYTHON="$VENV_DIR/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
    fail "Virtual environment Python not found at $VENV_PYTHON"
fi
STEP_VENV="passed"

CURRENT_STEP="installing dependencies"
print_header 3 "Installing dependencies..."
if [[ "$SKIP_INSTALL" -eq 1 ]]; then
    printf 'Skipping dependency installation (--skip-install).\n'
    STEP_INSTALL="skipped"
else
    "$VENV_PYTHON" -m pip install --upgrade pip
    "$VENV_PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt"
    STEP_INSTALL="passed"
fi

CURRENT_STEP="running test suite"
print_header 4 "Running test suite..."
if [[ "$SKIP_TESTS" -eq 1 ]]; then
    printf 'Skipping tests (--skip-tests).\n'
    STEP_TESTS="skipped"
else
    "$VENV_PYTHON" -m pytest "$SCRIPT_DIR/tests.py" -v
    STEP_TESTS="passed"
fi

CURRENT_STEP="running debug pipeline"
print_header 5 "Running debug pipeline..."
cd "$SCRIPT_DIR"
"$VENV_PYTHON" "$SCRIPT_DIR/main.py" --debug
STEP_DEBUG="passed"

print_summary
printf '\n✓ Reproduction complete. All steps passed.\n'