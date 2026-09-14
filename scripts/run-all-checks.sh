#!/usr/bin/env bash
#
# rms-spindoctor - Run All Checks Script
#
# This script runs linting, type checking, tests, and documentation build
# for the rms-spindoctor project.
#
# Usage:
#   ./scripts/run-all-checks.sh [options]
#
# Options:
#   -p, --parallel    Run all enabled scopes (code, docs, markdown) in parallel (faster, default)
#   -s, --sequential Run all checks sequentially (easier to debug)
#   -c, --code       Run only code checks (ruff, mypy, pytest)
#   -d, --docs       Run only documentation build (Sphinx + PyMarkdown lint)
#   -m, --markdown   Run only Markdown lint (PyMarkdown)
#   -i, --integration Include integration tests (slow; require PDS3_HOLDINGS_DIR
#                    + SPICE kernels).  Default: skipped.
#   -P, --postgres   Include PostgreSQL results-index tests (require a server named
#                    by SPINDOCTOR_TEST_POSTGRES_URL).  Default: skipped.
#   -h, --help       Show this help message
#
# Environment:
#   VENV or VENV_PATH  Path to virtualenv (default: $PROJECT_ROOT/venv)
#
# Code checks (run from project root with venv activated):
#   - ruff check (src, tests)
#   - ruff format --check (src, tests)
#   - mypy (src, tests)
#   - pytest (tests)
#
# Documentation checks:
#   - Sphinx build (docs/)
#   - PyMarkdown scan (docs/, root *.md, .cursor/)
#
# Exit codes:
#   0 - All checks passed
#   1 - One or more checks failed
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
RESET='\033[0m'

# Default options
PARALLEL=true
RUN_CODE=false
RUN_DOCS=false
RUN_MARKDOWN=false
SCOPE_SPECIFIED=false
RUN_INTEGRATION=false
RUN_POSTGRES=false

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Virtualenv path: set VENV or VENV_PATH to override default project venv
VENV="${VENV:-${VENV_PATH:-$PROJECT_ROOT/venv}}"

# Track failures
FAILED_CHECKS=()
EXIT_CODE=0

# Create temp directory for parallel output
TEMP_DIR=$(mktemp -d)
code_pid=""
docs_pid=""
markdown_pid=""

# Grace period (seconds) to wait for process to exit after SIGTERM before SIGKILL
CLEANUP_GRACE_PERIOD=${CLEANUP_GRACE_PERIOD:-5}
if ! echo "$CLEANUP_GRACE_PERIOD" | grep -qE '^[0-9]+$'; then
    echo "Error: CLEANUP_GRACE_PERIOD must be a non-negative integer (got: $CLEANUP_GRACE_PERIOD)" >&2
    exit 1
fi

_cleanup() {
    _wait_or_kill() {
        local pid=$1
        [ -z "$pid" ] && return
        kill -TERM "$pid" 2>/dev/null || true
        local waited=0
        while [ "$waited" -lt "$CLEANUP_GRACE_PERIOD" ]; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
            waited=$((waited + 1))
        done
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
        wait "$pid" 2>/dev/null || true
    }
    if [ -n "${code_pid}" ]; then
        _wait_or_kill "$code_pid"
        code_pid=""
    fi
    if [ -n "${docs_pid}" ]; then
        _wait_or_kill "$docs_pid"
        docs_pid=""
    fi
    if [ -n "${markdown_pid}" ]; then
        _wait_or_kill "$markdown_pid"
        markdown_pid=""
    fi
    rm -rf "$TEMP_DIR"
}
trap _cleanup EXIT SIGINT SIGTERM

# Print functions
print_header() {
    echo -e "\n${BOLD}${BLUE}===================================================${RESET}"
    echo -e "${BOLD}${BLUE}  $1${RESET}"
    echo -e "${BOLD}${BLUE}===================================================${RESET}\n"
}

print_section() {
    echo -e "\n${BOLD}${YELLOW}>>> $1${RESET}\n"
}

print_success() {
    echo -e "${GREEN}✓${RESET} $1"
}

print_error() {
    echo -e "${RED}✗${RESET} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${RESET} $1"
}

# Show usage
show_usage() {
    sed -n '/^# Usage:/,/^$/p' "$0" | sed 's/^# //g' | sed 's/^#//g'
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -s|--sequential)
            PARALLEL=false
            shift
            ;;
        -c|--code)
            RUN_CODE=true
            SCOPE_SPECIFIED=true
            shift
            ;;
        -d|--docs)
            RUN_DOCS=true
            SCOPE_SPECIFIED=true
            shift
            ;;
        -m|--markdown)
            RUN_MARKDOWN=true
            SCOPE_SPECIFIED=true
            shift
            ;;
        -i|--integration)
            RUN_INTEGRATION=true
            shift
            ;;
        -P|--postgres)
            RUN_POSTGRES=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${RESET}" >&2
            show_usage
            exit 1
            ;;
    esac
done

# If no scope flag was given, run code and docs (docs includes markdown lint)
if [ "$SCOPE_SPECIFIED" = false ]; then
    RUN_CODE=true
    RUN_DOCS=true
fi

# Start timer
START_TIME=$(date +%s)

print_header "rms-spindoctor - Running All Checks"

if [ "$PARALLEL" = true ]; then
    print_info "Running checks in PARALLEL mode"
else
    print_info "Running checks in SEQUENTIAL mode"
fi

if [ "$RUN_INTEGRATION" = true ]; then
    print_info "Integration tests: ENABLED (will run; require PDS3_HOLDINGS_DIR + SPICE)"
else
    print_info "Integration tests: SKIPPED (default; pass -i / --integration to include)"
fi

# An explicit -P with no server named would select the tier and then skip every
# test in it, reporting a pass that ran no PostgreSQL coverage at all.  The
# request is refused instead, because that is the one reading of -P nobody wants.
if [ "$RUN_POSTGRES" = true ] && [ -z "${SPINDOCTOR_TEST_POSTGRES_URL:-}" ]; then
    print_error "PostgreSQL tests were requested with -P / --postgres, but SPINDOCTOR_TEST_POSTGRES_URL is not set, so every test in that tier would skip."
    print_info "Export it, for example: export SPINDOCTOR_TEST_POSTGRES_URL=postgresql+psycopg://USER@HOST:5432/spindoctor"
    exit 1
fi

if [ "$RUN_POSTGRES" = true ]; then
    print_info "PostgreSQL tests: ENABLED (SPINDOCTOR_TEST_POSTGRES_URL is set)"
else
    print_info "PostgreSQL tests: SKIPPED (default; pass -P / --postgres to include)"
fi

# Function to run code checks (ruff, mypy, pytest)
run_code_checks() {
    local check_name="Code Checks"
    local output_file="${1:-}"
    local status_file="${2:-}"

    if [ -n "$output_file" ]; then
        exec > "$output_file" 2>&1
    fi

    print_section "$check_name"

    cd "$PROJECT_ROOT"

    if [ ! -f "$VENV/bin/activate" ]; then
        print_error "Virtual environment not found at $VENV"
        [ -n "$status_file" ] && echo "Code - Virtual environment not found" >> "$status_file"
        return 1
    fi

    source "$VENV/bin/activate"

    local failed=false
    local failed_checks=""

    # Ruff check
    print_info "Running ruff check..."
    if python -m ruff check src tests; then
        print_success "Ruff check passed"
    else
        print_error "Ruff check failed"
        failed=true
        failed_checks="${failed_checks}Code - Ruff check"$'\n'
    fi

    # Ruff format check
    print_info "Running ruff format --check..."
    if python -m ruff format --check src tests; then
        print_success "Ruff format check passed"
    else
        print_error "Ruff format check failed"
        failed=true
        failed_checks="${failed_checks}Code - Ruff format"$'\n'
    fi

    # Mypy (MYPYPATH=src so tests can resolve nav package)
    print_info "Running mypy..."
    if MYPYPATH=src python -m mypy src tests; then
        print_success "Mypy passed"
    else
        print_error "Mypy failed"
        failed=true
        failed_checks="${failed_checks}Code - Mypy"$'\n'
    fi

    # Pytest.  ``addopts = ["-m", "not integration and not postgres"]`` in
    # pyproject.toml excludes both opt-in tiers by default; -i and -P each
    # override that filter by dropping their own term from it, and opting into
    # both leaves an empty filter that accepts every test.
    # The render performance budgets assert single-core CPU time; the
    # parallel battery's workers contend for memory bandwidth and inflate
    # CPU time past the budgets, so the perf file is excluded from the
    # parallel run and executed as its own serial step afterwards.
    marker_expr=""
    pytest_marker_args=()
    if [ "$RUN_INTEGRATION" = true ] && [ "$RUN_POSTGRES" = true ]; then
        print_info "Running pytest (with integration and PostgreSQL tests)..."
    elif [ "$RUN_INTEGRATION" = true ]; then
        print_info "Running pytest (with integration tests)..."
        marker_expr="not postgres"
    elif [ "$RUN_POSTGRES" = true ]; then
        print_info "Running pytest (with PostgreSQL tests)..."
        marker_expr="not integration"
    else
        print_info "Running pytest (integration and PostgreSQL tests skipped — pass -i / -P to include)..."
    fi
    if [ "$RUN_INTEGRATION" = true ] || [ "$RUN_POSTGRES" = true ]; then
        pytest_marker_args=("-m" "$marker_expr")
    fi
    if [ "$RUN_INTEGRATION" = true ]; then
        pytest_marker_args+=("--ignore=tests/integration/test_sim_perf.py")
    fi
    # loadfile keeps every test of one file on one worker; the default scheduling
    # splits a file across processes and has crashed the PyQt6 workers.
    if python -m pytest tests -q --cov -n auto --dist=loadfile "${pytest_marker_args[@]}"; then
        print_success "Pytest passed"
    else
        print_error "Pytest failed"
        failed=true
        failed_checks="${failed_checks}Code - Pytest"$'\n'
    fi

    if [ "$RUN_INTEGRATION" = true ]; then
        print_info "Running render performance budgets (serial)..."
        if python -m pytest tests/integration/test_sim_perf.py -q -m ""; then
            print_success "Render performance budgets passed"
        else
            print_error "Render performance budgets failed"
            failed=true
            failed_checks="${failed_checks}Code - Render performance budgets"$'\n'
        fi
    fi

    deactivate 2>/dev/null || true

    if [ "$failed" = true ]; then
        if [ -n "$status_file" ]; then
            echo -n "$failed_checks" >> "$status_file"
        else
            while IFS= read -r line; do
                [ -n "$line" ] && FAILED_CHECKS+=("$line")
            done <<< "$failed_checks"
        fi
        return 1
    fi

    return 0
}

# Function to run Markdown lint only (PyMarkdown)
run_markdown_checks() {
    local check_name="Markdown Lint (PyMarkdown)"
    local output_file="${1:-}"
    local status_file="${2:-}"

    if [ -n "$output_file" ]; then
        exec > "$output_file" 2>&1
    fi

    print_section "$check_name"

    cd "$PROJECT_ROOT"

    if [ ! -f "$VENV/bin/activate" ]; then
        print_error "Virtual environment not found at $VENV"
        [ -n "$status_file" ] && echo "Markdown - Virtual environment not found" >> "$status_file"
        return 1
    fi

    source "$VENV/bin/activate"

    print_info "Running codespell (typos and British spellings)..."
    if python -m codespell_lib src tests docs util experiments scripts README.md CONTRIBUTING.md .cursor; then
        print_success "codespell passed"
    else
        print_error "codespell failed"
        FAILED_CHECKS+=("Markdown - codespell")
    fi

    print_info "Running PyMarkdown scan (docs/, .cursor/, root *.md)..."
    if python -m pymarkdown scan docs/ .cursor/ README.md CONTRIBUTING.md; then
        print_success "PyMarkdown scan passed"
    else
        print_error "PyMarkdown scan failed"
        if [ -n "$status_file" ]; then
            echo "Markdown - PyMarkdown scan" >> "$status_file"
        else
            FAILED_CHECKS+=("Markdown - PyMarkdown scan")
        fi
        deactivate 2>/dev/null || true
        return 1
    fi

    deactivate 2>/dev/null || true
    return 0
}

# Function to run documentation build
run_docs_build() {
    local check_name="Documentation Build"
    local output_file="${1:-}"
    local status_file="${2:-}"

    if [ -n "$output_file" ]; then
        exec > "$output_file" 2>&1
    fi

    print_section "$check_name"

    cd "$PROJECT_ROOT"

    if [ ! -f "$VENV/bin/activate" ]; then
        print_error "Virtual environment not found at $VENV"
        [ -n "$status_file" ] && echo "Documentation - Virtual environment not found" >> "$status_file"
        return 1
    fi

    source "$VENV/bin/activate"

    local sphinx_failed=false
    local pymarkdown_failed=false

    print_info "Building documentation (warnings treated as errors)..."
    if (cd docs && make clean && make html SPHINXOPTS="-W"); then
        print_success "Sphinx build passed (no errors or warnings)"
    else
        print_error "Sphinx build failed (errors or warnings present)"
        sphinx_failed=true
    fi

    local codespell_failed=false
    print_info "Running codespell (typos and British spellings)..."
    if python -m codespell_lib src tests docs util experiments scripts README.md CONTRIBUTING.md .cursor; then
        print_success "codespell passed"
    else
        print_error "codespell failed"
        codespell_failed=true
    fi

    if [ "$RUN_MARKDOWN" != true ]; then
        print_info "Running PyMarkdown scan (docs/, .cursor/, root *.md)..."
        if python -m pymarkdown scan docs/ .cursor/ README.md CONTRIBUTING.md; then
            print_success "PyMarkdown scan passed"
        else
            print_error "PyMarkdown scan failed"
            pymarkdown_failed=true
        fi
    fi

    deactivate 2>/dev/null || true

    if [ "$sphinx_failed" = true ] || [ "$pymarkdown_failed" = true ] \
        || [ "$codespell_failed" = true ]; then
        if [ -n "$status_file" ]; then
            [ "$sphinx_failed" = true ] && echo "Documentation - Sphinx build" >> "$status_file"
            [ "$pymarkdown_failed" = true ] && echo "Documentation - PyMarkdown scan" >> "$status_file"
            [ "$codespell_failed" = true ] && echo "Documentation - codespell" >> "$status_file"
        else
            [ "$sphinx_failed" = true ] && FAILED_CHECKS+=("Documentation - Sphinx build")
            [ "$pymarkdown_failed" = true ] && FAILED_CHECKS+=("Documentation - PyMarkdown scan")
        fi
        return 1
    fi

    return 0
}

# Run checks based on mode
if [ "$PARALLEL" = true ] && { [ "$RUN_CODE" = true ] || [ "$RUN_DOCS" = true ] || [ "$RUN_MARKDOWN" = true ]; }; then
    # Run all enabled scopes in parallel with distinct output/status files
    code_output="$TEMP_DIR/code.log"
    code_status="$TEMP_DIR/code.status"
    docs_output="$TEMP_DIR/docs.log"
    docs_status="$TEMP_DIR/docs.status"
    markdown_output="$TEMP_DIR/markdown.log"
    markdown_status="$TEMP_DIR/markdown.status"

    print_info "Running enabled checks in parallel, please wait..."

    if [ "$RUN_CODE" = true ]; then
        run_code_checks "$code_output" "$code_status" &
        code_pid=$!
    fi
    if [ "$RUN_DOCS" = true ]; then
        run_docs_build "$docs_output" "$docs_status" &
        docs_pid=$!
    fi
    if [ "$RUN_MARKDOWN" = true ]; then
        run_markdown_checks "$markdown_output" "$markdown_status" &
        markdown_pid=$!
    fi

    [ -n "$code_pid" ]    && { wait "$code_pid"    || EXIT_CODE=1; code_pid=""; }
    [ -n "$docs_pid" ]    && { wait "$docs_pid"    || EXIT_CODE=1; docs_pid=""; }
    [ -n "$markdown_pid" ] && { wait "$markdown_pid" || EXIT_CODE=1; markdown_pid=""; }
    
    for status_file in "$code_status" "$docs_status" "$markdown_status"; do
        if [ -f "$status_file" ]; then
            while IFS= read -r line; do
                [ -n "$line" ] && FAILED_CHECKS+=("$line")
            done < "$status_file"
        fi
    done

    echo ""
    [ -f "$code_output" ]    && cat "$code_output"
    [ -f "$docs_output" ]    && cat "$docs_output"
    [ -f "$markdown_output" ] && cat "$markdown_output"
else
    # Sequential (or only one scope)
    if [ "$RUN_CODE" = true ]; then
        if ! run_code_checks; then
            EXIT_CODE=1
        fi
    fi

    if [ "$RUN_DOCS" = true ]; then
        if ! run_docs_build; then
            EXIT_CODE=1
        fi
    fi

    if [ "$RUN_MARKDOWN" = true ]; then
        if ! run_markdown_checks; then
            EXIT_CODE=1
        fi
    fi
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
ELAPSED_SECONDS=$((ELAPSED % 60))

# Print summary (use EXIT_CODE as source of truth so we never show success when exiting 1)
print_header "Summary"

if [ "$EXIT_CODE" -eq 0 ]; then
    print_success "All checks passed!"
    echo -e "${GREEN}${BOLD}✓ SUCCESS${RESET} - All checks completed successfully"
else
    print_error "Some checks failed:"
    if [ ${#FAILED_CHECKS[@]} -eq 0 ]; then
        echo -e "  ${RED}✗${RESET} One or more checks failed (see output above)"
    else
        for check in "${FAILED_CHECKS[@]}"; do
            echo -e "  ${RED}✗${RESET} $check"
        done
    fi
    echo -e "${RED}${BOLD}✗ FAILURE${RESET} - One or more check(s) failed"
fi

echo ""
print_info "Total time: ${MINUTES}m ${ELAPSED_SECONDS}s"
echo ""

exit $EXIT_CODE
