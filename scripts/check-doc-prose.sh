#!/usr/bin/env bash
#
# Check the documentation prose against .cursor/rules/doc_prose_style.mdc.
#
# Two tiers:
#   ERRORS   -- high-confidence violations. These fail the script.
#   WARNINGS -- heuristics that need a human to judge. Reported, never fatal.
#
# Usage:
#   scripts/check-doc-prose.sh              # check every documentation file
#   scripts/check-doc-prose.sh FILE...      # check only the named files
#   scripts/check-doc-prose.sh --warnings   # include the heuristic tier
#
# The rule that defines each check, and the review checklist no script can run,
# is .cursor/rules/doc_prose_style.mdc.

set -uo pipefail

cd "$(dirname "$0")/.."

SHOW_WARNINGS=0
FILES=()
for arg in "$@"; do
    case "$arg" in
        --warnings|-w) SHOW_WARNINGS=1 ;;
        -h|--help) sed -n '3,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) FILES+=("$arg") ;;
    esac
done

if [[ ${#FILES[@]} -eq 0 ]]; then
    mapfile -t FILES < <(
        find docs -name '*.rst' -not -path 'docs/_build/*' -not -path 'docs/api_reference/*'
        printf '%s\n' README.md CONTRIBUTING.md
    )
fi

# The user guide and the quick start are written for users; the developer guide
# and the reports are not, so the audience-separation checks skip them. The
# reprojection API chapter documents an importable package, so module and class
# names are legitimate there.
user_facing() {
    case "$1" in
        docs/user_guide/user_guide_reprojection_api.rst) return 1 ;;
        docs/user_guide/*|docs/quick_start.rst|README.md) return 0 ;;
        *) return 1 ;;
    esac
}

ERRORS=0
WARNINGS=0

# report TIER MESSAGE MATCHES  -- MATCHES holds "path:line:text" lines.
# The matches arrive as an argument rather than on stdin: a pipeline would run
# this function in a subshell and the counters would not survive it.
report() {
    local tier="$1" msg="$2" matches="$3"
    [[ -z "${matches//[[:space:]]/}" ]] && return 0
    printf '\n%s: %s\n' "$tier" "$msg"
    printf '%s\n' "$matches" | sed 's/^/  /'
    if [[ "$tier" == ERROR ]]; then
        ERRORS=$((ERRORS + 1))
    else
        WARNINGS=$((WARNINGS + 1))
    fi
}

# Literal blocks hold captured program output, real JSON documents, and shell
# transcripts. They are evidence, not prose, so the prose checks must not read
# them: a genuine metadata document naturally contains config filenames and
# tracebacks. prose_stream emits "path:line:text" for prose lines only.
PROSE_ALL=""
PROSE_USER=""

prose_stream() {
    [[ $# -eq 0 ]] && return 0
    python3 -c '
import sys

for path in sys.argv[1:]:
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
    except OSError:
        continue
    md = path.endswith(".md")
    fence = False
    block_indent = None
    for n, line in enumerate(lines, 1):
        stripped = line.strip()
        if md:
            if stripped.startswith("```"):
                fence = not fence
                continue
            if fence:
                continue
            if line.startswith("    ") and stripped:
                continue
            print(f"{path}:{n}:{line}")
            continue
        indent = len(line) - len(line.lstrip())
        if block_indent is not None:
            if not stripped:
                continue
            if indent > block_indent:
                continue
            block_indent = None
        if not stripped:
            continue
        # A directive that takes literal content, or a paragraph ending in "::".
        low = stripped.lower()
        if low.startswith((".. code-block::", ".. code::", ".. literalinclude::",
                           ".. parsed-literal::", ".. math::")) or stripped.endswith("::"):
            block_indent = indent
            continue
        print(f"{path}:{n}:{line}")
' "$@"
    return 0
}

# hits PATTERN STREAM  -- PCRE, case sensitive, over a prose stream
hits() {
    local pat="$1" stream="$2"
    [[ -z "$stream" ]] && return 0
    printf '%s\n' "$stream" | grep -P -- "$pat" 2>/dev/null
    return 0
}

USER_FILES=()
ALL_FILES=()
for f in "${FILES[@]}"; do
    [[ -f "$f" ]] || continue
    ALL_FILES+=("$f")
    user_facing "$f" && USER_FILES+=("$f")
done

if [[ ${#ALL_FILES[@]} -eq 0 ]]; then
    echo 'check-doc-prose: no documentation files to check'
    exit 0
fi

echo "check-doc-prose: ${#ALL_FILES[@]} file(s), ${#USER_FILES[@]} user-facing"

PROSE_ALL="$(prose_stream "${ALL_FILES[@]}")"
[[ ${#USER_FILES[@]} -gt 0 ]] && PROSE_USER="$(prose_stream "${USER_FILES[@]}")"

# Packaged config filenames belong in the configuration chapter alone.
CFG_FILES=()
for f in "${ALL_FILES[@]}"; do
    [[ "$f" == docs/user_guide/user_guide_configuration.rst ]] && continue
    [[ "$f" == docs/dev_guide/* ]] && continue
    [[ "$f" == docs/*_report/* ]] && continue
    CFG_FILES+=("$f")
done
PROSE_CFG="$(prose_stream "${CFG_FILES[@]}")"

# ------------------------------------------------------------------ errors ----

# "nowhere near" is ordinary English; the rule is about doing something nowhere.
report ERROR 'you cannot do something "nowhere" -- say it is not done anywhere' \
    "$(hits '\bnowhere\b(?! near)' "$PROSE_ALL")"

# The pattern is a stem so that it matches both inflections.
COROTATION_HYPHENATED='\bco-rotat'  # codespell:ignore rotat -- deliberate stem
report ERROR 'spell corotation and corotating without a hyphen' \
    "$(hits "$COROTATION_HYPHENATED" "$PROSE_ALL")"

report ERROR 'PDS3 is not an image format -- images are VICAR, or FITS for LORRI' \
    "$(hits '(?i)PDS3[- ](format|formatted)' "$PROSE_ALL")"

report ERROR 'a shape is a physical form -- a metadata document has contents or a structure' \
    "$(hits '\bdocument shapes?\b' "$PROSE_ALL")"

report ERROR 'packaged config filename outside the configuration chapter' \
    "$(hits 'config_\d{3}_[a-z0-9_]*\.yaml' "$PROSE_CFG")"

if [[ ${#USER_FILES[@]} -gt 0 ]]; then
    report ERROR 'no "registry" in user-facing prose; a name is "named after" its class' \
        "$(hits '\bregistr(y|ies)\b|\bregisters? under\b' "$PROSE_USER")"

    report ERROR 'internal class or function name in user-facing prose' \
        "$(hits '\b(navigate_image_files|build_metadata_dict|compute_pointing|select_pointing|apply_pointing_to_obs|NavResult|NavContext|NavBase|ObsSnapshotInst|TreeRecordSource|IndexRecordSource)\b' "$PROSE_USER")"

    report ERROR 'attribute-style identifier in user-facing prose (obj.member)' \
        "$(hits '\b[a-z][a-z0-9_]*\.[a-z][a-z0-9_]{2,}(\(|\b)' "$PROSE_USER" \
            | grep -vP '\.(rst|md|py|json|yaml|yml|png|html|txt|csv|lbl|img|fits?|tab|xml|db|gz|sh|cfg|toml|in|bak)\b' \
            | grep -vP '(?i)\b(e\.g|i\.e|etc|vs|fig|no|sec|ch|eq)\.' \
            | grep -vP '^\S+:\d+:\s*(\.\.|:doc:|:ref:|\|)')"
fi

# docutils does not nest inline markup, so a literal opened inside a bold span
# renders its backticks as text. Real section titles take inline markup
# perfectly well -- it is the bold pseudo-heading that breaks. Regexes cannot
# tell a nested literal from an adjacent one, or from a "**" inside a literal
# (``rho_n**2``), so scan each line and track the two states.
nested_markup() {
    printf '%s\n' "$PROSE_ALL" | python3 -c '
import re
import sys

# Bold and literal spans wrap across lines but never across a paragraph. The
# prose stream drops blank lines, so a gap in the line numbers is a paragraph
# break: reset the state there, and at every change of file.
prev_file = None
prev_line = None
in_literal = in_bold = flagged = False

for raw in sys.stdin:
    line = raw.rstrip("\n")
    parts = line.split(":", 2)
    if len(parts) < 3:
        continue
    path, num, text = parts[0], parts[1], parts[2]
    try:
        num = int(num)
    except ValueError:
        continue
    # A bullet or enumerated item starts a new block even with no blank line
    # before it, so it resets the state too.
    bullet = re.match(r"\s*([*+-]|\#\.|\d+\.)\s", text) is not None
    if path != prev_file or prev_line is None or num != prev_line + 1 or bullet:
        in_literal = in_bold = flagged = False
    prev_file, prev_line = path, num
    if flagged:
        continue

    i = 0
    while i < len(text):
        if text.startswith("``", i):
            if not in_literal and in_bold:
                print(line)
                flagged = True
                break
            in_literal = not in_literal
            i += 2
            continue
        if text.startswith("**", i) and not in_literal:
            in_bold = not in_bold
            i += 2
            continue
        i += 1
'
    return 0
}
report ERROR 'inline markup does not nest -- a literal inside **bold** renders its backticks' \
    "$(nested_markup)"

# ---------------------------------------------------------------- warnings ----

if [[ $SHOW_WARNINGS -eq 1 ]]; then
    report WARNING 'name what is being tested rather than calling it a gate' \
        "$(hits '\bgat(e|es|ed|ing)\b' "$PROSE_ALL")"

    report WARNING 'you cannot be "with" an absent thing -- use "that has no"' \
        "$(hits '\b(images?|frames?|runs?|records?|rows?|products?)\s+with\s+no\b' "$PROSE_ALL")"

    report WARNING 'possible missing Oxford comma -- check whether this is a list of three' \
        "$(hits ',\s+[^,]{3,60}\s+(and|or)\s+\w' "$PROSE_ALL" | grep -vP ',\s+(and|or)\s')"

    report WARNING 'X-not-Y: keep only if a reader would genuinely expect Y' \
        "$(hits ',\s+not\s+\w' "$PROSE_ALL")"

    report WARNING 'a clause after "so" that restates the obvious should be deleted' \
        "$(hits '\bso (what|that means|you|it) \w+' "$PROSE_ALL")"

    if [[ ${#USER_FILES[@]} -gt 0 ]]; then
        report WARNING 'qualify "document" -- metadata document, PDS4 label, and so on' \
            "$(hits '(?<!metadata )(?<!PDS4 )(?<!PDS3 )(?<!this )(?<!the )\bdocuments?\b' "$PROSE_USER" \
                | grep -vP '(?i)\bdocument(ed|ing|ation)\b')"

        report WARNING 'say which index -- the results index or the PDS3 index' \
            "$(hits '(?<!results )(?<!PDS3 )(?<!PDS4 )\bindex\b' "$PROSE_USER" \
                | grep -vP '(?i)\bindex (file|table|column|row|of)\b')"

        report WARNING 'shape means dimensions and nothing else' \
            "$(hits '\bshapes?\b' "$PROSE_USER" \
                | grep -vP '(?i)\b(image|array|grid|mosaic|data|output)\s+shape\b')"
    fi
fi

# ----------------------------------------------------------------- summary ----

plural() { [[ "$1" -eq 1 ]] && echo category || echo categories; }

echo
if [[ $ERRORS -gt 0 ]]; then
    echo "check-doc-prose: FAILED -- $ERRORS error $(plural $ERRORS)"
    [[ $WARNINGS -gt 0 ]] && echo "check-doc-prose: $WARNINGS warning $(plural $WARNINGS)"
    exit 1
fi

if [[ $WARNINGS -gt 0 ]]; then
    echo "check-doc-prose: passed with $WARNINGS warning $(plural $WARNINGS) to review by hand"
else
    echo 'check-doc-prose: passed'
    [[ $SHOW_WARNINGS -eq 0 ]] && echo 'check-doc-prose: rerun with --warnings for the heuristic tier'
fi
exit 0
