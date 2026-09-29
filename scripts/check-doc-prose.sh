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

# hits PATTERN FILE...  -- PCRE, case sensitive, prints "path:line:text"
hits() {
    local pat="$1"
    shift
    [[ $# -eq 0 ]] && return 0
    grep -nPH -- "$pat" "$@" 2>/dev/null
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

# Packaged config filenames belong in the configuration chapter alone.
CFG_FILES=()
for f in "${ALL_FILES[@]}"; do
    [[ "$f" == docs/user_guide/user_guide_configuration.rst ]] && continue
    [[ "$f" == docs/dev_guide/* ]] && continue
    CFG_FILES+=("$f")
done

# ------------------------------------------------------------------ errors ----

report ERROR 'you cannot do something "nowhere" -- say it is not done anywhere' \
    "$(hits '\bnowhere\b' "${ALL_FILES[@]}")"

# The pattern is a stem so that it matches both inflections.
COROTATION_HYPHENATED='\bco-rotat'  # codespell:ignore rotat -- deliberate stem
report ERROR 'spell corotation and corotating without a hyphen' \
    "$(hits "$COROTATION_HYPHENATED" "${ALL_FILES[@]}")"

report ERROR 'PDS3 is not an image format -- images are VICAR, or FITS for LORRI' \
    "$(hits '(?i)PDS3[- ](format|formatted)' "${ALL_FILES[@]}")"

report ERROR 'a shape is a physical form -- a metadata document has contents or a structure' \
    "$(hits '\bdocument shapes?\b' "${ALL_FILES[@]}")"

report ERROR 'packaged config filename outside the configuration chapter' \
    "$(hits 'config_\d{3}_[a-z0-9_]*\.yaml' "${CFG_FILES[@]}")"

if [[ ${#USER_FILES[@]} -gt 0 ]]; then
    report ERROR 'no "registry" in user-facing prose; a name is "named after" its class' \
        "$(hits '\bregistr(y|ies)\b|\bregisters? under\b' "${USER_FILES[@]}")"

    report ERROR 'internal class or function name in user-facing prose' \
        "$(hits '\b(navigate_image_files|build_metadata_dict|compute_pointing|select_pointing|apply_pointing_to_obs|NavResult|NavContext|NavBase|ObsSnapshotInst|TreeRecordSource|IndexRecordSource)\b' "${USER_FILES[@]}")"

    report ERROR 'attribute-style identifier in user-facing prose (obj.member)' \
        "$(hits '\b[a-z][a-z0-9_]*\.[a-z][a-z0-9_]{2,}(\(|\b)' "${USER_FILES[@]}" \
            | grep -vP '\.(rst|md|py|json|yaml|yml|png|html|txt|csv|lbl|img|fits?|tab|xml|db|gz|sh|cfg|toml|in|bak)\b' \
            | grep -vP '(?i)\b(e\.g|i\.e|etc|vs|fig|no|sec|ch|eq)\.' \
            | grep -vP '^\S+:\d+:\s*(\.\.|:doc:|:ref:|\|)')"
fi

# Double backticks inside a heading do not render. A heading is a line whose
# successor is an underline of punctuation at least as long as it.
heading_backticks() {
    local f
    for f in "${ALL_FILES[@]}"; do
        [[ "$f" == *.rst ]] || continue
        awk -v F="$f" '
            NR > 1 && $0 ~ /^(=|-|~|\^|"|\+|#|\*)+$/ && length($0) >= length(prev) &&
                prev ~ /``/ { printf "%s:%d:%s\n", F, NR - 1, prev }
            { prev = $0 }
        ' "$f"
    done
}
report ERROR 'double backticks do not render inside a heading' "$(heading_backticks)"

# ---------------------------------------------------------------- warnings ----

if [[ $SHOW_WARNINGS -eq 1 ]]; then
    report WARNING 'name what is being tested rather than calling it a gate' \
        "$(hits '\bgat(e|es|ed|ing)\b' "${ALL_FILES[@]}")"

    report WARNING 'you cannot be "with" an absent thing -- use "that has no"' \
        "$(hits '\b(images?|frames?|runs?|records?|rows?|products?)\s+with\s+no\b' "${ALL_FILES[@]}")"

    report WARNING 'possible missing Oxford comma -- check whether this is a list of three' \
        "$(hits ',\s+[^,]{3,60}\s+(and|or)\s+\w' "${ALL_FILES[@]}" | grep -vP ',\s+(and|or)\s')"

    report WARNING 'X-not-Y: keep only if a reader would genuinely expect Y' \
        "$(hits ',\s+not\s+\w' "${ALL_FILES[@]}")"

    report WARNING 'a clause after "so" that restates the obvious should be deleted' \
        "$(hits '\bso (what|that means|you|it) \w+' "${ALL_FILES[@]}")"

    if [[ ${#USER_FILES[@]} -gt 0 ]]; then
        report WARNING 'qualify "document" -- metadata document, PDS4 label, and so on' \
            "$(hits '(?<!metadata )(?<!PDS4 )(?<!PDS3 )(?<!this )(?<!the )\bdocuments?\b' "${USER_FILES[@]}" \
                | grep -vP '(?i)\bdocument(ed|ing|ation)\b')"

        report WARNING 'say which index -- the results index or the PDS3 index' \
            "$(hits '(?<!results )(?<!PDS3 )(?<!PDS4 )\bindex\b' "${USER_FILES[@]}" \
                | grep -vP '(?i)\bindex (file|table|column|row|of)\b')"

        report WARNING 'shape means dimensions and nothing else' \
            "$(hits '\bshapes?\b' "${USER_FILES[@]}" \
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
