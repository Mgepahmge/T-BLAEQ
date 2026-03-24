#!/usr/bin/env bash
# buildDoc.sh
# Builds the T-BLAEQ Doxygen HTML documentation.
# Run from the project root or from the docs/ directory.
# Output is written to docs/html/.

set -euo pipefail

# Resolve the docs/ directory regardless of where the script is invoked from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check that Doxygen is available
if ! command -v doxygen &>/dev/null; then
    echo "Error: doxygen not found. Install it with:"
    echo "  sudo apt-get install doxygen"
    echo "  or: brew install doxygen"
    exit 1
fi

# Optional: warn if Graphviz is missing (dot graphs will be skipped)
if ! command -v dot &>/dev/null; then
    echo "Warning: graphviz 'dot' not found. Class and include graphs will be disabled."
    echo "  Install with: sudo apt-get install graphviz"
    # Patch Doxyfile on the fly to disable dot without modifying the committed file
    DOXY_ARGS="-d doxygen_warnings.log"
    HAVE_DOT_OVERRIDE="HAVE_DOT=NO"
    doxygen - <<< "$(cat Doxyfile; echo "$HAVE_DOT_OVERRIDE")"
else
    doxygen Doxyfile
fi

echo ""
echo "Documentation built successfully."
echo "Open: $SCRIPT_DIR/html/index.html"
