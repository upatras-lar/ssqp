# Detect if the script is being sourced correctly
if [ "$0" = "setup_env.sh" ] || [ "$0" = "./examples/setup_env.sh" ] || [ "$0" = "examples/setup_env.sh" ]; then
    echo "❌ Error: Do not run this script directly with 'sh', 'bash', or './'."
    echo "Use instead: source setup_env.sh"
    return 1 2>/dev/null || exit 1
fi

# Get absolute path of script directory ([root]/examples)
if [ -n "$BASH_SOURCE" ]; then
    SCRIPT_DIR=$(dirname "$(readlink -f "$BASH_SOURCE")")
elif [ -n "$ZSH_VERSION" ]; then
    SCRIPT_DIR=$(dirname "$(readlink -f "${(%):-%N}")")
else
    # Fallback for plain sh/dash (relies on running from project root)
    SCRIPT_DIR=$(pwd)/examples
fi

# Get absolute path of project root ([root]/)
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# Build PYTHONPATH
export PYTHONPATH="$ROOT_DIR/build/bindings:$SCRIPT_DIR:$PYTHONPATH"

echo "✅ Environment configured!"
echo "------------------------------------------------"
echo "PYTHONPATH now includes:"
echo "  - $ROOT_DIR/build/bindings"
echo "  - $SCRIPT_DIR"
echo "------------------------------------------------"
