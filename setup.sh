# Put this project on PYTHONPATH for the current terminal, so every script runs with plain
# `python path/to/script.py` (no -m, nothing installed). Must be sourced, not executed:
#
#     source setup.sh
#
# Also activates the deep-learning uv environment if no virtualenv is active yet.

if [ -n "$ZSH_VERSION" ]; then
    _src="${(%):-%x}"
    case "$ZSH_EVAL_CONTEXT" in *:file*) _sourced=1 ;; *) _sourced=0 ;; esac
else
    _src="${BASH_SOURCE[0]}"
    [ "${BASH_SOURCE[0]}" != "$0" ] && _sourced=1 || _sourced=0
fi

if [ "$_sourced" = 0 ]; then
    echo "setup.sh must be sourced to change your terminal:  source setup.sh"
    exit 1
fi

_root="$(cd "$(dirname "$_src")" && pwd)"
case ":$PYTHONPATH:" in
    *":$_root:"*) ;;                                              # already there
    *) export PYTHONPATH="$_root${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

if [ -z "$VIRTUAL_ENV" ] && [ -f "$HOME/deep-learning/.venv/bin/activate" ]; then
    . "$HOME/deep-learning/.venv/bin/activate"
fi

echo "PYTHONPATH += $_root   (python: $(command -v python))"
unset _src _sourced _root
