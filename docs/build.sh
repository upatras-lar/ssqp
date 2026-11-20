#!/usr/bin/env bash
# Builds the documentation site into site/, failing on warnings, then serves it with live reload
#
#     ./build.sh                          # serves at dev_addr from zensical.toml
#     ./build.sh -a 127.0.0.1:8000        # options go to zensical serve
#
# Needs Doxygen, the ssqpy target built (the Python reference reads its stubs) and the venv from
# requirements.txt in docvenv/
set -euo pipefail
cd "$(dirname "$0")"

zensical=docvenv/bin/zensical
if [[ ! -x $zensical ]]; then
  echo "No $zensical, create it with: python3 -m venv docvenv && docvenv/bin/pip install -r requirements.txt" >&2
  exit 1
fi
if [[ ! -d ../build/bindings ]]; then
  echo "No ../build/bindings, build the ssqpy target first" >&2
  exit 1
fi

# The C++ reference pages, from Doxygen's XML
rm -rf api
doxygen Doxyfile
python3 cpp_reference.py api/xml docs/cpp-reference/ssqp

# zensical serve does not support strict mode, so warnings are caught by a build first
"$zensical" build --clean --strict
exec "$zensical" serve "$@"
