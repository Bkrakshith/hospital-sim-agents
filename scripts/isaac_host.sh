#!/usr/bin/env bash
# The WardZero Isaac host, run with Isaac Sim's own Python (3.12). `hhumans run --sim isaac` starts it.
#
#   scripts/isaac_host.sh --run-dir DIR --scenario FILE [--headed|--headless] [--record] [--full-site] ...
#
# - Isaac Sim at $ISAAC_SIM_ROOT (default ~/isaacsim).
# - CPUs 0-7,10-31 (taskset), RAM capped with a systemd user scope (MemoryMax, no swap), like hh.sh.
# - conda, PYTHONPATH, the sourced system ROS and every API key are stripped; Isaac's internal ROS 2
#   Humble libraries (isaacsim.ros2.core) with CycloneDDS are used instead, the same RMW as the system.
# - ROS_DOMAIN_ID is not passed to Kit (the bridge's ROS2Context would let it override its domain);
#   it becomes --private-domain unless that is given.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTS_ROOT="$(dirname "$HERE")"
ISAAC="${ISAAC_SIM_ROOT:-$HOME/isaacsim}"
CPUS="${WARDZERO_ISAAC_CPUS:-0-7,10-31}"
MEM="${WARDZERO_ISAAC_MEMORY_MAX:-40G}"
if [[ ! -x "$ISAAC/python.sh" ]]; then
  echo "Isaac Sim not found at $ISAAC (set ISAAC_SIM_ROOT)" >&2
  exit 2
fi
PRIVATE=()
if [[ " $* " != *" --private-domain "* ]]; then
  PRIVATE=(--private-domain "${ROS_DOMAIN_ID:-0}")
fi
CAP=()
if command -v systemd-run >/dev/null && systemd-run --user --scope -q true 2>/dev/null; then
  CAP=(systemd-run --user --scope -q -p "MemoryMax=$MEM" -p MemorySwapMax=0)
fi
PIN=()
command -v taskset >/dev/null && PIN=(taskset -c "$CPUS")
export OMNI_KIT_ACCEPT_EULA=YES
exec "${CAP[@]}" "${PIN[@]}" env \
  -u LD_LIBRARY_PATH -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u PYTHONPATH -u PYTHONHOME \
  -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH -u ROS_DOMAIN_ID -u VIRTUAL_ENV \
  -u ANTHROPIC_API_KEY -u OPENAI_API_KEY -u ANTHROPIC_AUTH_TOKEN -u WARDZERO_KEY_FILE \
  ROS_DISTRO=humble RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  LD_LIBRARY_PATH="$ISAAC/exts/isaacsim.ros2.core/humble/lib" \
  PYTHONPATH="$AGENTS_ROOT" PYTHONUNBUFFERED=1 \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  "$ISAAC/python.sh" -m wardzero_agents.isaac_host.app "${PRIVATE[@]}" "$@"
