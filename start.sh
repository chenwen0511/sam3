#!/usr/bin/env bash
# Manage the SAM3 inference HTTP server (run_server.py).
#
# Usage:
#   ./start.sh start|stop|status|restart [extra args...]
#
# Examples:
#   ./start.sh start
#   ./start.sh start --port 18002
#   ./start.sh restart --host 0.0.0.0 --port 18002
#   ./start.sh status
#   ./start.sh stop

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${SAM3_CONDA_ENV:-sam3}"
CONDA_BASE="${CONDA_BASE:-/home/ubuntu/miniconda3}"
PID_FILE="${SAM3_PID_FILE:-${REPO_ROOT}/.run/run_server.pid}"
LOG_FILE="${SAM3_LOG_FILE:-${REPO_ROOT}/.run/run_server.log}"
HOST="${SAM3_HOST:-0.0.0.0}"
PORT="${SAM3_PORT:-18002}"

mkdir -p "$(dirname "${PID_FILE}")"

_activate_env() {
  if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
  elif command -v conda >/dev/null 2>&1; then
    # shellcheck source=/dev/null
    eval "$(conda shell.bash hook)"
  else
    echo "conda not found (looked for ${CONDA_BASE}/etc/profile.d/conda.sh)" >&2
    exit 1
  fi
  conda activate "${CONDA_ENV}"
}

_is_running() {
  local pid
  if [[ ! -f "${PID_FILE}" ]]; then
    return 1
  fi
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -z "${pid}" ]]; then
    return 1
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi
  return 1
}

_pid() {
  cat "${PID_FILE}" 2>/dev/null || true
}

cmd_start() {
  if _is_running; then
    echo "already running (pid $(_pid))"
    exit 0
  fi

  if [[ ! -f "${REPO_ROOT}/run_server.py" ]]; then
    echo "run_server.py not found in ${REPO_ROOT}" >&2
    exit 1
  fi

  rm -f "${PID_FILE}"
  _activate_env

  # Unbuffered stdout/stderr so nohup log shows lines immediately.
  export PYTHONUNBUFFERED=1
  nohup python -u run_server.py --host "${HOST}" --port "${PORT}" "$@" \
    >>"${LOG_FILE}" 2>&1 &
  local pid=$!
  echo "${pid}" >"${PID_FILE}"

  sleep 1
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "failed to start; see ${LOG_FILE}" >&2
    rm -f "${PID_FILE}"
    exit 1
  fi

  echo "started (pid ${pid})"
  echo "  listen: http://${HOST}:${PORT}"
  echo "  log:    ${LOG_FILE}"
}

cmd_stop() {
  if ! _is_running; then
    echo "not running"
    rm -f "${PID_FILE}"
    return 0
  fi

  local pid
  pid="$(_pid)"
  echo "stopping pid ${pid} ..."
  kill "${pid}" 2>/dev/null || true

  local i
  for i in $(seq 1 20); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      rm -f "${PID_FILE}"
      echo "stopped"
      return 0
    fi
    sleep 0.5
  done

  echo "force kill pid ${pid}"
  kill -9 "${pid}" 2>/dev/null || true
  rm -f "${PID_FILE}"
  echo "stopped"
}

cmd_status() {
  if _is_running; then
    local pid
    pid="$(_pid)"
    echo "running (pid ${pid})"
    echo "  listen: http://${HOST}:${PORT}"
    echo "  log:    ${LOG_FILE}"
    if command -v ps >/dev/null 2>&1; then
      ps -p "${pid}" -o pid,etime,cmd --no-headers 2>/dev/null || true
    fi
    exit 0
  fi
  echo "not running"
  exit 1
}

cmd_restart() {
  cmd_stop || true
  cmd_start "$@"
}

usage() {
  cat <<EOF
Usage: $0 {start|stop|status|restart} [extra run_server.py args...]

Env overrides:
  SAM3_CONDA_ENV   conda env name (default: sam3)
  SAM3_HOST        listen host (default: 0.0.0.0)
  SAM3_PORT        listen port (default: 18002)
  SAM3_PID_FILE    pid file path
  SAM3_LOG_FILE    log file path
  CONDA_BASE       conda install prefix (default: /home/ubuntu/miniconda3)
EOF
}

main() {
  local action="${1:-}"
  if [[ -z "${action}" ]]; then
    usage
    exit 1
  fi
  shift || true

  case "${action}" in
    start)   cmd_start "$@" ;;
    stop)    cmd_stop ;;
    status)  cmd_status ;;
    restart) cmd_restart "$@" ;;
    -h|--help|help) usage ;;
    *)
      echo "unknown command: ${action}" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
