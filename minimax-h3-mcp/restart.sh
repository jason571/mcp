#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
unit_name="minimax-h3-mcp.service"
mode="restart"
case "${1:-}" in
  "") ;;
  --force) mode="force" ;;
  --check) mode="check" ;;
  -h|--help)
    printf '%s\n' \
      '用法：./restart.sh [--force | --check]' \
      '  无参数    重启 MCP 服务，等待健康检查通过；有任务时停止操作。' \
      '  --force   强制重启，正在生成和排队的任务会中断。' \
      '  --check   只检查服务和健康状态，不重启。'
    exit 0
    ;;
  *) printf '未知参数：%s（使用 --help 查看用法）\n' "$1" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  printf '%s\n' '一次只能使用一个选项。' >&2
  exit 2
fi

if (( EUID == 0 )); then
  service_user="$(stat -c '%U' "$project_dir")"
  if [[ "$service_user" != "root" && "$service_user" != "UNKNOWN" ]]; then
    printf '切换到服务用户 %s 执行。\n' "$service_user"
    exec runuser -u "$service_user" -- bash "$project_dir/restart.sh" "$@"
  fi
fi
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
export DBUS_SESSION_BUS_ADDRESS="unix:path=$XDG_RUNTIME_DIR/bus"

for dependency in systemctl curl; do
  if ! command -v "$dependency" >/dev/null 2>&1; then
    printf '缺少命令：%s\n' "$dependency" >&2
    exit 1
  fi
done
python_binary="/home/clhh/fastvideo-h3-mcp/.venv/bin/python"
if [[ ! -x "$python_binary" ]]; then
  python_binary="$(command -v python3 || true)"
fi
if [[ -z "$python_binary" ]]; then
  printf '%s\n' '缺少 Python，无法校验健康检查响应。' >&2
  exit 1
fi

config_file="$project_dir/deploy/production.env"
if [[ ! -r "$config_file" ]]; then
  printf '缺少生产配置：%s\n' "$config_file" >&2
  exit 1
fi
service_port="$(source "$config_file"; printf '%s' "${H3_MCP_PORT:-8766}")"
if [[ ! "$service_port" =~ ^[0-9]+$ ]] || (( 10#$service_port < 1 || 10#$service_port > 65535 )); then
  printf '无效的 MCP 端口：%s\n' "$service_port" >&2
  exit 1
fi
health_url="http://127.0.0.1:$service_port/health"

if ! load_state="$(systemctl --user show "$unit_name" --property=LoadState --value)"; then
  printf '无法连接用户级 systemd；请以项目所属用户登录服务器后执行 %s/restart.sh。\n' "$project_dir" >&2
  exit 1
fi
if [[ "$load_state" != "loaded" ]]; then
  printf '服务未安装或无法加载（%s）。首次安装请执行：%s/scripts/install-user-service.sh\n' "$load_state" "$project_dir" >&2
  exit 1
fi

read_health() {
  curl --noproxy '*' --silent --show-error \
    --connect-timeout 2 --max-time 2 "$health_url" 2>/dev/null |
    "$python_binary" -c '
import json, sys
try:
    data = json.load(sys.stdin)
    if not isinstance(data, dict) or data.get("status") not in {"ok", "degraded"}:
        raise ValueError("unhealthy response")
    counts = [data.get("active_jobs"), data.get("queue_size")]
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("invalid job counts")
    print(data["status"], *counts)
except (ValueError, TypeError):
    sys.exit(1)
'
}

diagnostics() {
  systemctl --user --no-pager --full status "$unit_name" || true
  if command -v journalctl >/dev/null 2>&1; then
    journalctl --user -u "$unit_name" --no-pager -n 40 || true
  fi
}

health_summary=""
service_health="unknown"
active_jobs=0
queued_jobs=0
if health_summary="$(read_health)"; then
  read -r service_health active_jobs queued_jobs <<< "$health_summary"
  printf '当前健康状态：%s；任务：运行中 %s，排队中 %s。\n' \
    "$service_health" "$active_jobs" "$queued_jobs"
fi
if [[ "$mode" == "check" ]]; then
  if systemctl --user is-active --quiet "$unit_name" && [[ "$service_health" == "ok" ]]; then
    printf 'MCP 服务运行正常：%s\n' "$health_url"
    exit 0
  fi
  printf '%s\n' 'MCP 服务或健康检查异常。' >&2
  diagnostics
  exit 1
fi
if [[ "$mode" != "force" && -n "$health_summary" ]] && (( active_jobs > 0 || queued_jobs > 0 )); then
  printf '%s\n' '还有视频任务，未执行重启。等待任务结束后重试；需要中断任务时使用 ./restart.sh --force。' >&2
  exit 2
fi
if [[ -z "$health_summary" ]]; then
  printf '%s\n' '健康检查未响应，将尝试通过 systemd 恢复服务；未完成任务可能中断。'
elif [[ "$mode" == "force" ]]; then
  printf '%s\n' '正在强制重启，未完成的视频任务将中断。'
fi

printf '正在重启 %s ...\n' "$unit_name"
if ! systemctl --user reset-failed "$unit_name" || ! systemctl --user restart "$unit_name"; then
  printf '%s\n' '重启命令失败，服务状态和最近日志如下：' >&2
  diagnostics
  exit 1
fi
deadline=$((SECONDS + 30))
while (( SECONDS < deadline )); do
  if systemctl --user is-active --quiet "$unit_name" && \
    restart_health="$(read_health)" && [[ "$restart_health" == ok\ * ]]; then
    service_pid="$(systemctl --user show "$unit_name" --property=MainPID --value)"
    printf 'MCP 服务重启成功，PID=%s\n健康检查：%s\n' "$service_pid" "$health_url"
    exit 0
  fi
  sleep 1
done
printf '%s\n' '重启后 30 秒内健康检查未通过，服务状态和最近日志如下：' >&2
diagnostics
exit 1
