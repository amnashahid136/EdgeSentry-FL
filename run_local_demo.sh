#!/usr/bin/env bash
# Local, no-Docker demo: dashboard + 3 edge-node processes talking over the REST
# fallback (no MQTT broker needed). For the real MQTT architecture use
# `docker compose up --build`, or start your own broker and run:
#     MQTT_BROKER=localhost python3 cloud_aggregator/dashboard_app.py
#     python3 edge_node/edge_app.py --machine_id edge01 --transport mqtt --broker localhost
set -e
cd "$(dirname "$0")"
PORT="${PORT:-5050}"
URL="http://localhost:${PORT}"

cleanup() { echo; echo "Stopping dashboard and edge nodes..."; jobs -p | xargs -r kill 2>/dev/null; }
trap cleanup EXIT INT TERM

echo "Starting dashboard on ${URL} ..."
MQTT_BROKER=__no_broker__ PORT="$PORT" python3 cloud_aggregator/dashboard_app.py &
DASH_PID=$!

echo -n "Waiting for dashboard"
for i in $(seq 1 30); do
  if ! kill -0 "$DASH_PID" 2>/dev/null; then echo; echo "ERROR: dashboard exited (port ${PORT} in use? try PORT=5051 ./run_local_demo.sh)"; exit 1; fi
  curl -sf "${URL}/healthz" >/dev/null 2>&1 && break
  echo -n "."; sleep 1
done
echo

echo "Starting 3 edge nodes (REST transport) ..."
for spec in "edge01 1.0" "edge02 1.3" "edge03 0.8"; do
  set -- $spec
  python3 edge_node/edge_app.py --machine_id "$1" --transport rest --dashboard_url "$URL" --delay "$2" &
done

echo; echo "Open ${URL}   (Ctrl+C stops everything)"
wait "$DASH_PID"
