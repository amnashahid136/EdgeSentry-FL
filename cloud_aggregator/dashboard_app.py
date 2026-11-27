"""
dashboard_app.py
------------------
EdgeSentry cloud website (Flask). It is both halves of the "cloud" side:

  LIVE MONITORING   subscribes to MQTT status messages from every edge node
                    (or accepts them via REST when no broker is running),
                    keeps per-machine state, raises debounced ALARMS, and can
                    send control commands back to the edge (fault injection,
                    pause, rate) -- edge orchestration.

  ANALYSIS PAGES    explainability lab (4 XAI methods + faithfulness test),
                    federated-learning results, compression benchmarks and the
                    differential-privacy study. All numbers are read from
                    results/*.json produced by the pipeline, so the website
                    can never drift from the experiments.

Pages: /  /explain  /learning  /compression  /privacy  /about
Run:   python cloud_aggregator/dashboard_app.py      -> http://localhost:5050
"""
import collections
import csv
import json
import os
import sys
import threading
import time

import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_from_directory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from common import (CLASSES, CLASS_LABELS, MACHINES, SEVERITY_BY_MACHINE, DATA_DIR, MODEL_DIR,  # noqa: E402
                    RESULTS_DIR, EDGE_MODEL_FILE, GLOBAL_MODEL_FILE, WINDOW_LEN, SAMPLE_RATE, add_awgn, new_mqtt_client)

app = Flask(__name__, template_folder="templates", static_folder="static")

LOCK = threading.Lock()
STATE = {}                                   # machine_id -> live state dict
HISTORY = collections.defaultdict(lambda: collections.deque(maxlen=120))
EVENTS = collections.deque(maxlen=200)
COMMANDS = {}                                # machine_id -> pending command (REST pollers)
COMMANDED = collections.defaultdict(dict)    # machine_id -> last commanded scenario / paused
NET = {"mqtt_connected": False, "broker": None, "bytes": 0, "msgs": 0, "t0": None}
NET_LOG = collections.deque()                # (time, bytes) of recent uplink messages -> rolling rate
OFFLINE_AFTER_S = 10.0
RAW_WINDOW_BYTES = WINDOW_LEN * 2            # 16-bit ADC samples, what streaming raw data would cost


# ------------------------------------------------------------------ live state
def record_update(payload, nbytes=0):
    mid = payload.get("machine_id")
    if mid not in MACHINES:
        return
    now = time.time()
    with LOCK:
        s = STATE.setdefault(mid, {"n": 0, "n_fault": 0, "n_correct": 0, "alarm": False,
                                   "alarm_class": None, "alarm_since": None, "lat_sum": 0.0})
        s["latest"], s["last_seen"] = payload, now
        s["n"] += 1
        s["n_fault"] += payload["status"] == "FAULT"
        s["n_correct"] += payload["prediction"] == payload.get("true_label")
        s["lat_sum"] += payload.get("latency_ms", 0.0)
        s["windows"] = payload.get("seq", s["n"]) + 1          # windows classified ON the device
        s["suppressed"] = payload.get("suppressed", 0)
        s["delayed"] = s.get("delayed", 0) + (1 if payload.get("delayed") else 0)
        h = HISTORY[mid]
        h.append({"t": payload["timestamp"], "c": payload["class_id"], "s": payload["status"],
                  "conf": payload["confidence"], "k": payload.get("kurtosis"), "lat": payload["latency_ms"]})
        # Debounced alarm: >=3 of the last 5 windows are FAULT -> raise; <=1 -> clear.
        last5 = list(h)[-5:]
        faults = [x for x in last5 if x["s"] == "FAULT"]
        if not s["alarm"] and len(faults) >= 3:
            s["alarm"], s["alarm_since"] = True, now
            s["alarm_class"] = collections.Counter(x["c"] for x in faults).most_common(1)[0][0]
        elif s["alarm"] and len(faults) <= 1:
            s["alarm"], s["alarm_class"], s["alarm_since"] = False, None, None
        elif s["alarm"] and faults:
            s["alarm_class"] = collections.Counter(x["c"] for x in faults).most_common(1)[0][0]
        EVENTS.appendleft(payload)
        NET["bytes"] += nbytes
        NET["msgs"] += 1
        NET["t0"] = NET["t0"] or now
        NET_LOG.append((now, nbytes))


def start_mqtt(broker, port):
    if not broker or broker == "__no_broker__":
        print("[dashboard] MQTT disabled (REST /api/ingest only)", flush=True)
        return
    NET["broker"] = f"{broker}:{port}"
    try:
        client = new_mqtt_client("edgesentry-dashboard")
    except ImportError:
        print("[dashboard] paho-mqtt missing; REST only", flush=True)
        return

    def on_connect(c, ud, flags, rc, *a):
        NET["mqtt_connected"] = rc == 0
        print(f"[dashboard] MQTT connected to {broker}:{port} (rc={rc})", flush=True)
        c.subscribe("edgesentry/+/status")

    def on_disconnect(c, ud, rc, *a):
        NET["mqtt_connected"] = False

    def on_message(c, ud, msg):
        try:
            record_update(json.loads(msg.payload.decode()), len(msg.payload))
        except Exception as e:
            print(f"[dashboard] bad MQTT payload: {e}", flush=True)

    client.on_connect, client.on_disconnect, client.on_message = on_connect, on_disconnect, on_message
    client.connect_async(broker, port, keepalive=30)     # auto-reconnects if the broker starts late
    client.loop_start()
    app.config["MQTT"] = client


def machine_view(mid, now):
    s = STATE.get(mid)
    cm = COMMANDED.get(mid, {})
    if not s:
        return {"machine_id": mid, "severity": SEVERITY_BY_MACHINE[mid], "seen": False}
    age = now - s["last_seen"]
    online = age < OFFLINE_AFTER_S
    return {
        "machine_id": mid, "severity": SEVERITY_BY_MACHINE[mid], "seen": True, "online": online,
        "paused": bool(cm.get("paused")), "age_s": round(age, 1), "latest": s["latest"],
        "alarm": s["alarm"] and online, "alarm_class": s["alarm_class"], "alarm_since": s["alarm_since"],
        "n": s["n"], "n_fault": s["n_fault"], "windows": s.get("windows", s["n"]),
        "suppressed": s.get("suppressed", 0), "delayed": s.get("delayed", 0), "commanded": cm,
        "live_acc": s["n_correct"] / s["n"] if s["n"] else None,
        "mean_latency_ms": s["lat_sum"] / s["n"] if s["n"] else None,
        "history": list(HISTORY[mid])[-60:],
    }


@app.route("/api/ingest", methods=["POST"])
def ingest():
    """REST fallback so everything works without an MQTT broker."""
    record_update(request.get_json(force=True), len(request.get_data()))
    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    now = time.time()
    with LOCK:
        machines = {m: machine_view(m, now) for m in MACHINES}
        events = list(EVENTS)[:30]
        seen = [v for v in machines.values() if v["seen"]]
        total = sum(v["n"] for v in seen)
        windows = sum(v["windows"] for v in seen)
        avg_msg = NET["bytes"] / NET["msgs"] if NET["msgs"] else None
        while NET_LOG and NET_LOG[0][0] < now - 20:
            NET_LOG.popleft()
        span = max(min(20.0, now - NET["t0"]), 1.0) if NET["t0"] else 1.0
        uplink_bps = sum(b for _, b in NET_LOG) / span          # actual bytes/s sent by the fleet (last 20 s)
        raw_bps = sum(1 for v in seen if v["online"]) * SAMPLE_RATE * 2   # 16-bit samples, if the raw sensor were streamed
        kpi = {
            "online": sum(1 for v in seen if v["online"]), "total": len(MACHINES),
            "alarms": sum(1 for v in seen if v["alarm"]), "windows": windows, "messages": total,
            "uplink_Bps": uplink_bps, "raw_Bps": raw_bps, "window_ms": WINDOW_LEN / 48000.0 * 1000,
            "suppressed": sum(v["suppressed"] for v in seen),
            "mean_latency_ms": (sum(v["mean_latency_ms"] * v["n"] for v in seen) / total) if total else None,
            "avg_msg_bytes": avg_msg, "raw_window_bytes": RAW_WINDOW_BYTES,
            "live_acc": (sum(v["live_acc"] * v["n"] for v in seen) / total) if total else None,
        }
        return jsonify({"server_time": now, "machines": machines, "events": events, "kpi": kpi,
                        "net": {"mqtt_connected": NET["mqtt_connected"], "broker": NET["broker"]}})


@app.route("/api/command", methods=["POST"])
def api_command():
    """Dashboard -> edge control channel (MQTT topic edgesentry/<id>/cmd, or REST poll)."""
    body = request.get_json(force=True)
    mid = body.get("machine_id")
    if mid not in MACHINES:
        abort(400, "unknown machine")
    cmd = {k: body[k] for k in ("scenario", "paused", "delay", "uplink", "model", "report", "snr") if k in body}
    if "scenario" in cmd and cmd["scenario"] not in ("auto", *CLASSES):
        abort(400, "bad scenario")
    if "model" in cmd and cmd["model"] not in ("full", "pruned"):
        abort(400, "bad model")
    if "report" in cmd and cmd["report"] not in ("all", "on_change"):
        abort(400, "bad report mode")
    if "snr" in cmd and cmd["snr"] is not None and not isinstance(cmd["snr"], (int, float)):
        abort(400, "bad snr")
    COMMANDED[mid].update(cmd)
    COMMANDS[mid] = {**COMMANDS.get(mid, {}), **cmd}
    client = app.config.get("MQTT")
    if client is not None and NET["mqtt_connected"]:
        client.publish(f"edgesentry/{mid}/cmd", json.dumps(cmd), qos=0)
    return jsonify({"ok": True, "sent": cmd})


@app.route("/api/export.csv")
def api_export():
    """Download the recent event log as CSV (for reports / offline analysis)."""
    import io
    buf = io.StringIO()
    cols = ["timestamp", "machine_id", "seq", "window_idx", "prediction", "true_label", "status", "confidence",
            "latency_ms", "kurtosis", "crest_factor", "model_kind", "model_sha", "local_alarm", "snr", "delayed"]
    w = csv.writer(buf)
    w.writerow(cols)
    with LOCK:
        for e in reversed(list(EVENTS)):
            w.writerow([e.get(c, "") for c in cols])
    return app.response_class(buf.getvalue(), mimetype="text/csv",
                              headers={"Content-Disposition": "attachment; filename=edgesentry_events.csv"})


@app.route("/api/command/<mid>")
def api_command_poll(mid):
    return jsonify({"command": COMMANDS.pop(mid, None)})


# ---------------------------------------------------------- models (lazy)
_MODELS = {"lock": threading.Lock()}


def _need_tf():
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        abort(503, "TensorFlow is not installed in this environment; explainability endpoints are unavailable")


def get_keras():
    with _MODELS["lock"]:
        if "keras" not in _MODELS:
            _need_tf()
            import tensorflow as tf
            _MODELS["keras"] = tf.keras.models.load_model(os.path.join(MODEL_DIR, GLOBAL_MODEL_FILE))
        return _MODELS["keras"]


def get_engine():
    with _MODELS["lock"]:
        if "engine" not in _MODELS:
            from edge_node.engine import EdgeInferenceEngine
            _MODELS["engine"] = EdgeInferenceEngine(os.path.join(MODEL_DIR, EDGE_MODEL_FILE))
        return _MODELS["engine"]


_TEST = {}


def get_test(machine):
    if machine not in MACHINES:
        abort(400, "unknown machine")
    if machine not in _TEST:
        npz = np.load(os.path.join(DATA_DIR, machine, "test.npz"))
        _TEST[machine] = (npz["X"].astype(np.float32), npz["y"])
    return _TEST[machine]


@app.route("/api/window_index")
def api_window_index():
    """Pick a (random) held-out window of a given class from a machine's test shard."""
    machine, cls = request.args.get("machine", "edge01"), request.args.get("cls", "any")
    X, y = get_test(machine)
    pool = np.arange(len(y)) if cls == "any" else np.where(y == CLASSES.index(cls))[0]
    return jsonify({"idx": int(np.random.choice(pool)), "n_windows": int(len(y))})


XAI_LOCK = threading.Lock()


@app.route("/api/explain", methods=["POST"])
def api_explain():
    """Explain one window: edge INT8 decision, FP32 decision, attribution map,
    and a deletion-test faithfulness curve. Optional sensor noise (snr dB)."""
    b = request.get_json(force=True)
    machine, idx = b.get("machine", "edge01"), int(b.get("idx", 0))
    method, snr = b.get("method", "ig"), b.get("snr")
    from model.explainability import METHODS, explain as xai_explain
    if method not in METHODS:
        abort(400, "unknown method")
    X, y = get_test(machine)
    if not 0 <= idx < len(X):
        abort(400, "window index out of range")
    x = X[idx]
    if snr is not None:
        x = add_awgn(x[None], float(snr), np.random.default_rng(idx))[0]
    with XAI_LOCK:
        engine = get_engine()
        p_edge, ms_edge = engine.probs(x)
        keras = get_keras()
        ex = xai_explain(keras, x, method=method)
    return jsonify({
        "machine": machine, "idx": idx, "snr": snr, "label": int(y[idx]),
        "window": [round(float(v), 4) for v in x[:, 0]],
        "edge": {"pred": int(np.argmax(p_edge)), "probs": [float(v) for v in p_edge], "latency_ms": ms_edge},
        "fp32": {"pred": ex["pred"], "probs": ex["probs"]},
        "explanation": {k: ex[k] for k in ("method", "attribution", "ms", "deletion", "seg")},
    })


# ------------------------------------------------------------------ results
JSON_RESULTS = {"fl": "exp_fl_comparison.json", "compression": "exp_compression.json", "dp": "exp_dp.json",
                "generalization": "exp_generalization.json", "xai": "xai_report.json",
                "benchmark": "benchmark_report.json", "official": "official_metrics.json"}


@app.route("/api/results/<key>")
def api_results(key):
    if key not in JSON_RESULTS:
        abort(404)
    path = os.path.join(RESULTS_DIR, JSON_RESULTS[key])
    if not os.path.exists(path):
        return jsonify({"available": False}), 200
    with open(path) as f:
        data = json.load(f)
    data["available"] = True
    return jsonify(data)


@app.route("/api/meta")
def api_meta():
    from model.architecture import BASE_CONFIG, count_macs, peak_activation_elems
    return jsonify({"classes": CLASSES, "labels": CLASS_LABELS, "machines": MACHINES,
                    "severity": SEVERITY_BY_MACHINE, "window_len": WINDOW_LEN,
                    "macs": count_macs(), "cfg": BASE_CONFIG,
                    "peak_act_elems": peak_activation_elems()}) \
        if _can_import_arch() else jsonify({"classes": CLASSES, "labels": CLASS_LABELS, "machines": MACHINES,
                                            "severity": SEVERITY_BY_MACHINE, "window_len": WINDOW_LEN})


def _can_import_arch():
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False


@app.route("/results/<path:fname>")
def results_file(fname):
    if not fname.lower().endswith((".png", ".md", ".csv")):
        abort(404)
    return send_from_directory(RESULTS_DIR, fname)


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True})


# -------------------------------------------------------------------- pages
@app.context_processor
def inject_nav():
    return {"NAV": [("/", "Live Fleet"), ("/explain", "Explainability"), ("/learning", "Federated Learning"),
                    ("/compression", "Edge Compression"), ("/privacy", "Privacy"), ("/about", "About")]}


@app.route("/")
def page_fleet():
    return render_template("fleet.html", page="/")


@app.route("/explain")
def page_explain():
    return render_template("explain.html", page="/explain")


@app.route("/learning")
def page_learning():
    return render_template("learning.html", page="/learning")


@app.route("/compression")
def page_compression():
    return render_template("compression.html", page="/compression")


@app.route("/privacy")
def page_privacy():
    return render_template("privacy.html", page="/privacy")


@app.route("/about")
def page_about():
    return render_template("about.html", page="/about")


if __name__ == "__main__":
    start_mqtt(os.environ.get("MQTT_BROKER", "localhost"), int(os.environ.get("MQTT_PORT", 1883)))
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5050)), debug=False, threaded=True)
