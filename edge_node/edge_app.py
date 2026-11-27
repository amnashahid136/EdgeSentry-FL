"""
edge_app.py
------------
One simulated edge device (e.g. a Raspberry Pi bolted to a motor housing). This
is the code that would run on the real hardware -- it needs only the TFLite
runtime, no cloud connection, to make a decision.

Per sensor window it:
    1. runs on-device INT8 inference (the fault decision happens HERE),
    2. computes two tiny condition-monitoring features (kurtosis, crest factor),
    3. publishes ONE small JSON status message (~300 bytes) over MQTT -- the
       raw 1024-sample window never leaves the device,
    4. listens for control commands from the dashboard (inject a fault
       scenario, pause, change rate) -- the edge-orchestration channel.

The sensor is simulated by replaying this machine's HELD-OUT test windows in a
realistic story: healthy for a while, then a bearing fault develops, then
healthy again ("auto" scenario). Swap `Scenario.next_window` for real ADC code
on real hardware.

Edge-computing features (all controllable live from the dashboard):
    * model hot-swap      switch between the full INT8 model and the pruned+INT8
                          model at runtime (an OTA-style model update)
    * store-and-forward   if the uplink is cut, the node KEEPS DECIDING locally,
                          buffers its status messages (bounded queue) and
                          flushes the backlog when the link returns
    * report-on-change    send only when the status changes (+ a heartbeat) to
                          save bandwidth/energy instead of reporting every window
    * on-device alarm     debounced local alarm that works with no cloud at all
    * sensor-noise inject add white noise at a chosen SNR to watch degradation
    * model integrity     SHA-256 prefix of the running model in every message

Transport: MQTT (the real architecture; used in docker compose) or a REST
fallback to the dashboard (used by run_local_demo.sh, no broker needed).

Run:  python edge_node/edge_app.py --machine_id edge01 --broker localhost
"""
import argparse
import collections
import hashlib
import json
import os
import sys
import time
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from common import (CLASSES, MACHINES, DATA_DIR, MODEL_DIR, EDGE_MODEL_FILE, PRUNED_EDGE_MODEL_FILE,  # noqa: E402
                    add_awgn, new_mqtt_client)
from edge_node.engine import EdgeInferenceEngine  # noqa: E402

UNCERTAIN_BELOW = 0.60          # softmax confidence under this -> "UNCERTAIN" (possible novel/OOD condition)
DEFAULT_FAULT = {"edge01": 1, "edge02": 2, "edge03": 3}   # which fault each machine develops in "auto"
PHASE = {"edge01": 0, "edge02": 14, "edge03": 28}         # de-synchronise the machines in the demo
HEALTHY_N, FAULT_N = 28, 22                               # windows per phase of the auto story
HEARTBEAT_EVERY = 10          # in report-on-change mode, still send every Nth window as a heartbeat
BUFFER_MAX = 500              # bounded store-and-forward queue (oldest dropped first)


def rss_mb():
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6
    except Exception:
        return 0.0


def kurtosis(x):
    x = x.ravel().astype(np.float64)
    s = x.std() + 1e-9
    return float(np.mean(((x - x.mean()) / s) ** 4))


def crest_factor(x):
    x = x.ravel().astype(np.float64)
    return float(np.max(np.abs(x)) / (np.sqrt(np.mean(x ** 2)) + 1e-9))


class Scenario:
    """Decides which kind of window the (simulated) sensor produces next."""

    def __init__(self, machine_id, X, y, seed=0):
        self.X, self.y, self.mid = X, y, machine_id
        self.by_class = {c: np.where(y == c)[0] for c in range(len(CLASSES))}
        self.mode = "auto"
        self.step = PHASE.get(machine_id, 0)
        self.rng = np.random.default_rng(seed)

    def set_mode(self, mode):
        self.mode = mode if mode in ("auto", *CLASSES) else "auto"

    def next_class(self):
        if self.mode == "auto":
            pos = self.step % (HEALTHY_N + FAULT_N)
            self.step += 1
            return 0 if pos < HEALTHY_N else DEFAULT_FAULT.get(self.mid, 1)
        return CLASSES.index(self.mode)

    def next_window(self):
        c = self.next_class()
        i = int(self.rng.choice(self.by_class[c]))
        return i, self.X[i], int(self.y[i])


class Transport:
    """MQTT when a broker is reachable, otherwise REST to the dashboard."""

    def __init__(self, machine_id, broker, port, mode, dashboard_url, on_command):
        self.mid, self.dash, self.on_command, self.mqtt = machine_id, dashboard_url, on_command, None
        self.topic = f"edgesentry/{machine_id}/status"
        self.cmd_topic = f"edgesentry/{machine_id}/cmd"
        if mode in ("mqtt", "auto"):
            try:
                c = new_mqtt_client(f"edge-{machine_id}")
                c.on_message = lambda cl, ud, msg: self._cmd(msg.payload)
                c.on_connect = lambda cl, ud, fl, rc, *a: cl.subscribe(self.cmd_topic)
                if mode == "mqtt":      # docker: broker may start a few seconds after us -> keep retrying
                    c.connect_async(broker, port, keepalive=30)
                else:                   # auto: give up quickly and fall back to REST
                    c.connect(broker, port, keepalive=30)
                c.loop_start()
                self.mqtt = c
                print(f"[{machine_id}] transport: MQTT -> {broker}:{port}", flush=True)
            except Exception as e:
                if mode == "mqtt":
                    raise
                print(f"[{machine_id}] no MQTT broker ({e.__class__.__name__}); using REST fallback", flush=True)
        if self.mqtt is None:
            print(f"[{machine_id}] transport: REST -> {self.dash}", flush=True)

    def _cmd(self, raw):
        try:
            self.on_command(json.loads(raw.decode()))
        except Exception as e:
            print(f"[{self.mid}] bad command: {e}")

    def send(self, payload):
        if self.mqtt is not None:
            self.mqtt.publish(self.topic, json.dumps(payload), qos=0)
            return
        try:
            req = urllib.request.Request(self.dash + "/api/ingest", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=1).read()
        except Exception:
            pass

    def poll_commands(self):
        if self.mqtt is not None:       # MQTT pushes commands via on_message
            return
        try:
            with urllib.request.urlopen(f"{self.dash}/api/command/{self.mid}", timeout=1) as r:
                cmd = json.loads(r.read().decode()).get("command")
            if cmd:
                self.on_command(cmd)
        except Exception:
            pass


def sha_prefix(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def run(machine_id, model_path, broker, port, delay, transport, dashboard_url, max_windows=None):
    npz = np.load(os.path.join(DATA_DIR, machine_id, "test.npz"))
    X, y = npz["X"].astype(np.float32), npz["y"]
    paths = {"full": model_path, "pruned": os.path.join(MODEL_DIR, PRUNED_EDGE_MODEL_FILE)}
    engines = {"full": EdgeInferenceEngine(paths["full"])}
    shas = {"full": sha_prefix(paths["full"])}
    scen = Scenario(machine_id, X, y, seed=int(machine_id[-2:]))
    noise_rng = np.random.default_rng(int(machine_id[-2:]) + 7)
    state = {"paused": False, "delay": delay, "uplink": True, "model": "full", "report": "all", "snr": None}
    pending = collections.deque(maxlen=BUFFER_MAX)       # store-and-forward queue

    def on_command(cmd):
        if "scenario" in cmd:
            scen.set_mode(cmd["scenario"])
        if "paused" in cmd:
            state["paused"] = bool(cmd["paused"])
        if "delay" in cmd:
            state["delay"] = max(0.05, float(cmd["delay"]))
        if "uplink" in cmd:
            state["uplink"] = bool(cmd["uplink"])
        if "report" in cmd and cmd["report"] in ("all", "on_change"):
            state["report"] = cmd["report"]
        if "snr" in cmd:
            state["snr"] = None if cmd["snr"] in (None, "off") else float(cmd["snr"])
        if cmd.get("model") in ("full", "pruned"):
            m = cmd["model"]
            if m not in engines and os.path.exists(paths[m]):       # lazy "OTA" load of the new model
                engines[m] = EdgeInferenceEngine(paths[m])
                shas[m] = sha_prefix(paths[m])
            if m in engines:
                state["model"] = m
        print(f"[{machine_id}] command received: {cmd}", flush=True)

    tx = Transport(machine_id, broker, port, transport, dashboard_url, on_command)
    print(f"[{machine_id}] edge node online | model={os.path.basename(model_path)} sha={shas['full']} | "
          f"{len(X)} local test windows | scenario=auto", flush=True)

    seq, suppressed, delayed_sent = 0, 0, 0
    recent = collections.deque(maxlen=5)
    local_alarm, last_sent = False, None
    while max_windows is None or seq < max_windows:
        tx.poll_commands()
        if state["paused"]:
            time.sleep(0.3)
            continue
        idx, window, true_c = scen.next_window()
        if state["snr"] is not None:
            window = add_awgn(window[None], state["snr"], noise_rng)[0]
        engine = engines[state["model"]]
        pred, conf, latency_ms, probs = engine.predict(window)           # <- the decision, fully on-device
        status = "UNCERTAIN" if conf < UNCERTAIN_BELOW else ("OK" if pred == 0 else "FAULT")

        # on-device debounced alarm: works even with the uplink cut
        recent.append(status == "FAULT")
        if not local_alarm and sum(recent) >= 3:
            local_alarm = True
            print(f"[{machine_id}] *** LOCAL ALARM raised on-device ***", flush=True)
        elif local_alarm and sum(recent) <= 1:
            local_alarm = False
            print(f"[{machine_id}] local alarm cleared", flush=True)

        payload = {
            "machine_id": machine_id, "timestamp": time.time(), "seq": seq, "window_idx": idx,
            "prediction": CLASSES[pred], "class_id": pred, "status": status,
            "confidence": round(conf, 4), "probs": [round(float(p), 4) for p in probs],
            "latency_ms": round(latency_ms, 4),
            "kurtosis": round(kurtosis(window), 3), "crest_factor": round(crest_factor(window), 3),
            "rss_mb": round(rss_mb(), 1), "scenario": scen.mode, "model": os.path.basename(paths[state["model"]]),
            "model_kind": state["model"], "model_sha": shas[state["model"]],
            "local_alarm": local_alarm, "uplink": state["uplink"], "report": state["report"],
            "snr": state["snr"], "suppressed": suppressed, "backlog": len(pending),
            "true_label": CLASSES[true_c],   # demo-only (lets the dashboard show live accuracy)
        }
        print(f"[{machine_id}] #{seq:04d} win={idx:03d} pred={payload['prediction']:<17s} {status:9s} "
              f"conf={conf:.2f} lat={latency_ms:.3f}ms model={state['model']}"
              f"{'  [UPLINK DOWN, buffered]' if not state['uplink'] else ''}", flush=True)

        changed = last_sent != (status, pred)
        if not state["uplink"]:
            pending.append(payload)                         # store ...
        else:
            while pending:                                  # ... and forward the backlog first
                old = pending.popleft()
                old["delayed"] = True
                tx.send(old)
                delayed_sent += 1
            if state["report"] == "all" or changed or seq % HEARTBEAT_EVERY == 0:
                tx.send(payload)
                last_sent = (status, pred)
            else:
                suppressed += 1                             # report-on-change: nothing new to say
        seq += 1
        time.sleep(state["delay"])
    print(f"[{machine_id}] finished.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine_id", required=True, choices=MACHINES)
    ap.add_argument("--model_path", default=os.path.join(MODEL_DIR, EDGE_MODEL_FILE))
    ap.add_argument("--broker", default=os.environ.get("MQTT_BROKER", "localhost"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("MQTT_PORT", 1883)))
    ap.add_argument("--transport", choices=["auto", "mqtt", "rest"], default=os.environ.get("TRANSPORT", "auto"))
    ap.add_argument("--dashboard_url", default=os.environ.get("DASHBOARD_URL", "http://localhost:5050"))
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between simulated sensor windows")
    ap.add_argument("--max_windows", type=int, default=None)
    a = ap.parse_args()
    run(a.machine_id, a.model_path, a.broker, a.port, a.delay, a.transport, a.dashboard_url, a.max_windows)
