from __future__ import annotations

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from training.hybrid_model import HybridIDS
from training.nsl_kdd import FEATURE_COLUMNS

try:
    from scapy.all import ICMP, IP, TCP, UDP, sniff
except ImportError:  # pragma: no cover - depends on local packet-capture setup.
    ICMP = IP = TCP = UDP = sniff = None


SERVICE_PORTS = {
    20: "ftp_data",
    21: "ftp",
    22: "ssh",
    23: "telnet",
    25: "smtp",
    53: "domain_u",
    67: "eco_i",
    68: "eco_i",
    80: "http",
    110: "pop_3",
    119: "nntp",
    123: "ntp_u",
    143: "imap4",
    443: "http_443",
}


@dataclass
class LiveMonitor:
    model: HybridIDS | None = None
    events: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=200))
    stats: Counter = field(default_factory=Counter)
    running: bool = False
    mode: str = "live"
    interface: str | None = None
    error: str | None = None
    _thread: threading.Thread | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set_model(self, model: HybridIDS) -> None:
        self.model = model

    def start(self, interface: str | None = None, mode: str = "live") -> tuple[bool, str]:
        if mode not in {"live", "demo"}:
            return False, "Mode must be live or demo."
        if mode == "live" and sniff is None:
            return False, "Scapy is not installed. Run pip install -r requirements.txt."
        if self.model is None:
            return False, "Model is not loaded. Train the IDS first."
        if self.running:
            return True, "Live monitoring is already running."

        self.interface = interface or None
        self.error = None
        self.running = True
        self.mode = mode
        target = self._demo_loop if mode == "demo" else self._sniff_loop
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()
        time.sleep(0.2)
        if self.error:
            return False, self.error
        return True, "Demo monitoring started." if mode == "demo" else "Live monitoring started."

    def stop(self) -> str:
        self.running = False
        return "Live monitoring stopped."

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self.running,
                "mode": self.mode,
                "interface": self.interface or "default",
                "error": self.error,
                "total": int(self.stats["total"]),
                "normal": int(self.stats["normal"]),
                "attack": int(self.stats["attack"]),
            }

    def recent_events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.events)

    def _sniff_loop(self) -> None:
        try:
            sniff(
                iface=self.interface,
                prn=self._handle_packet,
                store=False,
                stop_filter=lambda _: not self.running,
            )
        except Exception as exc:  # pragma: no cover - hardware/driver specific.
            with self._lock:
                self.error = str(exc)
                self.running = False

    def _handle_packet(self, packet) -> None:
        if self.model is None or IP is None or not packet.haslayer(IP):
            return

        row, packet_info = packet_to_nsl_kdd_row(packet)
        frame = pd.DataFrame([row], columns=FEATURE_COLUMNS + ["label", "difficulty"])
        prediction = self.model.predict_frame(frame).iloc[0]
        event = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "source": packet_info["source"],
            "destination": packet_info["destination"],
            "protocol": packet_info["protocol"],
            "service": row["service"],
            "rf_attack_probability": float(prediction["rf_attack_probability"]),
            "vae_reconstruction_error": float(prediction["vae_reconstruction_error"]),
            "hybrid_score": float(prediction["hybrid_score"]),
            "prediction": str(prediction["prediction"]),
        }

        with self._lock:
            self.stats["total"] += 1
            self.stats[event["prediction"]] += 1
            self.events.appendleft(event)

    def _demo_loop(self) -> None:
        samples = [
            ("10.0.0.12:51544", "192.168.1.10:80", "tcp", "http", 180, "normal"),
            ("10.0.0.22:44321", "192.168.1.15:22", "tcp", "ssh", 420, "unknown"),
            ("203.0.113.8:49152", "192.168.1.20:23", "tcp", "telnet", 9000, "unknown"),
            ("198.51.100.44", "192.168.1.5", "icmp", "eco_i", 64, "unknown"),
        ]
        index = 0
        while self.running:
            source, destination, protocol, service, src_bytes, label = samples[index % len(samples)]
            index += 1
            row = demo_row(protocol=protocol, service=service, src_bytes=src_bytes, label=label)
            frame = pd.DataFrame([row], columns=FEATURE_COLUMNS + ["label", "difficulty"])
            prediction = self.model.predict_frame(frame).iloc[0] if self.model is not None else None
            if prediction is not None:
                event = {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "source": source,
                    "destination": destination,
                    "protocol": protocol,
                    "service": service,
                    "rf_attack_probability": float(prediction["rf_attack_probability"]),
                    "vae_reconstruction_error": float(prediction["vae_reconstruction_error"]),
                    "hybrid_score": float(prediction["hybrid_score"]),
                    "prediction": str(prediction["prediction"]),
                }
                with self._lock:
                    self.stats["total"] += 1
                    self.stats[event["prediction"]] += 1
                    self.events.appendleft(event)
            time.sleep(1.5)


def packet_to_nsl_kdd_row(packet) -> tuple[dict[str, Any], dict[str, str]]:
    ip_layer = packet[IP]
    protocol = protocol_name(packet)
    src_port = int(getattr(packet[TCP], "sport", 0)) if TCP is not None and packet.haslayer(TCP) else 0
    dst_port = int(getattr(packet[TCP], "dport", 0)) if TCP is not None and packet.haslayer(TCP) else 0
    if UDP is not None and packet.haslayer(UDP):
        src_port = int(packet[UDP].sport)
        dst_port = int(packet[UDP].dport)

    row = {column: 0 for column in FEATURE_COLUMNS}
    row.update(
        {
            "duration": 0,
            "protocol_type": protocol,
            "service": service_name(src_port, dst_port, protocol),
            "flag": tcp_flag(packet),
            "src_bytes": int(len(bytes(packet.payload))),
            "dst_bytes": 0,
            "count": 1,
            "srv_count": 1,
            "same_srv_rate": 1.0,
            "dst_host_count": 1,
            "dst_host_srv_count": 1,
            "dst_host_same_srv_rate": 1.0,
            "label": "unknown",
            "difficulty": -1,
        }
    )
    packet_info = {
        "source": f"{ip_layer.src}:{src_port}" if src_port else ip_layer.src,
        "destination": f"{ip_layer.dst}:{dst_port}" if dst_port else ip_layer.dst,
        "protocol": protocol,
    }
    return row, packet_info


def demo_row(protocol: str, service: str, src_bytes: int, label: str) -> dict[str, Any]:
    row = {column: 0 for column in FEATURE_COLUMNS}
    row.update(
        {
            "duration": 0,
            "protocol_type": protocol,
            "service": service,
            "flag": "SF",
            "src_bytes": src_bytes,
            "dst_bytes": 120 if label == "normal" else 0,
            "count": 2 if label == "normal" else 90,
            "srv_count": 2 if label == "normal" else 5,
            "serror_rate": 0.0 if label == "normal" else 0.8,
            "srv_serror_rate": 0.0 if label == "normal" else 0.8,
            "same_srv_rate": 1.0 if label == "normal" else 0.1,
            "diff_srv_rate": 0.0 if label == "normal" else 0.7,
            "dst_host_count": 20 if label == "normal" else 255,
            "dst_host_srv_count": 20 if label == "normal" else 8,
            "dst_host_same_srv_rate": 1.0 if label == "normal" else 0.03,
            "dst_host_diff_srv_rate": 0.0 if label == "normal" else 0.8,
            "dst_host_serror_rate": 0.0 if label == "normal" else 0.8,
            "dst_host_srv_serror_rate": 0.0 if label == "normal" else 0.8,
            "label": label,
            "difficulty": -1,
        }
    )
    return row


def protocol_name(packet) -> str:
    if TCP is not None and packet.haslayer(TCP):
        return "tcp"
    if UDP is not None and packet.haslayer(UDP):
        return "udp"
    if ICMP is not None and packet.haslayer(ICMP):
        return "icmp"
    return "tcp"


def service_name(src_port: int, dst_port: int, protocol: str) -> str:
    if protocol == "icmp":
        return "eco_i"
    return SERVICE_PORTS.get(dst_port) or SERVICE_PORTS.get(src_port) or "other"


def tcp_flag(packet) -> str:
    if TCP is None or not packet.haslayer(TCP):
        return "SF"
    flags = str(packet[TCP].flags)
    if "R" in flags:
        return "REJ"
    if "S" in flags and "A" not in flags:
        return "S0"
    if "F" in flags:
        return "SF"
    return "SF"


def block_ip_command(ip_address: str) -> str:
    rule_name = f"AI_IDS_Block_{ip_address}"
    return (
        f'netsh advfirewall firewall add rule name="{rule_name}" '
        f'dir=in action=block remoteip={ip_address}'
    )
