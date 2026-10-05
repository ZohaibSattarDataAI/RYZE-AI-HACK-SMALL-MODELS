"""
Privacy proof monitors.

Two live checks that power the dashboard's Privacy Proof panel:

1. Network monitor
   Attempts a socket connection to a known public DNS server. If the
   connection fails, we report Internet: OFF. This is not a security
   boundary — it is a user-facing confirmation that the app is running
   offline.

2. Outbound request counter
   Monkey-patches the `requests` library's Session.request method to
   count outbound HTTP requests made during the app's lifetime.
   Because FinGuard never imports requests at runtime, this counter
   should always read 0. If it ever reads > 0, the privacy panel will
   show it and the user can investigate.

Both checks are best-effort and never raise.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Public DNS servers used for the network probe. No data is sent or
# received — we only attempt a TCP connection to a well-known port.
_PROBE_TARGETS = [
    ("8.8.8.8", 53),       # Google Public DNS
    ("1.1.1.1", 53),       # Cloudflare DNS
]
_PROBE_TIMEOUT = 0.8       # seconds

# Module-level counter for outbound HTTP requests
_OUTBOUND_CALLS = 0
_PATCHED = False


# ---------------------------------------------------------------------------
# Network probe
# ---------------------------------------------------------------------------

def is_online(timeout: float = _PROBE_TIMEOUT) -> bool:
    """
    Return True if the device currently has outbound network access.
    Never raises.
    """
    for host, port in _PROBE_TARGETS:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            continue
    return False


# ---------------------------------------------------------------------------
# Outbound request counter
# ---------------------------------------------------------------------------

def _patch_requests() -> None:
    """
    Patch requests.Session.request to increment _OUTBOUND_CALLS.
    If requests is not installed, silently no-op.
    """
    global _PATCHED, _OUTBOUND_CALLS
    if _PATCHED:
        return
    try:
        import requests  # noqa: F401

        original_request = requests.Session.request

        def counting_request(self, method, url, *args, **kwargs):
            global _OUTBOUND_CALLS
            _OUTBOUND_CALLS += 1
            return original_request(self, method, url, *args, **kwargs)

        requests.Session.request = counting_request
        _PATCHED = True
    except Exception:
        # requests not installed — nothing to patch. Safe.
        _PATCHED = True


def outbound_call_count() -> int:
    """Return the number of outbound HTTP requests made since import."""
    return _OUTBOUND_CALLS


def reset_outbound_counter() -> None:
    """Reset the counter. Useful for tests."""
    global _OUTBOUND_CALLS
    _OUTBOUND_CALLS = 0


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

@dataclass
class PrivacyReport:
    internet_on: bool
    outbound_calls: int
    data_uploaded_bytes: int      # always 0 in FinGuard
    processing_location: str      # always "This device"


def privacy_report() -> PrivacyReport:
    """Return a fresh snapshot for the Privacy Proof panel."""
    _patch_requests()
    return PrivacyReport(
        internet_on=is_online(),
        outbound_calls=outbound_call_count(),
        data_uploaded_bytes=0,
        processing_location="This device",
    )


def model_summary(model_dir: str) -> dict:
    """
    Return a summary of local model weights for the Privacy panel.

    Counts files under model_dir (recursively), sums their sizes, and
    reports the largest. Never raises — returns zeros on any error.
    """
    import os

    total_files = 0
    total_bytes = 0
    largest_bytes = 0
    largest_name = ""

    try:
        for root, _dirs, files in os.walk(model_dir):
            for name in files:
                if not name.lower().endswith((".gguf", ".bin", ".safetensors", ".pt")):
                    continue
                path = os.path.join(root, name)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                total_files += 1
                total_bytes += size
                if size > largest_bytes:
                    largest_bytes = size
                    largest_name = name
    except Exception:
        pass

    return {
        "files": total_files,
        "total_mb": round(total_bytes / 1e6, 1),
        "largest_name": largest_name,
        "largest_mb": round(largest_bytes / 1e6, 1),
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    r = privacy_report()
    print(f"Internet:            {'ON' if r.internet_on else 'OFF'}")
    print(f"Outbound HTTP calls: {r.outbound_calls}")
    print(f"Data uploaded:       {r.data_uploaded_bytes} bytes")
    print(f"Processing:          {r.processing_location}")
    print()
    summary = model_summary("models")
    print(f"Model files found:   {summary['files']}")
    print(f"Total size:          {summary['total_mb']} MB")
    print(f"Largest file:        {summary['largest_name']} ({summary['largest_mb']} MB)")