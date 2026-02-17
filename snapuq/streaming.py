from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

@dataclass
class StreamSegment:
    name: str
    length: int
    severity: int = 0          # 0 means clean
    is_ood: bool = False

def build_monotone_severity_stream(
    base_length: int = 1000,
    severities: List[int] = [0,1,2,3,4,5],
    segment_len: int = 200,
    ood_bursts: Optional[List[Tuple[int,int]]] = None,
) -> List[StreamSegment]:
    segs: List[StreamSegment] = []
    for s in severities:
        segs.append(StreamSegment(name=f"sev{s}", length=segment_len, severity=s, is_ood=False))
    remaining = max(0, base_length - segment_len * len(severities))
    if remaining > 0:
        segs.append(StreamSegment(name="tail_clean", length=remaining, severity=0, is_ood=False))
    if ood_bursts:
        # Insert OOD bursts by splitting segments; simplistic but deterministic.
        # ood_bursts: [(start_idx, length), ...] in sample indices.
        expanded: List[StreamSegment] = []
        pos = 0
        for seg in segs:
            i = 0
            while i < seg.length:
                # Check if any burst starts here
                burst = next((b for b in ood_bursts if b[0] == pos), None)
                if burst:
                    expanded.append(StreamSegment(name="ood", length=burst[1], severity=seg.severity, is_ood=True))
                    i += burst[1]
                    pos += burst[1]
                    continue
                expanded.append(StreamSegment(name=seg.name, length=1, severity=seg.severity, is_ood=False))
                i += 1
                pos += 1
        # Re-compress consecutive 1-length segments back into blocks
        segs2: List[StreamSegment] = []
        cur = expanded[0]
        for s in expanded[1:]:
            if s.name == cur.name and s.severity == cur.severity and s.is_ood == cur.is_ood:
                cur.length += s.length
            else:
                segs2.append(cur)
                cur = s
        segs2.append(cur)
        return segs2
    return segs

def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = (c[window:] - c[:-window]) / float(window)
    # pad with NaNs to keep length consistent
    pad = np.full(window-1, np.nan, dtype=np.float32)
    return np.concatenate([pad, out.astype(np.float32)])
