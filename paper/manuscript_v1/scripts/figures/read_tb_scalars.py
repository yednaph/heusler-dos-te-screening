"""Minimal TensorBoard event-file reader (no tensorflow/tensorboard needed) for the surviving full-cubic training logs.

Event files are TFRecords: [uint64 length][uint32 crc][Event bytes][uint32 crc]. An Event (protobuf) carries
wall_time (field 1, double), step (field 2, int64) and summary (field 5); a Summary holds repeated Value messages
with tag (field 1, string) and simple_value (field 2, float). Only these fields are decoded.
Writes figures/data/tb_<family>_<fold>.csv with columns tag, step, value, wall_time.
"""
import struct, sys
from pathlib import Path
import pandas as pd


def varint(b, i):
    """Decode a protobuf varint at offset i; return (value, new offset)."""
    v = s = 0
    while True:
        c = b[i]; i += 1
        v |= (c & 0x7F) << s; s += 7
        if c < 0x80:
            return v, i


def fields(b):
    """Yield (field_number, wire_type, value) for one protobuf message (value = bytes for length-delimited)."""
    i = 0
    while i < len(b):
        key, i = varint(b, i); f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = varint(b, i)
        elif wt == 1:
            v = b[i:i + 8]; i += 8
        elif wt == 2:
            n, i = varint(b, i); v = b[i:i + n]; i += n
        elif wt == 5:
            v = b[i:i + 4]; i += 4
        else:
            raise ValueError(f'wire type {wt}')
        yield f, wt, v


def read_events(path):
    """Return a list of dicts (tag, step, value, wall_time) for scalar simple_value summaries in one event file."""
    rows, data = [], Path(path).read_bytes()
    i = 0
    while i + 12 <= len(data):
        n = struct.unpack('<Q', data[i:i + 8])[0]; ev = data[i + 12:i + 12 + n]; i += 12 + n + 4
        wall, step, summ = None, 0, None
        for f, wt, v in fields(ev):
            if f == 1 and wt == 1: wall = struct.unpack('<d', v)[0]
            elif f == 2 and wt == 0: step = v
            elif f == 5 and wt == 2: summ = v
        if summ is None:
            continue
        for f, wt, v in fields(summ):
            if f != 1 or wt != 2:
                continue
            tag, val = None, None
            for g, wt2, u in fields(v):
                if g == 1 and wt2 == 2: tag = u.decode()
                elif g == 2 and wt2 == 5: val = struct.unpack('<f', u)[0]
            if tag is not None and val is not None:
                rows.append(dict(tag=tag, step=step, value=val, wall_time=wall))
    return rows


if __name__ == '__main__':
    logdir, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = [r for f in sorted(logdir.rglob('events.out.tfevents.*')) for r in read_events(f)]
    df = pd.DataFrame(rows).sort_values(['tag', 'step']); df.to_csv(out, index=False)
    print(df.groupby('tag').step.agg(['count', 'min', 'max']))
