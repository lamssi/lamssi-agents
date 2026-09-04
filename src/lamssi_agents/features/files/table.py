"""Structured-table extraction used by the Files feature's ``read_file`` tool."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from lamssi_tools import err

log = logging.getLogger(__name__)


# ``.xls`` (legacy BIFF) is absent: openpyxl only reads .xlsx/.xlsm and xlrd isn't a dependency.
_DELIMITED = frozenset({".csv", ".tsv", ".dat", ".txt", ".asc"})
_EXCEL = frozenset({".xlsx", ".xlsm"})
_JSON = frozenset({".json", ".jsonl", ".ndjson"})
_NPY = frozenset({".npy", ".npz"})
_MAT = frozenset({".mat"})
_HDF5 = frozenset({".h5", ".hdf5", ".he5"})
_TDMS = frozenset({".tdms"})

# Beyond this element count, skip full stats (would load the whole array) and report shape/dtype + a sample only.
_MAX_LOAD_ELEMS = 5_000_000

SUPPORTED_SUFFIXES = _DELIMITED | _EXCEL | _JSON | _NPY | _MAT | _HDF5 | _TDMS

#: What ``read_file`` sends here unasked: narrower than :data:`SUPPORTED_SUFFIXES` because ambiguous text is usually prose, not data.
STRUCTURED_BY_DEFAULT = (
    # Binary: a text read returns bytes, so there is nothing to weigh up.
    _EXCEL | _NPY | _MAT | _HDF5 | _TDMS
    # Text, but unambiguously tabular by name.
    | frozenset({".csv", ".tsv", ".jsonl", ".ndjson"})
)


def extract_table(
    target: Path,
    *,
    sheet: str = "0",
    max_rows: int = 0,
    sample_rows: int = 5,
    rel_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Read *target* into a structured, JSON-safe summary.

    Dispatches by suffix to the matching reader. Tabular formats return
    ``kind="table"`` (``rows``/``columns``/``dtypes``/``stats``/``sample``/
    ``truncated``); NumPy/MATLAB return ``kind="array"`` or ``"arrays"``
    (per-array ``shape``/``dtype``/stats/sample); non-tabular JSON returns
    ``kind="json"`` (a compact structural view). ``{"error": ...}`` on an
    unsupported format or a parse failure.
    """
    suffix = target.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return err(
            f"Unsupported format: {suffix or '(none)'}",
            hint="read_file handles .csv/.tsv/.dat/.txt/.xlsx/.json/.jsonl/.npy/.npz/.mat/.h5/.tdms. "
                 "For PDF/DOCX/PPTX it extracts document text instead.",
            retriable=False,
        )

    display_path = rel_path if rel_path is not None else str(target)
    nrows = max_rows if max_rows > 0 else None

    try:
        if suffix in _EXCEL:
            return _read_excel(target, sheet, nrows, sample_rows, display_path)
        if suffix in _DELIMITED:
            return _read_delimited(target, suffix, nrows, sample_rows, display_path)
        if suffix in _JSON:
            return _read_json(target, suffix, nrows, sample_rows, display_path)
        if suffix in _NPY:
            return _read_npy(target, sample_rows, display_path)
        if suffix in _MAT:
            return _read_mat(target, sample_rows, display_path)
        if suffix in _HDF5:
            return _read_hdf5(target, sample_rows, display_path)
        if suffix in _TDMS:
            return _read_tdms(target, sample_rows, display_path)
    except Exception as exc:
        return err(
            f"Could not parse {target.name}: {type(exc).__name__}: {exc}",
            retriable=False,
        )
    return err(f"No reader for {suffix}", retriable=False)  # unreachable


def _missing_dep(hint: str) -> Dict[str, Any]:
    """Return an install hint for a missing table dependency."""
    return err(f"{hint}: pip install 'lamssi-agents[tables]'.", retriable=False)


def _load_pandas() -> Tuple[Any, Optional[Dict[str, Any]]]:
    try:
        import pandas as pd
        return pd, None
    except ImportError:
        return None, _missing_dep("pandas is required to read this format")


def _read_excel(target, sheet, nrows, sample_rows, display_path):
    pd, load_err = _load_pandas()
    if load_err:
        return load_err
    sheet_arg: Any = int(sheet) if str(sheet).strip().lstrip("-").isdigit() else sheet
    try:
        df = pd.read_excel(target, sheet_name=sheet_arg, nrows=nrows)
    except ImportError:
        return _missing_dep("Reading .xlsx/.xlsm needs openpyxl")
    return _summarize_dataframe(df, display_path, sample_rows, truncated=nrows is not None)


def _read_delimited(target, suffix, nrows, sample_rows, display_path):
    pd, load_err = _load_pandas()
    if load_err:
        return load_err
    if suffix == ".csv":
        df = pd.read_csv(target, nrows=nrows)
    elif suffix == ".tsv":
        df = pd.read_csv(target, sep="\t", nrows=nrows)
    else:
        # .dat / .txt / .asc: instrument dumps: whitespace-delimited columns, '#'-prefixed header lines.
        df = pd.read_csv(target, sep=r"\s+", engine="python", comment="#", nrows=nrows)
    return _summarize_dataframe(df, display_path, sample_rows, truncated=nrows is not None)


def _read_json(target, suffix, nrows, sample_rows, display_path):
    pd, _ = _load_pandas()  # pandas optional: non-tabular JSON uses stdlib

    if suffix in (".jsonl", ".ndjson"):
        if pd is None:
            return _missing_dep("pandas is required to read .jsonl")
        df = pd.read_json(str(target), lines=True)
        return _cap_and_summarize(df, nrows, sample_rows, display_path)

    # .json: load with stdlib so non-tabular records don't hard-fail.
    with open(target, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # list of records -> table
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj) and pd is not None:
        return _cap_and_summarize(pd.DataFrame(obj), nrows, sample_rows, display_path)
    # dict of equal-length columns -> table
    if (
        isinstance(obj, dict)
        and obj
        and all(isinstance(v, list) for v in obj.values())
        and len({len(v) for v in obj.values()}) == 1
        and pd is not None
    ):
        return _cap_and_summarize(pd.DataFrame(obj), nrows, sample_rows, display_path)

    return _summarize_json_structure(obj, display_path, sample_rows)


def _read_npy(target, sample_rows, display_path):
    try:
        import numpy as np
    except ImportError:
        return _missing_dep("numpy is required for .npy/.npz reads")
    arr = np.load(target, allow_pickle=False)
    if hasattr(arr, "files"):  # NpzFile (archive of named arrays)
        arrays = {name: _summarize_array(np, arr[name], sample_rows) for name in arr.files}
        return {"kind": "arrays", "path": display_path, "n_arrays": len(arrays), "arrays": arrays}
    return {"kind": "array", "path": display_path, **_summarize_array(np, arr, sample_rows)}


def _read_mat(target, sample_rows, display_path):
    try:
        import numpy as np
        from scipy.io import loadmat
    except ImportError:
        return _missing_dep("scipy and numpy are required for .mat reads")
    try:
        mat = loadmat(target)
    except NotImplementedError:
        return err(
            "This is a MATLAB v7.3 (HDF5) .mat file: scipy.io.loadmat can't read it. "
            "Add 'h5py' and read it as HDF5.",
            retriable=False,
        )
    arrays: Dict[str, Any] = {}
    for name, val in mat.items():
        if name.startswith("__"):  # __header__ / __version__ / __globals__
            continue
        if hasattr(val, "shape"):
            arrays[name] = _summarize_array(np, val, sample_rows)
        else:
            arrays[name] = {"type": type(val).__name__, "repr": str(val)[:200]}
    return {
        "kind": "arrays", "path": display_path, "format": "matlab",
        "n_arrays": len(arrays), "arrays": arrays,
    }


def _read_hdf5(target, sample_rows, display_path, max_datasets=200):
    try:
        import numpy as np
        import h5py
    except ImportError:
        return _missing_dep("Reading HDF5 needs h5py and numpy")

    datasets: Dict[str, Any] = {}
    state = {"truncated": False}

    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            if len(datasets) >= max_datasets:
                state["truncated"] = True
                return True  # any non-None return stops the walk
            datasets[name] = _summarize_h5_dataset(np, obj, sample_rows)
        return None

    with h5py.File(target, "r") as f:
        f.visititems(_visit)
        root_attrs = _coerce_attrs(f.attrs)

    out: Dict[str, Any] = {
        "kind": "arrays", "format": "hdf5", "path": display_path,
        "n_datasets": len(datasets), "datasets": datasets,
        "truncated": state["truncated"],
    }
    if root_attrs:
        out["attrs"] = root_attrs
    return out


def _read_tdms(target, sample_rows, display_path, max_channels=500):
    try:
        import numpy as np
        from nptdms import TdmsFile
    except ImportError:
        return _missing_dep("Reading TDMS needs npTDMS and numpy")

    channels: Dict[str, Any] = {}
    truncated = False
    with TdmsFile.open(target) as tdms:
        file_attrs = _coerce_attrs(getattr(tdms, "properties", {}) or {})
        for group in tdms.groups():
            for channel in group.channels():
                if len(channels) >= max_channels:
                    truncated = True
                    break
                channels[f"{group.name}/{channel.name}"] = _summarize_tdms_channel(
                    np, channel, sample_rows
                )
            if truncated:
                break

    out: Dict[str, Any] = {
        "kind": "arrays", "format": "tdms", "path": display_path,
        "n_channels": len(channels), "channels": channels, "truncated": truncated,
    }
    if file_attrs:
        out["attrs"] = file_attrs
    return out


def _cap_and_summarize(df, nrows, sample_rows, display_path):
    """Truncate *df* to *nrows* (precise ``truncated`` flag) then summarize."""
    truncated = False
    if nrows is not None and len(df) > nrows:
        df = df.head(nrows)
        truncated = True
    return _summarize_dataframe(df, display_path, sample_rows, truncated=truncated)


def _summarize_dataframe(
    df: Any,
    display_path: str,
    sample_rows: int,
    *,
    truncated: bool = False,
) -> Dict[str, Any]:
    """Return a bounded, JSON-safe DataFrame summary."""
    import pandas as pd

    columns = [str(c) for c in df.columns]
    dtypes = {str(c): str(df[c].dtype) for c in df.columns}

    stats: Dict[str, Any] = {}
    for c in df.columns:
        col = df[c]
        entry: Dict[str, Any] = {"nulls": int(col.isna().sum())}
        if pd.api.types.is_numeric_dtype(col) and bool(col.notna().any()):
            entry["min"] = _json_scalar(col.min())
            entry["max"] = _json_scalar(col.max())
            entry["mean"] = _json_scalar(col.mean())
            entry["std"] = _json_scalar(col.std())
        else:
            entry["unique"] = int(col.nunique(dropna=True))
            mode = col.mode(dropna=True)
            if len(mode):
                entry["top"] = _json_scalar(mode.iloc[0])
        stats[str(c)] = entry

    sample = [
        {str(k): _json_scalar(val) for k, val in row.items()}
        for row in df.head(max(sample_rows, 0)).to_dict(orient="records")
    ]

    return {
        "kind": "table",
        "path": display_path,
        "rows": len(df),
        "columns": columns,
        "dtypes": dtypes,
        "stats": stats,
        "sample": sample,
        "truncated": bool(truncated),
    }


def _summarize_array(np: Any, arr: Any, sample_rows: int) -> Dict[str, Any]:
    """Summarize a single numpy array: shape, dtype, numeric stats, sample."""
    info: Dict[str, Any] = {
        "shape": list(getattr(arr, "shape", ())),
        "dtype": str(getattr(arr, "dtype", type(arr).__name__)),
        "size": int(getattr(arr, "size", 0) or 0),
    }
    try:
        if arr.size and arr.dtype.kind in "fiub":  # float/int/uint/bool
            finite = arr[np.isfinite(arr)] if arr.dtype.kind == "f" else arr
            if finite.size:
                info["min"] = float(finite.min())
                info["max"] = float(finite.max())
                info["mean"] = float(finite.mean())
                info["std"] = float(finite.std())
            if arr.dtype.kind == "f":
                info["nans"] = int(np.isnan(arr).sum())
    except Exception:  # pragma: no cover - defensive against exotic dtypes
        pass

    try:
        if arr.ndim == 1:
            info["sample"] = [_np_scalar(x) for x in arr[:sample_rows].tolist()]
        elif arr.ndim == 2:
            info["sample"] = [
                [_np_scalar(x) for x in row] for row in arr[:sample_rows].tolist()
            ]
        # >2D: stats are enough; a verbatim slice would be noise.
    except Exception:  # pragma: no cover
        pass
    return info


def _fill_stats(np: Any, info: Dict[str, Any], size: int,
                read_full: Any, read_head: Any, sample_rows: int) -> None:
    """Add full array statistics or a bounded sample to *info*."""
    if size == 0:
        return
    try:
        if size <= _MAX_LOAD_ELEMS:
            full = _summarize_array(np, read_full(), sample_rows)
            for k in ("min", "max", "mean", "std", "nans", "sample"):
                if k in full:
                    info[k] = full[k]
        else:
            info["stats_skipped"] = f"too large ({size} > {_MAX_LOAD_ELEMS})"
            s = _summarize_array(np, read_head(), sample_rows)
            if "sample" in s:
                info["sample"] = s["sample"]
    except Exception:  # pragma: no cover - defensive
        pass


def _summarize_h5_dataset(np: Any, dset: Any, sample_rows: int) -> Dict[str, Any]:
    """Summarize one HDF5 dataset; skip full stats for very large arrays."""
    info: Dict[str, Any] = {
        "shape": list(dset.shape),
        "dtype": str(dset.dtype),
        "size": int(dset.size),
    }
    attrs = _coerce_attrs(dset.attrs)
    if attrs:
        info["attrs"] = attrs
    _fill_stats(
        np, info, int(dset.size),
        lambda: np.asarray(dset[()]),
        lambda: np.asarray(dset[:sample_rows]) if dset.ndim else np.asarray(dset[()]),
        sample_rows,
    )
    return info


def _summarize_tdms_channel(np: Any, channel: Any, sample_rows: int) -> Dict[str, Any]:
    """Summarize one TDMS channel; skip full stats for very large waveforms."""
    n = len(channel)
    info: Dict[str, Any] = {
        "shape": [n],
        "dtype": str(getattr(channel, "dtype", "") or "?"),
        "size": n,
    }
    props = _coerce_attrs(dict(channel.properties)) if channel.properties else {}
    if props:
        info["properties"] = props
    _fill_stats(
        np, info, n,
        lambda: np.asarray(channel[:]),
        lambda: np.asarray(channel[0:sample_rows]),
        sample_rows,
    )
    return info


def _coerce_attrs(attrs: Any) -> Dict[str, Any]:
    """Coerce an HDF5/TDMS attribute mapping to a small JSON-safe dict."""
    out: Dict[str, Any] = {}
    try:
        items = list(attrs.items())
    except Exception:
        return out
    for k, v in items:
        out[str(k)] = _attr_value(v)
    return out


def _attr_value(v: Any) -> Any:
    """JSON-safe coercion for a single HDF5/TDMS attribute value."""
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if getattr(v, "shape", None) == () and hasattr(v, "item"):  # numpy 0-d scalar
        try:
            return _attr_value(v.item())
        except Exception:
            pass
    if hasattr(v, "tolist"):  # array-like attribute
        try:
            lst = v.tolist()
            return lst if len(str(lst)) <= 300 else f"<array shape={list(getattr(v, 'shape', []))}>"
        except Exception:
            pass
    return str(v)[:200]


def _summarize_json_structure(obj: Any, display_path: str, sample_rows: int) -> Dict[str, Any]:
    """Compact structural view of non-tabular JSON (e.g. an Experiment record)."""

    def describe(v: Any) -> Dict[str, Any]:
        if isinstance(v, dict):
            return {"type": "object", "size": len(v), "keys": list(v.keys())[:50]}
        if isinstance(v, list):
            d: Dict[str, Any] = {"type": "array", "len": len(v)}
            if v:
                d["item_type"] = type(v[0]).__name__
            return d
        return {"type": type(v).__name__}

    sample: Any = None
    if isinstance(obj, list):
        sample = [describe(x) if isinstance(x, (dict, list)) else x for x in obj[:sample_rows]]
    elif isinstance(obj, dict):
        sample = {str(k): describe(v) for k, v in list(obj.items())[:sample_rows]}

    return {
        "kind": "json",
        "path": display_path,
        "structure": describe(obj),
        "sample": sample,
    }


def _json_scalar(v: Any) -> Any:
    """Coerce a pandas/numpy scalar to a JSON-safe Python value (NaN/NaT -> None)."""
    import pandas as pd
    try:
        if bool(pd.isna(v)):
            return None
    except (TypeError, ValueError):
        pass  # arrays / non-scalars never reach here, but stay defensive
    item = getattr(v, "item", None)
    if callable(item):
        try:
            v = v.item()
        except Exception:
            pass
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return str(v)


def _np_scalar(x: Any) -> Any:
    """Coerce an array scalar into a JSON-safe value."""
    import math
    if isinstance(x, bool) or x is None or isinstance(x, int):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, str):
        return x
    return str(x)  # complex, datetime, bytes, etc.
