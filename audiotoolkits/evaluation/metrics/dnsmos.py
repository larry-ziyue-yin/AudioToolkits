import atexit
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

from .base import MetricBase
from .utils import ensure_dir


class DNSMOSMetric(MetricBase):
    name = "dnsmos"
    supports_src = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.personalized = bool(cfg.get("personalized", False))
        self.include_raw = bool(cfg.get("include_raw", False))
        self.infer_batch = bool(cfg.get("infer_batch", True))
        self.ort_intra_threads = int(cfg.get("ort_intra_threads", 0) or 0)
        self.ort_inter_threads = int(cfg.get("ort_inter_threads", 0) or 0)
        self.cache_scores = bool(cfg.get("cache_scores", True))
        self.cache_shards = bool(cfg.get("cache_shards", True))
        self.cache_init_retries = max(1, int(cfg.get("cache_init_retries", 20) or 20))
        self.cache_timeout = max(1.0, float(cfg.get("cache_timeout", 60.0) or 60.0))
        self.cache_write_retries = max(1, int(cfg.get("cache_write_retries", 4) or 4))
        self._cache_commit_every_configured = "cache_commit_every" in cfg
        self.cache_commit_every = max(1, int(cfg.get("cache_commit_every", 32) or 32))
        self.scorer = None
        self.sampling_rate = None
        self.model_signature = {}
        self.cache_conn = None
        self.cache_read_conn = None
        self.cache_write_conn = None
        self.cache_path = None
        self.cache_write_path = None
        self.cache_shard_dir = None
        self._cache_pending = 0
        self._cache_exit_hook_registered = False
        self._cache_warned = False
        self._supports_infer_batch = None

    def prepare(self, context):
        from audiotoolkits.libs.mos_kits.dnsmos.dnsmos_local import ComputeScore, SAMPLING_RATE

        base_dir = Path(__file__).resolve().parents[2]
        model_dir = Path(self.cfg.get("model_dir") or (base_dir / "libs" / "mos_kits" / "dnsmos"))
        p808_model_path = model_dir / "DNSMOS" / "model_v8.onnx"
        if self.personalized:
            primary_model_path = model_dir / "pDNSMOS" / "sig_bak_ovr.onnx"
        else:
            primary_model_path = model_dir / "DNSMOS" / "sig_bak_ovr.onnx"
        self.scorer = self._build_scorer(
            ComputeScore,
            primary_model_path,
            p808_model_path,
            context.device,
            context,
        )
        if (
            self.cache_scores
            and not self._cache_commit_every_configured
            and context.cfg.get("parallel", {}).get("enabled", False)
        ):
            self.cache_commit_every = 1
        self.sampling_rate = SAMPLING_RATE
        self.model_signature = {
            "personalized": self.personalized,
            "primary_model": self._path_signature(primary_model_path),
            "p808_model": self._path_signature(p808_model_path),
            "sampling_rate": int(self.sampling_rate),
        }
        self._prepare_cache(context)

    def compute(self, item, context, role="gen"):
        if self.scorer is None:
            self.prepare(context)
        audio_path = item.gen_path if role == "gen" else item.src_path
        if audio_path is None:
            return {}

        cache_key = None
        cached = None
        if self.cache_scores and (self.cache_read_conn is not None or self.cache_write_conn is not None):
            cache_key = self._build_cache_key(audio_path)
            cached = self._cache_get(cache_key, context)

        if cached is None:
            result = self._score_audio(audio_path)
            cached = self._pack_score(result)
            if cache_key is not None:
                self._cache_set(cache_key, cached, context)

        return self._format_output(cached, role)

    def summary(self):
        self._close_cache()
        return {}

    def _prepare_cache(self, context):
        if not self.cache_scores:
            return
        self.cache_path = self._resolve_cache_path(context)
        self.cache_shard_dir = self._build_shard_dir(self.cache_path, context)
        if self._use_worker_shard(context):
            self.cache_read_conn = self._open_cache_conn(self.cache_path, writable=False, context=context)
            self.cache_write_path = self._build_worker_shard_path(self.cache_path, context)
        else:
            self.cache_write_path = self.cache_path
        self.cache_write_conn = self._open_cache_conn(self.cache_write_path, writable=True, context=context)
        self.cache_conn = self.cache_write_conn
        if self.cache_write_conn is None and self.cache_read_conn is None:
            self._warn_cache(context, "初始化失败，已禁用缓存")
            return
        if not self._cache_exit_hook_registered:
            atexit.register(self._close_cache)
            self._cache_exit_hook_registered = True

    def _resolve_cache_path(self, context):
        path = self.cfg.get("output_path") or self.cfg.get("cache_path")
        if path:
            return Path(path).expanduser()
        return Path(context.intermediate_dir) / "dnsmos" / "scores.sqlite3"

    def _open_cache_conn(self, path, writable, context):
        if path is None:
            return None
        path = Path(path)
        if not writable and not path.exists():
            return None
        if writable:
            ensure_dir(path.parent)
        last_exc = None
        for attempt in range(self.cache_init_retries):
            conn = None
            try:
                conn = sqlite3.connect(str(path), timeout=self.cache_timeout)
                conn.execute(f"PRAGMA busy_timeout={int(self.cache_timeout * 1000)}")
                if writable:
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    self._ensure_cache_schema(conn)
                    conn.commit()
                return conn
            except Exception as exc:
                last_exc = exc
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                if self._is_cache_locked_error(exc) and attempt + 1 < self.cache_init_retries:
                    time.sleep(min(0.2 * (attempt + 1), 2.0))
                    continue
                break
        mode = "写入" if writable else "读取"
        self._warn_cache(context, f"{mode}连接初始化失败: {last_exc}")
        return None

    @staticmethod
    def _ensure_cache_schema(conn):
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dnsmos_scores (
                cache_key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
            )
            """
        )

    def _use_worker_shard(self, context):
        return (
            self.cache_shards
            and context is not None
            and bool(context.cfg.get("parallel", {}).get("worker_shard_cache", False))
        )

    def _build_shard_dir(self, cache_path, context):
        run_id = None
        if context is not None:
            run_id = context.cfg.get("parallel", {}).get("run_id")
        run_id = str(run_id or f"manual-{os.getppid()}")
        return Path(cache_path).parent / f"{Path(cache_path).stem}_shards" / self._safe_path_token(run_id)

    def _build_worker_shard_path(self, cache_path, context):
        device = getattr(context, "device", "device") if context is not None else "device"
        device_token = self._safe_path_token(str(device))
        pid_token = self._safe_path_token(str(os.getpid()))
        stem = Path(cache_path).stem
        return self.cache_shard_dir / f"{stem}.{device_token}.pid{pid_token}.sqlite3"

    def _warn_cache(self, context, message):
        if self._cache_warned:
            return
        if context and context.logger:
            context.logger.warning("DNSMOS score cache %s", message)
        self._cache_warned = True

    def _build_scorer(self, scorer_cls, primary_model_path, p808_model_path, device, context):
        try:
            import onnxruntime as ort
        except Exception:
            return scorer_cls(str(primary_model_path), str(p808_model_path))
        def _build_with_providers(providers):
            sess_options = ort.SessionOptions()
            if self.ort_intra_threads > 0:
                sess_options.intra_op_num_threads = self.ort_intra_threads
            if self.ort_inter_threads > 0:
                sess_options.inter_op_num_threads = self.ort_inter_threads
            scorer = scorer_cls.__new__(scorer_cls)
            scorer.onnx_sess = ort.InferenceSession(
                str(primary_model_path),
                sess_options=sess_options,
                providers=providers,
            )
            scorer.p808_onnx_sess = ort.InferenceSession(
                str(p808_model_path),
                sess_options=sess_options,
                providers=providers,
            )
            return scorer
        last_exc = None
        for providers in self._resolve_onnx_provider_attempts(ort, device):
            try:
                scorer = _build_with_providers(providers)
                if context and context.logger:
                    context.logger.info("DNSMOS ONNX providers: %s", providers)
                return scorer
            except Exception as exc:
                last_exc = exc
                continue
        if context and context.logger:
            context.logger.warning("DNSMOS ONNX session 自定义配置失败，回退默认配置: %s", last_exc)
        try:
            return _build_with_providers(["CPUExecutionProvider"])
        except Exception:
            return scorer_cls(str(primary_model_path), str(p808_model_path))

    def _score_audio(self, audio_path):
        if self._supports_infer_batch is not False:
            try:
                result = self.scorer(
                    str(audio_path),
                    self.sampling_rate,
                    self.personalized,
                    infer_batch=self.infer_batch,
                )
                self._supports_infer_batch = True
                return result
            except TypeError as exc:
                if "infer_batch" not in str(exc):
                    raise
                self._supports_infer_batch = False
        return self.scorer(str(audio_path), self.sampling_rate, self.personalized)

    @staticmethod
    def _resolve_onnx_providers(ort, device):
        return DNSMOSMetric._resolve_onnx_provider_attempts(ort, device)[0]

    @staticmethod
    def _resolve_onnx_provider_attempts(ort, device):
        available = set(ort.get_available_providers())
        if isinstance(device, str):
            lowered = device.strip().lower()
        else:
            lowered = ""

        device_id = DNSMOSMetric._parse_device_id(lowered)
        attempts = []

        def add_provider(provider, include_device_id=True):
            if provider not in available:
                return
            if include_device_id:
                attempts.append([(provider, {"device_id": device_id}), "CPUExecutionProvider"])
            attempts.append([provider, "CPUExecutionProvider"])

        if lowered.startswith("cuda"):
            add_provider("CUDAExecutionProvider")
            add_provider("ROCMExecutionProvider")
            add_provider("MIGraphXExecutionProvider")
        if lowered in ("", "auto"):
            if "CUDAExecutionProvider" in available:
                add_provider("CUDAExecutionProvider")
            elif "ROCMExecutionProvider" in available:
                add_provider("ROCMExecutionProvider")
            elif "MIGraphXExecutionProvider" in available:
                add_provider("MIGraphXExecutionProvider")
        if lowered.startswith("rocm"):
            add_provider("ROCMExecutionProvider")
        if lowered.startswith("migraphx"):
            add_provider("MIGraphXExecutionProvider")
        attempts.append(["CPUExecutionProvider"])

        deduped = []
        seen = set()
        for providers in attempts:
            key = repr(providers)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(providers)
        return deduped

    @staticmethod
    def _parse_device_id(device):
        if not isinstance(device, str) or ":" not in device:
            return 0
        try:
            return max(0, int(device.rsplit(":", 1)[1]))
        except Exception:
            return 0

    @staticmethod
    def _to_float(value):
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _is_cache_locked_error(exc):
        text = str(exc).lower()
        return "database is locked" in text or "database schema is locked" in text

    @staticmethod
    def _path_signature(path):
        path = Path(path)
        try:
            stat = path.stat()
            resolved = path.resolve()
            return {
                "path": str(resolved),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        except Exception:
            return {"path": str(path), "size": None, "mtime_ns": None}

    def _build_cache_key(self, audio_path):
        key_obj = {
            "version": 1,
            "audio": self._path_signature(audio_path),
            "model": self.model_signature,
        }
        key_text = json.dumps(key_obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha1(key_text.encode("utf-8")).hexdigest()

    def _pack_score(self, result):
        return {
            "ovrl": self._to_float(result.get("OVRL")),
            "sig": self._to_float(result.get("SIG")),
            "bak": self._to_float(result.get("BAK")),
            "p808": self._to_float(result.get("P808_MOS")),
            "ovrl_raw": self._to_float(result.get("OVRL_raw")),
            "sig_raw": self._to_float(result.get("SIG_raw")),
            "bak_raw": self._to_float(result.get("BAK_raw")),
        }

    def _cache_get(self, cache_key, context):
        row = None
        seen = set()
        for conn in (self.cache_read_conn, self.cache_write_conn):
            if conn is None:
                continue
            conn_id = id(conn)
            if conn_id in seen:
                continue
            seen.add(conn_id)
            try:
                row = conn.execute(
                    "SELECT value_json FROM dnsmos_scores WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()
            except Exception as exc:
                self._warn_cache(context, f"读取失败，已忽略该缓存连接: {exc}")
                continue
            if row:
                break
        if not row:
            return None
        try:
            payload = json.loads(row[0])
        except Exception as exc:
            self._warn_cache(context, f"反序列化失败，已忽略缓存: {exc}")
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _cache_set(self, cache_key, payload, context):
        if self.cache_write_conn is None:
            return
        value_json = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        last_exc = None
        for attempt in range(self.cache_write_retries):
            try:
                self.cache_write_conn.execute(
                    """
                    INSERT OR REPLACE INTO dnsmos_scores (cache_key, value_json, updated_at)
                    VALUES (?, ?, strftime('%s','now'))
                    """,
                    (cache_key, value_json),
                )
                self._cache_pending += 1
                if self._cache_pending >= self.cache_commit_every:
                    self.cache_write_conn.commit()
                    self._cache_pending = 0
                return
            except Exception as exc:
                last_exc = exc
                try:
                    self.cache_write_conn.rollback()
                except Exception:
                    pass
                self._cache_pending = 0
                if self._is_cache_locked_error(exc) and attempt + 1 < self.cache_write_retries:
                    time.sleep(min(0.1 * (attempt + 1), 0.5))
                    continue
                break
        if last_exc is not None:
            try:
                self.cache_write_conn.rollback()
            except Exception:
                pass
            self._disable_write_cache()
            self._warn_cache(context, f"写入失败，后续继续计算不缓存: {last_exc}")

    def _flush_cache_pending(self):
        if self.cache_write_conn is None or self._cache_pending <= 0:
            return
        try:
            self.cache_write_conn.commit()
            self._cache_pending = 0
        except Exception:
            pass

    def _disable_write_cache(self):
        if self.cache_write_conn is not None:
            try:
                self.cache_write_conn.close()
            except Exception:
                pass
        self.cache_write_conn = None
        self.cache_conn = None
        self._cache_pending = 0

    def _close_cache(self):
        self._flush_cache_pending()
        seen = set()
        for conn in (self.cache_read_conn, self.cache_write_conn):
            if conn is None:
                continue
            conn_id = id(conn)
            if conn_id in seen:
                continue
            seen.add(conn_id)
            try:
                conn.close()
            except Exception:
                pass
        self.cache_read_conn = None
        self.cache_write_conn = None
        self.cache_conn = None
        self._cache_pending = 0

    @staticmethod
    def _safe_path_token(value):
        text = str(value).strip()
        chars = []
        for ch in text:
            if ch.isalnum() or ch in ("-", "_", "."):
                chars.append(ch)
            elif ch == ":":
                continue
            else:
                chars.append("_")
        token = "".join(chars).strip("._")
        return token or "unknown"

    def merge_cache_shards(self, context):
        if not self.cache_scores or not self.cache_shards:
            return
        if context is None or not bool(context.cfg.get("parallel", {}).get("enabled", False)):
            return
        cache_path = self._resolve_cache_path(context)
        shard_dir = self._build_shard_dir(cache_path, context)
        if not shard_dir.exists():
            return
        shard_paths = sorted(path for path in shard_dir.glob("*.sqlite3") if path.resolve() != cache_path.resolve())
        if not shard_paths:
            return

        merged_files = 0
        merged_rows = 0
        dst_conn = self._open_cache_conn(cache_path, writable=True, context=context)
        if dst_conn is None:
            self._warn_cache(context, f"shard 合并失败，无法打开主缓存: {cache_path}")
            return
        try:
            for shard_path in shard_paths:
                rows = self._iter_shard_rows(shard_path, context)
                if rows is None:
                    continue
                shard_rows = 0
                for batch in rows:
                    if not batch:
                        continue
                    dst_conn.executemany(
                        """
                        INSERT OR REPLACE INTO dnsmos_scores (cache_key, value_json, updated_at)
                        VALUES (?, ?, ?)
                        """,
                        batch,
                    )
                    shard_rows += len(batch)
                dst_conn.commit()
                merged_files += 1
                merged_rows += shard_rows
        except Exception as exc:
            try:
                dst_conn.rollback()
            except Exception:
                pass
            self._warn_cache(context, f"shard 合并失败: {exc}")
            return
        finally:
            try:
                dst_conn.close()
            except Exception:
                pass
        if context.logger:
            context.logger.info(
                "DNSMOS score cache shards merged: files=%d, rows=%d, cache=%s",
                merged_files,
                merged_rows,
                cache_path,
            )

    def _iter_shard_rows(self, shard_path, context):
        try:
            conn = sqlite3.connect(str(shard_path), timeout=self.cache_timeout)
            conn.execute(f"PRAGMA busy_timeout={int(self.cache_timeout * 1000)}")
        except Exception as exc:
            self._warn_cache(context, f"跳过 shard {shard_path}: {exc}")
            return None

        def _iterator():
            try:
                cursor = conn.execute(
                    "SELECT cache_key, value_json, updated_at FROM dnsmos_scores"
                )
                while True:
                    batch = cursor.fetchmany(1000)
                    if not batch:
                        break
                    yield batch
            except Exception as exc:
                self._warn_cache(context, f"跳过 shard {shard_path}: {exc}")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        return _iterator()

    def _format_output(self, payload, role):
        prefix = self.name if role == "gen" else f"{self.name}_src"
        out = {
            f"{prefix}_ovrl": self._to_float(payload.get("ovrl")),
            f"{prefix}_sig": self._to_float(payload.get("sig")),
            f"{prefix}_bak": self._to_float(payload.get("bak")),
            f"{prefix}_p808": self._to_float(payload.get("p808")),
        }
        if self.include_raw:
            out.update({
                f"{prefix}_ovrl_raw": self._to_float(payload.get("ovrl_raw")),
                f"{prefix}_sig_raw": self._to_float(payload.get("sig_raw")),
                f"{prefix}_bak_raw": self._to_float(payload.get("bak_raw")),
            })
        return out
