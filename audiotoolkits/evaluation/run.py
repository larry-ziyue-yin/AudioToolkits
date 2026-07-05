import argparse
import copy
import importlib.util
import json
import logging
import multiprocessing as mp
import os
import time

_MODULE_START = time.perf_counter()

from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from .config import load_config, normalize_config
from .context import EvalContext
from .data import scan_items
from .metrics import build_metrics
from .metrics.utils import MetricSkip
from .results import write_results_csv, summarize_rows, write_summary_csv


_WORKER_METRIC_CACHE = {}
_WORKER_ASR_CACHE = {}
_NLTK_RESOURCE_PATHS = {
    "punkt": "tokenizers/punkt",
}
_NLTK_GUARDED_RESOURCES = set()
_NLTK_ORIGINAL_DOWNLOAD = None


def _get_logger():
    logger = logging.getLogger("audiotoolkits.eval")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def _resolve_device(cfg):
    if cfg.get("device"):
        return cfg.get("device")
    asr_cfg = cfg.get("asr", {})
    return asr_cfg.get("device", "auto")


def _ensure_output_path(path, overwrite, logger):
    path = Path(path)
    if path.exists() and not overwrite:
        raise RuntimeError(f"Output already exists: {path}")
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _skip_reasons(metric, item, role, eval_src):
    reasons = []
    if role == "src" and not eval_src:
        reasons.append("eval_src 未开启")
    if role == "src" and item.src_path is None:
        reasons.append("缺少 src 音频")
    if metric.requires_gt_audio and item.gt_path is None:
        reasons.append("缺少 gt 音频")
    if metric.requires_ref_text and not item.ref_text:
        reasons.append("缺少参考文本")
    return reasons


def _iter_enabled_metric_cfgs(metric_cfgs):
    for metric_cfg in metric_cfgs or []:
        if not metric_cfg:
            continue
        if metric_cfg.get("enabled") is False:
            continue
        name = str(metric_cfg.get("name", "")).strip().lower()
        if not name:
            continue
        yield metric_cfg


def _collect_required_nltk_resources(metric_cfgs):
    needs_speechbertscore = False
    for metric_cfg in _iter_enabled_metric_cfgs(metric_cfgs):
        name = str(metric_cfg.get("name", "")).strip().lower()
        if name == "speechbertscore":
            needs_speechbertscore = True
            break
    if not needs_speechbertscore:
        return set()
    try:
        has_discrete_speech_metrics = importlib.util.find_spec("discrete_speech_metrics") is not None
    except Exception:
        has_discrete_speech_metrics = False
    if not has_discrete_speech_metrics:
        return set()
    # discrete_speech_metrics.speechbleu imports nltk and calls nltk.download("punkt") at import time.
    return {"punkt"}


def _has_nltk_resource(nltk_module, resource):
    locator = _NLTK_RESOURCE_PATHS.get(str(resource).strip().lower())
    if not locator:
        return False
    try:
        nltk_module.data.find(locator)
        return True
    except LookupError:
        return False


def _can_skip_nltk_download(nltk_module, info_or_id, guarded_resources):
    if isinstance(info_or_id, (list, tuple, set)):
        requests = list(info_or_id)
    else:
        requests = [info_or_id]
    if not requests:
        return False
    for request in requests:
        if not isinstance(request, str):
            return False
        rid = request.strip().lower()
        if rid not in guarded_resources:
            return False
        if not _has_nltk_resource(nltk_module, rid):
            return False
    return True


def _install_nltk_download_guard(nltk_module, resources):
    global _NLTK_ORIGINAL_DOWNLOAD
    normalized = {str(x).strip().lower() for x in resources if str(x).strip()}
    if not normalized:
        return
    _NLTK_GUARDED_RESOURCES.update(normalized)
    if _NLTK_ORIGINAL_DOWNLOAD is None:
        original = getattr(nltk_module, "download", None)
        if not callable(original):
            return
        _NLTK_ORIGINAL_DOWNLOAD = original
    guarded_snapshot = set(_NLTK_GUARDED_RESOURCES)
    original_download = _NLTK_ORIGINAL_DOWNLOAD

    def _guarded_download(info_or_id=None, *args, **kwargs):
        if _can_skip_nltk_download(nltk_module, info_or_id, guarded_snapshot):
            return True
        return original_download(info_or_id, *args, **kwargs)

    nltk_module.download = _guarded_download


def _initialize_nltk_resources(metric_cfgs, logger, download_if_missing):
    resources = _collect_required_nltk_resources(metric_cfgs)
    if not resources:
        return
    try:
        import nltk
    except Exception as exc:
        if download_if_missing:
            logger.warning("需要 NLTK 资源 %s，但导入 nltk 失败: %s", sorted(resources), exc)
        return
    for resource in sorted(resources):
        if _has_nltk_resource(nltk, resource):
            continue
        if not download_if_missing:
            raise RuntimeError(
                f"缺少 NLTK 资源: {resource}。请在主进程启动阶段单线程预下载后再启用并行 worker。"
            )
        logger.info("启动阶段单线程下载 NLTK 资源: %s", resource)
        ok = nltk.download(resource, quiet=True)
        if not ok or not _has_nltk_resource(nltk, resource):
            raise RuntimeError(f"NLTK 资源下载失败: {resource}")
    _install_nltk_download_guard(nltk, resources)


def _is_parallel_candidate(metric):
    return metric.name not in ("wer", "cer")


def _load_tqdm():
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kwargs):
            return x
    return tqdm


def _cache_key(value):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return repr(value)


def _build_worker_cfg(cfg, device):
    worker_cfg = copy.deepcopy(cfg)
    worker_cfg.setdefault("output", {})
    worker_cfg["output"]["save_intermediate"] = False
    worker_cfg.setdefault("parallel", {})
    worker_cfg["parallel"]["worker_shard_cache"] = True
    worker_cfg["device"] = device
    if "asr" in worker_cfg:
        worker_cfg["asr"]["device"] = device
    return worker_cfg


def _available_cuda_devices():
    try:
        import torch
    except Exception:
        return []
    try:
        if not torch.cuda.is_available():
            return []
        return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
    except Exception:
        return []


def _normalize_device_list(raw):
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        if "," in text:
            return [x.strip() for x in text.split(",") if x.strip()]
        return [text]
    if isinstance(raw, (list, tuple)):
        values = []
        for value in raw:
            text = str(value).strip()
            if text:
                values.append(text)
        return values
    return []


def _resolve_parallel_devices(parallel_cfg, default_device, logger):
    requested = parallel_cfg.get("devices", "auto")
    cuda_devices = _available_cuda_devices()
    if isinstance(requested, str) and requested.strip().lower() == "auto":
        if cuda_devices:
            return cuda_devices
        if default_device and default_device != "auto":
            return [default_device]
        return ["cpu"]

    resolved = []
    for device in _normalize_device_list(requested):
        lowered = device.lower()
        if lowered == "auto":
            if cuda_devices:
                resolved.extend(cuda_devices)
            elif default_device and default_device != "auto":
                resolved.append(default_device)
            else:
                resolved.append("cpu")
            continue
        if lowered == "cuda":
            if cuda_devices:
                resolved.extend(cuda_devices)
            else:
                logger.warning("parallel.devices=cuda 但未检测到 CUDA，回退到 cpu")
                resolved.append("cpu")
            continue
        if lowered.startswith("cuda") and not cuda_devices:
            logger.warning("parallel.devices=%s 但未检测到 CUDA，回退到 cpu", device)
            resolved.append("cpu")
            continue
        resolved.append(device)

    if not resolved:
        if default_device and default_device != "auto":
            resolved.append(default_device)
        else:
            resolved.append("cpu")

    deduped = []
    seen = set()
    for device in resolved:
        if device not in seen:
            deduped.append(device)
            seen.add(device)
    return deduped


def _build_parallel_plan(cfg, default_device, logger):
    parallel_cfg = cfg.get("parallel", {})
    enabled = bool(parallel_cfg.get("enabled", False))
    devices = _resolve_parallel_devices(parallel_cfg, default_device, logger)
    workers_per_device = max(1, int(parallel_cfg.get("workers_per_device", 1)))
    chunk_size = max(1, int(parallel_cfg.get("chunk_size", 8)))
    worker_devices = []
    for device in devices:
        worker_devices.extend([device] * workers_per_device)
    max_workers = len(worker_devices)
    if not enabled or max_workers <= 1:
        return {
            "enabled": False,
            "devices": devices,
            "worker_devices": worker_devices,
            "max_workers": max_workers,
            "chunk_size": chunk_size,
            "precompute_asr": bool(parallel_cfg.get("precompute_asr", True)),
        }
    if len(devices) == 1 and workers_per_device > 1 and str(devices[0]).startswith("cuda"):
        logger.warning(
            "检测到单卡多进程并发 (device=%s, workers_per_device=%d)。"
            "在部分 ROCm 环境中可能触发 VMFault，运行时会自动降级重试。",
            devices[0],
            workers_per_device,
        )
    return {
        "enabled": True,
        "devices": devices,
        "worker_devices": worker_devices,
        "max_workers": max_workers,
        "chunk_size": chunk_size,
        "precompute_asr": bool(parallel_cfg.get("precompute_asr", True)),
    }


def _chunk_list(values, chunk_size):
    for start in range(0, len(values), chunk_size):
        yield values[start:start + chunk_size]


def _merge_skip_samples(dst, src, limit):
    for reason, rows in src.items():
        bucket = dst.setdefault(reason, [])
        for row in rows:
            if len(bucket) >= limit:
                break
            bucket.append(row)


def _log_skip_stats(logger, skip_reason_counts, skip_samples):
    if not skip_reason_counts:
        return
    for reason, count in skip_reason_counts.items():
        logger.info("  跳过原因: %s -> %d", reason, count)
        samples = skip_samples.get(reason, [])
        if samples:
            logger.info("  跳过样例(%s): %s", reason, "; ".join(samples))


def _log_timing_stats(logger, total_elapsed, metric_elapsed_map, phase_elapsed_map=None, metric_detail_map=None):
    logger.info("耗时统计: 总耗时 %.2fs", total_elapsed)
    if phase_elapsed_map:
        for name, elapsed in phase_elapsed_map.items():
            logger.info("  阶段耗时 %s: %.2fs", name, elapsed)
    if not metric_elapsed_map:
        return
    for name, elapsed in sorted(metric_elapsed_map.items(), key=lambda x: x[1], reverse=True):
        ratio = (elapsed / total_elapsed * 100.0) if total_elapsed > 0 else 0.0
        detail = (metric_detail_map or {}).get(name, {})
        if detail:
            detail_text = ", ".join(f"{key}={value:.2f}s" for key, value in detail.items())
            logger.info("  指标耗时 %s: %.2fs (%.1f%%; %s)", name, elapsed, ratio, detail_text)
        else:
            logger.info("  指标耗时 %s: %.2fs (%.1f%%)", name, elapsed, ratio)


def _get_worker_metric_state(metric_cfg, cfg, output_dir, model_cache_dir, device):
    _initialize_nltk_resources([metric_cfg], _get_logger(), download_if_missing=False)
    cache_key = (_cache_key(metric_cfg), str(device), str(output_dir), str(model_cache_dir))
    state = _WORKER_METRIC_CACHE.get(cache_key)
    if state is not None:
        return state
    worker_cfg = _build_worker_cfg(cfg, device)
    context = EvalContext(
        cfg=worker_cfg,
        output_dir=Path(output_dir),
        model_cache_dir=Path(model_cache_dir),
        device=device,
        logger=_get_logger(),
    )
    metric = build_metrics([metric_cfg])[0]
    try:
        metric.prepare(context)
    except MetricSkip as exc:
        state = {"skip_reason": str(exc) or "metric_skipped"}
        _WORKER_METRIC_CACHE[cache_key] = state
        return state
    state = {"metric": metric, "context": context}
    _WORKER_METRIC_CACHE[cache_key] = state
    return state


def _metric_chunk_worker(
    metric_cfg,
    cfg,
    output_dir,
    model_cache_dir,
    device,
    role,
    eval_src,
    missing_policy,
    max_skip_samples,
    indexed_chunk,
):
    state = _get_worker_metric_state(metric_cfg, cfg, output_dir, model_cache_dir, device)
    if "skip_reason" in state:
        return {"metric_skip_reason": state["skip_reason"]}
    metric = state["metric"]
    context = state["context"]
    updates = []
    ok_count = 0
    skip_count = 0
    fail_count = 0
    skip_reason_counts = {}
    skip_samples = {}
    errors = []
    for idx, item in indexed_chunk:
        reasons = _skip_reasons(metric, item, role, eval_src)
        if reasons:
            if missing_policy == "error":
                raise RuntimeError(f"Missing inputs for metric {metric.name} on {item.utt_id}")
            skip_count += 1
            for reason in reasons:
                skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + 1
                samples = skip_samples.setdefault(reason, [])
                if len(samples) < max_skip_samples:
                    samples.append(
                        f"{item.utt_id} | gen={item.gen_path} | gt={item.gt_path} | src={item.src_path}"
                    )
            continue
        try:
            result = metric.compute(item, context, role=role)
        except MetricSkip as exc:
            reason = str(exc) or "metric_skipped"
            skip_count += 1
            skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + 1
            samples = skip_samples.setdefault(reason, [])
            if len(samples) < max_skip_samples:
                samples.append(
                    f"{item.utt_id} | gen={item.gen_path} | gt={item.gt_path} | src={item.src_path}"
                )
            continue
        except Exception as exc:
            fail_count += 1
            if len(errors) < 10:
                errors.append(f"{item.utt_id}: {exc}")
            continue
        updates.append((idx, result))
        ok_count += 1
    return {
        "updates": updates,
        "ok_count": ok_count,
        "skip_count": skip_count,
        "fail_count": fail_count,
        "skip_reason_counts": skip_reason_counts,
        "skip_samples": skip_samples,
        "errors": errors,
    }


def _get_worker_asr_state(asr_cfg, intermediate_dir, device):
    from .metrics.asr import build_asr

    cache_key = (_cache_key(asr_cfg), str(intermediate_dir), str(device))
    state = _WORKER_ASR_CACHE.get(cache_key)
    if state is not None:
        return state
    try:
        asr = build_asr(asr_cfg, intermediate_dir, device, _get_logger(), save_intermediate=False)
    except MetricSkip as exc:
        state = {"skip_reason": str(exc) or "asr_skipped"}
        _WORKER_ASR_CACHE[cache_key] = state
        return state
    state = {"asr": asr}
    _WORKER_ASR_CACHE[cache_key] = state
    return state


def _asr_chunk_worker(asr_cfg, intermediate_dir, device, role, items_chunk):
    state = _get_worker_asr_state(asr_cfg, intermediate_dir, device)
    if "skip_reason" in state:
        return {"skip_reason": state["skip_reason"]}
    asr = state["asr"]
    texts = []
    errors = []
    for item in items_chunk:
        audio_path = item.gen_path if role == "gen" else item.src_path
        if audio_path is None:
            continue
        try:
            text = asr.transcribe(item, role=role, persist=False)
        except Exception as exc:
            if len(errors) < 10:
                errors.append(f"{item.utt_id}: {exc}")
            continue
        texts.append((item.utt_id, text))
    return {"role": role, "texts": texts, "errors": errors}


def _run_metric_role_sequential(
    metric,
    items,
    context,
    role,
    eval_src,
    missing_policy,
    max_error_logs,
    max_skip_samples,
    results,
    tqdm,
):
    ok_count = 0
    skip_count = 0
    fail_count = 0
    skip_reason_counts = {}
    skip_samples = {}
    error_logs = 0
    iterator = tqdm(enumerate(items), total=len(items), desc=f"{metric.name}:{role}")
    for idx, item in iterator:
        reasons = _skip_reasons(metric, item, role, eval_src)
        if reasons:
            if missing_policy == "error":
                raise RuntimeError(f"Missing inputs for metric {metric.name} on {item.utt_id}")
            skip_count += 1
            for reason in reasons:
                skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + 1
                samples = skip_samples.setdefault(reason, [])
                if len(samples) < max_skip_samples:
                    samples.append(f"{item.utt_id} | gen={item.gen_path} | gt={item.gt_path} | src={item.src_path}")
            continue
        try:
            result = metric.compute(item, context, role=role)
        except MetricSkip as exc:
            reason = str(exc) or "metric_skipped"
            skip_count += 1
            skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + 1
            samples = skip_samples.setdefault(reason, [])
            if len(samples) < max_skip_samples:
                samples.append(f"{item.utt_id} | gen={item.gen_path} | gt={item.gt_path} | src={item.src_path}")
            continue
        except Exception as exc:
            fail_count += 1
            if error_logs < max_error_logs:
                context.logger.exception("指标 %s 在 %s 上失败: %s", metric.name, item.utt_id, exc)
                error_logs += 1
            else:
                context.logger.warning("指标 %s 在 %s 上失败: %s", metric.name, item.utt_id, exc)
            continue
        results[idx].update(result)
        ok_count += 1
    return {
        "ok_count": ok_count,
        "skip_count": skip_count,
        "fail_count": fail_count,
        "skip_reason_counts": skip_reason_counts,
        "skip_samples": skip_samples,
    }


def _run_metric_role_parallel(
    metric,
    metric_cfg,
    items,
    context,
    role,
    eval_src,
    missing_policy,
    max_error_logs,
    max_skip_samples,
    results,
    parallel_plan,
    tqdm,
):
    indexed_items = list(enumerate(items))
    chunks = list(_chunk_list(indexed_items, parallel_plan["chunk_size"]))
    if not chunks:
        return {
            "ok_count": 0,
            "skip_count": 0,
            "fail_count": 0,
            "skip_reason_counts": {},
            "skip_samples": {},
        }

    worker_devices = parallel_plan["worker_devices"]
    max_workers = min(parallel_plan["max_workers"], len(chunks))
    attempt_workers = max_workers
    last_error = None
    while attempt_workers >= 1:
        active_devices = worker_devices[:attempt_workers] or worker_devices or ["cpu"]
        ok_count = 0
        skip_count = 0
        fail_count = 0
        skip_reason_counts = {}
        skip_samples = {}
        pending_updates = []
        error_logs = 0
        metric_skip_reason = None
        mp_ctx = mp.get_context("spawn")
        try:
            with ProcessPoolExecutor(max_workers=attempt_workers, mp_context=mp_ctx) as executor:
                futures = []
                for task_idx, chunk in enumerate(chunks):
                    device = active_devices[task_idx % len(active_devices)]
                    futures.append(
                        executor.submit(
                            _metric_chunk_worker,
                            metric_cfg,
                            context.cfg,
                            str(context.output_dir),
                            str(context.model_cache_dir),
                            device,
                            role,
                            eval_src,
                            missing_policy,
                            max_skip_samples,
                            chunk,
                        )
                    )
                iterator = tqdm(as_completed(futures), total=len(futures), desc=f"{metric.name}:{role}")
                for future in iterator:
                    payload = future.result()
                    if payload.get("metric_skip_reason"):
                        metric_skip_reason = payload["metric_skip_reason"]
                        continue
                    pending_updates.extend(payload.get("updates", []))
                    ok_count += int(payload.get("ok_count", 0))
                    skip_count += int(payload.get("skip_count", 0))
                    fail_count += int(payload.get("fail_count", 0))
                    for reason, count in payload.get("skip_reason_counts", {}).items():
                        skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + int(count)
                    _merge_skip_samples(skip_samples, payload.get("skip_samples", {}), max_skip_samples)
                    for message in payload.get("errors", []):
                        if error_logs < max_error_logs:
                            context.logger.warning("指标 %s 在并行 worker 上失败: %s", metric.name, message)
                            error_logs += 1
                        else:
                            break
            if metric_skip_reason:
                return {"metric_skip_reason": metric_skip_reason}
            for idx, result in pending_updates:
                results[idx].update(result)
            return {
                "ok_count": ok_count,
                "skip_count": skip_count,
                "fail_count": fail_count,
                "skip_reason_counts": skip_reason_counts,
                "skip_samples": skip_samples,
            }
        except Exception as exc:
            if isinstance(exc, RuntimeError) and str(exc).startswith("Missing inputs for metric"):
                raise
            last_error = exc
            if attempt_workers <= 1:
                break
            next_workers = max(1, attempt_workers // 2)
            if isinstance(exc, BrokenProcessPool):
                context.logger.warning(
                    "指标 %s 并行进程池崩溃，worker 从 %d 降到 %d 重试。",
                    metric.name,
                    attempt_workers,
                    next_workers,
                )
            else:
                context.logger.warning(
                    "指标 %s 并行执行异常 (%s: %s)，worker 从 %d 降到 %d 重试。",
                    metric.name,
                    exc.__class__.__name__,
                    exc,
                    attempt_workers,
                    next_workers,
                )
            attempt_workers = next_workers

    raise RuntimeError(f"指标 {metric.name} 的并行执行失败 (role={role})") from last_error


def _needs_gen_asr(metric, ref_items):
    hyp_map = getattr(metric, "hyp_map", None) or {}
    if not hyp_map:
        return True
    for item in ref_items:
        if item.utt_id not in hyp_map:
            return True
    return False


def _precompute_asr_for_wer(metric_entries, items, context, eval_src, parallel_plan, max_error_logs, tqdm):
    if not parallel_plan["enabled"] or not parallel_plan["precompute_asr"]:
        return
    asr = context.resources.get("asr")
    if asr is None:
        return
    wer_metrics = [entry["metric"] for entry in metric_entries if entry["metric"].name in ("wer", "cer")]
    if not wer_metrics:
        return
    ref_items = [item for item in items if item.ref_text]
    if not ref_items:
        return

    roles = []
    if any(_needs_gen_asr(metric, ref_items) for metric in wer_metrics):
        roles.append("gen")
    if eval_src and any(getattr(metric, "supports_src", True) for metric in wer_metrics):
        roles.append("src")
    if not roles:
        return

    tasks = []
    for role in roles:
        cached_ids = set(asr.text_maps.get(role, {}).keys())
        candidates = []
        for item in ref_items:
            audio_path = item.gen_path if role == "gen" else item.src_path
            if audio_path is None:
                continue
            if item.utt_id in cached_ids:
                continue
            candidates.append(item)
        if not candidates:
            continue
        context.logger.info("ASR 预转写: role=%s, 待处理=%d", role, len(candidates))
        for chunk in _chunk_list(candidates, parallel_plan["chunk_size"]):
            tasks.append((role, chunk))
    if not tasks:
        return

    worker_devices = parallel_plan["worker_devices"]
    max_workers = min(parallel_plan["max_workers"], len(tasks))
    attempt_workers = max_workers
    role_texts = None
    last_error = None
    while attempt_workers >= 1:
        active_devices = worker_devices[:attempt_workers] or worker_devices or ["cpu"]
        attempt_texts = {role: {} for role in roles}
        skip_reason = None
        error_logs = 0
        mp_ctx = mp.get_context("spawn")
        try:
            with ProcessPoolExecutor(max_workers=attempt_workers, mp_context=mp_ctx) as executor:
                futures = []
                for task_idx, (role, chunk) in enumerate(tasks):
                    device = active_devices[task_idx % len(active_devices)]
                    futures.append(
                        executor.submit(
                            _asr_chunk_worker,
                            context.cfg.get("asr", {}),
                            str(context.intermediate_dir),
                            device,
                            role,
                            chunk,
                        )
                    )
                iterator = tqdm(as_completed(futures), total=len(futures), desc="asr:precompute")
                for future in iterator:
                    payload = future.result()
                    if payload.get("skip_reason"):
                        skip_reason = payload["skip_reason"]
                        continue
                    role = payload.get("role")
                    if role:
                        for utt_id, text in payload.get("texts", []):
                            attempt_texts.setdefault(role, {})[utt_id] = text
                    for message in payload.get("errors", []):
                        if error_logs < max_error_logs:
                            context.logger.warning("ASR 预转写失败: %s", message)
                            error_logs += 1
                        else:
                            break
            if skip_reason:
                context.logger.warning("ASR 预转写跳过: %s", skip_reason)
                return
            role_texts = attempt_texts
            break
        except Exception as exc:
            last_error = exc
            if attempt_workers <= 1:
                break
            next_workers = max(1, attempt_workers // 2)
            if isinstance(exc, BrokenProcessPool):
                context.logger.warning(
                    "ASR 预转写进程池崩溃，worker 从 %d 降到 %d 重试。",
                    attempt_workers,
                    next_workers,
                )
            else:
                context.logger.warning(
                    "ASR 预转写并行异常 (%s: %s)，worker 从 %d 降到 %d 重试。",
                    exc.__class__.__name__,
                    exc,
                    attempt_workers,
                    next_workers,
                )
            attempt_workers = next_workers

    if role_texts is None:
        raise RuntimeError("ASR 预转写并行失败") from last_error

    for role, mapping in role_texts.items():
        added = asr.update_texts(role, mapping, persist=context.save_intermediate)
        context.logger.info("ASR 预转写完成: role=%s, 新增缓存=%d", role, added)
    if hasattr(asr, "flush"):
        asr.flush()


def main():
    main_start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to evaluation config YAML")
    parser.add_argument("--root", help="Root directory containing generated audio")
    parser.add_argument("--gt-root", help="Root directory containing ground-truth audio")
    parser.add_argument("--src-root", help="Root directory containing source audio")
    parser.add_argument("--manifest", help="Manifest file path (csv/json/jsonl)")
    parser.add_argument("--gt-text-path", help="Ground-truth text file path")
    parser.add_argument("--results-csv", help="Results CSV output path")
    parser.add_argument("--summary-csv", help="Summary CSV output path")
    parser.add_argument("--cache-dir", help="Cache directory for models and intermediate artifacts")
    parser.add_argument("--output-dir", help="Output directory for results/summary CSVs when not specified")
    args = parser.parse_args()

    cfg = load_config(args.config)
    original_root = cfg["data"].get("root")
    if args.root:
        cfg["data"]["root"] = args.root
    if args.gt_root:
        cfg["data"]["gt_root"] = args.gt_root
    elif args.root and cfg["data"].get("gt_root") == original_root:
        cfg["data"]["gt_root"] = None
    if args.src_root:
        cfg["data"]["src_root"] = args.src_root
    elif args.root and cfg["data"].get("src_root") == original_root:
        cfg["data"]["src_root"] = None
    if args.manifest:
        cfg["data"]["manifest"] = args.manifest
    if args.gt_text_path:
        cfg["data"]["gt_text_path"] = args.gt_text_path
    if args.results_csv:
        cfg["output"]["results_csv"] = args.results_csv
    if args.summary_csv:
        cfg["output"]["summary_csv"] = args.summary_csv
    if args.cache_dir:
        cfg["output"]["cache_dir"] = args.cache_dir
    if args.output_dir:
        cfg["output"]["output_dir"] = args.output_dir
    cfg = normalize_config(cfg)
    output_dir = cfg["output"].get("output_dir")
    if output_dir:
        output_dir = Path(output_dir)
        if not args.results_csv:
            cfg["output"]["results_csv"] = str(output_dir / Path(cfg["output"]["results_csv"]).name)
        if not args.summary_csv:
            cfg["output"]["summary_csv"] = str(output_dir / Path(cfg["output"]["summary_csv"]).name)
    else:
        output_dir = Path(cfg["output"]["results_csv"]).parent
        cfg["output"]["output_dir"] = str(output_dir)
    logger = _get_logger()

    items = scan_items(cfg)
    if not items:
        raise RuntimeError("No evaluation items found.")

    metric_cfgs = list(_iter_enabled_metric_cfgs(cfg.get("metrics", [])))
    _initialize_nltk_resources(metric_cfgs, logger, download_if_missing=True)
    metrics = build_metrics(cfg.get("metrics", []))
    if not metrics:
        raise RuntimeError("No metrics configured.")
    metric_entries = [{"metric": metric, "cfg": metric_cfg} for metric, metric_cfg in zip(metrics, metric_cfgs)]

    cache_dir = Path(cfg["output"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_cache_dir = cache_dir / "model_cache"
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg)
    context = EvalContext(
        cfg=cfg,
        output_dir=output_dir,
        model_cache_dir=model_cache_dir,
        device=device,
        logger=logger,
    )
    parallel_plan = _build_parallel_plan(cfg, device, logger)
    if parallel_plan["enabled"]:
        cfg.setdefault("parallel", {}).setdefault("run_id", f"{int(time.time())}-{os.getpid()}")
    if parallel_plan["enabled"]:
        logger.info(
            "并行模式启用: devices=%s, workers_per_device=%d, total_workers=%d, chunk_size=%d",
            parallel_plan["devices"],
            int(cfg.get("parallel", {}).get("workers_per_device", 1)),
            parallel_plan["max_workers"],
            parallel_plan["chunk_size"],
        )
    elif bool(cfg.get("parallel", {}).get("enabled", False)):
        logger.info("并行模式已配置，但当前可用 worker 不足，按串行执行。")

    prepared_entries = []
    prepare_elapsed_map = {}
    prepare_phase_start = time.perf_counter()
    for entry in metric_entries:
        metric = entry["metric"]
        try:
            if parallel_plan["enabled"] and _is_parallel_candidate(metric):
                logger.info("指标 %s 将在并行 worker 中初始化。", metric.name)
                prepare_elapsed_map[metric.name] = 0.0
            else:
                logger.info("准备指标: %s", metric.name)
                prepare_start = time.perf_counter()
                metric.prepare(context)
                prepare_elapsed = time.perf_counter() - prepare_start
                prepare_elapsed_map[metric.name] = prepare_elapsed
                logger.info("准备完成: %s (%.2fs)", metric.name, prepare_elapsed)
        except MetricSkip as exc:
            logger.warning("跳过指标 %s: %s", metric.name, exc)
            continue
        prepared_entries.append(entry)
    prepare_phase_elapsed = time.perf_counter() - prepare_phase_start
    metric_entries = prepared_entries
    if not metric_entries:
        raise RuntimeError("No metrics prepared (all skipped).")

    tqdm = _load_tqdm()

    missing_policy = cfg["data"].get("missing_policy", "skip")
    eval_src = bool(cfg["data"].get("eval_src", False))

    results = []
    for item in items:
        results.append({
            "utt_id": item.utt_id,
            "gen_path": str(item.gen_path),
            "gt_path": str(item.gt_path) if item.gt_path else "",
            "src_path": str(item.src_path) if item.src_path else "",
            "ref_text": item.ref_text or "",
        })

    logger.info("待评测条数: %d", len(items))
    logger.info("指标列表: %s", ", ".join([entry["metric"].name for entry in metric_entries]))
    logger.info("输出目录: %s", output_dir)
    logger.info("模型缓存目录: %s", model_cache_dir)
    logger.info(
        "数据配置: root=%s, gt_root=%s, src_root=%s, gen_suffix=%s, gt_suffix=%s, src_suffix=%s, exts=%s",
        cfg["data"].get("root"),
        cfg["data"].get("gt_root"),
        cfg["data"].get("src_root"),
        cfg["data"].get("gen_suffix"),
        cfg["data"].get("gt_suffix"),
        cfg["data"].get("src_suffix"),
        cfg["data"].get("audio_extensions"),
    )

    roles = ("gen", "src") if eval_src else ("gen",)
    max_error_logs = int(cfg.get("output", {}).get("max_error_logs", 3))
    max_skip_samples = int(cfg.get("output", {}).get("max_skip_samples", 3))
    asr_precompute_start = time.perf_counter()
    _precompute_asr_for_wer(
        metric_entries,
        items,
        context,
        eval_src,
        parallel_plan,
        max_error_logs,
        tqdm,
    )
    asr_precompute_elapsed = time.perf_counter() - asr_precompute_start

    metric_loop_start = time.perf_counter()
    metric_elapsed_map = {}
    metric_compute_elapsed_map = {}
    for entry in metric_entries:
        metric = entry["metric"]
        metric_cfg = entry["cfg"]
        logger.info("开始计算指标: %s", metric.name)
        metric_start = time.perf_counter()
        metric_skipped = False
        for role in roles:
            if role == "src" and not getattr(metric, "supports_src", True):
                logger.info("指标 %s 不支持 src，跳过", metric.name)
                continue
            logger.info("  角色: %s", role)
            role_start = time.perf_counter()
            if parallel_plan["enabled"] and _is_parallel_candidate(metric):
                role_stats = _run_metric_role_parallel(
                    metric,
                    metric_cfg,
                    items,
                    context,
                    role,
                    eval_src,
                    missing_policy,
                    max_error_logs,
                    max_skip_samples,
                    results,
                    parallel_plan,
                    tqdm,
                )
                if role_stats.get("metric_skip_reason"):
                    logger.warning("跳过指标 %s: %s", metric.name, role_stats["metric_skip_reason"])
                    metric_skipped = True
                    break
            else:
                role_stats = _run_metric_role_sequential(
                    metric,
                    items,
                    context,
                    role,
                    eval_src,
                    missing_policy,
                    max_error_logs,
                    max_skip_samples,
                    results,
                    tqdm,
                )
            role_elapsed = time.perf_counter() - role_start
            logger.info(
                "  角色 %s 完成: 成功 %d, 跳过 %d, 失败 %d, 耗时 %.2fs",
                role,
                role_stats["ok_count"],
                role_stats["skip_count"],
                role_stats["fail_count"],
                role_elapsed,
            )
            _log_skip_stats(logger, role_stats["skip_reason_counts"], role_stats["skip_samples"])
            if role_stats["ok_count"] == 0 and role_stats["skip_count"] > 0:
                logger.warning("  角色 %s 没有成功样本，请检查数据配对或配置。", role)
        if metric_skipped:
            continue
        if parallel_plan["enabled"] and _is_parallel_candidate(metric) and hasattr(metric, "merge_cache_shards"):
            metric.merge_cache_shards(context)
        metric_compute_elapsed = time.perf_counter() - metric_start
        metric_total_elapsed = prepare_elapsed_map.get(metric.name, 0.0) + metric_compute_elapsed
        metric_compute_elapsed_map[metric.name] = metric_compute_elapsed
        metric_elapsed_map[metric.name] = metric_total_elapsed
        logger.info(
            "完成指标: %s (compute %.2fs, total %.2fs)",
            metric.name,
            metric_compute_elapsed,
            metric_total_elapsed,
        )
    metric_compute_phase_elapsed = time.perf_counter() - metric_loop_start

    finalize_start = time.perf_counter()
    asr_flush_elapsed = 0.0
    asr_resource = context.resources.get("asr")
    if asr_resource is not None and hasattr(asr_resource, "flush"):
        asr_flush_start = time.perf_counter()
        asr_resource.flush()
        asr_flush_elapsed = time.perf_counter() - asr_flush_start

    results_path = _ensure_output_path(cfg["output"]["results_csv"], cfg["output"].get("overwrite", True), logger)
    results_write_start = time.perf_counter()
    write_results_csv(results, results_path)
    results_write_elapsed = time.perf_counter() - results_write_start

    extra_summary = {}
    metric_summary_start = time.perf_counter()
    for entry in metric_entries:
        metric = entry["metric"]
        extra_summary.update(metric.summary())
    metric_summary_elapsed = time.perf_counter() - metric_summary_start
    finalize_elapsed = time.perf_counter() - finalize_start

    total_elapsed = time.perf_counter() - _MODULE_START
    main_elapsed = time.perf_counter() - main_start
    startup_elapsed = max(0.0, main_start - _MODULE_START)
    phase_elapsed_map = {
        "startup_import": startup_elapsed,
        "prepare": prepare_phase_elapsed,
        "asr_precompute": asr_precompute_elapsed,
        "metric_compute": metric_compute_phase_elapsed,
        "finalize": finalize_elapsed,
        "asr_flush": asr_flush_elapsed,
        "results_write": results_write_elapsed,
        "metric_summary": metric_summary_elapsed,
    }
    metric_detail_map = {
        name: {
            "prepare": prepare_elapsed_map.get(name, 0.0),
            "compute": metric_compute_elapsed_map.get(name, 0.0),
        }
        for name in metric_elapsed_map
    }
    _log_timing_stats(logger, total_elapsed, metric_elapsed_map, phase_elapsed_map, metric_detail_map)

    extra_summary["timing_total_seconds"] = float(total_elapsed)
    extra_summary["timing_main_seconds"] = float(main_elapsed)
    for name, elapsed in phase_elapsed_map.items():
        extra_summary[f"timing_phase_{name}_seconds"] = float(elapsed)
    for name, elapsed in metric_elapsed_map.items():
        extra_summary[f"timing_metric_{name}_seconds"] = float(elapsed)
        extra_summary[f"timing_metric_{name}_prepare_seconds"] = float(prepare_elapsed_map.get(name, 0.0))
        extra_summary[f"timing_metric_{name}_compute_seconds"] = float(metric_compute_elapsed_map.get(name, 0.0))
    summary_rows = summarize_rows(results, extra_summary=extra_summary)
    summary_path = _ensure_output_path(cfg["output"]["summary_csv"], cfg["output"].get("overwrite", True), logger)
    write_summary_csv(summary_rows, summary_path)

    logger.info("Saved results to %s", results_path)
    logger.info("Saved summary to %s", summary_path)


if __name__ == "__main__":
    main()
