import csv
import sys
import unicodedata
from pathlib import Path

from .base import MetricBase
from .asr import build_asr
from .utils import ensure_dir


SPACES = {" ", "\t", "\r", "\n"}
PUNCTS = [
    "!",
    ",",
    "?",
    "\u3001",
    "\u3002",
    "\uff01",
    "\uff0c",
    "\uff1b",
    "\uff1f",
    "\uff1a",
    "\u300c",
    "\u300d",
    "\ufe30",
    "\u300e",
    "\u300f",
    "\u300a",
    "\u300b",
]


def characterize(text):
    res = []
    i = 0
    while i < len(text):
        char = text[i]
        if char in PUNCTS:
            i += 1
            continue
        cat1 = unicodedata.category(char)
        if cat1 == "Zs" or cat1 == "Cn" or char in SPACES:
            i += 1
            continue
        # Keep <...> as one token so normalize() can strip it as a whole tag.
        if char == "<":
            j = i + 1
            while j < len(text) and text[j] != ">":
                j += 1
            if j < len(text) and text[j] == ">":
                j += 1
            res.append(text[i:j])
            i = j
            continue
        # True character-level tokenization for CER (including English).
        res.append(char)
        i += 1
    return res


def stripoff_tags(text):
    if not text:
        return ""
    chars = []
    i = 0
    while i < len(text):
        if text[i] == "<":
            while i < len(text) and text[i] != ">":
                i += 1
            i += 1
        else:
            chars.append(text[i])
            i += 1
    return "".join(chars)


def normalize(tokens, ignore_words, case_sensitive):
    new_sentence = []
    for token in tokens:
        x = token
        if not case_sensitive:
            x = x.upper()
        if x in ignore_words:
            continue
        x = stripoff_tags(x)
        if not x:
            continue
        new_sentence.append(x)
    return new_sentence


def default_cluster(word):
    unicode_names = [unicodedata.name(char) for char in word]
    for i in reversed(range(len(unicode_names))):
        name = unicode_names[i]
        if name.startswith("DIGIT"):
            unicode_names[i] = "Number"
        elif name.startswith("CJK UNIFIED IDEOGRAPH") or name.startswith("CJK COMPATIBILITY IDEOGRAPH"):
            unicode_names[i] = "Mandarin"
        elif name.startswith("LATIN CAPITAL LETTER") or name.startswith("LATIN SMALL LETTER"):
            unicode_names[i] = "English"
        elif name.startswith("HIRAGANA LETTER"):
            unicode_names[i] = "Japanese"
        elif name.startswith((
            "AMPERSAND",
            "APOSTROPHE",
            "COMMERCIAL AT",
            "DEGREE CELSIUS",
            "EQUALS SIGN",
            "FULL STOP",
            "HYPHEN-MINUS",
            "LOW LINE",
            "NUMBER SIGN",
            "PLUS SIGN",
            "SEMICOLON",
        )):
            del unicode_names[i]
        else:
            return "Other"
    if len(unicode_names) == 0:
        return "Other"
    if len(unicode_names) == 1:
        return unicode_names[0]
    for i in range(len(unicode_names) - 1):
        if unicode_names[i] != unicode_names[i + 1]:
            return "Other"
    return unicode_names[0]


def width(string):
    return sum(1 + (unicodedata.east_asian_width(c) in "AFW") for c in string)


def _rate(stats):
    if stats["all"] > 0:
        return 100.0 * float(stats["ins"] + stats["sub"] + stats["del"]) / stats["all"]
    return 0.0


def _tokenize(text, char_level, case_sensitive):
    if char_level:
        tokens = characterize(text)
    else:
        tokens = text.strip().split()
    return normalize(tokens, set(), case_sensitive)


def _load_text_map(path):
    if not path:
        return {}
    mapping = {}
    path = Path(path)
    if not path.exists():
        return mapping
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            utt_id = parts[0]
            text = " ".join(parts[1:])
            mapping[utt_id] = text
    return mapping


class Calculator:
    def __init__(self):
        self.data = {}
        self.cost = {"cor": 0, "sub": 1, "del": 1, "ins": 1}
        self.space = []

    def calculate(self, lab, rec):
        lab = [""] + list(lab)
        rec = [""] + list(rec)
        while len(self.space) < len(lab):
            self.space.append([])
        for row in self.space:
            for element in row:
                element["dist"] = 0
                element["error"] = "non"
            while len(row) < len(rec):
                row.append({"dist": 0, "error": "non"})

        for i in range(len(lab)):
            self.space[i][0]["dist"] = i
            self.space[i][0]["error"] = "del"
        for j in range(len(rec)):
            self.space[0][j]["dist"] = j
            self.space[0][j]["error"] = "ins"
        self.space[0][0]["error"] = "non"

        for token in lab:
            if token not in self.data and len(token) > 0:
                self.data[token] = {"all": 0, "cor": 0, "sub": 0, "ins": 0, "del": 0}
        for token in rec:
            if token not in self.data and len(token) > 0:
                self.data[token] = {"all": 0, "cor": 0, "sub": 0, "ins": 0, "del": 0}

        for i, lab_token in enumerate(lab):
            for j, rec_token in enumerate(rec):
                if i == 0 or j == 0:
                    continue
                min_dist = 10**9
                min_error = "none"
                dist = self.space[i - 1][j]["dist"] + self.cost["del"]
                error = "del"
                if dist < min_dist:
                    min_dist = dist
                    min_error = error
                dist = self.space[i][j - 1]["dist"] + self.cost["ins"]
                error = "ins"
                if dist < min_dist:
                    min_dist = dist
                    min_error = error
                if lab_token == rec_token:
                    dist = self.space[i - 1][j - 1]["dist"] + self.cost["cor"]
                    error = "cor"
                else:
                    dist = self.space[i - 1][j - 1]["dist"] + self.cost["sub"]
                    error = "sub"
                if dist < min_dist:
                    min_dist = dist
                    min_error = error
                self.space[i][j]["dist"] = min_dist
                self.space[i][j]["error"] = min_error

        result = {"lab": [], "rec": [], "all": 0, "cor": 0, "sub": 0, "ins": 0, "del": 0}
        i = len(lab) - 1
        j = len(rec) - 1
        while True:
            error = self.space[i][j]["error"]
            if error == "cor":
                if len(lab[i]) > 0:
                    self.data[lab[i]]["all"] += 1
                    self.data[lab[i]]["cor"] += 1
                    result["all"] += 1
                    result["cor"] += 1
                result["lab"].insert(0, lab[i])
                result["rec"].insert(0, rec[j])
                i -= 1
                j -= 1
            elif error == "sub":
                if len(lab[i]) > 0:
                    self.data[lab[i]]["all"] += 1
                    self.data[lab[i]]["sub"] += 1
                    result["all"] += 1
                    result["sub"] += 1
                result["lab"].insert(0, lab[i])
                result["rec"].insert(0, rec[j])
                i -= 1
                j -= 1
            elif error == "del":
                if len(lab[i]) > 0:
                    self.data[lab[i]]["all"] += 1
                    self.data[lab[i]]["del"] += 1
                    result["all"] += 1
                    result["del"] += 1
                result["lab"].insert(0, lab[i])
                result["rec"].insert(0, "")
                i -= 1
            elif error == "ins":
                if len(rec[j]) > 0:
                    self.data[rec[j]]["ins"] += 1
                    result["ins"] += 1
                result["lab"].insert(0, "")
                result["rec"].insert(0, rec[j])
                j -= 1
            elif error == "non":
                break
            else:
                break
        return result


    def overall(self):
        result = {"all": 0, "cor": 0, "sub": 0, "ins": 0, "del": 0}
        for token in self.data:
            result["all"] += self.data[token]["all"]
            result["cor"] += self.data[token]["cor"]
            result["sub"] += self.data[token]["sub"]
            result["ins"] += self.data[token]["ins"]
            result["del"] += self.data[token]["del"]
        return result

    def cluster(self, tokens):
        result = {"all": 0, "cor": 0, "sub": 0, "ins": 0, "del": 0}
        for token in tokens:
            if token in self.data:
                result["all"] += self.data[token]["all"]
                result["cor"] += self.data[token]["cor"]
                result["sub"] += self.data[token]["sub"]
                result["ins"] += self.data[token]["ins"]
                result["del"] += self.data[token]["del"]
        return result


class WerCerMetric(MetricBase):
    name = "wer"
    requires_ref_text = True
    supports_src = True

    def __init__(self, cfg, char_level=False):
        super().__init__(cfg)
        self.char_level = char_level
        self.case_sensitive = bool(cfg.get("case_sensitive", False))
        self.cluster_name = cfg.get("cluster")
        self.use_cluster_for_item = bool(cfg.get("use_cluster_for_item", False))
        self.hyp_text_path = cfg.get("hyp_text_path")
        self.report_output_path = cfg.get("report_output_path")
        self.report_output_path_src = cfg.get("report_output_path_src")
        self.report_csv_path = cfg.get("report_csv_path")
        self.report_csv_path_src = cfg.get("report_csv_path_src")
        self.report_include_alignment = bool(cfg.get("report_include_alignment", True))
        max_words = cfg.get("report_max_words_per_line")
        if max_words is None:
            max_words = sys.maxsize
        self.report_max_words_per_line = int(max_words)
        self.summary_clusters = bool(cfg.get("summary_clusters", True))
        self.hyp_map = {}
        self.calculators = {"gen": Calculator(), "src": Calculator()}
        self.cluster_tokens = {"gen": set(), "src": set()}
        self.report_paths = {"gen": None, "src": None}
        self.report_csv_paths = {"gen": None, "src": None}
        self.seen_roles = set()
        if char_level:
            self.name = "cer"

    def prepare(self, context):
        if self.hyp_text_path:
            self.hyp_map = _load_text_map(self.hyp_text_path)
        if not self.hyp_map:
            if not context.cfg.get("asr", {}).get("enabled", True):
                raise RuntimeError("ASR is disabled but WER/CER is enabled.")
        if "asr" not in context.resources:
            if context.cfg.get("asr", {}).get("enabled", True):
                context.resources["asr"] = build_asr(
                    context.cfg.get("asr", {}),
                    context.intermediate_dir,
                    context.device,
                    context.logger,
                    save_intermediate=context.save_intermediate,
                )
        if context.save_intermediate:
            base_dir = context.intermediate_dir / "wer"
            ensure_dir(base_dir)
            if self.report_output_path:
                self.report_paths["gen"] = Path(self.report_output_path).expanduser()
            else:
                self.report_paths["gen"] = base_dir / f"{self.name}_report.txt"
            if self.report_output_path_src:
                self.report_paths["src"] = Path(self.report_output_path_src).expanduser()
            else:
                self.report_paths["src"] = base_dir / f"{self.name}_src_report.txt"
            if self.report_csv_path:
                self.report_csv_paths["gen"] = Path(self.report_csv_path).expanduser()
            else:
                self.report_csv_paths["gen"] = base_dir / f"{self.name}_summary.csv"
            if self.report_csv_path_src:
                self.report_csv_paths["src"] = Path(self.report_csv_path_src).expanduser()
            else:
                self.report_csv_paths["src"] = base_dir / f"{self.name}_src_summary.csv"
            if context.cfg.get("output", {}).get("overwrite", True):
                for path in self.report_paths.values():
                    if path:
                        ensure_dir(path.parent)
                        path.write_text("", encoding="utf-8")

    def compute(self, item, context, role="gen"):
        if not item.ref_text:
            return {}
        hyp = ""
        if role == "gen" and self.hyp_map:
            hyp = self.hyp_map.get(item.utt_id, "")
        if not hyp:
            asr = context.resources.get("asr")
            hyp = asr.transcribe(item, role=role) if asr else ""
        lab = _tokenize(item.ref_text, self.char_level, self.case_sensitive)
        rec = _tokenize(hyp, self.char_level, self.case_sensitive)
        calc = self.calculators[role]
        stats = calc.calculate(lab, rec)
        self.seen_roles.add(role)
        if self.cluster_name:
            cluster_tokens = self.cluster_tokens[role]
            for token in lab + rec:
                if default_cluster(token) == self.cluster_name:
                    cluster_tokens.add(token)
        if self.use_cluster_for_item and self.cluster_name:
            local_calc = Calculator()
            local_calc.calculate(lab, rec)
            tokens = [t for t in lab + rec if default_cluster(t) == self.cluster_name]
            rate = _rate(local_calc.cluster(tokens))
        else:
            rate = _rate(stats)
        if context.save_intermediate:
            self._append_report(role, item.utt_id, stats)
        key = self.name if role == "gen" else f"{self.name}_src"
        return {key: rate}

    def summary(self):
        summary = {}
        for role, calc in self.calculators.items():
            if role not in self.seen_roles:
                continue
            overall_stats = calc.overall()
            if overall_stats["all"] == 0:
                continue
            prefix = f"overall_{self.name}" if role == "gen" else f"overall_{self.name}_src"
            if self.cluster_name:
                cluster_stats = calc.cluster(self.cluster_tokens[role])
                summary[prefix] = _rate(cluster_stats)
                summary[f"{prefix}_all"] = _rate(overall_stats)
            else:
                summary[prefix] = _rate(overall_stats)
            if self.summary_clusters:
                clusters = self._collect_clusters(calc)
                for cluster_id, tokens in clusters.items():
                    cluster_stats = calc.cluster(tokens)
                    summary[f"{prefix}_{cluster_id}"] = _rate(cluster_stats)
            if self.report_paths.get(role):
                self._append_report_summary(role, calc)
            if self.report_csv_paths.get(role):
                self._write_report_summary_csv(role, calc)
        return summary

    def _collect_clusters(self, calc):
        clusters = {}
        for token in calc.data:
            cluster_id = default_cluster(token)
            clusters.setdefault(cluster_id, set()).add(token)
        return clusters

    def _append_report(self, role, utt_id, stats):
        path = self.report_paths.get(role)
        if not path:
            return
        ensure_dir(path.parent)
        rate = _rate(stats)
        lines = [
            f"utt: {utt_id}",
            "WER: %4.2f %% N=%d C=%d S=%d D=%d I=%d"
            % (rate, stats["all"], stats["cor"], stats["sub"], stats["del"], stats["ins"]),
        ]
        if self.report_include_alignment:
            lines.extend(self._format_alignment(stats["lab"], stats["rec"]))
        lines.append("")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    def _append_report_summary(self, role, calc):
        path = self.report_paths.get(role)
        if not path:
            return
        clusters = self._collect_clusters(calc)
        overall_stats = calc.overall()
        lines = [
            "=" * 75,
            "",
            "Overall -> %4.2f %% N=%d C=%d S=%d D=%d I=%d"
            % (
                _rate(overall_stats),
                overall_stats["all"],
                overall_stats["cor"],
                overall_stats["sub"],
                overall_stats["del"],
                overall_stats["ins"],
            ),
        ]
        for cluster_id in sorted(clusters.keys()):
            stats = calc.cluster(clusters[cluster_id])
            lines.append(
                "%s -> %4.2f %% N=%d C=%d S=%d D=%d I=%d"
                % (
                    cluster_id,
                    _rate(stats),
                    stats["all"],
                    stats["cor"],
                    stats["sub"],
                    stats["del"],
                    stats["ins"],
                )
            )
        lines.append("=" * 75)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    def _write_report_summary_csv(self, role, calc):
        path = self.report_csv_paths.get(role)
        if not path:
            return
        ensure_dir(path.parent)
        clusters = self._collect_clusters(calc)
        rows = []
        overall_stats = calc.overall()
        rows.append({
            "cluster": "Overall",
            "wer": _rate(overall_stats),
            "N": overall_stats["all"],
            "C": overall_stats["cor"],
            "S": overall_stats["sub"],
            "D": overall_stats["del"],
            "I": overall_stats["ins"],
        })
        for cluster_id in sorted(clusters.keys()):
            stats = calc.cluster(clusters[cluster_id])
            rows.append({
                "cluster": cluster_id,
                "wer": _rate(stats),
                "N": stats["all"],
                "C": stats["cor"],
                "S": stats["sub"],
                "D": stats["del"],
                "I": stats["ins"],
            })
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["cluster", "wer", "N", "C", "S", "D", "I"])
            writer.writeheader()
            writer.writerows(rows)

    def _format_alignment(self, lab_tokens, rec_tokens):
        space = {"lab": [], "rec": []}
        for idx in range(len(lab_tokens)):
            len_lab = width(lab_tokens[idx])
            len_rec = width(rec_tokens[idx])
            length = max(len_lab, len_rec)
            space["lab"].append(length - len_lab)
            space["rec"].append(length - len_rec)
        upper_lab = len(lab_tokens)
        upper_rec = len(rec_tokens)
        lab1 = 0
        rec1 = 0
        lines = []
        while lab1 < upper_lab or rec1 < upper_rec:
            lines.append("lab: " + self._format_tokens(lab_tokens, space["lab"], lab1, upper_lab))
            lines.append("rec: " + self._format_tokens(rec_tokens, space["rec"], rec1, upper_rec))
            lines.append("")
            lab1 = min(upper_lab, lab1 + self.report_max_words_per_line)
            rec1 = min(upper_rec, rec1 + self.report_max_words_per_line)
        return lines

    def _format_tokens(self, tokens, pads, start, upper):
        end = min(upper, start + self.report_max_words_per_line)
        parts = []
        for idx in range(start, end):
            token = tokens[idx]
            pad = " " * pads[idx]
            parts.append(f"{token}{pad}")
        return " ".join(parts)
