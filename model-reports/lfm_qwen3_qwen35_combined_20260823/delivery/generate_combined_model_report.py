#!/usr/bin/env python3

import csv
import math
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from docx import Document
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader


TITLE = "LFM2.5-8B-A1B、Qwen3-8B 与 Qwen3.5 系列 Q4_0 端侧 CPU 性能与资源占用对比报告（OnePlus PLK110）"
ROOT = Path("/home/qwe/workspace/llama.cpp")
REPORT_ROOT = ROOT / "model-reports/lfm_qwen3_qwen35_combined_20260823"
WORK = REPORT_ROOT / "work"
DELIVERY = REPORT_ROOT / "delivery"
FIGURES = WORK / "figures"
RENDERED = WORK / "rendered"
OLD_NORMALIZED = ROOT / "model-reports/lfm25_qwen35_plk110_20260821/work/normalized_measurements.csv"
QWEN3_PREFILL_DIR = ROOT / "cpu-text-bench-logs/20260821-095519"
QWEN3_RESUME_DIR = ROOT / "cpu-text-bench-logs/20260821-174004-qwen3-8b-resume"
QWEN35_08_DIR = ROOT / "cpu-text-bench-logs/20260822-013033-remaining/qwen35-08b"
BUILDER = Path("/home/qwe/.codex/skills/external-model-capability-report/scripts/build_customer_docx.py")
SOFFICE_FALLBACK = Path("/home/qwe/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/soffice")

PLATFORM = "OnePlus PLK110 / Qualcomm SM8850 / Android 16"
LFM = "LFM2.5-8B-A1B（8.47B）"
QWEN3 = "Qwen3-8B（8.19B）"
QWEN35_9 = "Qwen3.5-9B（9.20B）"
QWEN35_08 = "Qwen3.5-0.8B（772.85M）"
MODELS = [LFM, QWEN3, QWEN35_9, QWEN35_08]
SHORT_NAMES = ["LFM2.5-8B-A1B", "Qwen3-8B", "Qwen3.5-9B", "Qwen3.5-0.8B"]
COLORS = ["#3370FF", "#1E5ED8", "#A9C4F5", "#C8CED7"]

TEXT = "#202124"
MUTED = "#667085"
GRID = "#E8EAEE"
AXIS = "#CDD2DA"

FIELDS = [
    "platform",
    "backend",
    "model",
    "parameter_class",
    "weight_quantization",
    "test_type",
    "length_tokens",
    "metric",
    "value",
    "unit",
    "direction",
    "statistic",
    "repetitions",
    "status",
    "source_file",
    "notes",
]

FONTS: dict[str, ImageFont.FreeTypeFont] = {}


def font_path(bold: bool = False) -> str:
    pattern = "Noto Sans CJK SC:style=Bold" if bold else "Noto Sans CJK SC"
    path = subprocess.check_output(["fc-match", "-f", "%{file}", pattern], text=True).strip()
    if not path:
        raise RuntimeError(f"font not found: {pattern}")
    return path


def configure_fonts() -> None:
    regular = font_path(False)
    bold = font_path(True)
    FONTS.update(
        {
            "title": ImageFont.truetype(bold, 47),
            "subtitle": ImageFont.truetype(regular, 25),
            "panel": ImageFont.truetype(bold, 28),
            "axis": ImageFont.truetype(regular, 22),
            "value": ImageFont.truetype(bold, 17),
            "legend": ImageFont.truetype(regular, 22),
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def new_row(
    model: str,
    parameter_class: str,
    test_type: str,
    length: int,
    metric: str,
    value: float,
    unit: str,
    direction: str,
    source: str,
    notes: str,
) -> dict[str, str]:
    return {
        "platform": PLATFORM,
        "backend": "CPU",
        "model": model,
        "parameter_class": parameter_class,
        "weight_quantization": "Q4_0",
        "test_type": test_type,
        "length_tokens": str(length),
        "metric": metric,
        "value": f"{value:.3f}",
        "unit": unit,
        "direction": direction,
        "statistic": "single_run_observed",
        "repetitions": "1",
        "status": "PASS",
        "source_file": source,
        "notes": notes,
    }


def append_model_rows(
    rows: list[dict[str, str]],
    model: str,
    parameter_class: str,
    perf_paths: list[Path],
    memory_paths: list[Path],
) -> None:
    perf_records: dict[tuple[str, int], dict[str, str]] = {}
    memory_records: dict[tuple[str, int], dict[str, str]] = {}
    for path in perf_paths:
        for record in read_tsv(path):
            perf_records[(record["case"], int(record["length"]))] = record
    for path in memory_paths:
        for record in read_tsv(path):
            memory_records[(record["case"], int(record["length"]))] = record

    for (case_name, length), record in sorted(perf_records.items(), key=lambda item: (item[0][0], item[0][1])):
        if case_name == "prefill_text":
            test_type = "prefill"
            metric = "prefill_throughput"
            value = float(record["prefill tok/s"])
            notes = "prompt evaluation; one generated token only closes the run"
        else:
            test_type = "long_generation_decode"
            metric = "long_generation_decode_throughput"
            value = float(record["decode tok/s"])
            notes = "128-token prompt followed by length_tokens generated tokens"
        rows.append(
            new_row(
                model,
                parameter_class,
                test_type,
                length,
                metric,
                value,
                "tokens/s",
                "higher_is_better",
                record["log"],
                notes,
            )
        )

    for (case_name, length), record in sorted(memory_records.items(), key=lambda item: (item[0][0], item[0][1])):
        test_type = "prefill" if case_name == "prefill_text" else "long_generation_decode"
        source = record["mem csv"]
        peak = float(record["peak/VmHWM"])
        state = float(record["state"])
        total_context = float(record["KV incl state"])
        attention_kv = total_context - state
        rows.extend(
            [
                new_row(
                    model,
                    parameter_class,
                    test_type,
                    length,
                    "process_peak_rss",
                    peak,
                    "MiB",
                    "lower_is_better",
                    source,
                    "process peak VmHWM",
                ),
                new_row(
                    model,
                    parameter_class,
                    test_type,
                    length,
                    "attention_kv_cache",
                    attention_kv,
                    "MiB",
                    "lower_is_better",
                    source,
                    "derived as KV incl state minus recurrent state",
                ),
                new_row(
                    model,
                    parameter_class,
                    test_type,
                    length,
                    "recurrent_state",
                    state,
                    "MiB",
                    "lower_is_better",
                    source,
                    "reported recurrent/state component",
                ),
                new_row(
                    model,
                    parameter_class,
                    test_type,
                    length,
                    "total_context_state",
                    total_context,
                    "MiB",
                    "lower_is_better",
                    source,
                    "source field: KV incl state",
                ),
            ]
        )


def build_normalized(output: Path) -> list[dict[str, str]]:
    reviewed = read_csv(OLD_NORMALIZED)
    rows = [row for row in reviewed if row["model"] in {LFM, QWEN35_9}]
    append_model_rows(
        rows,
        QWEN3,
        "8.19B",
        [QWEN3_PREFILL_DIR / "perf.tsv", QWEN3_RESUME_DIR / "perf.tsv"],
        [QWEN3_PREFILL_DIR / "memory.tsv", QWEN3_RESUME_DIR / "memory.tsv"],
    )
    append_model_rows(
        rows,
        QWEN35_08,
        "772.85M",
        [QWEN35_08_DIR / "perf.tsv"],
        [QWEN35_08_DIR / "memory.tsv"],
    )
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def metric_series(
    rows: list[dict[str, str]], model: str, test_type: str, metric: str
) -> tuple[list[int], list[float]]:
    selected = [
        row
        for row in rows
        if row["model"] == model
        and row["test_type"] == test_type
        and row["metric"] == metric
        and row["status"] == "PASS"
        and row["value"]
    ]
    selected.sort(key=lambda row: int(row["length_tokens"]))
    return [int(row["length_tokens"]) for row in selected], [float(row["value"]) for row in selected]


def value_at(
    rows: list[dict[str, str]], model: str, test_type: str, metric: str, length: int
) -> float:
    for row in rows:
        if (
            row["model"] == model
            and row["test_type"] == test_type
            and row["metric"] == metric
            and int(row["length_tokens"]) == length
            and row["status"] == "PASS"
        ):
            return float(row["value"])
    raise KeyError((model, test_type, metric, length))


def context_label(value: int) -> str:
    return f"{value // 1024}K" if value % 1024 == 0 else str(value)


def nice_max(value: float) -> float:
    if value <= 0:
        return 1.0
    power = 10 ** math.floor(math.log10(value))
    normalized = value / power
    step = 1 if normalized <= 1 else 2 if normalized <= 2 else 5 if normalized <= 5 else 10
    return step * power


def centered(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, font, fill: str) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((x - (box[2] - box[0]) / 2, y), text, font=font, fill=fill)


def chart_canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (2400, 1120), "white")
    draw = ImageDraw.Draw(image)
    draw.text((110, 40), title, font=FONTS["title"], fill=TEXT)
    draw.text((110, 112), subtitle, font=FONTS["subtitle"], fill=MUTED)
    return image, draw


def draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    cursor = x
    for label, color in zip(SHORT_NAMES, COLORS):
        draw.rounded_rectangle((cursor, y, cursor + 31, y + 22), radius=4, fill=color)
        draw.text((cursor + 42, y - 5), label, font=FONTS["legend"], fill=TEXT)
        box = draw.textbbox((0, 0), label, font=FONTS["legend"])
        cursor += 42 + (box[2] - box[0]) + 52


def draw_axes(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    maximum: float,
) -> None:
    left, top, right, bottom = bounds
    for tick in range(6):
        value = maximum * tick / 5
        y = bottom - (bottom - top) * tick / 5
        draw.line((left, y, right, y), fill=GRID, width=2)
        label = f"{value:.0f}" if maximum >= 100 else f"{value:.1f}"
        box = draw.textbbox((0, 0), label, font=FONTS["axis"])
        draw.text((left - 17 - (box[2] - box[0]), y - 13), label, font=FONTS["axis"], fill=MUTED)
    draw.line((left, top, left, bottom), fill=AXIS, width=2)
    draw.line((left, bottom, right, bottom), fill=AXIS, width=2)


def grouped_bar_chart(
    output: Path,
    title: str,
    subtitle: str,
    labels: list[str],
    series: list[list[float]],
    decimals: int,
) -> None:
    image, draw = chart_canvas(title, subtitle)
    draw_legend(draw, 150, 176)
    bounds = (155, 265, 2320, 920)
    maximum = nice_max(max(max(values) for values in series) * 1.19)
    draw_axes(draw, bounds, maximum)
    left, top, right, bottom = bounds
    group_width = (right - left) / len(labels)
    bar_width = group_width * 0.76 / len(series)
    for group_index, label in enumerate(labels):
        group_left = left + group_index * group_width + group_width * 0.12
        for series_index, values in enumerate(series):
            value = values[group_index]
            x0 = group_left + series_index * bar_width
            x1 = x0 + bar_width * 0.84
            y = bottom - (bottom - top) * value / maximum
            draw.rounded_rectangle((x0, y, x1, bottom), radius=5, fill=COLORS[series_index])
            centered(
                draw,
                (x0 + x1) / 2,
                max(top + 3, y - 28),
                f"{value:.{decimals}f}",
                FONTS["value"],
                TEXT,
            )
        centered(draw, left + group_index * group_width + group_width / 2, bottom + 22, label, FONTS["axis"], MUTED)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, dpi=(240, 240), optimize=True)


def line_chart(
    output: Path,
    title: str,
    subtitle: str,
    labels: list[str],
    series: list[list[float]],
) -> None:
    image, draw = chart_canvas(title, subtitle)
    draw_legend(draw, 150, 176)
    bounds = (165, 275, 2310, 915)
    maximum = nice_max(max(max(values) for values in series) * 1.18)
    draw_axes(draw, bounds, maximum)
    left, top, right, bottom = bounds
    step = (right - left) / (len(labels) - 1)
    label_offsets = [-55, -33, -11, 9]
    for series_index, values in enumerate(series):
        points = []
        for index, value in enumerate(values):
            x = left + index * step
            y = bottom - (bottom - top) * value / maximum
            points.append((x, y))
        draw.line(points, fill=COLORS[series_index], width=6, joint="curve")
        for (x, y), value in zip(points, values):
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=COLORS[series_index], outline="white", width=2)
            centered(
                draw,
                x,
                max(top + 2, min(bottom - 31, y + label_offsets[series_index])),
                f"{value:.0f}",
                FONTS["value"],
                TEXT,
            )
    for index, label in enumerate(labels):
        centered(draw, left + index * step, bottom + 22, label, FONTS["axis"], MUTED)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, dpi=(240, 240), optimize=True)


def rss_chart(
    output: Path,
    labels: list[str],
    prefill_series: list[list[float]],
    decode_series: list[list[float]],
) -> None:
    image, draw = chart_canvas(
        "四配置的进程 Peak RSS 对比",
        "单位：MiB；左侧为 Prefill，右侧为长文本连续生成；数值越低越节省内存",
    )
    draw_legend(draw, 150, 176)

    panels = [
        ((160, 310, 1110, 915), "Prefill Peak RSS", prefill_series),
        ((1370, 310, 2320, 915), "Decode Peak RSS", decode_series),
    ]
    for bounds, panel_title, series in panels:
        left, top, right, bottom = bounds
        draw.text((left, top - 55), panel_title, font=FONTS["panel"], fill=TEXT)
        maximum = nice_max(max(max(values) for values in series) * 1.2)
        draw_axes(draw, bounds, maximum)
        group_width = (right - left) / len(labels)
        bar_width = group_width * 0.78 / len(series)
        for group_index, label in enumerate(labels):
            group_left = left + group_index * group_width + group_width * 0.11
            for series_index, values in enumerate(series):
                value = values[group_index]
                x0 = group_left + series_index * bar_width
                x1 = x0 + bar_width * 0.84
                y = bottom - (bottom - top) * value / maximum
                draw.rounded_rectangle((x0, y, x1, bottom), radius=4, fill=COLORS[series_index])
                centered(draw, (x0 + x1) / 2, max(top + 2, y - 25), f"{value:.0f}", FONTS["value"], TEXT)
            centered(draw, left + group_index * group_width + group_width / 2, bottom + 20, label, FONTS["axis"], MUTED)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, dpi=(240, 240), optimize=True)


def make_charts(rows: list[dict[str, str]]) -> dict[str, Path]:
    labels = ["1K", "2K", "3K", "4K", "8K", "16K", "32K"]
    prefill = [metric_series(rows, model, "prefill", "prefill_throughput")[1] for model in MODELS]
    decode = [
        metric_series(rows, model, "long_generation_decode", "long_generation_decode_throughput")[1]
        for model in MODELS
    ]
    kv = [metric_series(rows, model, "prefill", "attention_kv_cache")[1] for model in MODELS]
    prefill_rss = [metric_series(rows, model, "prefill", "process_peak_rss")[1] for model in MODELS]
    decode_rss = [
        metric_series(rows, model, "long_generation_decode", "process_peak_rss")[1]
        for model in MODELS
    ]

    charts = {
        "prefill": FIGURES / "combined_prefill.png",
        "decode": FIGURES / "combined_decode.png",
        "kv": FIGURES / "combined_attention_kv_cache.png",
        "rss": FIGURES / "combined_peak_rss.png",
    }
    grouped_bar_chart(
        charts["prefill"],
        "四配置的 Prefill 吞吐对比",
        "单位：tokens/s；数值越高表示 Prompt 处理越快；单次观测",
        labels,
        prefill,
        2,
    )
    grouped_bar_chart(
        charts["decode"],
        "四配置的长文本连续生成吞吐对比",
        "单位：tokens/s；128-token Prompt 后连续生成对应长度；数值越高越好",
        labels,
        decode,
        2,
    )
    line_chart(
        charts["kv"],
        "四配置的 Attention KV Cache 对比",
        "单位：MiB；已扣除递归状态；数值越低越节省上下文内存",
        labels,
        kv,
    )
    rss_chart(charts["rss"], labels, prefill_rss, decode_rss)
    return charts


def values_text(rows: list[dict[str, str]], test_type: str, metric: str, length: int, decimals: int = 2) -> str:
    values = [value_at(rows, model, test_type, metric, length) for model in MODELS]
    return "、".join(f"{name} **{value:.{decimals}f}**" for name, value in zip(SHORT_NAMES, values))


def make_markdown(path: Path, rows: list[dict[str, str]], charts: dict[str, Path]) -> None:
    prefill_1k = values_text(rows, "prefill", "prefill_throughput", 1024)
    prefill_32k = values_text(rows, "prefill", "prefill_throughput", 32768)
    decode_1k = values_text(rows, "long_generation_decode", "long_generation_decode_throughput", 1024)
    decode_32k = values_text(rows, "long_generation_decode", "long_generation_decode_throughput", 32768)
    kv_32k = values_text(rows, "prefill", "attention_kv_cache", 32768, 0)
    prefill_rss_32k = values_text(rows, "prefill", "process_peak_rss", 32768)
    decode_rss_32k = values_text(rows, "long_generation_decode", "process_peak_rss", 32768)

    markdown = f"""# {TITLE}

　　**总结：**本次报告将 OnePlus PLK110（Qualcomm SM8850）CPU-only 路径下四个 Q4_0 配置的全部有效运行结果统一比较。32K 档位的 Prefill 吞吐依次为：{prefill_32k} tokens/s；长文本连续生成吞吐依次为：{decode_32k} tokens/s。四配置参数规模与架构不同，数值用于呈现各自在相同设备和测试方法下的实测运行特征，不计算跨模型提升率或排名。

---

# 一、核心结论

本报告按指标统一展示四个配置，不再分别展开模型章节。核心结果覆盖 1K-32K Prefill、长文本连续生成、Attention KV Cache 与进程 Peak RSS。

- **Prefill：**1K 档位为 {prefill_1k} tokens/s；32K 档位为 {prefill_32k} tokens/s。
- **Decode：**1K 连续生成为 {decode_1k} tokens/s；32K 连续生成为 {decode_32k} tokens/s。
- **Attention KV Cache：**32K 档位为 {kv_32k} MiB；该指标已扣除各模型单独分配的递归状态。
- **进程 Peak RSS：**32K Prefill 为 {prefill_rss_32k} MiB；32K Decode 为 {decode_rss_32k} MiB。

---

# 二、测试范围与指标说明

本次报告汇总四个已完成配置的单次运行记录，并在每项指标中使用同一组 1K、2K、3K、4K、8K、16K、32K 档位进行并排展示。

**硬件与系统环境：**以下信息用于说明测试依赖的平台和运行路径。

- 测试平台：OnePlus PLK110，Qualcomm SM8850，Android 16，arm64-v8a。
- 后端路径：llama.cpp CPU-only，6 线程，CPU Mask `0xfc`，`NGL=0`，`DEVICE=none`。

**部署实验设置与指标定义：**以下信息用于说明配置范围和统计方式。

- 模型配置：LFM2.5-8B-A1B（8.47B）、Qwen3-8B（8.19B）、Qwen3.5-9B（9.20B）、Qwen3.5-0.8B（772.85M），权重均为 Q4_0。
- 推理设置：关闭 Flash Attention、关闭 KV 与算子卸载，`batch-size=2048`，`ubatch-size=512`。
- Prefill：输入对应长度 Prompt，吞吐取 llama.cpp prompt evaluation 统计。
- Decode：约 128-token Prompt 后连续生成对应长度文本，速度为整个生成区间平均值。
- 资源：Peak RSS 取 Android VmHWM；Attention KV Cache 为总 Context 状态扣除递归状态后的注意力缓存。
- 统计方式：每个配置、阶段和长度档位执行 1 次；不同配置来自连续测试批次，长时间运行期间的供电和温控状态可能影响观测值。

<pagebreak/>

# 三、Prefill 吞吐统一对比

Prefill 吞吐反映模型处理输入 Prompt 并建立上下文状态的速度。下图将四个配置在全部七个 Prompt 档位中并排展示。

![四配置 Prefill 吞吐对比]({charts['prefill']})

图 1：LFM2.5、Qwen3 与 Qwen3.5 系列 Prefill 吞吐统一对比

在 1K 档位，四配置观测值为 {prefill_1k} tokens/s；在 32K 档位为 {prefill_32k} tokens/s。随着 Prompt 从 1K 增至 32K，四条结果序列整体均呈下降趋势。

　　**本节结论：**四配置均完成 32K Prefill，32K 观测范围为 **2.94-50.61 tokens/s**；该范围描述不同规模与架构配置的实测分布，不表示同模型方案间的提升关系。

<pagebreak/>

# 四、长文本连续生成吞吐统一对比

Decode 指标表示约 128-token Prompt 后连续生成对应长度文本的区间平均吞吐，生成长度增加时，已累积上下文同步增长。

![四配置 Decode 吞吐对比]({charts['decode']})

图 2：LFM2.5、Qwen3 与 Qwen3.5 系列长文本连续生成吞吐统一对比

在 1K 连续生成档位，四配置观测值为 {decode_1k} tokens/s；在 32K 档位为 {decode_32k} tokens/s。四配置在更长生成区间内的平均吞吐均低于各自 1K 结果。

　　**本节结论：**四配置均完成 32K 连续生成，32K 区间平均吞吐范围为 **0.73-21.12 tokens/s**；该指标不等同于固定 Context 深度下生成 TG128 的瞬时速度。

<pagebreak/>

# 五、Attention KV Cache 统一对比

Attention KV Cache 表示注意力历史状态随上下文长度增长产生的内存占用。为保持语义一致，下图从总 Context 状态中扣除了各配置的递归状态。

![四配置 Attention KV Cache 对比]({charts['kv']})

图 3：LFM2.5、Qwen3 与 Qwen3.5 系列 Attention KV Cache 统一对比

在 32K 档位，Attention KV Cache 为 {kv_32k} MiB。LFM2.5-8B-A1B 与 Qwen3.5-0.8B 的 Attention KV Cache 在七个档位中数值相同，但两者递归状态和模型结构并不相同。

　　**本节结论：**四配置的 Attention KV Cache 均随长度增长，32K 观测范围为 **387-4644 MiB**；Context 状态规模需与模型结构及递归状态共同理解。

<pagebreak/>

# 六、进程 Peak RSS 统一对比

Peak RSS 是进程生命周期内的内存高水位，包含模型加载、计算缓冲、KV Cache 与其他运行状态。下图在同一画布中分别比较 Prefill 和 Decode。

![四配置 Peak RSS 对比]({charts['rss']})

图 4：LFM2.5、Qwen3 与 Qwen3.5 系列 Prefill/Decode Peak RSS 统一对比

在 32K 档位，Prefill Peak RSS 为 {prefill_rss_32k} MiB；Decode Peak RSS 为 {decode_rss_32k} MiB。Peak RSS 不等于 KV Cache 单项占用，两者不可直接互换。

　　**本节结论：**32K Prefill Peak RSS 的观测范围为 **2018.27-10054.95 MiB**，32K Decode 为 **1192.35-9045.38 MiB**；结果反映各配置在本次运行批次中的进程内存高水位。

<pagebreak/>

# 七、综合结论与适用范围

综合四配置的 Prefill、长文本连续生成、Attention KV Cache 与 Peak RSS 结果，所有配置均形成 1K-32K 完整记录。统一图表显示，随着处理或生成长度增加，各配置均出现吞吐下降，同时 Attention KV Cache 持续增长；32K Prefill 吞吐范围为 **2.94-50.61 tokens/s**，32K Decode 范围为 **0.73-21.12 tokens/s**，32K Attention KV Cache 范围为 **387-4644 MiB**。

　　**适用条件：**结果基于 OnePlus PLK110、Qualcomm SM8850、Android 16、6 CPU 线程、Q4_0、关闭 Flash Attention 和单次运行统计。模型参数规模、架构和运行批次不同，本文仅比较实测绝对值与随长度变化的趋势，不计算跨模型提升率或严格排名。
"""
    path.write_text(markdown, encoding="utf-8")


def write_source_notes(path: Path, charts: dict[str, Path]) -> None:
    lines = [
        "# Source notes",
        "",
        "## Sources",
        "",
        f"- Reviewed LFM2.5 and Qwen3.5-9B measurements: {OLD_NORMALIZED}",
        f"- Qwen3-8B 1k-16k Prefill: {QWEN3_PREFILL_DIR}",
        f"- Qwen3-8B 32k Prefill and 1k-32k Decode: {QWEN3_RESUME_DIR}",
        f"- Qwen3.5-0.8B complete run: {QWEN35_08_DIR}",
        "",
        "## Mapping",
        "",
        "- Public names use exact parameter counts from llama.cpp logs: 8.47B, 8.19B, 9.20B, and 772.85M.",
        "- Prefill throughput maps from perf.tsv prefill tok/s.",
        "- Long-generation Decode maps from perf.tsv decode tok/s.",
        "- Peak RSS maps from memory.tsv peak/VmHWM.",
        "- Attention KV Cache equals KV incl state minus recurrent state.",
        "- Interrupted Qwen3-8B attempts and nan timing rows are excluded; only the later complete PASS records are used.",
        "- No cross-model speedup, reduction, or ranking is calculated because parameter size and architecture differ.",
        "",
        "## Chart map",
        "",
    ]
    for name, chart in charts.items():
        lines.append(f"- {name}: {chart}")
    lines.extend(
        [
            "",
            "## QA",
            "",
            "- All chart values are generated from normalized_measurements.csv.",
            "- Every valid bar and line point has a numeric label.",
            "- All four configurations use the same context labels and stable color mapping across figures.",
            "- DOCX uses the shared customer builder and PDF is converted from the validated DOCX.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_docx_and_pdf(markdown: Path, docx_path: Path, pdf_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(BUILDER), str(markdown), str(docx_path), "--title", TITLE],
        check=True,
    )
    soffice = shutil.which("soffice")
    if not soffice and SOFFICE_FALLBACK.is_file():
        soffice = str(SOFFICE_FALLBACK)
    if not soffice:
        raise RuntimeError("LibreOffice soffice was not found")

    convert_dir = WORK / "pdf_convert"
    profile_dir = WORK / "libreoffice_profile"
    shutil.rmtree(convert_dir, ignore_errors=True)
    shutil.rmtree(profile_dir, ignore_errors=True)
    convert_dir.mkdir(parents=True)
    profile_dir.mkdir(parents=True)
    subprocess.run(
        [
            soffice,
            "--headless",
            f"-env:UserInstallation=file://{profile_dir}",
            "--convert-to",
            "pdf",
            "--outdir",
            str(convert_dir),
            str(docx_path),
        ],
        check=True,
    )
    converted = convert_dir / f"{TITLE}.pdf"
    if not converted.is_file():
        raise RuntimeError("LibreOffice did not create the expected PDF")
    shutil.copy2(converted, pdf_path)


def validate_outputs(docx_path: Path, pdf_path: Path, script_path: Path) -> None:
    with zipfile.ZipFile(docx_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("DOCX package validation failed")
        numbering = archive.read("word/numbering.xml").decode("utf-8")
        if "3370FF" not in numbering or "•" not in numbering:
            raise ValueError("DOCX native blue bullet definition is missing")
    document = Document(docx_path)
    if document.core_properties.title != TITLE:
        raise ValueError("DOCX title metadata mismatch")
    reader = PdfReader(str(pdf_path))
    if not reader.pages:
        raise ValueError("PDF has no pages")
    if (reader.metadata or {}).get("/Title", "") != TITLE:
        raise ValueError("PDF title metadata mismatch")
    expected = {docx_path.name, pdf_path.name, script_path.name}
    actual = {item.name for item in DELIVERY.iterdir() if item.is_file()}
    if actual != expected:
        raise ValueError(f"delivery directory mismatch: {sorted(actual)}")


def render_pdf(pdf_path: Path) -> None:
    shutil.rmtree(RENDERED, ignore_errors=True)
    RENDERED.mkdir(parents=True)
    subprocess.run(
        ["pdftoppm", "-png", "-r", "120", str(pdf_path), str(RENDERED / "page")],
        check=True,
    )
    pages = sorted(RENDERED.glob("page-*.png"))
    if not pages:
        raise RuntimeError("PDF rendering produced no pages")
    thumbs = []
    for page in pages:
        image = Image.open(page).convert("RGB")
        image.thumbnail((360, 510))
        thumbs.append(image.copy())
    columns = 3
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (columns * 380, rows * 530), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % columns) * 380 + 10, (index // columns) * 530 + 10))
    sheet.save(RENDERED / "contact_sheet.png")


def main() -> None:
    configure_fonts()
    WORK.mkdir(parents=True, exist_ok=True)
    DELIVERY.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    for existing in DELIVERY.iterdir():
        if existing.is_file() and existing.resolve() != script_path:
            existing.unlink()

    normalized = WORK / "normalized_measurements.csv"
    source_notes = WORK / "source_notes.md"
    report_md = WORK / "report.md"
    rows = build_normalized(normalized)
    charts = make_charts(rows)
    make_markdown(report_md, rows, charts)
    write_source_notes(source_notes, charts)

    docx_path = DELIVERY / f"{TITLE}.docx"
    pdf_path = DELIVERY / f"{TITLE}.pdf"
    build_docx_and_pdf(report_md, docx_path, pdf_path)
    validate_outputs(docx_path, pdf_path, script_path)
    render_pdf(pdf_path)
    print(docx_path)
    print(pdf_path)
    print(RENDERED / "contact_sheet.png")


if __name__ == "__main__":
    main()
