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


TITLE = "LFM2.5-8B-A1B、Qwen3.5-9B 与 Qwen3-8B Q4_0 端侧 CPU 测试阶段总结报告（OnePlus PLK110）"
ROOT = Path("/home/qwe/workspace/llama.cpp")
REPORT_ROOT = ROOT / "model-reports/android_q4_0_stage_20260821"
WORK = REPORT_ROOT / "work"
DELIVERY = REPORT_ROOT / "delivery"
FIGURES = WORK / "figures"
RENDERED = WORK / "rendered"
OLD_NORMALIZED = ROOT / "model-reports/lfm25_qwen35_plk110_20260821/work/normalized_measurements.csv"
QWEN3_DIR = ROOT / "cpu-text-bench-logs/20260821-095519"
BUILDER = Path("/home/qwe/.codex/skills/external-model-capability-report/scripts/build_customer_docx.py")
SOFFICE_FALLBACK = Path("/home/qwe/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/soffice")

PLATFORM = "OnePlus PLK110 / Qualcomm SM8850 / Android 16"
LFM = "LFM2.5-8B-A1B（8.47B）"
QWEN35 = "Qwen3.5-9B（9.20B）"
QWEN3 = "Qwen3-8B（8.19B）"

BLUE = "#3370FF"
DARK_BLUE = "#1E5ED8"
LIGHT_BLUE = "#A9C4F5"
GRAY = "#C8CED7"
TEXT = "#202124"
MUTED = "#667085"
GRID = "#E8EAEE"
AXIS = "#CDD2DA"


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
            "title": ImageFont.truetype(bold, 48),
            "subtitle": ImageFont.truetype(regular, 25),
            "panel": ImageFont.truetype(bold, 29),
            "axis": ImageFont.truetype(regular, 23),
            "value": ImageFont.truetype(bold, 20),
            "legend": ImageFont.truetype(regular, 23),
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def measurement_row(
    *,
    test_type: str,
    length: int,
    metric: str,
    value: str,
    unit: str,
    direction: str,
    status: str,
    source: str,
    notes: str,
) -> dict[str, str]:
    return {
        "platform": PLATFORM,
        "backend": "CPU",
        "model": QWEN3,
        "parameter_class": "8.19B",
        "weight_quantization": "Q4_0",
        "test_type": test_type,
        "length_tokens": str(length),
        "metric": metric,
        "value": value,
        "unit": unit,
        "direction": direction,
        "statistic": "single_run_observed",
        "repetitions": "1",
        "status": status,
        "source_file": source,
        "notes": notes,
    }


def build_normalized_data(output: Path, runtime_observations: Path) -> list[dict[str, str]]:
    rows = read_csv(OLD_NORMALIZED)
    perf = read_tsv(QWEN3_DIR / "perf.tsv")
    memory = {int(row["length"]): row for row in read_tsv(QWEN3_DIR / "memory.tsv")}

    for perf_row in perf:
        length = int(perf_row["length"])
        mem_row = memory[length]
        log_path = perf_row["log"]
        mem_path = mem_row["mem csv"]
        rows.append(
            measurement_row(
                test_type="prefill",
                length=length,
                metric="prefill_throughput",
                value=f'{float(perf_row["prefill tok/s"]):.3f}',
                unit="tokens/s",
                direction="higher_is_better",
                status="PASS",
                source=log_path,
                notes="prompt evaluation; one generated token only closes the run",
            )
        )
        rows.append(
            measurement_row(
                test_type="prefill",
                length=length,
                metric="process_peak_rss",
                value=f'{float(mem_row["peak/VmHWM"]):.3f}',
                unit="MiB",
                direction="lower_is_better",
                status="PASS",
                source=mem_path,
                notes="prefill process peak VmHWM",
            )
        )
        rows.append(
            measurement_row(
                test_type="prefill",
                length=length,
                metric="attention_kv_cache",
                value=f'{float(mem_row["KV incl state"]):.3f}',
                unit="MiB",
                direction="lower_is_better",
                status="PASS",
                source=mem_path,
                notes="state component is 0.00 MiB, so KV incl state equals Attention KV Cache",
            )
        )
        rows.append(
            measurement_row(
                test_type="prefill",
                length=length,
                metric="recurrent_state",
                value=f'{float(mem_row["state"]):.3f}',
                unit="MiB",
                direction="lower_is_better",
                status="PASS",
                source=mem_path,
                notes="reported recurrent/state component",
            )
        )
        rows.append(
            measurement_row(
                test_type="prefill",
                length=length,
                metric="total_context_state",
                value=f'{float(mem_row["KV incl state"]):.3f}',
                unit="MiB",
                direction="lower_is_better",
                status="PASS",
                source=mem_path,
                notes="source field: KV incl state",
            )
        )

    interrupted_log = QWEN3_DIR / "Qwen3-8B_m32897_prefill_text_32768_r1.log"
    rows.append(
        measurement_row(
            test_type="prefill",
            length=32768,
            metric="prefill_run_status",
            value="",
            unit="status",
            direction="neutral",
            status="INTERRUPTED",
            source=str(interrupted_log),
            notes="user interruption after about 215.59 minutes; last completed n_past marker was 28672; no valid prompt throughput",
        )
    )

    runtime_rows = [
        ("battery_level", "1", "percent", "neutral", "battery level during the 32k run"),
        ("battery_temperature", "34.4", "degC", "neutral", "battery temperature during the 32k run"),
        ("usb_max_charging_power", "2.5", "W", "higher_is_better", "5 V x 0.5 A USB charging broadcast limit"),
        ("cpu_policy0_frequency_limit", "2.2272", "GHz", "higher_is_better", "hardware maximum 3.6288 GHz"),
        ("cpu_policy6_frequency_limit", "1.6320", "GHz", "higher_is_better", "hardware maximum 4.6080 GHz"),
    ]
    for metric, value, unit, direction, notes in runtime_rows:
        rows.append(
            measurement_row(
                test_type="runtime_observation",
                length=32768,
                metric=metric,
                value=value,
                unit=unit,
                direction=direction,
                status="OBSERVED",
                source=str(runtime_observations),
                notes=notes,
            )
        )

    fieldnames = list(rows[0].keys())
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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


def context_label(value: int) -> str:
    if value % 1024 == 0:
        return f"{value // 1024}K"
    return str(value)


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
    draw.text((110, 42), title, font=FONTS["title"], fill=TEXT)
    draw.text((110, 116), subtitle, font=FONTS["subtitle"], fill=MUTED)
    return image, draw


def draw_legend(draw: ImageDraw.ImageDraw, items: list[tuple[str, str]], x: int, y: int) -> None:
    cursor = x
    for label, color in items:
        draw.rounded_rectangle((cursor, y, cursor + 34, y + 24), radius=5, fill=color)
        draw.text((cursor + 46, y - 5), label, font=FONTS["legend"], fill=TEXT)
        box = draw.textbbox((0, 0), label, font=FONTS["legend"])
        cursor += 46 + (box[2] - box[0]) + 65


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
        draw.text((left - 18 - (box[2] - box[0]), y - 14), label, font=FONTS["axis"], fill=MUTED)
    draw.line((left, top, left, bottom), fill=AXIS, width=2)
    draw.line((left, bottom, right, bottom), fill=AXIS, width=2)


def draw_bar_panel(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    labels: list[str],
    values: list[float],
    panel_title: str,
    decimals: int = 2,
) -> None:
    left, top, right, bottom = bounds
    maximum = nice_max(max(values) * 1.22)
    draw.text((left, top - 58), panel_title, font=FONTS["panel"], fill=TEXT)
    draw_axes(draw, bounds, maximum)
    span = (right - left) / len(values)
    width = span * 0.54
    for index, (label, value) in enumerate(zip(labels, values)):
        x0 = left + index * span + (span - width) / 2
        x1 = x0 + width
        y = bottom - (bottom - top) * value / maximum
        draw.rounded_rectangle((x0, y, x1, bottom), radius=7, fill=BLUE)
        centered(draw, (x0 + x1) / 2, max(top + 2, y - 31), f"{value:.{decimals}f}", FONTS["value"], TEXT)
        centered(draw, (x0 + x1) / 2, bottom + 20, label, FONTS["axis"], MUTED)


def draw_line_panel(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    labels: list[str],
    values: list[float],
    panel_title: str,
) -> None:
    left, top, right, bottom = bounds
    maximum = nice_max(max(values) * 1.22)
    draw.text((left, top - 58), panel_title, font=FONTS["panel"], fill=TEXT)
    draw_axes(draw, bounds, maximum)
    step = (right - left) / (len(values) - 1)
    points = []
    for index, value in enumerate(values):
        x = left + index * step
        y = bottom - (bottom - top) * value / maximum
        points.append((x, y))
    draw.line(points, fill=DARK_BLUE, width=7, joint="curve")
    for (x, y), label, value in zip(points, labels, values):
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=BLUE, outline="white", width=3)
        centered(draw, x, max(top + 2, y - 38), f"{value:.2f}", FONTS["value"], TEXT)
        centered(draw, x, bottom + 20, label, FONTS["axis"], MUTED)


def draw_grouped_panel(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    labels: list[str],
    first: list[float],
    second: list[float] | None,
    panel_title: str,
    first_label: str,
    second_label: str | None,
    decimals: int = 0,
) -> None:
    left, top, right, bottom = bounds
    all_values = first + (second or [])
    maximum = nice_max(max(all_values) * 1.25)
    draw.text((left, top - 58), panel_title, font=FONTS["panel"], fill=TEXT)
    draw_axes(draw, bounds, maximum)
    span = (right - left) / len(labels)
    width = span * (0.29 if second else 0.52)
    for index, label in enumerate(labels):
        center = left + index * span + span / 2
        pairs = [(first[index], BLUE, center - width if second else center - width / 2)]
        if second:
            pairs.append((second[index], GRAY, center))
        for value, color, x0 in pairs:
            x1 = x0 + width
            y = bottom - (bottom - top) * value / maximum
            draw.rounded_rectangle((x0, y, x1, bottom), radius=6, fill=color)
            centered(draw, (x0 + x1) / 2, max(top + 2, y - 30), f"{value:.{decimals}f}", FONTS["value"], TEXT)
        centered(draw, center, bottom + 20, label, FONTS["axis"], MUTED)
    legend = [(first_label, BLUE)]
    if second and second_label:
        legend.append((second_label, GRAY))
    draw_legend(draw, legend, left, top - 104)


def save_chart(image: Image.Image, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, dpi=(240, 240), optimize=True)


def make_speed_chart(
    rows: list[dict[str, str]], model: str, short_name: str, output: Path, include_decode: bool
) -> None:
    p_lengths, prefill = metric_series(rows, model, "prefill", "prefill_throughput")
    image, draw = chart_canvas(
        f"{short_name} 的端侧 CPU 推理速度",
        "单位：tokens/s；数值越高越好；单次观测",
    )
    if include_decode:
        d_lengths, decode = metric_series(rows, model, "long_generation_decode", "long_generation_decode_throughput")
        draw_bar_panel(draw, (170, 300, 1110, 925), [context_label(v) for v in p_lengths], prefill, "Prefill")
        draw_bar_panel(draw, (1370, 300, 2310, 925), [context_label(v) for v in d_lengths], decode, "长文本连续生成")
        draw_legend(draw, [(short_name, BLUE)], 170, 188)
    else:
        draw_bar_panel(draw, (210, 280, 2290, 925), [context_label(v) for v in p_lengths], prefill, "Prefill")
        draw_legend(draw, [(short_name, BLUE)], 210, 170)
    save_chart(image, output)


def make_resource_chart(
    rows: list[dict[str, str]], model: str, short_name: str, output: Path, include_decode: bool
) -> None:
    lengths, kv = metric_series(rows, model, "prefill", "attention_kv_cache")
    rss_lengths, prefill_rss = metric_series(rows, model, "prefill", "process_peak_rss")
    image, draw = chart_canvas(
        f"{short_name} 的 KV Cache 与内存占用",
        "单位：MiB；KV Cache 与 Peak RSS 数值越低越节省资源",
    )
    draw_line_panel(draw, (170, 300, 1110, 925), [context_label(v) for v in lengths], kv, "Attention KV Cache")
    if include_decode:
        _, decode_rss = metric_series(rows, model, "long_generation_decode", "process_peak_rss")
        draw_grouped_panel(
            draw,
            (1370, 300, 2310, 925),
            [context_label(v) for v in rss_lengths],
            prefill_rss,
            decode_rss,
            "进程 Peak RSS",
            "Prefill",
            "连续生成",
        )
    else:
        draw_grouped_panel(
            draw,
            (1370, 300, 2310, 925),
            [context_label(v) for v in rss_lengths],
            prefill_rss,
            None,
            "进程 Peak RSS",
            "Prefill",
            None,
        )
    save_chart(image, output)


def make_frequency_chart(output: Path) -> None:
    labels = ["CPU policy0", "CPU policy6"]
    limited = [2.2272, 1.6320]
    hardware = [3.6288, 4.6080]
    image, draw = chart_canvas(
        "Qwen3-8B 32K 运行期间的 CPU 频率限制",
        "单位：GHz；蓝色为系统当时允许的最高频率",
    )
    draw_grouped_panel(
        draw,
        (260, 290, 2260, 920),
        labels,
        limited,
        hardware,
        "运行时上限与硬件最高频率",
        "运行时频率上限",
        "硬件最高频率",
        2,
    )
    save_chart(image, output)


def value_at(rows: list[dict[str, str]], model: str, test_type: str, metric: str, length: int) -> float:
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


def make_markdown(path: Path, rows: list[dict[str, str]], charts: dict[str, Path]) -> None:
    q3_prefill = [value_at(rows, QWEN3, "prefill", "prefill_throughput", length) for length in (1024, 8192, 16384)]
    q3_kv = [value_at(rows, QWEN3, "prefill", "attention_kv_cache", length) for length in (1024, 8192, 16384)]
    q3_rss = [value_at(rows, QWEN3, "prefill", "process_peak_rss", length) for length in (1024, 8192, 16384)]

    common_prefill_rows = []
    for length in (1024, 8192, 16384):
        common_prefill_rows.append(
            f"- **{context_label(length)}：**LFM2.5-8B-A1B {value_at(rows, LFM, 'prefill', 'prefill_throughput', length):.2f} tokens/s；"
            f"Qwen3.5-9B {value_at(rows, QWEN35, 'prefill', 'prefill_throughput', length):.2f} tokens/s；"
            f"Qwen3-8B {value_at(rows, QWEN3, 'prefill', 'prefill_throughput', length):.2f} tokens/s。"
        )

    markdown = f"""# {TITLE}

　　**总结：**本次报告基于 OnePlus PLK110（Qualcomm SM8850）CPU-only 路径的 Q4_0 模型测试结果整理。LFM2.5-8B-A1B（8.47B）与 Qwen3.5-9B（9.20B）均形成 1K-32K Prefill、长文本连续生成、Attention KV Cache 与 Peak RSS 完整结果；Qwen3-8B（8.19B）当前有效范围为 1K-16K Prefill，其吞吐由 **44.31 tokens/s** 变化至 **7.51 tokens/s**，Attention KV Cache 由 **180.00 MiB** 增至 **2340.00 MiB**。Qwen3-8B 32K Prefill 在运行约 **3小时35分**、推进至 **28672 tokens** 后中断，当时设备电量为 **1%**，CPU policy6 频率上限为 **1.632 GHz**。

---

# 一、工作背景与目标

本次阶段性验证用于建立三款 Q4_0 纯文本模型在 Android CPU 路径上的速度、KV Cache 与进程内存基线，并记录长上下文运行期间对结果有效性有直接影响的设备状态。

- **模型范围：**LFM2.5-8B-A1B（8.47B）、Qwen3.5-9B（9.20B）与 Qwen3-8B（8.19B）。
- **测试目标：**测量 Prefill、长文本连续生成、Attention KV Cache 和进程 Peak RSS 随长度增加的变化。
- **一致性要求：**三款模型均采用 Q4_0 权重、llama.cpp CPU-only 路径、6 CPU 线程和关闭 Flash Attention 的设置。

---

# 二、核心结论

当前结果同时体现了已完成模型的 32K 运行能力，以及 Qwen3-8B 在供电与频率受限状态下的有效测试边界。

- **LFM2.5-8B-A1B：**1K-32K Prefill 为 **100.83-29.59 tokens/s**，长文本连续生成平均吞吐为 **37.75-13.67 tokens/s**；32K Attention KV Cache 为 **387.00 MiB**。
- **Qwen3.5-9B：**1K-32K Prefill 为 **30.90-12.67 tokens/s**，长文本连续生成平均吞吐为 **7.83-1.54 tokens/s**；32K Attention KV Cache 为 **1032.00 MiB**。
- **Qwen3-8B：**1K、8K、16K Prefill 分别为 **{q3_prefill[0]:.2f}、{q3_prefill[1]:.2f}、{q3_prefill[2]:.2f} tokens/s**；对应 Attention KV Cache 为 **{q3_kv[0]:.00f}、{q3_kv[1]:.00f}、{q3_kv[2]:.00f} MiB**。
- **运行边界：**Qwen3-8B 32K 运行时，USB 供电广播上限约 **2.5 W**，CPU policy6 的频率上限仅为硬件最高频率的 **35.4%**，该次中断日志不计入吞吐结果。

---

# 三、测试范围与指标说明

本次报告覆盖 OnePlus PLK110 上三款模型已经形成有效记录的长度档位，按模型自身的长度变化解释结果，不计算不同参数规模和架构之间的提升率。

**硬件与系统环境：**以下信息用于说明测试依赖的平台和运行路径。

- 测试平台：OnePlus PLK110，Qualcomm SM8850，Android 16，arm64-v8a。
- 后端路径：llama.cpp CPU-only，6 线程，CPU Mask `0xfc`，`NGL=0`，`DEVICE=none`。

**部署实验设置与指标定义：**以下信息用于说明模型配置与测量含义。

- 量化配置：三款模型权重均为 Q4_0；Flash Attention 关闭。
- Prefill：输入对应长度的文本 Prompt，吞吐取 llama.cpp prompt evaluation 统计。
- 长文本连续生成：约 128-token Prompt 后连续生成对应长度文本，速度为整个生成区间平均值。
- 资源指标：Peak RSS 取 Android `/proc/<PID>/status` 的 VmHWM；Attention KV Cache 从 llama.cpp Context 内存日志提取。
- 统计方式：每个模型、阶段和长度档位执行 1 次，结果为单次观测值。

---

# 四、推理速度表现：Prefill 与长文本连续生成

Prefill 吞吐反映模型处理输入 Prompt 并建立上下文状态的速度；长文本连续生成表示约 128-token Prompt 后持续生成对应长度文本的区间平均吞吐。两项指标均为数值越高表示处理速度越快。

## 4.1 LFM2.5-8B-A1B

LFM2.5-8B-A1B 的有效结果覆盖 1K-32K，2K 与 3K 之间存在单次运行波动，长 Prompt 下整体吞吐下降。

![LFM2.5-8B-A1B 推理速度]({charts['lfm_speed']})

图 1：LFM2.5-8B-A1B Prefill 与长文本连续生成吞吐

在 1K、8K 和 32K Prompt 下，Prefill 分别为 **100.83、69.04 和 29.59 tokens/s**；对应长度的连续生成平均吞吐为 **37.75、24.47 和 13.67 tokens/s**。

　　**本节结论：**LFM2.5-8B-A1B 在本次设备上完成 32K Prompt 与 32K 连续生成，对应吞吐为 **29.59 tokens/s** 和 **13.67 tokens/s**。

<pagebreak/>

## 4.2 Qwen3.5-9B

Qwen3.5-9B 的 Prefill 随 Prompt 长度增加连续下降，32K Prompt 的单次处理耗时为 2586.24 秒。

![Qwen3.5-9B 推理速度]({charts['qwen35_speed']})

图 2：Qwen3.5-9B Prefill 与长文本连续生成吞吐

在 1K、8K 和 32K Prompt 下，Prefill 分别为 **30.90、24.20 和 12.67 tokens/s**；对应长度的连续生成平均吞吐为 **7.83、5.76 和 1.54 tokens/s**。

　　**本节结论：**Qwen3.5-9B 在本次设备上完成 32K Prompt 与 32K 连续生成，对应吞吐为 **12.67 tokens/s** 和 **1.54 tokens/s**。

<pagebreak/>

## 4.3 Qwen3-8B

Qwen3-8B 当前有效 Prefill 结果覆盖 1K-16K，吞吐随 Prompt 增长持续下降。

![Qwen3-8B Prefill 速度]({charts['qwen3_speed']})

图 3：Qwen3-8B 的 1K-16K Prefill 吞吐

在 1K、8K 和 16K Prompt 下，Prefill 分别为 **{q3_prefill[0]:.2f}、{q3_prefill[1]:.2f} 和 {q3_prefill[2]:.2f} tokens/s**；对应单次 Prefill 耗时为 23.11、617.48 和 2181.42 秒。

　　**本节结论：**Qwen3-8B 当前有效结果延伸至 16K，16K Prefill 为 **{q3_prefill[2]:.2f} tokens/s**。

三款模型共同完成档位的 Prefill 单次观测值汇总如下；数值仅用于并排读取，不计算跨架构提升率。

{chr(10).join(common_prefill_rows)}

　　**本节结论：**共同档位下，三款模型均表现出长 Prompt 吞吐下降；LFM2.5-8B-A1B 与 Qwen3.5-9B 的长文本连续生成也随生成长度增加而下降。连续生成指标不等同于固定 Context 深度下生成 TG128 的瞬时速度。

<pagebreak/>

# 五、运行期资源占用：Peak RSS 与 Attention KV Cache

Attention KV Cache 反映注意力历史状态随长度增长的内存成本；Peak RSS 是进程生命周期内的内存高水位，包含模型加载、计算缓冲与运行状态。

## 5.1 LFM2.5-8B-A1B

LFM2.5-8B-A1B 的 Attention KV Cache 随长度近似线性增长，递归状态在各档位保持约 0.28 MiB。

![LFM2.5-8B-A1B 资源占用]({charts['lfm_resource']})

图 4：LFM2.5-8B-A1B Attention KV Cache 与 Peak RSS

在 1K、8K 和 32K 档位，Attention KV Cache 分别为 **15.00、99.00 和 387.00 MiB**；Prefill Peak RSS 的已测上限为 **6160.18 MiB**。

　　**本节结论：**LFM2.5-8B-A1B 的 32K Attention KV Cache 为 **387.00 MiB**，本次进程 Peak RSS 上限为 **6160.18 MiB**。

<pagebreak/>

## 5.2 Qwen3.5-9B

Qwen3.5-9B 的 Attention KV Cache 随长度近似线性增长，递归状态在各档位保持约 50.25 MiB。

![Qwen3.5-9B 资源占用]({charts['qwen35_resource']})

图 5：Qwen3.5-9B Attention KV Cache 与 Peak RSS

在 1K、8K 和 32K 档位，Attention KV Cache 分别为 **40.00、264.00 和 1032.00 MiB**；Prefill Peak RSS 的已测上限为 **7252.77 MiB**。

　　**本节结论：**Qwen3.5-9B 的 32K Attention KV Cache 为 **1032.00 MiB**，本次进程 Peak RSS 上限为 **7252.77 MiB**。

<pagebreak/>

## 5.3 Qwen3-8B

Qwen3-8B 在 1K-16K 有效档位内，Attention KV Cache 和 Prefill Peak RSS 均随长度增长。

![Qwen3-8B 资源占用]({charts['qwen3_resource']})

图 6：Qwen3-8B 的 1K-16K Attention KV Cache 与 Prefill Peak RSS

在 1K、8K 和 16K 档位，Attention KV Cache 分别为 **{q3_kv[0]:.00f}、{q3_kv[1]:.00f} 和 {q3_kv[2]:.00f} MiB**；Peak RSS 分别为 **{q3_rss[0]:.2f}、{q3_rss[1]:.2f} 和 {q3_rss[2]:.2f} MiB**。

　　**本节结论：**Qwen3-8B 在 16K 的 Attention KV Cache 为 **{q3_kv[2]:.00f} MiB**，Prefill Peak RSS 为 **{q3_rss[2]:.2f} MiB**。

<pagebreak/>

# 六、Qwen3-8B 32K 运行边界

Qwen3-8B 32K Prefill 日志记录了长时间运行过程中设备供电和 CPU 频率限制对测试有效性的影响。该次运行没有形成可用于吞吐统计的完整 Prompt evaluation 结果。

![Qwen3-8B 32K CPU 频率限制]({charts['frequency']})

图 7：Qwen3-8B 32K 运行期间的 CPU 频率上限

日志最后确认处理进度达到 **28672 tokens**，总运行时间约 **215.59 分钟** 后由用户终止。当时设备电量为 **1%**、USB 充电广播上限约 **5 V/0.5 A**；CPU policy0 与 policy6 的允许最高频率分别为 **2.2272 GHz** 和 **1.6320 GHz**，对应硬件最高频率的 61.4% 和 35.4%。

　　**本节结论：**该次 32K 运行处于明显的供电与系统频率限制状态，结果状态记为 **INTERRUPTED**，不进入吞吐曲线或模型速度结论。

---

# 七、综合结论与适用范围

综合当前结果，LFM2.5-8B-A1B 和 Qwen3.5-9B 已形成 1K-32K 完整观测；Qwen3-8B 的有效范围为 1K-16K，其中 16K Prefill 为 **7.51 tokens/s**、Attention KV Cache 为 **2340.00 MiB**、Peak RSS 为 **8158.60 MiB**。32K 中断记录表明，长时间满载测试需要同时控制设备电量、持续供电能力与 CPU 频率上限。

　　**适用条件：**结果基于 OnePlus PLK110、Qualcomm SM8850、Android 16、6 CPU 线程、Q4_0、关闭 Flash Attention 和单次运行统计。不同架构模型仅分析自身长度变化；长文本连续生成不等同于固定 Context 深度下的 TG128 测试。
"""
    path.write_text(markdown, encoding="utf-8")


def write_runtime_observations(path: Path) -> None:
    path.write_text(
        """Runtime observations captured through ADB on 2026-08-21 during Qwen3-8B 32k Prefill
battery_level_percent=1
battery_status=2
usb_powered=true
usb_max_charging_current_uA=500000
usb_max_charging_voltage_uV=5000000
battery_voltage_mV=2782
battery_temperature_tenths_C=344
cpu_policy0_current_kHz=787200
cpu_policy0_scaling_max_kHz=2227200
cpu_policy0_hardware_max_kHz=3628800
cpu_policy6_current_kHz=883200
cpu_policy6_scaling_max_kHz=1632000
cpu_policy6_hardware_max_kHz=4608000
benchmark_stage=Qwen3-8B_m32897_prefill_text_32768_r1
last_completed_n_past=28672
reported_total_time_ms=12926828.27
termination=Interrupted by user
""",
        encoding="utf-8",
    )


def write_source_notes(path: Path, charts: dict[str, Path]) -> None:
    lines = [
        "# Source notes",
        "",
        "## Source identity",
        "",
        f"- Reviewed baseline: {OLD_NORMALIZED}",
        f"- Qwen3-8B performance table: {QWEN3_DIR / 'perf.tsv'}",
        f"- Qwen3-8B memory table: {QWEN3_DIR / 'memory.tsv'}",
        f"- Qwen3-8B interrupted 32k log: {QWEN3_DIR / 'Qwen3-8B_m32897_prefill_text_32768_r1.log'}",
        "- Runtime battery and CPU frequency values were captured with ADB during the active 32k run on 2026-08-21.",
        "",
        "## Mapping and calculations",
        "",
        "- Public model names: LFM2.5-8B-A1B (8.47B), Qwen3.5-9B (9.20B), Qwen3-8B (8.19B).",
        "- Qwen3-8B prefill_throughput maps from perf.tsv field prefill tok/s.",
        "- Qwen3-8B process_peak_rss maps from memory.tsv field peak/VmHWM.",
        "- Qwen3-8B recurrent state is 0.00 MiB, so KV incl state equals Attention KV Cache.",
        "- CPU limit ratio is scaling_max_freq / cpuinfo_max_freq.",
        "- No cross-model improvement, reduction, speedup, or rank is calculated because model size and architecture differ.",
        "",
        "## Omissions and status handling",
        "",
        "- Qwen3-8B 32k Prefill is marked INTERRUPTED. Its zero/inf timing footer is invalid and excluded from charts and conclusions.",
        "- Qwen3-8B Decode and Qwen3.5-0.8B results are absent from the source set and omitted from the customer body.",
        "- Battery and CPU values are one-time runtime observations, not steady-state averages.",
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
            "- All chart values are derived from normalized_measurements.csv.",
            "- Every valid bar and point carries a numeric label.",
            "- Interrupted 32k throughput is displayed as a status in text, never as zero.",
            "- DOCX is generated with the shared customer builder and converted to PDF with LibreOffice.",
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
    pdf_title = (reader.metadata or {}).get("/Title", "")
    if pdf_title != TITLE:
        raise ValueError(f"PDF title metadata mismatch: {pdf_title!r}")

    expected = {docx_path.name, pdf_path.name, script_path.name}
    actual = {path.name for path in DELIVERY.iterdir() if path.is_file()}
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
        x = (index % columns) * 380 + 10
        y = (index // columns) * 530 + 10
        sheet.paste(thumb, (x, y))
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

    runtime_observations = WORK / "runtime_observations.txt"
    normalized = WORK / "normalized_measurements.csv"
    source_notes = WORK / "source_notes.md"
    report_md = WORK / "report.md"
    write_runtime_observations(runtime_observations)
    rows = build_normalized_data(normalized, runtime_observations)

    charts = {
        "lfm_speed": FIGURES / "lfm25_speed.png",
        "qwen35_speed": FIGURES / "qwen35_speed.png",
        "qwen3_speed": FIGURES / "qwen3_speed.png",
        "lfm_resource": FIGURES / "lfm25_resource.png",
        "qwen35_resource": FIGURES / "qwen35_resource.png",
        "qwen3_resource": FIGURES / "qwen3_resource.png",
        "frequency": FIGURES / "qwen3_32k_frequency_limit.png",
    }
    make_speed_chart(rows, LFM, "LFM2.5-8B-A1B", charts["lfm_speed"], True)
    make_speed_chart(rows, QWEN35, "Qwen3.5-9B", charts["qwen35_speed"], True)
    make_speed_chart(rows, QWEN3, "Qwen3-8B", charts["qwen3_speed"], False)
    make_resource_chart(rows, LFM, "LFM2.5-8B-A1B", charts["lfm_resource"], True)
    make_resource_chart(rows, QWEN35, "Qwen3.5-9B", charts["qwen35_resource"], True)
    make_resource_chart(rows, QWEN3, "Qwen3-8B", charts["qwen3_resource"], False)
    make_frequency_chart(charts["frequency"])

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
