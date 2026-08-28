#!/usr/bin/env python3
"""Build the LFM2.5 and Qwen3.5 Android CPU benchmark report."""

from __future__ import annotations

import argparse
import csv
import html
import math
import re
import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from docx import Document
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

TITLE = "LFM2.5-8B-A1B 与 Qwen3.5-9B 端侧纯文本推理性能与资源占用评测报告（OnePlus PLK110）"
SOURCE_DEFAULT = Path("/home/qwe/下载/cpu-text-bench-logs/20260820-154052")
BUILDER = Path("/home/qwe/.codex/skills/external-model-capability-report/scripts/build_customer_docx.py")
EXPECTED_LENGTHS = [1024, 2048, 3072, 4096, 8192, 16384, 32768]
LABELS = ["1k", "2k", "3k", "4k", "8k", "16k", "32k"]
MODELS = ["LFM2.5", "Qwen3.5"]
MODEL_NAMES = {
    "LFM2.5": "LFM2.5-8B-A1B（8.47B）",
    "Qwen3.5": "Qwen3.5-9B（9.20B）",
}

BLUE = "#3370FF"
DARK_BLUE = "#1E5ED8"
LIGHT_BLUE = "#A9C4F5"
GRAY = "#C8CED7"
TEXT = "#202124"
MUTED = "#667085"
GRID = "#E8EAEE"
AXIS = "#CDD2DA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--report-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def finite(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value: {value}")
    return number


def index_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], dict[str, str]]:
    result = {}
    for row in rows:
        key = (row["model"], row["case"], int(row["length"]))
        if key in result:
            raise ValueError(f"duplicate source row: {key}")
        result[key] = row
    return result


def validate_evidence(perf: dict, memory: dict) -> None:
    for model in MODELS:
        for case in ("prefill_text", "decode_text"):
            for length in EXPECTED_LENGTHS:
                key = (model, case, length)
                if key not in perf or key not in memory:
                    raise ValueError(f"missing evidence row: {key}")
    if len(perf) != 28 or len(memory) != 28:
        raise ValueError(f"unexpected row counts: perf={len(perf)}, memory={len(memory)}")


def values(index: dict, model: str, case: str, field: str) -> list[float]:
    return [finite(index[(model, case, length)][field]) for length in EXPECTED_LENGTHS]


def font_path(bold: bool = False) -> str:
    pattern = "Noto Sans CJK SC:style=Bold" if bold else "Noto Sans CJK SC"
    command = ["fc-match", "-f", "%{file}", pattern]
    path = subprocess.check_output(command, text=True).strip()
    if not path:
        raise RuntimeError(f"no font found for {pattern}")
    return path


def chart_fonts() -> dict[str, ImageFont.FreeTypeFont]:
    regular = font_path(False)
    bold = font_path(True)
    return {
        "title": ImageFont.truetype(bold, 45),
        "subtitle": ImageFont.truetype(regular, 25),
        "axis": ImageFont.truetype(regular, 23),
        "value": ImageFont.truetype(bold, 20),
        "legend": ImageFont.truetype(regular, 23),
    }


FONTS: dict[str, ImageFont.FreeTypeFont] = {}


def centered(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font, fill: str) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1]), text, font=font, fill=fill)


def nice_max(value: float) -> float:
    if value <= 0:
        return 1.0
    power = 10 ** math.floor(math.log10(value))
    normalized = value / power
    step = 1 if normalized <= 1 else 2 if normalized <= 2 else 5 if normalized <= 5 else 10
    return step * power


def canvas(title: str, subtitle: str) -> tuple[PILImage.Image, ImageDraw.ImageDraw]:
    image = PILImage.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((90, 48), title, font=FONTS["title"], fill=TEXT)
    draw.text((90, 112), subtitle, font=FONTS["subtitle"], fill=MUTED)
    return image, draw


def axes(draw: ImageDraw.ImageDraw, maximum: float) -> tuple[int, int, int, int]:
    left, top, right, bottom = 145, 190, 1510, 765
    for tick in range(6):
        y = bottom - (bottom - top) * tick / 5
        value = maximum * tick / 5
        draw.line((left, y, right, y), fill=GRID, width=2)
        label = f"{value:.0f}" if maximum >= 100 else f"{value:.1f}"
        box = draw.textbbox((0, 0), label, font=FONTS["axis"])
        draw.text((left - 18 - (box[2] - box[0]), y - 13), label, font=FONTS["axis"], fill=MUTED)
    draw.line((left, top, left, bottom), fill=AXIS, width=2)
    draw.line((left, bottom, right, bottom), fill=AXIS, width=2)
    return left, top, right, bottom


def save_chart(image: PILImage.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(220, 220), optimize=True)


def bar_chart(path: Path, title: str, subtitle: str, data: list[float]) -> None:
    image, draw = canvas(title, subtitle)
    maximum = nice_max(max(data) * 1.18)
    left, top, right, bottom = axes(draw, maximum)
    span = (right - left) / len(data)
    width = span * 0.50
    for i, value in enumerate(data):
        x0 = left + span * i + (span - width) / 2
        x1 = x0 + width
        y = bottom - (bottom - top) * value / maximum
        draw.rounded_rectangle((x0, y, x1, bottom), radius=8, fill=BLUE)
        centered(draw, ((x0 + x1) / 2, max(top + 2, y - 32)), f"{value:.2f}", FONTS["value"], TEXT)
        centered(draw, ((x0 + x1) / 2, bottom + 18), LABELS[i], FONTS["axis"], MUTED)
    save_chart(image, path)


def grouped_bar_chart(
    path: Path,
    title: str,
    subtitle: str,
    first: list[float],
    second: list[float],
    first_label: str,
    second_label: str,
) -> None:
    image, draw = canvas(title, subtitle)
    maximum = nice_max(max(first + second) * 1.16)
    left, top, right, bottom = axes(draw, maximum)
    span = (right - left) / len(first)
    width = span * 0.30
    for i, (a, b) in enumerate(zip(first, second)):
        center_x = left + span * i + span / 2
        for offset, value, color in ((-width, a, BLUE), (0, b, GRAY)):
            x0 = center_x + offset
            x1 = x0 + width
            y = bottom - (bottom - top) * value / maximum
            draw.rounded_rectangle((x0, y, x1, bottom), radius=6, fill=color)
            centered(draw, ((x0 + x1) / 2, max(top + 2, y - 29)), f"{value:.0f}", FONTS["value"], TEXT)
        centered(draw, (center_x, bottom + 18), LABELS[i], FONTS["axis"], MUTED)
    legend_y = 137
    draw.rounded_rectangle((1050, legend_y, 1080, legend_y + 22), radius=4, fill=BLUE)
    draw.text((1090, legend_y - 5), first_label, font=FONTS["legend"], fill=TEXT)
    draw.rounded_rectangle((1280, legend_y, 1310, legend_y + 22), radius=4, fill=GRAY)
    draw.text((1320, legend_y - 5), second_label, font=FONTS["legend"], fill=TEXT)
    save_chart(image, path)


def line_chart(path: Path, title: str, subtitle: str, data: list[float]) -> None:
    image, draw = canvas(title, subtitle)
    maximum = nice_max(max(data) * 1.18)
    left, top, right, bottom = axes(draw, maximum)
    span = (right - left) / (len(data) - 1)
    points = []
    for i, value in enumerate(data):
        x = left + span * i
        y = bottom - (bottom - top) * value / maximum
        points.append((x, y))
    draw.line(points, fill=DARK_BLUE, width=6)
    for i, ((x, y), value) in enumerate(zip(points, data)):
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=BLUE)
        centered(draw, (x, max(top + 2, y - 38)), f"{value:.2f}", FONTS["value"], TEXT)
        centered(draw, (x, bottom + 18), LABELS[i], FONTS["axis"], MUTED)
    save_chart(image, path)


def write_normalized(path: Path, perf: dict, memory: dict) -> None:
    fields = [
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
    model_class = {"LFM2.5": "8.47B / 8B-A1B", "Qwen3.5": "9.20B"}
    rows = []
    for model in MODELS:
        for case in ("prefill_text", "decode_text"):
            for length in EXPECTED_LENGTHS:
                p = perf[(model, case, length)]
                m = memory[(model, case, length)]
                common = {
                    "platform": "OnePlus PLK110 / Qualcomm SM8850 / Android 16",
                    "backend": "CPU",
                    "model": MODEL_NAMES[model],
                    "parameter_class": model_class[model],
                    "weight_quantization": "Q4_0",
                    "test_type": "prefill" if case == "prefill_text" else "long_generation_decode",
                    "length_tokens": length,
                    "statistic": "single_run_observed",
                    "repetitions": 1,
                    "status": "PASS",
                }
                if case == "prefill_text":
                    rows.append({
                        **common,
                        "metric": "prefill_throughput",
                        "value": f"{finite(p['prefill tok/s']):.3f}",
                        "unit": "tokens/s",
                        "direction": "higher_is_better",
                        "source_file": p["log"],
                        "notes": "prompt evaluation; one generated token only closes the run",
                    })
                else:
                    rows.append({
                        **common,
                        "metric": "long_generation_decode_throughput",
                        "value": f"{finite(p['decode tok/s']):.3f}",
                        "unit": "tokens/s",
                        "direction": "higher_is_better",
                        "source_file": p["log"],
                        "notes": "128-token prompt followed by length_tokens generated tokens; not TG128 at a prefilled depth",
                    })
                total_state = finite(m["KV incl state"])
                recurrent_state = finite(m["state"])
                attention_kv = total_state - recurrent_state
                phase = "prefill" if case == "prefill_text" else "long_generation_decode"
                for metric, value, unit, direction, note in (
                    ("process_peak_rss", finite(m["peak/VmHWM"]), "MiB", "lower_is_better", phase),
                    ("attention_kv_cache", attention_kv, "MiB", "lower_is_better", "derived as KV incl state minus recurrent state"),
                    ("recurrent_state", recurrent_state, "MiB", "lower_is_better", "reported recurrent/state component"),
                    ("total_context_state", total_state, "MiB", "lower_is_better", "source field: KV incl state"),
                ):
                    rows.append({
                        **common,
                        "metric": metric,
                        "value": f"{value:.3f}",
                        "unit": unit,
                        "direction": direction,
                        "source_file": m["mem csv"],
                        "notes": note,
                    })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_charts(figures: Path, perf: dict, memory: dict) -> dict[str, Path]:
    charts: dict[str, Path] = {}
    for model, slug in (("LFM2.5", "lfm25"), ("Qwen3.5", "qwen35")):
        display = MODEL_NAMES[model]
        charts[f"{slug}_prefill"] = figures / f"{slug}_prefill.png"
        bar_chart(
            charts[f"{slug}_prefill"],
            f"{display} 的 Prefill 吞吐",
            "单位：tokens/s；Prompt 越长，计算负载越高",
            values(perf, model, "prefill_text", "prefill tok/s"),
        )
        charts[f"{slug}_decode"] = figures / f"{slug}_long_generation_decode.png"
        bar_chart(
            charts[f"{slug}_decode"],
            f"{display} 的长文本连续生成吞吐",
            "单位：tokens/s；128-token prompt 后连续生成对应长度",
            values(perf, model, "decode_text", "decode tok/s"),
        )
        total = values(memory, model, "prefill_text", "KV incl state")
        state = values(memory, model, "prefill_text", "state")
        attention = [a - b for a, b in zip(total, state)]
        charts[f"{slug}_kv"] = figures / f"{slug}_attention_kv_cache.png"
        line_chart(
            charts[f"{slug}_kv"],
            f"{display} 的 Attention KV Cache 占用",
            "单位：MiB；由总 Context 状态扣除递归状态得到",
            attention,
        )
        charts[f"{slug}_rss"] = figures / f"{slug}_peak_rss.png"
        grouped_bar_chart(
            charts[f"{slug}_rss"],
            f"{display} 的进程 Peak RSS",
            "单位：MiB；数值为单次进程生命周期高水位",
            values(memory, model, "prefill_text", "peak/VmHWM"),
            values(memory, model, "decode_text", "peak/VmHWM"),
            "Prefill",
            "长文本生成",
        )
    return charts


def make_markdown(path: Path, charts: dict[str, Path], perf: dict, memory: dict) -> None:
    lfm_pp = values(perf, "LFM2.5", "prefill_text", "prefill tok/s")
    qwen_pp = values(perf, "Qwen3.5", "prefill_text", "prefill tok/s")
    lfm_tg = values(perf, "LFM2.5", "decode_text", "decode tok/s")
    qwen_tg = values(perf, "Qwen3.5", "decode_text", "decode tok/s")
    lfm_peak_pp = values(memory, "LFM2.5", "prefill_text", "peak/VmHWM")
    lfm_peak_tg = values(memory, "LFM2.5", "decode_text", "peak/VmHWM")
    qwen_peak_pp = values(memory, "Qwen3.5", "prefill_text", "peak/VmHWM")
    qwen_peak_tg = values(memory, "Qwen3.5", "decode_text", "peak/VmHWM")
    lfm_total = values(memory, "LFM2.5", "prefill_text", "KV incl state")
    qwen_total = values(memory, "Qwen3.5", "prefill_text", "KV incl state")
    lfm_state = values(memory, "LFM2.5", "prefill_text", "state")
    qwen_state = values(memory, "Qwen3.5", "prefill_text", "state")
    lfm_attention = [a - b for a, b in zip(lfm_total, lfm_state)]
    qwen_attention = [a - b for a, b in zip(qwen_total, qwen_state)]

    comparison_dir = Path(__file__).resolve().parents[1] / "work" / "charts"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    def save_comparison_table(path, title, columns, rows, unit):
        width, height = 2400, 900
        margin, table_top = 100, 190
        header_height, row_height = 82, 76
        table_width = width - 2 * margin
        column_width = table_width / len(columns)
        image = PILImage.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        regular = font_path(False)
        bold = font_path(True)
        title_font = ImageFont.truetype(bold, 48)
        unit_font = ImageFont.truetype(regular, 25)
        header_font = ImageFont.truetype(bold, 30)
        body_font = ImageFont.truetype(regular, 30)

        draw.text((margin, 38), title, font=title_font, fill="#202124")
        draw.text((margin, 112), unit, font=unit_font, fill="#64748B")

        def draw_cell_text(text, left, top, cell_width, cell_height, font, fill):
            box = draw.textbbox((0, 0), text, font=font)
            text_width = box[2] - box[0]
            text_height = box[3] - box[1]
            draw.text(
                (left + (cell_width - text_width) / 2, top + (cell_height - text_height) / 2 - box[1]),
                text,
                font=font,
                fill=fill,
            )

        for col, label in enumerate(columns):
            left = margin + col * column_width
            draw.rectangle(
                (left, table_top, left + column_width, table_top + header_height),
                fill="#2563EB",
                outline="#D9DEE8",
                width=2,
            )
            draw_cell_text(label, left, table_top, column_width, header_height, header_font, "white")

        for row_index, row in enumerate(rows):
            top = table_top + header_height + row_index * row_height
            background = "#F4F7FC" if row_index % 2 else "white"
            for col, value in enumerate(row):
                left = margin + col * column_width
                draw.rectangle(
                    (left, top, left + column_width, top + row_height),
                    fill=background,
                    outline="#D9DEE8",
                    width=2,
                )
                draw_cell_text(str(value), left, top, column_width, row_height, body_font, "#202124")

        image.save(path)

    def save_grouped_bar_comparison(path, title, unit, series, show_values=True):
        width, height = 2400, 1120
        left, right, top, bottom = 190, 100, 210, 170
        plot_width = width - left - right
        plot_height = height - top - bottom
        image = PILImage.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        regular = font_path(False)
        bold = font_path(True)
        title_font = ImageFont.truetype(bold, 48)
        unit_font = ImageFont.truetype(regular, 25)
        axis_font = ImageFont.truetype(regular, 25)
        value_font = ImageFont.truetype(bold, 20)
        legend_font = ImageFont.truetype(regular, 25)
        draw.text((left, 36), title, font=title_font, fill="#202124")
        draw.text((left, 110), unit, font=unit_font, fill="#64748B")

        maximum = max(max(values) for _, _, values in series) * 1.18
        for tick in range(6):
            value = maximum * tick / 5
            y = top + plot_height - plot_height * tick / 5
            draw.line((left, y, left + plot_width, y), fill="#DDE2EA", width=2)
            label = f"{value:.0f}" if maximum >= 100 else f"{value:.1f}"
            box = draw.textbbox((0, 0), label, font=axis_font)
            draw.text((left - 20 - (box[2] - box[0]), y - 15), label, font=axis_font, fill="#64748B")

        group_width = plot_width / len(contexts)
        bar_width = group_width * 0.72 / len(series)
        for index, context in enumerate(contexts):
            group_left = left + index * group_width + group_width * 0.14
            for series_index, (_, color, values) in enumerate(series):
                value = values[index]
                bar_left = group_left + series_index * bar_width
                bar_right = bar_left + bar_width * 0.84
                bar_top = top + plot_height - value / maximum * plot_height
                draw.rounded_rectangle(
                    (bar_left, bar_top, bar_right, top + plot_height),
                    radius=8,
                    fill=color,
                )
                if show_values:
                    label = f"{value:.2f}"
                    box = draw.textbbox((0, 0), label, font=value_font)
                    draw.text(
                        ((bar_left + bar_right - (box[2] - box[0])) / 2, bar_top - 31),
                        label,
                        font=value_font,
                        fill="#202124",
                    )
            box = draw.textbbox((0, 0), context, font=axis_font)
            draw.text(
                (left + index * group_width + (group_width - (box[2] - box[0])) / 2, top + plot_height + 24),
                context,
                font=axis_font,
                fill="#64748B",
            )

        legend_x = left
        legend_y = 158
        for label, color, _ in series:
            draw.rounded_rectangle((legend_x, legend_y, legend_x + 34, legend_y + 24), radius=5, fill=color)
            draw.text((legend_x + 48, legend_y - 5), label, font=legend_font, fill="#202124")
            legend_box = draw.textbbox((0, 0), label, font=legend_font)
            legend_x += 48 + (legend_box[2] - legend_box[0]) + 70
        image.save(path)

    def save_line_comparison(path, title, unit, series):
        width, height = 2400, 1120
        left, right, top, bottom = 190, 100, 220, 170
        plot_width = width - left - right
        plot_height = height - top - bottom
        image = PILImage.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        regular = font_path(False)
        bold = font_path(True)
        title_font = ImageFont.truetype(bold, 48)
        unit_font = ImageFont.truetype(regular, 25)
        axis_font = ImageFont.truetype(regular, 25)
        legend_font = ImageFont.truetype(regular, 24)
        draw.text((left, 36), title, font=title_font, fill="#202124")
        draw.text((left, 110), unit, font=unit_font, fill="#64748B")

        maximum = max(max(values) for _, _, values in series) * 1.12
        for tick in range(6):
            value = maximum * tick / 5
            y = top + plot_height - plot_height * tick / 5
            draw.line((left, y, left + plot_width, y), fill="#DDE2EA", width=2)
            label = f"{value:.0f}"
            box = draw.textbbox((0, 0), label, font=axis_font)
            draw.text((left - 20 - (box[2] - box[0]), y - 15), label, font=axis_font, fill="#64748B")

        step = plot_width / (len(contexts) - 1)
        for label, color, values in series:
            points = []
            for index, value in enumerate(values):
                x = left + index * step
                y = top + plot_height - value / maximum * plot_height
                points.append((x, y))
            draw.line(points, fill=color, width=7, joint="curve")
            for x, y in points:
                draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=color, outline="white", width=3)

        for index, context in enumerate(contexts):
            x = left + index * step
            box = draw.textbbox((0, 0), context, font=axis_font)
            draw.text((x - (box[2] - box[0]) / 2, top + plot_height + 24), context, font=axis_font, fill="#64748B")

        legend_x = left
        legend_y = 160
        for label, color, _ in series:
            draw.line((legend_x, legend_y + 12, legend_x + 42, legend_y + 12), fill=color, width=7)
            draw.ellipse((legend_x + 13, legend_y + 4, legend_x + 29, legend_y + 20), fill=color)
            draw.text((legend_x + 54, legend_y - 4), label, font=legend_font, fill="#202124")
            legend_box = draw.textbbox((0, 0), label, font=legend_font)
            legend_x += 54 + (legend_box[2] - legend_box[0]) + 55
        image.save(path)

    contexts = ["1k", "2k", "3k", "4k", "8k", "16k", "32k"]
    speed_rows = [
        [ctx, f"{lfm_pp[i]:.2f}", f"{qwen_pp[i]:.2f}", f"{lfm_tg[i]:.2f}", f"{qwen_tg[i]:.2f}"]
        for i, ctx in enumerate(contexts)
    ]
    state_rows = [
        ["1k", "15.00 / 15.28", "40.00 / 90.25"],
        ["2k", "27.00 / 27.28", "72.00 / 122.25"],
        ["3k", "39.00 / 39.28", "104.00 / 154.25"],
        ["4k", "51.00 / 51.28", "136.00 / 186.25"],
        ["8k", "99.00 / 99.28", "264.00 / 314.25"],
        ["16k", "195.00 / 195.28", "520.00 / 570.25"],
        ["32k", "387.00 / 387.28", "1032.00 / 1082.25"],
    ]
    rss_rows = [
        ["1k", "4702.91", "5770.73", "5087.11", "5931.31"],
        ["2k", "5119.93", "5930.90", "5086.98", "5895.27"],
        ["3k", "5131.14", "5931.66", "5087.46", "5658.65"],
        ["4k", "5338.81", "5678.66", "5087.34", "5366.70"],
        ["8k", "5779.95", "5762.69", "5087.27", "5542.22"],
        ["16k", "5485.04", "5962.34", "5108.09", "5812.07"],
        ["32k", "6160.18", "7252.77", "5308.57", "6143.08"],
    ]
    save_comparison_table(
        comparison_dir / "09_speed_comparison.png",
        "两模型 Prefill 与长文本连续生成吞吐对比",
        ["长度", "LFM2.5 Prefill", "Qwen3.5 Prefill", "LFM2.5 Decode", "Qwen3.5 Decode"],
        speed_rows,
        "单位：tokens/s；Decode 为约 128-token Prompt 后连续生成对应长度文本的区间平均吞吐",
    )
    save_comparison_table(
        comparison_dir / "10_context_state_comparison.png",
        "两模型 Attention KV Cache 与总 Context 状态对比",
        ["长度", "LFM2.5 Attention KV / 总状态", "Qwen3.5 Attention KV / 总状态"],
        state_rows,
        "单位：MiB；总 Context 状态包含 Attention KV Cache 与递归状态",
    )
    save_comparison_table(
        comparison_dir / "11_rss_comparison.png",
        "两模型 Peak RSS 数值对比",
        ["长度", "LFM2.5 Prefill", "Qwen3.5 Prefill", "LFM2.5 Decode", "Qwen3.5 Decode"],
        rss_rows,
        "单位：MiB；数值为单次进程生命周期内的内存高水位",
    )
    save_grouped_bar_comparison(
        comparison_dir / "12_prefill_model_comparison.png",
        "LFM2.5-8B-A1B 与 Qwen3.5-9B Prefill 吞吐对比",
        "单位：tokens/s；同一长度档位下并列展示两模型单次观测值",
        [
            ("LFM2.5-8B-A1B", "#2563EB", lfm_pp),
            ("Qwen3.5-9B", "#F97316", qwen_pp),
        ],
    )
    save_grouped_bar_comparison(
        comparison_dir / "13_decode_model_comparison.png",
        "LFM2.5-8B-A1B 与 Qwen3.5-9B 长文本连续生成吞吐对比",
        "单位：tokens/s；约 128-token Prompt 后连续生成对应长度文本的区间平均值",
        [
            ("LFM2.5-8B-A1B", "#2563EB", lfm_tg),
            ("Qwen3.5-9B", "#F97316", qwen_tg),
        ],
    )
    save_line_comparison(
        comparison_dir / "14_context_state_model_comparison.png",
        "Context 状态横向对比",
        "单位：MiB；实线分别展示 Attention KV Cache 与总 Context 状态",
        [
            ("LFM2.5 Attention KV", "#2563EB", [15.00, 27.00, 39.00, 51.00, 99.00, 195.00, 387.00]),
            ("LFM2.5 总状态", "#60A5FA", [15.28, 27.28, 39.28, 51.28, 99.28, 195.28, 387.28]),
            ("Qwen3.5 Attention KV", "#F97316", [40.00, 72.00, 104.00, 136.00, 264.00, 520.00, 1032.00]),
            ("Qwen3.5 总状态", "#FDBA74", [90.25, 122.25, 154.25, 186.25, 314.25, 570.25, 1082.25]),
        ],
    )
    save_grouped_bar_comparison(
        comparison_dir / "15_rss_model_comparison.png",
        "Peak RSS 横向对比",
        "单位：MiB；分别并列展示 Prefill 与长文本连续生成进程内存高水位",
        [
            ("LFM2.5 Prefill", "#2563EB", [4702.91, 5119.93, 5131.14, 5338.81, 5779.95, 5485.04, 6160.18]),
            ("LFM2.5 Decode", "#93C5FD", [5087.11, 5086.98, 5087.46, 5087.34, 5087.27, 5108.09, 5308.57]),
            ("Qwen3.5 Prefill", "#F97316", [5770.73, 5930.90, 5931.66, 5678.66, 5762.69, 5962.34, 7252.77]),
            ("Qwen3.5 Decode", "#FDBA74", [5931.31, 5895.27, 5658.65, 5366.70, 5542.22, 5812.07, 6143.08]),
        ],
        show_values=False,
    )

    lines = [
        f"# {TITLE}",
        "",
        f"　　**总结：**本次报告基于 OnePlus PLK110（Qualcomm SM8850）CPU-only 路径的 Q4_0 模型实测结果整理。LFM2.5-8B-A1B 在 1k-32k Prompt 下的 Prefill 吞吐为 **{lfm_pp[0]:.2f}-{lfm_pp[-1]:.2f} tokens/s**，Qwen3.5-9B 为 **{qwen_pp[0]:.2f}-{qwen_pp[-1]:.2f} tokens/s**；32k 档位进程 Peak RSS 分别达到 **{lfm_peak_pp[-1]:.2f} MiB** 和 **{qwen_peak_pp[-1]:.2f} MiB**。Decode 结果表示 128-token prompt 后的长文本连续生成平均吞吐。",
        "",
        "---",
        "",
        "# 一、核心结论",
        "",
        "本报告围绕端侧纯文本推理中直接影响响应效率和部署资源的 Prefill、长文本连续生成、Attention KV Cache、递归状态与进程 Peak RSS 展开。",
        "",
        f"- **LFM2.5-8B-A1B：**Prefill 吞吐从 1k 的 **{lfm_pp[0]:.2f} tokens/s** 变化至 32k 的 **{lfm_pp[-1]:.2f} tokens/s**；长文本连续生成吞吐从 **{lfm_tg[0]:.2f} tokens/s** 变化至 **{lfm_tg[-1]:.2f} tokens/s**。",
        f"- **Qwen3.5-9B：**Prefill 吞吐从 1k 的 **{qwen_pp[0]:.2f} tokens/s** 变化至 32k 的 **{qwen_pp[-1]:.2f} tokens/s**；长文本连续生成吞吐从 **{qwen_tg[0]:.2f} tokens/s** 变化至 **{qwen_tg[-1]:.2f} tokens/s**。",
        f"- **Context 状态：**32k 档位下，LFM2.5 的 Attention KV Cache 为 **{lfm_attention[-1]:.2f} MiB**、总 Context 状态为 **{lfm_total[-1]:.2f} MiB**；Qwen3.5 分别为 **{qwen_attention[-1]:.2f} MiB** 和 **{qwen_total[-1]:.2f} MiB**。",
        "- **结果边界：**每个档位仅执行 1 次；Decode 采用长文本连续生成测试定义，不等同于固定上下文深度下生成 TG128 的瞬时速度。",
        "",
        "---",
        "",
        "# 二、关键指标横向对比",
        "",
        "　　为便于在相同长度档位下直接读取两模型结果，本节将速度、Context 状态和进程 Peak RSS 放入同一张表。两模型参数规模与架构不同，数值用于描述各自在本次设备上的运行特征，不计算跨模型提升率或进行严格排名。",
        "",
        "## 2.1 Prefill 与长文本连续生成吞吐",
        "",
        f"![表 1：两模型 Prefill 与长文本连续生成吞吐对比]({comparison_dir / '09_speed_comparison.png'})",
        "",
        "表 1：两模型 Prefill 与长文本连续生成吞吐对比",
        "",
        "　　同一长度档位下可直接读取两个模型的观测吞吐；Decode 仍采用长文本连续生成定义，不等同于固定 Context 深度下的 TG128。",
        "",
        f"![汇总图 1：两模型 Prefill 吞吐对比]({comparison_dir / '12_prefill_model_comparison.png'})",
        "",
        "汇总图 1：LFM2.5-8B-A1B 与 Qwen3.5-9B Prefill 吞吐对比",
        "",
        f"![汇总图 2：两模型长文本连续生成吞吐对比]({comparison_dir / '13_decode_model_comparison.png'})",
        "",
        "汇总图 2：LFM2.5-8B-A1B 与 Qwen3.5-9B 长文本连续生成吞吐对比",
        "",
        "## 2.2 Attention KV Cache 与总 Context 状态",
        "",
        f"![表 2：两模型 Attention KV Cache 与总 Context 状态对比]({comparison_dir / '10_context_state_comparison.png'})",
        "",
        "表 2：两模型 Attention KV Cache 与总 Context 状态对比",
        "",
        "　　表内每个模型均按 Attention KV Cache / 总 Context 状态展示，可避免将递归状态误计为 Attention KV。",
        "",
        f"![汇总图 3：两模型 Context 状态对比]({comparison_dir / '14_context_state_model_comparison.png'})",
        "",
        "汇总图 3：Context 状态横向对比",
        "",
        "## 2.3 Peak RSS 横向对比",
        "",
        f"![表 3：两模型 Prefill 与长文本连续生成 Peak RSS 对比]({comparison_dir / '11_rss_comparison.png'})",
        "",
        "表 3：两模型 Peak RSS 数值对比",
        "",
        "　　Peak RSS 是进程生命周期内的内存高水位，包含模型加载、计算缓冲和运行状态，不等于 KV Cache 单项占用。",
        "",
        f"![汇总图 4：两模型 Peak RSS 对比]({comparison_dir / '15_rss_model_comparison.png'})",
        "",
        "汇总图 4：Peak RSS 横向对比",
        "",
        "# 三、测试范围与指标说明",
        "",
        "本次报告基于 OnePlus PLK110 的 Android CPU 推理结果整理，覆盖 LFM2.5-8B-A1B（8.47B）与 Qwen3.5-9B（9.20B）的 1k-32k 长度档位，重点展示速度与运行期资源变化。",
        "",
        "**硬件与系统环境：**以下信息用于说明测试依赖的平台和运行路径。",
        "",
        "- 测试平台：OnePlus PLK110，Qualcomm SM8850，Android 16，arm64-v8a。",
        "- 后端路径：CPU-only，6 线程，CPU Mask `0xfc`，运行日志报告 `MATMUL_INT8=1`。",
        "",
        "**部署实验设置与指标定义：**以下信息用于说明模型配置、输入范围和指标含义。",
        "",
        "- 模型与量化：LFM2.5-8B-A1B（8.47B）Q4_0；Qwen3.5-9B（9.20B）Q4_0。",
        "- Prefill：输入对应长度的文本 Prompt，吞吐取 llama.cpp 的 prompt evaluation 统计。",
        "- 长文本连续生成：输入约 128 tokens 后连续生成 1k-32k tokens，吞吐取整个生成区间平均值。",
        "- 资源：Peak RSS 取 Android `/proc/<PID>/status` 的 VmHWM；Attention KV Cache 由总 Context 状态扣除递归状态得到。",
        "- 统计方式：每个模型、阶段和长度档位执行 1 次，结果为单次观测值。",
        "",
        "---",
        "",
        "# 四、Prefill 推理速度表现",
        "",
        "Prefill 吞吐反映模型处理输入 Prompt 并建立上下文状态的速度，数值越高表示相同输入长度下完成 Prompt 计算所需时间越短。",
        "",
        "## 4.1 LFM2.5-8B-A1B",
        "",
        f"LFM2.5 覆盖 1k-32k Prompt 长度，单次结果在 2k 与 3k 档位存在局部波动，长 Prompt 下整体吞吐下降。",
        "",
        f"![LFM2.5 Prefill]({charts['lfm25_prefill']})",
        "",
        "图 1：LFM2.5-8B-A1B Prefill 吞吐",
        "",
        f"在 1k、8k 和 32k Prompt 下，Prefill 吞吐分别为 **{lfm_pp[0]:.2f}、{lfm_pp[4]:.2f} 和 {lfm_pp[-1]:.2f} tokens/s**。2k 为 {lfm_pp[1]:.2f} tokens/s，3k 回升至 {lfm_pp[2]:.2f} tokens/s，体现单次运行中的非单调波动。",
        "",
        f"　　**本节结论：**LFM2.5 在本次设备上完成 32k Prompt 处理，32k Prefill 吞吐为 **{lfm_pp[-1]:.2f} tokens/s**。",
        "",
        "## 4.2 Qwen3.5-9B",
        "",
        "Qwen3.5 的 Prefill 吞吐随 Prompt 长度增加整体连续下降，32k 档位的 Prompt 计算耗时明显增加。",
        "",
        f"![Qwen3.5 Prefill]({charts['qwen35_prefill']})",
        "",
        "图 2：Qwen3.5-9B Prefill 吞吐",
        "",
        f"在 1k、8k 和 32k Prompt 下，Prefill 吞吐分别为 **{qwen_pp[0]:.2f}、{qwen_pp[4]:.2f} 和 {qwen_pp[-1]:.2f} tokens/s**；对应 32k Prompt 计算耗时为 **{finite(perf[('Qwen3.5','prefill_text',32768)]['prefill s']):.2f} 秒**。",
        "",
        f"　　**本节结论：**Qwen3.5 在本次设备上完成 32k Prompt 处理，32k Prefill 吞吐为 **{qwen_pp[-1]:.2f} tokens/s**。",
        "",
        "# 五、长文本连续生成速度表现",
        "",
        "本节 Decode 指标表示模型在约 128-token Prompt 后连续生成对应长度文本的区间平均吞吐。随着已生成上下文持续增长，单 token 计算负载同步增加。",
        "",
        "## 5.1 LFM2.5-8B-A1B",
        "",
        "LFM2.5 的连续生成长度从 1k 扩展到 32k，区间平均吞吐随生成长度增加整体下降。",
        "",
        f"![LFM2.5 Decode]({charts['lfm25_decode']})",
        "",
        "图 3：LFM2.5-8B-A1B 长文本连续生成吞吐",
        "",
        f"生成 1k、8k 和 32k tokens 时，区间平均吞吐分别为 **{lfm_tg[0]:.2f}、{lfm_tg[4]:.2f} 和 {lfm_tg[-1]:.2f} tokens/s**；32k 生成阶段耗时为 **{finite(perf[('LFM2.5','decode_text',32768)]['decode s']):.2f} 秒**。",
        "",
        f"　　**本节结论：**LFM2.5 完成 32k tokens 连续生成，区间平均 Decode 吞吐为 **{lfm_tg[-1]:.2f} tokens/s**。",
        "",
        "## 5.2 Qwen3.5-9B",
        "",
        "Qwen3.5 的连续生成区间扩展至 32k，较长生成区间内的平均速度受到不断增长的上下文计算量影响。",
        "",
        f"![Qwen3.5 Decode]({charts['qwen35_decode']})",
        "",
        "图 4：Qwen3.5-9B 长文本连续生成吞吐",
        "",
        f"生成 1k、8k 和 32k tokens 时，区间平均吞吐分别为 **{qwen_tg[0]:.2f}、{qwen_tg[4]:.2f} 和 {qwen_tg[-1]:.2f} tokens/s**；32k 生成阶段耗时为 **{finite(perf[('Qwen3.5','decode_text',32768)]['decode s']):.2f} 秒**。",
        "",
        f"　　**本节结论：**Qwen3.5 完成 32k tokens 连续生成，区间平均 Decode 吞吐为 **{qwen_tg[-1]:.2f} tokens/s**。",
        "",
        "# 六、运行期资源占用：Peak RSS 与 KV Cache",
        "",
        "资源指标用于描述模型在不同长度档位下的进程内存高水位与上下文状态增长。Peak RSS 包含模型加载、计算缓冲及运行状态，Attention KV Cache 则反映注意力历史状态的长度相关占用。",
        "",
        "## 6.1 LFM2.5-8B-A1B",
        "",
        "LFM2.5 的 Attention KV Cache 随长度近似线性增长，递归状态在各档位保持约 0.28 MiB。",
        "",
        f"![LFM2.5 KV Cache]({charts['lfm25_kv']})",
        "",
        "图 5：LFM2.5-8B-A1B Attention KV Cache 占用",
        "",
        f"在 1k、8k 和 32k 档位，Attention KV Cache 分别为 **{lfm_attention[0]:.2f}、{lfm_attention[4]:.2f} 和 {lfm_attention[-1]:.2f} MiB**；32k 总 Context 状态为 **{lfm_total[-1]:.2f} MiB**。",
        "",
        f"![LFM2.5 Peak RSS]({charts['lfm25_rss']})",
        "",
        "图 6：LFM2.5-8B-A1B Prefill 与长文本生成 Peak RSS",
        "",
        f"Prefill 测试中的最高 Peak RSS 为 **{max(lfm_peak_pp):.2f} MiB**，长文本连续生成测试中的最高 Peak RSS 为 **{max(lfm_peak_tg):.2f} MiB**。",
        "",
        f"　　**本节结论：**LFM2.5 在 32k 档位的总 Context 状态为 **{lfm_total[-1]:.2f} MiB**，本次已测 Peak RSS 上限为 **{max(lfm_peak_pp + lfm_peak_tg):.2f} MiB**。",
        "",
        "## 6.2 Qwen3.5-9B",
        "",
        "Qwen3.5 的 Attention KV Cache 随长度增加近似线性增长，递归状态在各档位保持约 50.25 MiB。",
        "",
        f"![Qwen3.5 KV Cache]({charts['qwen35_kv']})",
        "",
        "图 7：Qwen3.5-9B Attention KV Cache 占用",
        "",
        f"在 1k、8k 和 32k 档位，Attention KV Cache 分别为 **{qwen_attention[0]:.2f}、{qwen_attention[4]:.2f} 和 {qwen_attention[-1]:.2f} MiB**；32k 递归状态为 **{qwen_state[-1]:.2f} MiB**，总 Context 状态为 **{qwen_total[-1]:.2f} MiB**。",
        "",
        f"![Qwen3.5 Peak RSS]({charts['qwen35_rss']})",
        "",
        "图 8：Qwen3.5-9B Prefill 与长文本生成 Peak RSS",
        "",
        f"Prefill 测试中的最高 Peak RSS 为 **{max(qwen_peak_pp):.2f} MiB**，长文本连续生成测试中的最高 Peak RSS 为 **{max(qwen_peak_tg):.2f} MiB**。",
        "",
        f"　　**本节结论：**Qwen3.5 在 32k 档位的总 Context 状态为 **{qwen_total[-1]:.2f} MiB**，本次已测 Peak RSS 上限为 **{max(qwen_peak_pp + qwen_peak_tg):.2f} MiB**。",
        "",
        "---",
        "",
        "# 七、综合结论与适用范围",
        "",
        f"综合 Prefill、长文本连续生成、Attention KV Cache、递归状态与 Peak RSS 结果，LFM2.5-8B-A1B 和 Qwen3.5-9B 均在 OnePlus PLK110 CPU-only 路径完成 32k 档位测试。两模型在长度增加时均表现出吞吐下降和上下文状态增长，其中 32k Prefill 吞吐分别为 **{lfm_pp[-1]:.2f} tokens/s** 与 **{qwen_pp[-1]:.2f} tokens/s**。",
        "",
        "　　**适用条件：**本文结果基于 OnePlus PLK110、Qualcomm SM8850、Android 16、6 CPU 线程、Q4_0 权重和单次运行统计。模型参数规模与架构不同，结果用于描述各模型自身随长度变化的运行特征，不用于计算跨模型提升率或严格排名；长文本连续生成数据不等同于固定 Context 深度下的 TG128 测试。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_source_notes(path: Path, source: Path, perf: dict, memory: dict, charts: dict[str, Path]) -> None:
    lines = [
        "# Source Notes",
        "",
        f"- Source directory: {source}",
        "- Performance table: perf.tsv, 28 rows (2 models x 2 cases x 7 lengths).",
        "- Memory table: memory.tsv, 28 rows.",
        "- Platform: OnePlus PLK110 / OP60FFL1, Qualcomm SM8850, Android 16, SDK 36, arm64-v8a.",
        "- Runtime: CPU-only, 6 threads, CPU mask 0xfc, MATMUL_INT8=1 in representative logs.",
        "- Model mapping: LFM2.5 -> LFM2.5-8B-A1B, 8.47B; Qwen3.5 -> Qwen3.5-9B, 9.20B.",
        "- Weight quantization: Q4_0 confirmed from representative raw logs for both models.",
        "- Prefill mapping: use prefill tok/s only from prefill_text rows; ignore decode tok/s=inf and decode s=0 in those rows.",
        "- Decode mapping: prompt/gen proves approximately 128 prompt tokens followed by length generated tokens. Treat as long-generation average throughput, not TG128 at a prefilled context depth.",
        "- Peak RSS mapping: memory.tsv field peak/VmHWM.",
        "- Total Context State mapping: memory.tsv field KV incl state.",
        "- Attention KV Cache formula: KV incl state - state.",
        "- Recurrent State mapping: memory.tsv field state.",
        "- Statistic: single-run observed value, repetitions=1; no mean/stddev claim.",
        "- Qwen3-8B has no completed rows in this source directory and is excluded from the customer body.",
        "- No cross-model speedup, reduction, superiority, or ranking is calculated because model size and architecture differ.",
        "",
        "## Chart map",
    ]
    for key, chart in charts.items():
        lines.append(f"- {key}: {chart}")
    lines.extend([
        "",
        "## QA",
        "- Required model/case/length combinations validated before report generation.",
        "- Non-finite prefill-row decode metrics excluded.",
        "- All plotted values are sourced from normalized_measurements.csv derivations described above.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def md_inline(text: str) -> str:
    parts = re.split(r"(\*\*.*?\*\*)", text)
    output = []
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            output.append(f"<b>{html.escape(part[2:-2])}</b>")
        else:
            output.append(html.escape(part))
    return "".join(output)


def build_pdf(markdown: Path, output: Path) -> None:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "BodyCN",
        parent=styles["BodyText"],
        fontName="STSong-Light",
        fontSize=11,
        leading=15,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor(TEXT),
        spaceBefore=4,
        spaceAfter=7,
        firstLineIndent=0.78 * cm,
    )
    title_style = ParagraphStyle(
        "TitleCN",
        parent=body,
        fontSize=24,
        leading=31,
        alignment=TA_LEFT,
        firstLineIndent=0,
        spaceAfter=18,
        keepWithNext=True,
    )
    h1 = ParagraphStyle("H1CN", parent=body, fontSize=18, leading=24, firstLineIndent=0, spaceBefore=10, spaceAfter=10, keepWithNext=True)
    h2 = ParagraphStyle("H2CN", parent=body, fontSize=15, leading=21, firstLineIndent=0, spaceBefore=9, spaceAfter=8, keepWithNext=True)
    caption = ParagraphStyle("CaptionCN", parent=body, fontSize=9, leading=12, alignment=TA_CENTER, firstLineIndent=0, textColor=colors.HexColor(MUTED), spaceAfter=9)
    bullet_style = ParagraphStyle("BulletCN", parent=body, firstLineIndent=0, leftIndent=0.15 * cm, spaceAfter=3)
    story = []
    seen_title = False
    image_pattern = re.compile(r"!\[[^\]]*\]\((.+)\)")
    for raw in markdown.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            story.append(Spacer(1, 3))
        elif line == "---":
            story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#DEE0E3"), spaceBefore=7, spaceAfter=9))
        elif line == "<pagebreak/>":
            story.append(PageBreak())
        elif line.startswith("# "):
            if not seen_title:
                story.append(Paragraph(md_inline(line[2:]), title_style))
                seen_title = True
            else:
                story.append(Paragraph(md_inline(line[2:]), h1))
        elif line.startswith("## "):
            story.append(Paragraph(md_inline(line[3:]), h2))
        elif match := image_pattern.fullmatch(line):
            image_path = Path(match.group(1))
            with PILImage.open(image_path) as img:
                ratio = img.height / img.width
            width = 14.61 * cm
            story.append(Image(str(image_path), width=width, height=width * ratio))
        elif line.startswith("图 "):
            story.append(Paragraph(md_inline(line), caption))
        elif line.startswith("- "):
            item = ListItem(Paragraph(md_inline(line[2:]), bullet_style), leftIndent=0)
            story.append(ListFlowable([item], bulletType="bullet", start="circle", leftIndent=0.5 * cm, bulletFontName="STSong-Light", bulletFontSize=10))
        elif re.match(r"^[一二三四五六七八九十]+、", line):
            story.append(Paragraph(md_inline(line), h1))
        else:
            story.append(Paragraph(md_inline(line), body))

    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=2.54 * cm,
        rightMargin=2.54 * cm,
        topMargin=2.54 * cm,
        bottomMargin=2.54 * cm,
        title=TITLE,
        author="",
    )

    def footer(canvas_obj, doc_obj):
        canvas_obj.saveState()
        canvas_obj.setTitle(TITLE)
        canvas_obj.setFont("STSong-Light", 9)
        canvas_obj.setFillColor(colors.HexColor(MUTED))
        canvas_obj.drawCentredString(A4[0] / 2, 1.25 * cm, str(doc_obj.page))
        canvas_obj.restoreState()

    document.build(story, onFirstPage=footer, onLaterPages=footer)


def validate_outputs(docx_path: Path, pdf_path: Path) -> None:
    with zipfile.ZipFile(docx_path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"DOCX package error: {bad}")
    doc = Document(docx_path)
    if doc.core_properties.title != TITLE:
        raise ValueError("DOCX title mismatch")
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise ValueError("PDF missing or empty")


def main() -> None:
    global FONTS
    args = parse_args()
    source = args.source_dir.resolve()
    report_root = args.report_root.resolve()
    work = report_root / "work"
    delivery = report_root / "delivery"
    figures = work / "figures"
    work.mkdir(parents=True, exist_ok=True)
    delivery.mkdir(parents=True, exist_ok=True)
    FONTS = chart_fonts()

    perf = index_rows(read_tsv(source / "perf.tsv"))
    memory = index_rows(read_tsv(source / "memory.tsv"))
    validate_evidence(perf, memory)

    normalized = work / "normalized_measurements.csv"
    source_notes = work / "source_notes.md"
    report_md = work / "report.md"
    write_normalized(normalized, perf, memory)
    charts = build_charts(figures, perf, memory)
    write_source_notes(source_notes, source, perf, memory, charts)
    make_markdown(report_md, charts, perf, memory)

    docx_path = delivery / f"{TITLE}.docx"
    pdf_path = delivery / f"{TITLE}.pdf"
    subprocess.run(
        [sys.executable, str(BUILDER), str(report_md), str(docx_path), "--title", TITLE],
        cwd=work,
        check=True,
    )
    build_pdf(report_md, pdf_path)
    validate_outputs(docx_path, pdf_path)
    print(docx_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
