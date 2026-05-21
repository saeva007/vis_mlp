#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw a paper-style StaticRNNLowVisNet flowchart as SVG and PNG."""

from __future__ import annotations

import html
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


W, H = 3840, 2160
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_STEM = "static_rnn_lowvis_flowchart"

COLORS = {
    "white": "#FFFFFF",
    "ink": "#17313D",
    "muted": "#5F7282",
    "line": "#A9B7C2",
    "panel": "#FBFCFE",
    "gray_l": "#F6F8FA",
    "blue": "#2B78C5",
    "blue_l": "#EAF3FC",
    "green": "#1FA187",
    "green_l": "#EAF8F4",
    "purple": "#7B61B6",
    "purple_l": "#F1ECFA",
    "teal": "#2E9AA3",
    "teal_l": "#EAF7F8",
}


FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/segoeui.ttf"),
]
BOLD_CANDIDATES = [
    Path("C:/Windows/Fonts/arialbd.ttf"),
    Path("C:/Windows/Fonts/segoeuib.ttf"),
]


def font_path(bold: bool = False) -> str | None:
    for path in BOLD_CANDIDATES if bold else FONT_CANDIDATES:
        if path.exists():
            return str(path)
    return None


REGULAR_FONT = font_path(False)
BOLD_FONT = font_path(True) or REGULAR_FONT


def pil_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = BOLD_FONT if bold else REGULAR_FONT
    if path:
        return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def wrap_lines(lines: list[str], width: int) -> list[str]:
    out: list[str] = []
    for line in lines:
        if not line:
            out.append("")
        else:
            out.extend(textwrap.wrap(line, width=width, break_long_words=False))
    return out


def hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int
    title: str
    lines: list[str]
    fill: str
    stroke: str
    title_color: str | None = None
    wrap: int = 34
    title_size: int = 40
    body_size: int = 34
    stroke_width: int = 5

    @property
    def left(self) -> tuple[int, int]:
        return self.x, self.y + self.h // 2

    @property
    def right(self) -> tuple[int, int]:
        return self.x + self.w, self.y + self.h // 2

    @property
    def top(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y

    @property
    def bottom(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h


class SvgCanvas:
    def __init__(self) -> None:
        self.parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
            "<defs>",
            '<marker id="arrow" markerWidth="14" markerHeight="14" refX="12" refY="7" orient="auto" markerUnits="strokeWidth">',
            f'<path d="M 0 0 L 14 7 L 0 14 z" fill="{COLORS["muted"]}"/>',
            "</marker>",
            '<marker id="arrowBlue" markerWidth="14" markerHeight="14" refX="12" refY="7" orient="auto" markerUnits="strokeWidth">',
            f'<path d="M 0 0 L 14 7 L 0 14 z" fill="{COLORS["blue"]}"/>',
            "</marker>",
            '<marker id="arrowGreen" markerWidth="14" markerHeight="14" refX="12" refY="7" orient="auto" markerUnits="strokeWidth">',
            f'<path d="M 0 0 L 14 7 L 0 14 z" fill="{COLORS["green"]}"/>',
            "</marker>",
            '<marker id="arrowPurple" markerWidth="14" markerHeight="14" refX="12" refY="7" orient="auto" markerUnits="strokeWidth">',
            f'<path d="M 0 0 L 14 7 L 0 14 z" fill="{COLORS["purple"]}"/>',
            "</marker>",
            '<marker id="arrowTeal" markerWidth="14" markerHeight="14" refX="12" refY="7" orient="auto" markerUnits="strokeWidth">',
            f'<path d="M 0 0 L 14 7 L 0 14 z" fill="{COLORS["teal"]}"/>',
            "</marker>",
            "</defs>",
            f'<rect x="0" y="0" width="{W}" height="{H}" fill="{COLORS["white"]}"/>',
        ]

    def text(
        self,
        x: int,
        y: int,
        value: str,
        size: int,
        color: str,
        *,
        bold: bool = False,
        anchor: str = "start",
        uppercase: bool = False,
    ) -> None:
        weight = "700" if bold else "400"
        family = "Arial, Segoe UI, Helvetica, sans-serif"
        if uppercase:
            value = value.upper()
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{html.escape(value)}</text>'
        )

    def multiline(
        self,
        x: int,
        y: int,
        lines: list[str],
        size: int,
        color: str,
        *,
        line_gap: int,
    ) -> None:
        family = "Arial, Segoe UI, Helvetica, sans-serif"
        escaped = [html.escape(line) for line in lines]
        body = "".join(
            f'<tspan x="{x}" dy="{0 if i == 0 else line_gap}">{line}</tspan>'
            for i, line in enumerate(escaped)
        )
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'font-weight="400" fill="{color}">{body}</text>'
        )

    def rounded_rect(self, x: int, y: int, w: int, h: int, fill: str, stroke: str, sw: int = 4, r: int = 28) -> None:
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" ry="{r}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def arrow(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        color: str,
        *,
        width: int = 7,
        dashed: bool = False,
        marker: str | None = None,
    ) -> None:
        marker_id = marker or {
            COLORS["blue"]: "arrowBlue",
            COLORS["green"]: "arrowGreen",
            COLORS["purple"]: "arrowPurple",
            COLORS["teal"]: "arrowTeal",
        }.get(color, "arrow")
        dash = ' stroke-dasharray="18 14"' if dashed else ""
        self.parts.append(
            f'<line x1="{start[0]}" y1="{start[1]}" x2="{end[0]}" y2="{end[1]}" '
            f'stroke="{color}" stroke-width="{width}" stroke-linecap="round" '
            f'marker-end="url(#{marker_id})"{dash}/>'
        )

    def polyline_arrow(
        self,
        points: list[tuple[int, int]],
        color: str,
        *,
        width: int = 6,
        dashed: bool = False,
    ) -> None:
        marker_id = {
            COLORS["blue"]: "arrowBlue",
            COLORS["green"]: "arrowGreen",
            COLORS["purple"]: "arrowPurple",
            COLORS["teal"]: "arrowTeal",
        }.get(color, "arrow")
        dash = ' stroke-dasharray="18 14"' if dashed else ""
        pts = " ".join(f"{x},{y}" for x, y in points)
        self.parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" '
            f'stroke-linecap="round" stroke-linejoin="round" marker-end="url(#{marker_id})"{dash}/>'
        )

    def add_box(self, box: Box) -> None:
        self.rounded_rect(box.x, box.y, box.w, box.h, box.fill, box.stroke, box.stroke_width)
        self.text(
            box.x + 48,
            box.y + 72,
            box.title,
            box.title_size,
            box.title_color or box.stroke,
            bold=True,
        )
        self.multiline(
            box.x + 48,
            box.y + 128,
            wrap_lines(box.lines, box.wrap),
            box.body_size,
            COLORS["ink"],
            line_gap=int(box.body_size * 1.42),
        )

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.parts + ["</svg>"]) + "\n", encoding="utf-8")


class PngCanvas:
    def __init__(self) -> None:
        self.im = Image.new("RGB", (W, H), hex_rgb(COLORS["white"]))
        self.draw = ImageDraw.Draw(self.im)

    def text(
        self,
        x: int,
        y: int,
        value: str,
        size: int,
        color: str,
        *,
        bold: bool = False,
        anchor: str = "la",
        uppercase: bool = False,
    ) -> None:
        if uppercase:
            value = value.upper()
        self.draw.text((x, y), value, fill=hex_rgb(color), font=pil_font(size, bold), anchor=anchor)

    def multiline(self, x: int, y: int, lines: list[str], size: int, color: str, *, line_gap: int) -> None:
        yy = y
        font = pil_font(size, False)
        for line in lines:
            self.draw.text((x, yy), line, fill=hex_rgb(color), font=font, anchor="la")
            yy += line_gap

    def rounded_rect(self, x: int, y: int, w: int, h: int, fill: str, stroke: str, sw: int = 4, r: int = 28) -> None:
        self.draw.rounded_rectangle((x, y, x + w, y + h), radius=r, fill=hex_rgb(fill), outline=hex_rgb(stroke), width=sw)

    def arrow(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        color: str,
        *,
        width: int = 7,
        dashed: bool = False,
    ) -> None:
        if dashed:
            self._dashed_line(start, end, color, width)
        else:
            self.draw.line((start, end), fill=hex_rgb(color), width=width)
        self._arrow_head(start, end, color)

    def polyline_arrow(
        self,
        points: list[tuple[int, int]],
        color: str,
        *,
        width: int = 6,
        dashed: bool = False,
    ) -> None:
        for a, b in zip(points[:-1], points[1:]):
            self.arrow(a, b, color, width=width, dashed=dashed)

    def _dashed_line(self, start: tuple[int, int], end: tuple[int, int], color: str, width: int) -> None:
        x1, y1 = start
        x2, y2 = end
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist == 0:
            return
        dash, gap = 22, 16
        steps = int(dist // (dash + gap)) + 1
        for i in range(steps):
            t0 = (i * (dash + gap)) / dist
            t1 = min((i * (dash + gap) + dash) / dist, 1)
            if t0 >= 1:
                break
            a = (int(x1 + (x2 - x1) * t0), int(y1 + (y2 - y1) * t0))
            b = (int(x1 + (x2 - x1) * t1), int(y1 + (y2 - y1) * t1))
            self.draw.line((a, b), fill=hex_rgb(color), width=width)

    def _arrow_head(self, start: tuple[int, int], end: tuple[int, int], color: str) -> None:
        x1, y1 = start
        x2, y2 = end
        ang = math.atan2(y2 - y1, x2 - x1)
        size = 26
        spread = 0.42
        p1 = (x2, y2)
        p2 = (x2 - size * math.cos(ang - spread), y2 - size * math.sin(ang - spread))
        p3 = (x2 - size * math.cos(ang + spread), y2 - size * math.sin(ang + spread))
        self.draw.polygon([p1, p2, p3], fill=hex_rgb(color))

    def add_box(self, box: Box) -> None:
        self.rounded_rect(box.x, box.y, box.w, box.h, box.fill, box.stroke, box.stroke_width)
        self.text(box.x + 48, box.y + 52, box.title, box.title_size, box.title_color or box.stroke, bold=True)
        self.multiline(
            box.x + 48,
            box.y + 106,
            wrap_lines(box.lines, box.wrap),
            box.body_size,
            COLORS["ink"],
            line_gap=int(box.body_size * 1.42),
        )

    def save(self, path: Path) -> None:
        self.im.save(path, quality=96)


def add_static_elements(canvas: SvgCanvas | PngCanvas) -> None:
    canvas.text(
        150,
        108,
        "End-to-End Low-Visibility Classification Forecast Model",
        66,
        COLORS["ink"],
        bold=True,
    )
    canvas.text(
        150,
        174,
        "StaticRNNLowVisNet: dynamic sequence encoder + static station branch + optional feature-engineering branch",
        36,
        COLORS["muted"],
    )
    canvas.text(155, 292, "Inference graph", 28, COLORS["blue"], bold=True, uppercase=True)
    canvas.text(155, 1285, "Two-stage training", 28, COLORS["purple"], bold=True, uppercase=True)
    canvas.text(
        150,
        2070,
        "Source: vis_mlp/train_static_rnn_lowvis.py | SCI-style vector diagram on white background",
        28,
        COLORS["muted"],
    )


def draw_boxes(canvas: SvgCanvas | PngCanvas) -> dict[str, Box]:
    boxes = {
        "input": Box(
            150,
            415,
            740,
            610,
            "Input tensor x",
            [
                "X_dyn: [B, T=12, D_dyn=24-27]",
                "Forecast fields + zenith + optional PM10/PM2.5",
                "x_static: [B, 5]",
                "Vegetation id: [B] -> embedding",
                "x_FE: [B, D_FE] optional",
            ],
            COLORS["blue_l"],
            COLORS["blue"],
            wrap=31,
        ),
        "prep": Box(
            1040,
            415,
            650,
            610,
            "Runtime preprocessing",
            [
                "Resolve feature layout",
                "Optional PM channel mask",
                "log1p for skewed dynamic variables",
                "RobustScaler + clipping",
            ],
            COLORS["gray_l"],
            COLORS["line"],
            title_color=COLORS["muted"],
            wrap=29,
        ),
        "dynamic": Box(
            1880,
            535,
            780,
            250,
            "Dynamic encoder",
            [
                "Linear(D_dyn -> H) + LN + GELU",
                "GRU/LSTM over 12 h sequence",
                "mean / last / attention pooling + LN",
            ],
            COLORS["green_l"],
            COLORS["green"],
            wrap=35,
            title_size=37,
            body_size=31,
        ),
        "static": Box(
            1880,
            830,
            780,
            225,
            "Static encoder",
            [
                "5 station/static variables",
                "concat vegetation embedding",
                "two-layer MLP -> h_static",
            ],
            COLORS["purple_l"],
            COLORS["purple"],
            wrap=36,
            title_size=37,
            body_size=31,
        ),
        "fe": Box(
            1880,
            1100,
            780,
            185,
            "FE encoder (optional)",
            [
                "Engineered physical features",
                "compact MLP -> h_FE",
            ],
            COLORS["teal_l"],
            COLORS["teal"],
            wrap=36,
            title_size=37,
            body_size=31,
        ),
        "fusion": Box(
            2865,
            590,
            465,
            430,
            "Fusion MLP",
            [
                "[h_dyn || h_static || h_FE]",
                "Linear + LN + GELU + dropout",
                "latent vector z",
            ],
            "#F7F4FC",
            COLORS["purple"],
            wrap=23,
            title_size=38,
            body_size=30,
        ),
        "class": Box(
            3455,
            575,
            305,
            300,
            "Class head",
            [
                "logits/probs [B, 3]",
                "Fog: <500 m",
                "Mist: 500-1000 m",
                "Clear: >=1000 m",
            ],
            COLORS["blue_l"],
            COLORS["blue"],
            wrap=18,
            title_size=33,
            body_size=27,
        ),
        "reg": Box(
            3455,
            960,
            305,
            240,
            "Aux head",
            [
                "optional",
                "log1p(visibility)",
                "regression output",
            ],
            COLORS["green_l"],
            COLORS["green"],
            wrap=18,
            title_size=33,
            body_size=27,
        ),
        "s1": Box(
            240,
            1485,
            750,
            305,
            "Stage 1 pretraining",
            [
                "S1 aligned 12 h dataset",
                "train all parameters",
                "balanced fog / mist / clear batches",
            ],
            COLORS["blue_l"],
            COLORS["blue"],
            wrap=34,
            title_size=36,
            body_size=29,
        ),
        "s2a": Box(
            1180,
            1485,
            750,
            305,
            "Stage 2 Phase A",
            [
                "load compatible S1 checkpoint",
                "head-only adaptation",
                "fusion, heads, FE/norm/attention + L2-SP",
            ],
            COLORS["purple_l"],
            COLORS["purple"],
            wrap=35,
            title_size=36,
            body_size=29,
        ),
        "s2b": Box(
            2120,
            1485,
            750,
            305,
            "Stage 2 Phase B",
            [
                "reload best Phase A",
                "unfreeze all parameters",
                "low backbone LR + head LR + L2-SP",
            ],
            COLORS["green_l"],
            COLORS["green"],
            wrap=35,
            title_size=36,
            body_size=29,
        ),
        "eval": Box(
            3060,
            1485,
            540,
            305,
            "Validation policy",
            [
                "weighted focal loss",
                "FP penalty + recall boost",
                "fog/mist threshold search",
            ],
            COLORS["gray_l"],
            COLORS["line"],
            title_color=COLORS["muted"],
            wrap=25,
            title_size=34,
            body_size=28,
        ),
    }

    canvas.rounded_rect(1810, 350, 910, 970, COLORS["panel"], "#D8E1E8", sw=3, r=30)
    canvas.text(1880, 440, "Parallel encoders", 36, COLORS["muted"], bold=True)
    canvas.rounded_rect(150, 1350, 3610, 565, COLORS["white"], "#DCE4EA", sw=3, r=30)

    for box in boxes.values():
        canvas.add_box(box)
    return boxes


def draw_arrows(canvas: SvgCanvas | PngCanvas, b: dict[str, Box]) -> None:
    canvas.arrow((890, 720), (1040, 720), COLORS["blue"], width=8)
    canvas.arrow((1690, 650), (1880, 660), COLORS["green"], width=7)
    canvas.arrow((1690, 735), (1880, 940), COLORS["purple"], width=7)
    canvas.arrow((1690, 820), (1880, 1195), COLORS["teal"], width=7)
    canvas.arrow((2660, 660), (2865, 710), COLORS["green"], width=7)
    canvas.arrow((2660, 940), (2865, 805), COLORS["purple"], width=7)
    canvas.arrow((2660, 1195), (2865, 905), COLORS["teal"], width=7)
    canvas.arrow((3330, 720), (3455, 725), COLORS["blue"], width=8)
    canvas.arrow((3330, 850), (3455, 1080), COLORS["green"], width=7)
    canvas.text(2870, 1085, "end-to-end differentiable", 28, COLORS["muted"])

    canvas.arrow((990, 1638), (1180, 1638), COLORS["muted"], width=7)
    canvas.arrow((1930, 1638), (2120, 1638), COLORS["muted"], width=7)
    canvas.arrow((2870, 1638), (3060, 1638), COLORS["muted"], width=7)
    canvas.polyline_arrow([(3375, 1485), (3375, 1325), (3110, 1020)], COLORS["purple"], width=5, dashed=True)
    canvas.text(2715, 1330, "checkpoint selection updates deployed parameters", 27, COLORS["muted"])


def draw_svg() -> None:
    c = SvgCanvas()
    add_static_elements(c)
    boxes = draw_boxes(c)
    draw_arrows(c, boxes)
    c.save(OUT_DIR / f"{OUT_STEM}.svg")


def draw_png() -> None:
    c = PngCanvas()
    add_static_elements(c)
    boxes = draw_boxes(c)
    draw_arrows(c, boxes)
    c.save(OUT_DIR / f"{OUT_STEM}.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_svg()
    draw_png()
    print(f"Wrote {OUT_DIR / (OUT_STEM + '.svg')}")
    print(f"Wrote {OUT_DIR / (OUT_STEM + '.png')}")


if __name__ == "__main__":
    main()
