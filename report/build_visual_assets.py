from __future__ import annotations

import math
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"

# Restrained, colorblind-aware palette inspired by Okabe-Ito / Paul Tol.
CONTROL = "#0072B2"  # pipeline/control blue
PASS = "#009E73"  # eligible/pass bluish green
WARN = "#E69F00"  # pending/offline amber
ROUTE = "#CC79A7"  # route/decision purple-magenta
NEG = "#D55E00"  # stress/negative vermillion
SKY = "#56B4E9"
INK = "#1B2733"
MUTED = "#666666"
GRID = "#D9DDE3"
SOFT = "#F4F4F4"
WHITE = "#FFFFFF"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\timesbd.ttf" if bold else r"C:\Windows\Fonts\times.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def wrap_lines(text: str, max_chars: int) -> list[str]:
    out: list[str] = []
    for part in text.split("\n"):
        out.extend(textwrap.wrap(part, max_chars) or [""])
    return out


def centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fnt, fill=INK, gap=7):
    x1, y1, x2, y2 = box
    max_chars = max(8, (x2 - x1) // max(9, fnt.size // 2))
    lines = wrap_lines(text, max_chars)
    sizes = [draw.textbbox((0, 0), line, font=fnt) for line in lines]
    widths = [b[2] - b[0] for b in sizes]
    heights = [b[3] - b[1] for b in sizes]
    total_h = sum(heights) + gap * (len(lines) - 1)
    y = y1 + ((y2 - y1) - total_h) / 2
    for line, w, h in zip(lines, widths, heights):
        draw.text((x1 + ((x2 - x1) - w) / 2, y), line, font=fnt, fill=fill)
        y += h + gap


def wrapped(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], width: int, fnt, fill=INK, line_gap=7):
    x, y = xy
    max_chars = max(10, width // max(9, fnt.size // 2))
    for line in wrap_lines(text, max_chars):
        draw.text((x, y), line, font=fnt, fill=fill)
        bbox = draw.textbbox((x, y), line, font=fnt)
        y += bbox[3] - bbox[1] + line_gap
    return y


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill=MUTED, width=6):
    draw.line([start, end], fill=fill, width=width)
    sx, sy = start
    ex, ey = end
    angle = math.atan2(ey - sy, ex - sx)
    size = 18
    left = (ex - size * math.cos(angle - math.pi / 6), ey - size * math.sin(angle - math.pi / 6))
    right = (ex - size * math.cos(angle + math.pi / 6), ey - size * math.sin(angle + math.pi / 6))
    draw.polygon([end, left, right], fill=fill)


def elbow_arrow(draw, points: list[tuple[int, int]], fill=MUTED, width=6):
    for a, b in zip(points, points[1:]):
        draw.line([a, b], fill=fill, width=width)
    arrow(draw, points[-2], points[-1], fill=fill, width=width)


def pill(draw, box, text, fill, outline=None, text_fill=WHITE, size=31):
    if outline is None:
        outline = fill
    draw.rounded_rectangle(box, radius=(box[3] - box[1]) // 2, fill=fill, outline=outline, width=4)
    centered(draw, box, text, font(size, True), fill=text_fill, gap=4)


def diamond(draw, cx, cy, w, h, text, fill, outline=None, text_fill=WHITE, size=28):
    if outline is None:
        outline = fill
    pts = [(cx, cy - h // 2), (cx + w // 2, cy), (cx, cy + h // 2), (cx - w // 2, cy)]
    draw.polygon(pts, fill=fill, outline=outline)
    draw.line(pts + [pts[0]], fill=outline, width=4)
    centered(draw, (cx - w // 2 + 70, cy - h // 2 + 48, cx + w // 2 - 70, cy + h // 2 - 48), text, font(size, True), fill=text_fill, gap=6)


def card(draw, box, title, body, fill=SOFT, outline=MUTED, title_size=34, body_size=25):
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=outline, width=3)
    x1, y1, x2, _ = box
    draw.text((x1 + 22, y1 + 18), title, font=font(title_size, True), fill=INK)
    wrapped(draw, body, (x1 + 22, y1 + 70), x2 - x1 - 44, font(body_size), fill=INK, line_gap=6)


def footer(draw, text: str, outline=CONTROL):
    draw.rounded_rectangle((80, 1715, 1420, 1792), radius=22, fill="#F7F8FA", outline=outline, width=3)
    centered(draw, (110, 1720, 1390, 1786), text, font(25, True), fill=INK)


def pipeline_flowchart():
    img = Image.new("RGB", (1500, 1830), WHITE)
    draw = ImageDraw.Draw(img)
    draw.text((70, 36), "Receipt-Safe Tool-Choice Diagnostic Flow", font=font(52, True), fill=INK)
    draw.text((72, 98), "Diagnosis is not interpreted before receipt gates pass.", font=font(28), fill=MUTED)

    top_y = 185
    pill(draw, (90, top_y, 380, top_y + 84), "State x_t", CONTROL, size=30)
    pill(draw, (475, top_y, 805, top_y + 84), "Gen candidates", CONTROL, size=30)
    pill(draw, (895, top_y, 1285, top_y + 84), "Gate(C_t, M_t)", CONTROL, size=30)
    arrow(draw, (382, top_y + 42), (470, top_y + 42), width=5)
    arrow(draw, (807, top_y + 42), (890, top_y + 42), width=5)

    diamond(draw, 750, 430, 560, 245, "Does oracle replay mark a\nlive tool-choice opportunity?", CONTROL, size=27)
    arrow(draw, (1090, top_y + 84), (805, 318), width=5)
    draw.text((816, 295), "candidate set", font=font(21, True), fill=MUTED)

    pill(draw, (125, 570, 430, 655), "No opportunity", PASS, text_fill=WHITE, size=28)
    pill(draw, (520, 650, 980, 742), "Select action + log JSONL", WARN, text_fill=INK, size=28)
    pill(draw, (1030, 650, 1365, 742), "Stop: not evidence", WARN, text_fill=INK, size=27)
    elbow_arrow(draw, [(470, 430), (275, 430), (275, 565)], fill=PASS, width=5)
    arrow(draw, (750, 552), (750, 645), fill=WARN, width=5)
    arrow(draw, (985, 696), (1025, 696), fill=WARN, width=5)
    draw.text((292, 456), "No", font=font(23, True), fill=PASS)
    draw.text((768, 580), "Yes", font=font(23, True), fill="#8A5E00")

    diamond(draw, 750, 910, 550, 250, "Do receipt hard gates pass?\nrun, JSONL, pairing, safe", CONTROL, size=27)
    arrow(draw, (750, 742), (750, 785), width=5)
    pill(draw, (125, 900, 455, 985), "Repair / stop", PASS, size=28)
    elbow_arrow(draw, [(480, 910), (455, 910)], fill=PASS, width=5)
    draw.text((506, 858), "No", font=font(23, True), fill=PASS)

    pill(draw, (520, 1118, 980, 1210), "Read diagnosis\nand battery summary", WARN, text_fill=INK, size=27)
    arrow(draw, (750, 1035), (750, 1112), fill=WARN, width=5)
    draw.text((768, 1060), "Yes", font=font(23, True), fill="#8A5E00")

    pill(draw, (135, 1360, 380, 1444), "Route A", ROUTE, size=28)
    pill(draw, (520, 1360, 980, 1444), "Route B: stable negative", ROUTE, size=28)
    pill(draw, (1120, 1360, 1370, 1444), "Route C", ROUTE, size=28)
    pill(draw, (85, 1560, 445, 1648), "continue confirmation", ROUTE, size=24)
    pill(draw, (505, 1560, 995, 1648), "task-pressure review\nor route closure", ROUTE, size=24)
    pill(draw, (1080, 1560, 1425, 1648), "layer diagnosis", ROUTE, size=24)
    elbow_arrow(draw, [(750, 1210), (750, 1312), (258, 1312), (258, 1355)], fill=ROUTE, width=5)
    arrow(draw, (750, 1210), (750, 1355), fill=ROUTE, width=5)
    elbow_arrow(draw, [(750, 1312), (1245, 1312), (1245, 1355)], fill=ROUTE, width=5)
    arrow(draw, (258, 1444), (258, 1555), fill=ROUTE, width=5)
    arrow(draw, (750, 1444), (750, 1555), fill=ROUTE, width=5)
    arrow(draw, (1245, 1444), (1245, 1555), fill=ROUTE, width=5)

    footer(draw, "Final v4 seed7 follows Route B: trigger_good = [], negative_inversion = [], blocked steps = task_pressure.")
    img.save(FIG_DIR / "pipeline_flowchart.png")


def evidence_ladder():
    img = Image.new("RGB", (1800, 860), WHITE)
    draw = ImageDraw.Draw(img)
    draw.text((70, 40), "Post-Midway Evidence Ladder", font=font(52, True), fill=INK)
    draw.text((72, 100), "Offline pack eligibility is separate from receipt-safe live evidence.", font=font(28), fill=MUTED)

    stages = [
        ("Smoke", "10-task\nopportunity visible\nclaim bounded", SKY, "offline/smoke"),
        ("v3 seed7", "dual battery\nreceipt-safe\npure task_pressure", SOFT, "live"),
        ("seed11 stress", "stress-only\nsignal reopens\nnot confirmation", "#FFF5DA", "live probe"),
        ("seed11 confirm", "repaired ref.\nRoute B\nno trigger-good", "#FAEAF3", "live"),
        ("v4 offline", "repair + stress\npreflight clean\nnot live evidence", "#EAF6F1", "offline"),
        ("v4 seed7 live", "Route B\nstable negative\nno claim promotion", "#F3E9F2", "live"),
    ]
    x0, y, bw, bh, gap = 55, 235, 255, 255, 38
    boxes = []
    for i, (title, body, fill, badge) in enumerate(stages):
        x = x0 + i * (bw + gap)
        box = (x, y, x + bw, y + bh)
        boxes.append(box)
        card(draw, box, title, body, fill=fill, outline=MUTED, title_size=31, body_size=24)
        pill(draw, (x + 22, y + bh - 58, x + bw - 22, y + bh - 18), badge, CONTROL if "live" in badge else WARN, text_fill=WHITE if "live" in badge else INK, size=20)
        if i:
            arrow(draw, (boxes[i - 1][2] + 8, y + bh // 2), (x - 10, y + bh // 2), width=5)

    draw.rounded_rectangle((130, 610, 1670, 712), radius=22, fill="#F7F8FA", outline=CONTROL, width=3)
    centered(draw, (160, 620, 1640, 702), "Claim boundary: whether-to-ask improvement accepted; stable which-tool improvement not accepted.", font(29, True), fill=INK)
    img.save(FIG_DIR / "evidence_ladder.png")


def metrics_bars():
    img = Image.new("RGB", (1800, 900), WHITE)
    draw = ImageDraw.Draw(img)
    draw.text((70, 40), "Final Locked Comparison: voi_memory_v1", font=font(52, True), fill=INK)
    draw.text((72, 100), "Reference vs stress does not improve the which-tool metrics.", font=font(28), fill=MUTED)

    metrics = [
        ("choose-swallow\nlower better", 0.5291, 0.5376, "worse"),
        ("family mismatch\nlower better", 0.5291, 0.5376, "worse"),
        ("verify usage\nhigher better", 0.2063, 0.0860, "drops"),
    ]
    x1, y1, x2, y2 = 145, 210, 1280, 720
    draw.line((x1, y2, x2, y2), fill=INK, width=3)
    draw.line((x1, y1, x1, y2), fill=INK, width=3)
    for tick in [0, 0.25, 0.50, 0.75, 1.0]:
        y = y2 - int((y2 - y1) * tick)
        draw.line((x1 - 9, y, x2, y), fill=GRID, width=2)
        draw.text((67, y - 15), f"{tick:.2f}", font=font(23), fill=MUTED)

    group_w, bar_w = (x2 - x1) / 3, 92
    for idx, (label, ref, stress, callout) in enumerate(metrics):
        cx = x1 + group_w * idx + group_w / 2
        for offset, value, color, name in [(-62, ref, CONTROL, "ref"), (62, stress, NEG, "stress")]:
            bx1, bx2 = int(cx + offset - bar_w / 2), int(cx + offset + bar_w / 2)
            by1 = y2 - int((y2 - y1) * value)
            draw.rounded_rectangle((bx1, by1, bx2, y2), radius=8, fill=color)
            centered(draw, (bx1 - 15, by1 - 42, bx2 + 15, by1 - 8), f"{value:.3f}", font(24, True), fill=INK)
            draw.text((bx1 + 7, y2 + 12), name, font=font(19, True), fill=MUTED)
        draw.text((int(cx - 145), y2 + 62), label, font=font(25, True), fill=INK)
        draw.text((int(cx - 50), y1 + 16), callout, font=font(24, True), fill=NEG)
        arrow(draw, (int(cx - 12), y1 + 46), (int(cx + 35), y1 + 46), fill=NEG, width=4)

    draw.rounded_rectangle((1330, 240, 1700, 675), radius=20, fill="#F7F8FA", outline=MUTED, width=3)
    wrapped(
        draw,
        "Route B\ntrigger_good = []\nnegative_inversion = []\nblocked-step audit:\n100/100 task_pressure\nin both packs",
        (1360, 272),
        315,
        font(28, True),
        fill=INK,
        line_gap=9,
    )
    img.save(FIG_DIR / "v4_final_metrics.png")


def case_walkthrough():
    img = Image.new("RGB", (1800, 820), WHITE)
    draw = ImageDraw.Draw(img)
    draw.text((70, 40), "One Logged Step: Opportunity Without Success", font=font(52, True), fill=INK)
    draw.text((72, 100), "Opportunity can exist even when the selected action fails the oracle tool family.", font=font(28), fill=MUTED)

    y = 210
    nodes = [
        ((70, y, 405, y + 175), "State text", "Ambiguous clue.\nScan is cheap.\nVerify resolves conflict.", "#E8F4FD"),
        ((510, y, 845, y + 175), "Candidates", "choose now\nscan\nverify", "#FFF5DA"),
        ((950, y, 1285, y + 175), "Oracle family", "verify is useful\nafter scan ambiguity", "#EAF6F1"),
        ((1390, y, 1725, y + 175), "Logged action", "direct choose\nor ASK blocked", "#FBECE5"),
    ]
    for box, title, body, fill in nodes:
        card(draw, box, title, body, fill=fill, outline=MUTED, title_size=33, body_size=22)
    for a, b in zip(nodes, nodes[1:]):
        arrow(draw, (a[0][2] + 12, y + 88), (b[0][0] - 12, y + 88), width=5)

    diamond(draw, 900, 555, 520, 190, "Does this count as\nwhich-tool improvement?", CONTROL, size=27)
    pill(draw, (165, 588, 520, 670), "Opportunity: yes", PASS, size=27)
    pill(draw, (1120, 588, 1590, 670), "Improvement evidence: no", ROUTE, size=27)
    elbow_arrow(draw, [(640, 555), (520, 555), (520, 628)], fill=PASS, width=5)
    elbow_arrow(draw, [(1160, 555), (1120, 555), (1120, 628)], fill=ROUTE, width=5)
    draw.rounded_rectangle((360, 735, 1440, 790), radius=18, fill=SOFT, outline=MUTED, width=3)
    centered(draw, (390, 738, 1410, 786), "Diagnosis routes the blocked opportunity to task_pressure, not to a memory/selector fix.", font(26, True), fill=INK)
    img.save(FIG_DIR / "case_walkthrough.png")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_flowchart()
    evidence_ladder()
    metrics_bars()
    case_walkthrough()
    print(FIG_DIR)


if __name__ == "__main__":
    main()
