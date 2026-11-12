"""Small dependency-free HTML and PDF exporters for the paper plots."""

from __future__ import annotations

import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def render_html(svg: str, svg_name: str) -> str:
    filename = json.dumps(svg_name)
    document_title = html.escape(svg_name.rsplit(".", 1)[0].replace("_", " ").title())
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{document_title}</title><style>
body {{ margin: 0; padding: 20px; background: #f5f5f5; font-family: "Times New Roman", serif; }}
.toolbar {{ max-width: 1180px; margin: 0 auto 10px; text-align: right; }}
button {{ border: 1px solid #bbb; border-radius: 4px; background: white; padding: 7px 12px; cursor: pointer; }}
.page {{ max-width: 1180px; margin: auto; background: white; border: 1px solid #ddd; }}
svg {{ display: block; width: 100%; height: auto; }}
@page {{ size: landscape; margin: 0; }}
@media print {{
  html, body {{ margin: 0; padding: 0; background: white; }}
  body {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
  .toolbar {{ display: none !important; }}
  .page {{ width: 100%; max-width: none; margin: 0; border: 0; }}
}}
</style></head><body><div class="toolbar"><button onclick="window.print()">Print PDF</button>
<button onclick="saveSvg()">Save SVG</button></div><main class="page">{svg}</main><script>
function saveSvg() {{ const source = new XMLSerializer().serializeToString(document.querySelector("svg"));
const blob = new Blob([source], {{type: "image/svg+xml;charset=utf-8"}}); const url = URL.createObjectURL(blob);
const link = document.createElement("a"); link.href = url; link.download = {filename}; link.click();
URL.revokeObjectURL(url); }}
</script></body></html>"""


def _pdf_color(value: str) -> str:
    value = value.lstrip("#")
    channels = [int(value[index:index + 2], 16) / 255 for index in (0, 2, 4)]
    return " ".join(f"{channel:.4f}" for channel in channels)


def _pdf_text(value: str) -> str:
    replacements = {
        "–": "-", "—": "-", "α": "alpha", "γ": "gamma", "⁻": "^-", "⁰": "^0",
        "¹": "^1", "²": "^2", "³": "^3", "⁴": "^4", "⁵": "^5",
        "⁶": "^6", "⁷": "^7", "⁸": "^8", "⁹": "^9",
    }
    value = "".join(replacements.get(character, character) for character in value)
    return value.encode("latin-1", errors="replace").decode("latin-1").replace(
        "\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _text_width(value: str, size: float) -> float:
    width = 0.0
    for character in value:
        if character == " ":
            width += 0.25
        elif character in "ilI.,:;|'":
            width += 0.28
        elif character in "mwMW@":
            width += 0.82
        elif character.isupper():
            width += 0.66
        elif character.isdigit():
            width += 0.50
        else:
            width += 0.48
    return width * size


def write_pdf_from_svg(svg: str, path: Path) -> None:
    """Translate the small SVG subset used by plot.py into a standalone PDF."""
    root = ET.fromstring(svg)
    width = float(root.attrib["viewBox"].split()[2])
    height = float(root.attrib["viewBox"].split()[3])
    commands: list[str] = []
    class_sizes = {
        "figure-heading": 20,
        "legend": 17,
        "tick": 15,
        "axis-label": 19,
        "panel-label": 17,
    }

    def number(element: ET.Element, name: str, default: float = 0) -> float:
        return float(element.attrib.get(name, default))

    def style(element: ET.Element, fill_default: str = "none") -> tuple[str, str, float]:
        css_class = element.attrib.get("class", "")
        stroke = element.attrib.get(
            "stroke", "#9a9a9a" if css_class == "grid-line" else "none")
        fill = element.attrib.get("fill", fill_default)
        stroke_width = number(
            element, "stroke-width", 0.75 if css_class == "grid-line" else 1)
        return fill, stroke, stroke_width

    def paint(
            fill: str, stroke: str, stroke_width: float, geometry: str,
            dashed: bool = False) -> None:
        commands.append("q")
        if fill != "none":
            commands.append(f"{_pdf_color(fill)} rg")
        if stroke != "none":
            commands.append(f"{_pdf_color(stroke)} RG {stroke_width:.3f} w")
            commands.append("[2 3] 0 d" if dashed else "[] 0 d")
        commands.append(geometry)
        commands.append(
            "B" if fill != "none" and stroke != "none" else
            "f" if fill != "none" else "S")
        commands.append("Q")

    def visit(element: ET.Element) -> None:
        tag = element.tag.rsplit("}", 1)[-1]
        if tag in {"defs", "clipPath", "title", "desc", "style"}:
            return
        if tag in {"svg", "g"}:
            for child in element:
                visit(child)
            return
        fill, stroke, stroke_width = style(element)
        if tag == "rect":
            x, y = number(element, "x"), number(element, "y")
            rectangle_width = number(element, "width")
            rectangle_height = number(element, "height")
            paint(
                fill, stroke, stroke_width,
                f"{x:.3f} {height-y-rectangle_height:.3f} "
                f"{rectangle_width:.3f} {rectangle_height:.3f} re")
        elif tag == "line":
            x1, y1 = number(element, "x1"), height - number(element, "y1")
            x2, y2 = number(element, "x2"), height - number(element, "y2")
            paint(
                "none", stroke, stroke_width,
                f"{x1:.3f} {y1:.3f} m {x2:.3f} {y2:.3f} l",
                element.attrib.get("class") == "grid-line")
        elif tag == "circle":
            x = number(element, "cx")
            y = height - number(element, "cy")
            radius = number(element, "r")
            factor = radius * 0.5522847498
            geometry = (
                f"{x+radius:.3f} {y:.3f} m "
                f"{x+radius:.3f} {y+factor:.3f} {x+factor:.3f} {y+radius:.3f} {x:.3f} {y+radius:.3f} c "
                f"{x-factor:.3f} {y+radius:.3f} {x-radius:.3f} {y+factor:.3f} {x-radius:.3f} {y:.3f} c "
                f"{x-radius:.3f} {y-factor:.3f} {x-factor:.3f} {y-radius:.3f} {x:.3f} {y-radius:.3f} c "
                f"{x+factor:.3f} {y-radius:.3f} {x+radius:.3f} {y-factor:.3f} {x+radius:.3f} {y:.3f} c h")
            paint(fill, stroke, stroke_width, geometry)
        elif tag in {"polygon", "polyline"}:
            points = [
                tuple(map(float, item.split(",")))
                for item in element.attrib["points"].split()
            ]
            geometry = (
                f"{points[0][0]:.3f} {height-points[0][1]:.3f} m " +
                " ".join(f"{x:.3f} {height-y:.3f} l" for x, y in points[1:]))
            if tag == "polygon":
                geometry += " h"
            paint(fill, stroke, stroke_width, geometry)
        elif tag == "text":
            safe_value = _pdf_text("".join(element.itertext()))
            css_class = element.attrib.get("class", "")
            size = float(element.attrib.get("font-size", class_sizes.get(css_class, 15)))
            font = "/F2" if css_class == "figure-heading" else "/F1"
            x, y = number(element, "x"), height - number(element, "y")
            text_width = _text_width(safe_value, size)
            anchor = element.attrib.get("text-anchor")
            offset = (-text_width / 2 if anchor == "middle"
                      else -text_width if anchor == "end" else 0)
            if element.attrib.get("transform", "").startswith("rotate(-90"):
                commands.append(
                    f"BT {font} {size:.2f} Tf 0 1 -1 0 {x:.3f} {y:.3f} Tm "
                    f"{offset:.3f} 0 Td ({safe_value}) Tj ET")
            else:
                commands.append(
                    f"BT {font} {size:.2f} Tf {x+offset:.3f} {y:.3f} Td "
                    f"({safe_value}) Tj ET")

    visit(root)
    content = ("\n".join(commands) + "\n").encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:g} {height:g}] "
         "/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> /Contents 4 0 R >>").encode(),
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Bold >>",
    ]
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, payload in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode())
        output.extend(payload)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects)+1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend((
        f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n").encode())
    path.write_bytes(output)
