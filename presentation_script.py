"""
presentation_script.py

Reads Report.docx and generates a presentation script Word document
(Presentation_Script.docx) that pairs each paragraph of text with
a display cue — telling the presenter what slide, figure, table,
or visual to show while reading that section aloud.
"""

from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import re
import os

INPUT_FILE = "Report.docx"
OUTPUT_FILE = "Presentation_Script.docx"


def _detect_display_cue(text, style_name, context):
    """Return a display-cue string describing what to show on screen
    while presenting the given paragraph.

    Parameters
    ----------
    text : str
        The paragraph text.
    style_name : str
        The Word style name (e.g. 'Heading 1', 'Normal', 'List Bullet').
    context : dict
        Mutable dict that tracks state across paragraphs (current section,
        figure counter, etc.).
    """

    lower = text.lower()

    # --- Heading → new slide ------------------------------------------------
    if "Heading" in style_name:
        context["current_section"] = text
        context["slide_number"] = context.get("slide_number", 0) + 1
        return (
            f">> DISPLAY: New slide (Slide {context['slide_number']}) — "
            f"Title: \"{text}\""
        )

    # --- Explicit figure references -----------------------------------------
    fig_match = re.search(r"figure\s*(\d+)", lower)
    if fig_match:
        fig_num = fig_match.group(1)
        return (
            f">> DISPLAY: Show Figure {fig_num} on screen. "
            f"Keep it visible while reading this paragraph."
        )

    # --- Explicit table references ------------------------------------------
    tbl_match = re.search(r"table\s*(\d+)", lower)
    if tbl_match:
        tbl_num = tbl_match.group(1)
        return (
            f">> DISPLAY: Show Table {tbl_num} on screen. "
            f"Keep it visible while reading this paragraph."
        )

    # --- Bullet / list items ------------------------------------------------
    if "List" in style_name or "Bullet" in style_name:
        return (
            ">> DISPLAY: Show this bullet point on the current slide "
            "(reveal one bullet at a time)."
        )

    # --- Paragraphs about visualisations / charts ---------------------------
    viz_keywords = [
        "heatmap", "boxplot", "bar chart", "histogram",
        "roc curve", "confusion matrix", "correlation",
        "distribution", "plot", "visualisation", "visualization",
    ]
    for kw in viz_keywords:
        if kw in lower:
            return (
                f">> DISPLAY: Show the relevant visualisation "
                f"({kw.title()}) while reading this paragraph."
            )

    # --- Paragraphs about model results / metrics ---------------------------
    metric_keywords = [
        "accuracy", "f1-score", "f1 score", "precision", "recall",
        "roc-auc", "auc", "cross-validation",
    ]
    for kw in metric_keywords:
        if kw in lower:
            return (
                ">> DISPLAY: Show the results / metrics summary "
                "on the current slide."
            )

    # --- References section -------------------------------------------------
    section = context.get("current_section", "")
    if "reference" in section.lower():
        return ">> DISPLAY: Show References slide."

    # --- Title / author line (before first heading) -------------------------
    if context.get("slide_number", 0) == 0:
        return ">> DISPLAY: Show Title slide with the report title and author."

    # --- Default: keep current slide ----------------------------------------
    return ">> DISPLAY: Keep the current slide visible."


def generate_presentation_script(input_path, output_path):
    """Read *input_path* (.docx) and write *output_path* (.docx) with
    each paragraph followed by a coloured display cue."""

    src = Document(input_path)
    dst = Document()

    # -- Title ---------------------------------------------------------------
    title = dst.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Presentation Script")
    run.bold = True
    run.font.size = Pt(18)

    subtitle = dst.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run(
        f"Generated from: {os.path.basename(input_path)}\n"
        "Read each paragraph aloud; the cue beneath tells you "
        "what to display on screen."
    )
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(100, 100, 100)

    dst.add_paragraph()  # spacer

    context = {"slide_number": 0, "current_section": ""}

    for para in src.paragraphs:
        text = para.text.strip()
        style_name = para.style.name if para.style else "Normal"

        # Skip completely empty paragraphs
        if not text:
            continue

        # --- Write the original text ----------------------------------------
        p = dst.add_paragraph()

        if "Heading" in style_name:
            run = p.add_run(text)
            run.bold = True
            run.font.size = Pt(14) if "1" in style_name else Pt(12)
        elif "List" in style_name or "Bullet" in style_name:
            run = p.add_run(f"• {text}")
            run.font.size = Pt(11)
        else:
            run = p.add_run(text)
            run.font.size = Pt(11)

        # --- Write the display cue ------------------------------------------
        cue_text = _detect_display_cue(text, style_name, context)
        cue_para = dst.add_paragraph()
        cue_run = cue_para.add_run(cue_text)
        cue_run.bold = True
        cue_run.italic = True
        cue_run.font.size = Pt(10)
        cue_run.font.color.rgb = RGBColor(0, 102, 204)  # blue

        # thin separator
        sep = dst.add_paragraph()
        sep_run = sep.add_run("─" * 60)
        sep_run.font.size = Pt(6)
        sep_run.font.color.rgb = RGBColor(200, 200, 200)

    dst.save(output_path)
    print(f"Presentation script saved to: {output_path}")


if __name__ == "__main__":
    generate_presentation_script(INPUT_FILE, OUTPUT_FILE)
