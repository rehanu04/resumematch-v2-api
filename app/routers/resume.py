from __future__ import annotations

from io import BytesIO

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.core.security import require_app_key

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

router = APIRouter(prefix="/v1", tags=["resume"])


class ResumePdfRequest(BaseModel):
    jd_text: str = Field(default="")
    template: str = Field(default="Classic ATS")

    first_name: str = Field(default="")
    last_name: str = Field(default="")
    email: str = Field(default="")
    phone: str = Field(default="")
    location: str = Field(default="")
    target_role: str = Field(default="")

    summary: str = Field(default="")
    skills: list[str] = Field(default_factory=list)

    experience_text: str = Field(default="")
    projects_text: str = Field(default="")
    education_text: str = Field(default="")
    extras_text: str = Field(default="")


def _wrap(text: str, font: str, size: int, max_width: float) -> list[str]:
    if not text:
        return []
    words = text.replace("\t", " ").split()
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        trial = (" ".join(cur + [w])).strip()
        if stringWidth(trial, font, size) <= max_width:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def _split_paragraphs(block: str) -> list[str]:
    if not block:
        return []
    raw = block.replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    buf: list[str] = []
    for line in raw.split("\n"):
        if line.strip() == "":
            if buf:
                out.append("\n".join(buf).strip())
                buf = []
        else:
            buf.append(line.rstrip())
    if buf:
        out.append("\n".join(buf).strip())
    return [p for p in out if p.strip()]


def _is_bullet(line: str) -> bool:
    s = line.strip()
    return s.startswith(("•", "-", "*"))


def _clean_bullet(line: str) -> str:
    s = line.strip()
    if s.startswith("•"):
        return s[1:].strip()
    if s.startswith(("-", "*")):
        return s[1:].strip()
    return s


@router.post("/resume/pdf")
def generate_resume_pdf(payload: ResumePdfRequest, _=Depends(require_app_key)):
    tpl = (payload.template or "Classic ATS").strip().lower()
    is_min = "min" in tpl

    # ATS-safe typography
    name_size = 18 if is_min else 16
    body_size = 10
    section_size = 11

    top = 54
    left = 54
    right = 54
    bottom = 54

    line_gap = 13 if is_min else 12
    section_gap = 10 if is_min else 8

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    width, height = LETTER

    x = left
    y = height - top
    maxw = width - left - right

    def new_page():
        nonlocal y
        c.showPage()
        y = height - top

    def ensure_space(needed: float):
        nonlocal y
        if y - needed < bottom:
            new_page()

    def draw_text_lines(lines: list[str], font: str, size: int, gap: int, indent: float = 0.0):
        nonlocal y
        c.setFont(font, size)
        for ln in lines:
            ensure_space(gap + 2)
            c.drawString(x + indent, y, ln)
            y -= gap

    def draw_section(title: str):
        nonlocal y
        ensure_space(28)
        y -= section_gap
        c.setFont("Helvetica-Bold", section_size)
        c.drawString(x, y, title.upper())
        y -= 6
        c.setLineWidth(0.6)
        c.setStrokeGray(0.6)
        c.line(x, y, x + maxw, y)
        c.setStrokeGray(0)
        y -= 10

    # Header
    full_name = (payload.first_name + " " + payload.last_name).strip() or "Resume"
    c.setFont("Helvetica-Bold", name_size)
    c.drawString(x, y, full_name)
    y -= 22 if is_min else 20

    role = payload.target_role.strip()
    if role:
        c.setFont("Helvetica", 11)
        c.drawString(x, y, role)
        y -= 16

    contact_parts = [payload.email.strip(), payload.phone.strip(), payload.location.strip()]
    contact = " • ".join([p for p in contact_parts if p])
    if contact:
        c.setFont("Helvetica", 9)
        c.setFillGray(0.15)
        c.drawString(x, y, contact)
        c.setFillGray(0)
        y -= 16

    # Summary
    summary = payload.summary.strip()
    if summary:
        draw_section("Summary")
        draw_text_lines(_wrap(summary, "Helvetica", body_size, maxw), "Helvetica", body_size, line_gap)

    # Skills (2 columns)
    skills = [s.strip() for s in payload.skills if s.strip()]
    if skills:
        skills = list(dict.fromkeys(skills))
        draw_section("Skills")
        col_gap = 18
        col_w = (maxw - col_gap) / 2
        c.setFont("Helvetica", body_size)
        rows = [skills[i:i + 2] for i in range(0, len(skills), 2)]
        for r in rows:
            ensure_space(line_gap + 2)
            c.drawString(x, y, r[0])
            if len(r) > 1:
                c.drawString(x + col_w + col_gap, y, r[1])
            y -= line_gap

    def render_block(title: str, block: str):
        blk = block.strip()
        if not blk:
            return
        draw_section(title)
        paras = _split_paragraphs(blk)
        for p in paras:
            for ln in p.split("\n"):
                if not ln.strip():
                    continue
                if _is_bullet(ln):
                    bullet = _clean_bullet(ln)
                    wrapped = _wrap(bullet, "Helvetica", body_size, maxw - 14)
                    if wrapped:
                        ensure_space(line_gap + 2)
                        c.setFont("Helvetica", body_size)
                        c.drawString(x, y, "•")
                        c.drawString(x + 12, y, wrapped[0])
                        y -= line_gap
                        if len(wrapped) > 1:
                            draw_text_lines(wrapped[1:], "Helvetica", body_size, line_gap, indent=12)
                else:
                    draw_text_lines(_wrap(ln.strip(), "Helvetica", body_size, maxw), "Helvetica", body_size, line_gap)
            y -= 6

    render_block("Experience", payload.experience_text)
    render_block("Projects", payload.projects_text)
    render_block("Education", payload.education_text)

    extras = payload.extras_text.strip()
    if extras:
        render_block("Additional", extras)

    c.showPage()
    c.save()

    pdf_bytes = buf.getvalue()
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'inline; filename="resume.pdf"'},
    )
