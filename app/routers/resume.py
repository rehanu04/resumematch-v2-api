from __future__ import annotations

import re
from io import BytesIO

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from app.core.security import require_app_key
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

router = APIRouter(prefix="/v1", tags=["resume"])

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9+.#/-]*")
_STOP = {
    "and", "or", "the", "a", "an", "to", "of", "in", "on", "for", "with", "as", "is", "are", "be", "by", "at",
    "we", "you", "your", "our", "they", "their", "need", "needs", "required", "requirements", "must", "should",
    "experience", "years", "year", "role", "job", "work", "working", "ability", "skills", "skill", "strong", "knowledge",
}


class ResumePdfRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    jd_text: str = Field(default="", validation_alias=AliasChoices("jd_text", "jd"))
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

    linkedin: str = Field(default="")
    github: str = Field(default="")
    portfolio: str = Field(default="")



def _normalize(text: str) -> str:
    return (text or "").lower()



def _wrap(text: str, font: str, size: int, max_width: float) -> list[str]:
    if not text:
        return []
    words = text.replace("\t", " ").split()
    lines: list[str] = []
    cur: list[str] = []
    for word in words:
        trial = (" ".join(cur + [word])).strip()
        if stringWidth(trial, font, size) <= max_width:
            cur.append(word)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [word]
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
    return line.strip().startswith(("•", "-", "*"))



def _clean_bullet(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("•"):
        return stripped[1:].strip()
    if stripped.startswith(("-", "*")):
        return stripped[1:].strip()
    return stripped



def _extract_jd_terms(jd_text: str) -> list[str]:
    counts: dict[str, int] = {}
    for token in _WORD_RE.findall(_normalize(jd_text)):
        if len(token) < 3 or token in _STOP:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [term for term, _ in ranked[:30]]



def _prioritize_skills(skills: list[str], jd_text: str) -> list[str]:
    jd_terms = set(_extract_jd_terms(jd_text))
    deduped = list(dict.fromkeys([item.strip() for item in skills if item.strip()]))

    def score(skill: str) -> tuple[int, str]:
        norm = _normalize(skill)
        token_hits = sum(1 for token in jd_terms if token in norm)
        exact = 2 if norm in jd_terms else 0
        return (exact + token_hits, norm)

    return sorted(deduped, key=lambda item: (-score(item)[0], score(item)[1]))



def _line_score(line: str, jd_terms: set[str]) -> int:
    norm = _normalize(line)
    return sum(1 for term in jd_terms if term in norm)



def _prioritize_paragraph_lines(paragraph: str, jd_terms: set[str]) -> list[str]:
    lines = [line for line in paragraph.split("\n") if line.strip()]
    if not lines:
        return []

    headers = [line for line in lines if not _is_bullet(line)]
    bullets = [line for line in lines if _is_bullet(line)]
    bullets_sorted = sorted(bullets, key=lambda line: (-_line_score(line, jd_terms), lines.index(line)))
    return headers + bullets_sorted


@router.post("/resume/pdf")
def generate_resume_pdf(payload: ResumePdfRequest, _=Depends(require_app_key)):
    template_value = (payload.template or "Classic ATS").strip().lower()
    is_minimal = "min" in template_value

    name_size = 18 if is_minimal else 16
    body_size = 10
    section_size = 11

    top = 58 if is_minimal else 54
    left = 54
    right = 54
    bottom = 54

    line_gap = 13 if is_minimal else 12
    section_gap = 10 if is_minimal else 8

    jd_terms = set(_extract_jd_terms(payload.jd_text))
    prioritized_skills = _prioritize_skills(payload.skills, payload.jd_text)

    buf = BytesIO()
    pdf = canvas.Canvas(buf, pagesize=LETTER)
    width, height = LETTER

    x = left
    y = height - top
    maxw = width - left - right

    def new_page():
        nonlocal y
        pdf.showPage()
        y = height - top

    def ensure_space(needed: float):
        nonlocal y
        if y - needed < bottom:
            new_page()

    def draw_text_lines(lines: list[str], font: str, size: int, gap: int, indent: float = 0.0):
        nonlocal y
        pdf.setFont(font, size)
        for line in lines:
            ensure_space(gap + 2)
            pdf.drawString(x + indent, y, line)
            y -= gap

    def draw_section(title: str):
        nonlocal y
        ensure_space(28)
        y -= section_gap
        pdf.setFont("Helvetica-Bold", section_size)
        pdf.drawString(x, y, title.upper())
        y -= 6
        pdf.setLineWidth(0.6)
        pdf.setStrokeGray(0.6)
        pdf.line(x, y, x + maxw, y)
        pdf.setStrokeGray(0)
        y -= 10

    full_name = (payload.first_name + " " + payload.last_name).strip() or "Resume"
    pdf.setFont("Helvetica-Bold", name_size)
    pdf.drawString(x, y, full_name)
    y -= 22 if is_minimal else 20

    role = payload.target_role.strip()
    if role:
        pdf.setFont("Helvetica", 11)
        pdf.drawString(x, y, role)
        y -= 16

    contact_parts = [payload.email.strip(), payload.phone.strip(), payload.location.strip()]
    contact = " • ".join([part for part in contact_parts if part])
    if contact:
        pdf.setFont("Helvetica", 9)
        pdf.setFillGray(0.15)
        draw_text_lines(_wrap(contact, "Helvetica", 9, maxw), "Helvetica", 9, 11)
        pdf.setFillGray(0)

    link_parts = []
    if payload.linkedin.strip():
        link_parts.append(f"LinkedIn: {payload.linkedin.strip()}")
    if payload.github.strip():
        link_parts.append(f"GitHub: {payload.github.strip()}")
    if payload.portfolio.strip():
        link_parts.append(f"Portfolio: {payload.portfolio.strip()}")
    if link_parts:
        pdf.setFont("Helvetica", 8)
        pdf.setFillGray(0.2)
        draw_text_lines(_wrap(" • ".join(link_parts), "Helvetica", 8, maxw), "Helvetica", 8, 10)
        pdf.setFillGray(0)
        y -= 2

    summary = payload.summary.strip()
    if summary:
        draw_section("Summary")
        draw_text_lines(_wrap(summary, "Helvetica", body_size, maxw), "Helvetica", body_size, line_gap)

    if prioritized_skills:
        draw_section("Skills")
        skills_line = ", ".join(prioritized_skills)
        draw_text_lines(_wrap(skills_line, "Helvetica", body_size, maxw), "Helvetica", body_size, line_gap)

    def render_block(title: str, block: str):
        nonlocal y
        blk = block.strip()
        if not blk:
            return
        draw_section(title)
        paragraphs = _split_paragraphs(blk)
        for paragraph in paragraphs:
            prioritized_lines = _prioritize_paragraph_lines(paragraph, jd_terms)
            for raw_line in prioritized_lines:
                if not raw_line.strip():
                    continue
                if _is_bullet(raw_line):
                    bullet = _clean_bullet(raw_line)
                    wrapped = _wrap(bullet, "Helvetica", body_size, maxw - 14)
                    if wrapped:
                        ensure_space(line_gap + 2)
                        pdf.setFont("Helvetica", body_size)
                        pdf.drawString(x, y, "•")
                        pdf.drawString(x + 12, y, wrapped[0])
                        y -= line_gap
                        if len(wrapped) > 1:
                            draw_text_lines(wrapped[1:], "Helvetica", body_size, line_gap, indent=12)
                else:
                    draw_text_lines(_wrap(raw_line.strip(), "Helvetica", body_size, maxw), "Helvetica", body_size, line_gap)
            y -= 6

    render_block("Experience", payload.experience_text)
    render_block("Projects", payload.projects_text)
    render_block("Education", payload.education_text)

    extras = payload.extras_text.strip()
    if extras:
        render_block("Additional", extras)

    pdf.save()
    return Response(
        content=buf.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": 'inline; filename="resume.pdf"'},
    )
