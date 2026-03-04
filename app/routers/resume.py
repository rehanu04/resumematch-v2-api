from fastapi import APIRouter, Depends
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.core.security import require_app_key

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from io import BytesIO


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


@router.post("/resume/pdf")
def generate_resume_pdf(payload: ResumePdfRequest, _=Depends(require_app_key)):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    width, height = LETTER

    x = 54
    y = height - 54

    def draw_line(text: str, size: int = 11, gap: int = 14, bold: bool = False):
        nonlocal y
        if not text:
            return
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        max_chars = 95
        for i in range(0, len(text), max_chars):
            c.drawString(x, y, text[i:i + max_chars])
            y -= gap

    full_name = (payload.first_name + " " + payload.last_name).strip() or "Resume"
    draw_line(full_name, size=16, gap=18, bold=True)

    contact = " • ".join([t for t in [payload.email, payload.phone, payload.location] if t.strip()])
    draw_line(contact, size=10, gap=14)

    if payload.target_role.strip():
        draw_line(payload.target_role.strip(), size=11, gap=16, bold=True)

    y -= 8
    draw_line("SUMMARY", size=12, gap=16, bold=True)
    draw_line(payload.summary.strip(), size=10, gap=14)

    y -= 8
    draw_line("SKILLS", size=12, gap=16, bold=True)
    if payload.skills:
        draw_line(", ".join(payload.skills), size=10, gap=14)

    def section(title: str, body: str):
        nonlocal y
        if not body.strip():
            return
        y -= 8
        draw_line(title, size=12, gap=16, bold=True)
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            draw_line(line, size=10, gap=14)

    section("EXPERIENCE", payload.experience_text)
    section("PROJECTS", payload.projects_text)
    section("EDUCATION", payload.education_text)
    section("EXTRAS", payload.extras_text)

    c.showPage()
    c.save()

    pdf_bytes = buf.getvalue()
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'inline; filename="resume.pdf"'}
    )
