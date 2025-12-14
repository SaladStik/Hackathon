from datetime import datetime
from pathlib import Path
from typing import Optional
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    HRFlowable,
)
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.lib.enums import TA_CENTER, TA_LEFT


def generate_pdf_report(
    output_path: str,
    detection_id: str,
    annotated_image_path: str,
    original_image_path: Optional[str] = None,
    detection_data: dict = None,
) -> str:
    """generates report"""
    detection_data = detection_data or {}
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.25 * inch,
    )

    styles = getSampleStyleSheet()

    # custom styling
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        spaceAfter=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1a1a2e"),
    )

    subtitle_style = ParagraphStyle(
        "CustomSubtitle",
        parent=styles["Heading2"],
        fontSize=14,
        spaceAfter=20,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4a4a6a"),
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceAfter=10,
        spaceBefore=15,
        textColor=colors.HexColor("#16213e"),
    )

    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["Normal"],
        fontSize=11,
        spaceAfter=8,
        leading=14,
    )

    alert_style = ParagraphStyle(
        "AlertStyle",
        parent=styles["Normal"],
        fontSize=14,
        textColor=colors.red,
        alignment=TA_CENTER,
        spaceAfter=10,
        spaceBefore=10,
    )
    compliant_style = ParagraphStyle(
        "CompliantStyle",
        parent=styles["Normal"],
        fontSize=14,
        textColor=colors.green,
        alignment=TA_CENTER,
        spaceAfter=10,
        spaceBefore=10,
    )

    warning_style = ParagraphStyle(
        "WarningStyle",
        parent=styles["Normal"],
        fontSize=14,
        textColor=colors.orange,
        alignment=TA_CENTER,
        spaceAfter=10,
        spaceBefore=10,
    )

    story = []

    # header
    story.append(Paragraph("PPE SAFETY COMPLIANCE REPORT", title_style))
    story.append(
        Paragraph("Personal Protective Equipment Detection Analysis", subtitle_style)
    )
    story.append(
        HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a1a2e"))
    )
    story.append(Spacer(1, 20))

    # report metadata
    total_persons = detection_data.get("total_persons", 0)
    meta_data = [
        ["Report ID:", detection_id],
        ["Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Persons Detected:", str(total_persons)],
    ]
    meta_table = Table(meta_data, colWidths=[1.5 * inch, 4 * inch])
    meta_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#333333")),
                ("ALIGN", (0, 0), (0, -1), "RIGHT"),
                ("ALIGN", (1, 0), (1, -1), "LEFT"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 20))

    # get persons data
    persons = detection_data.get("persons", [])

    # count compliance status
    compliant_count = sum(1 for p in persons if p.get("compliance") == "compliant")
    non_compliant_count = sum(
        1 for p in persons if p.get("compliance") == "non_compliant"
    )
    partial_count = sum(
        1 for p in persons if p.get("compliance") in ["partial", "unknown"]
    )

    # overall status
    story.append(Paragraph("Overall Compliance Status", heading_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))

    if non_compliant_count > 0:
        story.append(
            Paragraph(
                f"[ ! ] NON-COMPLIANT: {non_compliant_count} Person(s) Missing Required PPE",
                alert_style,
            )
        )
    elif partial_count > 0:
        story.append(
            Paragraph(
                f"[ ~ ] PARTIAL COMPLIANCE: {partial_count} Person(s) with Incomplete PPE",
                warning_style,
            )
        )
    elif compliant_count > 0:
        story.append(
            Paragraph(
                f"[ OK ] COMPLIANT: All {compliant_count} Person(s) Wearing Required PPE",
                compliant_style,
            )
        )
    else:
        story.append(
            Paragraph("[ ? ]No persons detected in the analysis.", warning_style)
        )
    story.append(Spacer(1, 20))

    # summary table
    summary_data = [
        ["Status", "Count", "Indicator"],
        ["Fully Compliant", str(compliant_count), "OK" if compliant_count > 0 else "-"],
        ["Partially Compliant", str(partial_count), "~" if partial_count > 0 else "-"],
        [
            "Non-Compliant",
            str(non_compliant_count),
            "!" if non_compliant_count > 0 else "-",
        ],
        ["Total Persons", str(total_persons), "-"],
    ]

    summary_table = Table(summary_data, colWidths=[2.5 * inch, 1.5 * inch, 1.5 * inch])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#d4edda")),
                ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#fff3cd")),
                (
                    "BACKGROUND",
                    (0, 3),
                    (-1, 3),
                    (
                        colors.HexColor("#f87da0")
                        if non_compliant_count > 0
                        else colors.HexColor("#d4edda")
                    ),
                ),
                ("BACKGROUND", (0, 4), (-1, 4), colors.HexColor("#F5F5F5")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 20))

    # per person details
    story.append(Paragraph("Individual PPE Assessment", heading_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 10))

    if persons:
        # create table with all persons
        person_table_data = [["Person", "Helmet", "Vest", "Mask", "Status"]]

        for person in persons:
            person_id = person.get("person_id", "?")
            ppe_status = person.get("ppe_status", {})
            compliance = person.get("compliance", "unknown")

            # get ppe statuses
            helmet = ppe_status.get("helmet", {})
            vest = ppe_status.get("vest", {})
            mask = ppe_status.get("mask", {})

            def get_ppe_text(status):
                """returns text based indicator ppe status"""
                if status.get("present") is True:
                    conf = status.get("confidence", 0)
                    return f"YES ({conf * 100:.1f}%)"
                elif status.get("present") is False:
                    conf = status.get("confidence", 0)
                    return f"NO ({conf * 100:.1f}%)"
                else:
                    return "N/A"

            helmet_text = get_ppe_text(helmet)
            vest_text = get_ppe_text(vest)
            mask_text = get_ppe_text(mask)

            if compliance == "compliant":
                status_text = "COMPLIANT"
            elif compliance == "non_compliant":
                status_text = "NON-COMPLIANT"
            elif compliance == "partial":
                status_text = "PARTIAL"
            elif compliance == "unknown":
                status_text = "UNKNOWN"
            else:
                status_text = "N/A"

            person_table_data.append(
                [f"Person {person_id}", helmet_text, vest_text, mask_text, status_text]
            )

        person_table = Table(
            person_table_data,
            colWidths=[1 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.4 * inch],
        )

        # base style
        base_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
        ]

        # add grid colours based on compliance
        for i, person in enumerate(persons, start=1):
            compliance = person.get("compliance", "unknown")
            if compliance == "compliant":
                bg_color = colors.HexColor("#d4edda")
                text_color = colors.green
            elif compliance == "partial":
                bg_color = colors.HexColor("#fff3cd")
                text_color = colors.orange
            elif compliance == "non_compliant":
                bg_color = colors.HexColor("#f87da0")
                text_color = colors.red
            else:
                bg_color = colors.HexColor("#fff3cd")
                text_color = colors.HexColor("#856404")

            base_style.append(("BACKGROUND", (0, i), (-1, i), bg_color))
            base_style.append(("TEXTCOLOR", (-1, i), (-1, i), text_color))

            # color indivdual ppe cells
            ppe_status = person.get("ppe_status", {})
            for col, ppe_type in enumerate(["helmet", "vest", "mask"], start=1):
                status = ppe_status.get(ppe_type, {})
                if status.get("present") is True:
                    cell_color = colors.HexColor("#d4edda")
                    cell_text_color = colors.green
                elif status.get("present") is False:
                    cell_color = colors.HexColor("#f87da0")
                    cell_text_color = colors.red
                else:
                    cell_color = colors.HexColor("#fff3cd")
                    cell_text_color = colors.orange

                base_style.append(("BACKGROUND", (col, i), (col, i), cell_color))
                base_style.append(("TEXTCOLOR", (col, i), (col, i), cell_text_color))
        person_table.setStyle(TableStyle(base_style))
        story.append(person_table)
    else:
        story.append(Paragraph("No persons detected in the analysis.", body_style))

    story.append(Spacer(1, 20))

    # annotated image
    story.append(Paragraph("Detection Analysis", heading_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 10))

    if Path(annotated_image_path).exists():
        img = Image(annotated_image_path)
        img_width = 6 * inch
        aspect = img.imageHeight / img.imageWidth
        img_height = img_width * aspect

        max_height = 4 * inch
        if img_height > max_height:
            img_height = max_height
            img_width = img_height / aspect
        img._restrictSize(img_width, img_height)
        story.append(img)
        story.append(Spacer(1, 10))
        story.append(
            Paragraph(
                "<i>Figure: Annotated image showing detected persons and PPE status.</i>",
                ParagraphStyle(
                    "Caption",
                    parent=styles["Normal"],
                    fontSize=9,
                    alignment=TA_CENTER,
                    textColor=colors.grey,
                ),
            )
        )
    else:
        story.append(Paragraph("Annotated image not available.", body_style))

    story.append(Spacer(1, 20))
    # Issues and recommendations
    story.append(Paragraph("Issues and Recommendations", heading_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 10))

    # list non-compliant persons with missing items
    non_compliant_persons = [
        p for p in persons if p.get("compliance") == "non_compliant"
    ]

    if non_compliant_persons:
        story.append(Paragraph("IMMEDIATE ACTION REQUIRED:", body_style))
        story.append(Spacer(1, 5))
        for person in non_compliant_persons:
            person_id = person.get("person_id", "?")
            ppe_status = person.get("ppe_status", {})
            missing = []
            ppe_names = {
                "helmet": "Hard Hat",
                "vest": "Safety Vest",
                "mask": "Face Mask",
            }
            for ppe_type, name in ppe_names.items():
                status = ppe_status.get(ppe_type, {})
                if status.get("present") is False:
                    missing.append(name)
                elif status.get("present") is None:
                    missing.append(f"{name} (Not verified)")
            if missing:
                missing_text = ", ".join(missing)
                story.append(
                    Paragraph(
                        f"<b>Person {person_id}:</b> Missing {missing_text}",
                        body_style,
                    )
                )
        story.append(Spacer(1, 15))

        recommendations = [
            "Stop work immediately and provide missing PPE to non-compliant personnel.",
            "Review Safety protocols with all staff to reinforce PPE requirements.",
            "Document the incident and actions taken for compliance records.",
            "Conduct follow-up inspections to ensure ongoing compliance.",
            "Consider additional training sessions on PPE importance and usage.",
        ]
    else:
        recommendations = [
            "Maintain current PPE compliance practices.",
            "Continue regular training and awareness programs on PPE importance.",
            "Conduct periodic audits to ensure ongoing compliance.",
            "Encourage a safety-first culture among all personnel.",
            "Stay updated with the latest safety regulations and standards.",
        ]
    story.append(Paragraph("<b>Recommendations</b>", body_style))
    for i, rec in enumerate(recommendations, start=1):
        story.append(Paragraph(f"{i}. {rec}", body_style))

    story.append(Spacer(1, 30))

    doc.build(story)
    return output_path
