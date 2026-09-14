"""Generates a concise, edge-system focused Word document (.docx)
specifically tailored for an on-premises edge appliance at a gate.
"""
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

DOC_PATH = Path("docs/PROJECT_STATUS.docx")
DOC_PATH_EDGE = Path("docs/ANPR_Edge_System_Status.docx")
DOC_PATH_ROOT = Path("ANPR_Edge_System_Status.docx")

def set_cell_background(cell, color_hex):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=80, bottom=80, left=120, right=120):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table, color="D0D5DD", sz="4"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>\n'
        f'  <w:bottom w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>\n'
        f'  <w:insideH w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>\n'
        f'  <w:left w:val="none"/>\n'
        f'  <w:right w:val="none"/>\n'
        f'  <w:insideV w:val="none"/>\n'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def build_short_edge_document():
    doc = docx.Document()

    # Margins (compact)
    for sec in doc.sections:
        sec.top_margin = Inches(0.7)
        sec.bottom_margin = Inches(0.7)
        sec.left_margin = Inches(0.75)
        sec.right_margin = Inches(0.75)

    # Style
    normal = doc.styles['Normal']
    normal.font.name = 'Calibri'
    normal.font.size = Pt(10)
    normal.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(3)

    # Title
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_after = Pt(2)
    r_title = title_p.add_run("ANPR Edge Appliance — Status & Pending Tasks")
    r_title.font.name = 'Arial'
    r_title.font.size = Pt(18)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79) # Deep Navy

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(10)
    r_sub = sub_p.add_run("On-Premises Gate System | CPU-Only Edge Box | 100% Local & Offline-First")
    r_sub.font.size = Pt(11)
    r_sub.font.bold = True
    r_sub.font.color.rgb = RGBColor(0x59, 0x59, 0x59)

    # Edge Context Box
    meta_table = doc.add_table(rows=3, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(meta_table, color="D0D5DD")
    meta_data = [
        ("Deployment Target:", "Standalone Mini-PC / Industrial Edge Box at Gate (Intel i5/i7, 8–16 GB RAM, CPU-only)"),
        ("Operational Model:", "100% Local & Air-Gapped. Zero cloud dependency. Local RTSP decode, local DB, local Web UI."),
        ("Camera Capacity:", "4–8 IP cameras (sub-stream D1/720p decode for AI, main-stream snap for plate evidence).")
    ]
    for i, (k, v) in enumerate(meta_data):
        row = meta_table.rows[i]
        c0, c1 = row.cells[0], row.cells[1]
        c0.text = k
        c0.paragraphs[0].runs[0].font.bold = True
        c0.paragraphs[0].runs[0].font.size = Pt(9)
        c0.paragraphs[0].runs[0].font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
        c1.text = v
        c1.paragraphs[0].runs[0].font.size = Pt(9)
        set_cell_background(c0, "F2F4F7")
        set_cell_background(c1, "FFFFFF")
        set_cell_margins(c0, 40, 40, 80, 80)
        set_cell_margins(c1, 40, 40, 80, 80)
        c0.width = Inches(2.0)
        c1.width = Inches(5.0)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # ---------------------------------------------------------
    # Snapshot Table
    # ---------------------------------------------------------
    h1 = doc.add_heading("1. Edge System Status Snapshot", level=1)
    h1.style.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    h1.paragraph_format.space_before = Pt(8)
    h1.paragraph_format.space_after = Pt(4)

    summary_rows = [
        ("Component", "Status", "Readiness for Edge Box"),
        ("CPU AI Inference Engine", "DONE (85%)", "ONNX Runtime + OpenVINO on Intel CPU. Fast NumPy letterbox & NMS. Zero GPU required. 400+ tests passing."),
        ("Video Capture & Cascade", "DONE (90%)", "RTSP sub-stream reader with drop queue (no lag buildup), ByteTrack tracking, multi-frame plate voting."),
        ("Edge Process Architecture", "DONE (80%)", "Isolated OS processes per camera (supervisor auto-restart) + offline local disk spooling (zero lost events)."),
        ("Local Database & Storage", "DONE (85%)", "PostgreSQL/SQLite with Alembic. Date-sharded local JPEG storage (YYYY/MM/DD) with sanitized names."),
        ("Local Web Dashboard", "DONE (65%)", "Dark/Light UI, live stats, entry/exit trends, whitelist/blacklist, audit logs, and daily Excel export."),
        ("Edge UI Enhancements", "IN PROGRESS", "Need interactive ROI/line drawing canvas on camera stream, and evidence drawer for plate disputes."),
        ("Edge Maintenance (Janitor)", "LEFT (0%)", "Auto-purge local images older than 30 days to avoid filling local disk."),
        ("Node-Locked Licensing", "LEFT (15%)", "Hardware fingerprinting (CPU/Motherboard UUID) to bind installation to edge machine.")
    ]

    st_table = doc.add_table(rows=len(summary_rows), cols=3)
    st_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(st_table, color="D0D5DD")
    col_w = [Inches(2.2), Inches(1.3), Inches(3.5)]

    for r_idx, row in enumerate(st_table.rows):
        is_h = (r_idx == 0)
        d_row = summary_rows[r_idx]
        for c_idx, cell in enumerate(row.cells):
            cell.text = d_row[c_idx]
            p = cell.paragraphs[0]
            r = p.runs[0]
            cell.width = col_w[c_idx]
            set_cell_margins(cell, 50, 50, 70, 70)
            if is_h:
                set_cell_background(cell, "1F4E79")
                r.font.bold = True
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                r.font.size = Pt(9)
            else:
                set_cell_background(cell, "F9FAFC" if r_idx % 2 == 1 else "FFFFFF")
                r.font.size = Pt(8.5)
                if c_idx == 1:
                    r.font.bold = True
                    if "DONE" in d_row[1]:
                        r.font.color.rgb = RGBColor(0x1E, 0x7E, 0x34)
                    elif "IN PROGRESS" in d_row[1]:
                        r.font.color.rgb = RGBColor(0x85, 0x64, 0x04)
                    else:
                        r.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # ---------------------------------------------------------
    # Section 2: What is DONE
    # ---------------------------------------------------------
    h1 = doc.add_heading("2. What is DONE (Tested & Working on Edge)", level=1)
    h1.style.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    h1.paragraph_format.space_before = Pt(8)
    h1.paragraph_format.space_after = Pt(3)

    done_items = [
        ("CPU-Only AI Inference: ", "Inference engine runs strictly on CPU via ONNX Runtime with OpenVINO acceleration. YOLOv8n vehicle detection, 320x320 license plate detection, and ByteTrack multi-object tracking all execute without PyTorch or CUDA."),
        ("Multi-Frame Positional Voting: ", "Never relies on single-frame OCR. Combines quality-weighted character consensus across video track history (UP32AB1234), eliminating typical edge gate errors from motion blur and bad angles."),
        ("Process-per-Camera Architecture: ", "Each RTSP camera runs in a dedicated OS worker process supervised by a watchdog with thread-budget allocation. A camera crash never crashes the web server or other cameras."),
        ("Air-Gapped Offline Spooling: ", "If the local database or API restarts, workers store detected events in a local disk spool queue and auto-flush on reconnection. Zero missed events."),
        ("Edge Media Storage: ", "Event vehicle and plate crop images are saved to local disk sharded by date (storage/media/YYYY/MM/DD/cam_plate_uid.jpg) ensuring fast filesystem lookups."),
        ("Gate Web Dashboard: ", "Runs locally on the edge box. Provides dark/light mode, real-time vehicle entry/exit counters, blacklist/whitelist lookup, vehicle search with photo history, RBAC user permissions, and 1-click Excel daily reports."),
        ("Automated Quality Gate: ", "400+ unit and integration tests passing locally covering the complete cascade.")
    ]
    for b_pre, text in done_items:
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(2)
        r0 = p.add_run(b_pre)
        r0.font.bold = True
        r0.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
        p.add_run(text)

    # ---------------------------------------------------------
    # Section 3: What is IN PROGRESS
    # ---------------------------------------------------------
    h1 = doc.add_heading("3. What is IN PROGRESS (Current Transition)", level=1)
    h1.style.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    h1.paragraph_format.space_before = Pt(8)
    h1.paragraph_format.space_after = Pt(3)

    in_prog = [
        ("Switching Default OCR to PP-OCRv4 ONNX: ", "The pipeline supports PP-OCRv4 ONNX (3x faster, 10 MB artifact). Script scripts/fetch_recognizer.py is ready; downloading the model will replace the legacy EasyOCR fallback and drop the heavy PyTorch edge dependency."),
        ("Deprecating In-Process Camera Threads: ", "Standalone supervisor works; the remaining prototype camera endpoints in main.py are being switched so only the out-of-process supervisor handles camera ingestion.")
    ]
    for b_pre, text in in_prog:
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(2)
        r0 = p.add_run(b_pre)
        r0.font.bold = True
        r0.font.color.rgb = RGBColor(0x85, 0x64, 0x04)
        p.add_run(text)

    # ---------------------------------------------------------
    # Section 4: What is LEFT
    # ---------------------------------------------------------
    h1 = doc.add_heading("4. What is LEFT (Edge Deployment Checklist)", level=1)
    h1.style.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    h1.paragraph_format.space_before = Pt(8)
    h1.paragraph_format.space_after = Pt(3)

    doc.add_paragraph("Essential features needed before shipping and installing the edge box on-site:")

    todo_gate = [
        ("1. ROI & Trigger Line Drawing Tool (Cameras.jsx): ", "Technician needs an on-screen canvas to draw the gate boundary polygon and virtual entry/exit trigger line over the live camera snapshot during installation."),
        ("2. Plate Evidence Drawer (Events.jsx): ", "Clicking an event must open a drawer showing: full vehicle picture, zoomed plate crop, confidence score, and the frame-by-frame OCR read history. Essential for resolving gate disputes with guards."),
        ("3. Operator Misread Quick-Fix (Events.jsx): ", "A 1-click 'Correct Plate' button that updates the record and automatically registers an alias in plate_aliases so the same misread never happens again."),
        ("4. Real-Time WebSockets (/ws/events): ", "Replace the 5-second polling interval with instant WebSockets so guards see cars appear on the dashboard the instant they cross the gate line."),
        ("5. Security Alerts UI Page: ", "A dedicated Alerts page and top-bar bell notification that rings whenever a Blacklisted or Unknown vehicle arrives at the gate."),
        ("6. Disk Retention Janitor (workers/janitor.py): ", "Automated cleanup daemon that deletes image JPEGs older than 30 days and prunes raw frame reads older than 7 days, preventing the edge box hard drive from filling up."),
        ("7. Excel Vehicle List Import: ", "Allow the society manager to upload a messy Excel/CSV of resident plates, normalizing spaces, O/0 typos, and uppercase."),
        ("8. Node-Locked Edge Licensing: ", "Generate a hardware fingerprint from the edge PC (CPU ID + Motherboard UUID + MAC) and check an encrypted vendor license key. Prevents distributors or clients from copying the software onto unauthorized machines."),
        ("9. One-Click Edge Boot & Auto-Start: ", "Create Docker Compose / systemd service units so the edge box starts everything automatically upon power-on without technician intervention.")
    ]
    for b_pre, text in todo_gate:
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(3)
        r0 = p.add_run(b_pre)
        r0.font.bold = True
        r0.font.color.rgb = RGBColor(0xC0, 0x00, 0x00) # Red
        p.add_run(text)

    # ---------------------------------------------------------
    # Section 5: Top 5 Immediate Action Steps
    # ---------------------------------------------------------
    h1 = doc.add_heading("5. Top 5 Immediate Priorities for Next Build", level=1)
    h1.style.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    h1.paragraph_format.space_before = Pt(8)
    h1.paragraph_format.space_after = Pt(3)

    priorities = [
        ("Priority 1: ", "Download PP-OCRv4 model (python scripts/fetch_recognizer.py --ppocr) to drop PyTorch and boost CPU inference speed."),
        ("Priority 2: ", "Build the interactive ROI & trigger line canvas editor in Cameras.jsx."),
        ("Priority 3: ", "Build the Event Evidence Drawer & misread correction in Events.jsx."),
        ("Priority 4: ", "Implement workers/janitor.py to safeguard edge disk capacity."),
        ("Priority 5: ", "Implement node-locked hardware license check for distributor protection.")
    ]
    for b_pre, text in priorities:
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(2)
        r0 = p.add_run(b_pre)
        r0.font.bold = True
        r0.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
        p.add_run(text)

    # Save
    try:
        doc.save(str(DOC_PATH))
        print(f"  - {DOC_PATH.resolve()}")
    except PermissionError:
        print(f"  Note: {DOC_PATH} is currently open in Word; saving new version as {DOC_PATH_EDGE}")

    try:
        doc.save(str(DOC_PATH_EDGE))
        print(f"  - {DOC_PATH_EDGE.resolve()}")
    except Exception as e:
        print(f"Error saving {DOC_PATH_EDGE}: {e}")

    try:
        doc.save(str(DOC_PATH_ROOT))
        print(f"  - {DOC_PATH_ROOT.resolve()}")
    except Exception as e:
        print(f"Error saving {DOC_PATH_ROOT}: {e}")

if __name__ == "__main__":
    build_short_edge_document()
