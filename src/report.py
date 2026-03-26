from fpdf import FPDF
import tempfile
import os

class EVReportPDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 15)
        self.cell(0, 10, "EV Charging Infrastructure Plan", border=False, ln=1, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def generate_pdf_report(report_dict):
    """
    Takes the structured report from the LangGraph agent and converts it to a PDF string/bytes.
    """
    pdf = EVReportPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    sections = [
        ("Demand Summary", report_dict.get("demand_summary", "")),
        ("Peak Load Analysis", report_dict.get("high_load_analysis", "")),
        ("Scheduling Insights", report_dict.get("scheduling_insights", "")),
        ("Infrastructure Recommendations", report_dict.get("infrastructure_recommendations", "")),
        ("References & Guidelines", report_dict.get("references", ""))
    ]
    
    for title, content in sections:
        if content:
            pdf.set_font("helvetica", "B", 12)
            pdf.cell(0, 10, title, ln=1)
            pdf.set_font("helvetica", "", 10)
            
            # Using multi_cell for text wrapping
            # Replace characters that might break typical fpdf encodings or just let it pass
            pdf.multi_cell(0, 6, str(content).replace("’", "'").replace("”", '"').replace("“", '"'))
            pdf.ln(5)
            
    if report_dict.get("data_warnings"):
        pdf.set_font("helvetica", "B", 10)
        pdf.set_text_color(200, 0, 0)
        pdf.multi_cell(0, 6, "WARNING: " + report_dict["data_warnings"])
        pdf.set_text_color(0, 0, 0)

    # create a temporary file to save the PDF, then read its bytes safely
    tmp_path = tempfile.mktemp(suffix=".pdf")
    pdf.output(tmp_path)
    
    with open(tmp_path, "rb") as f:
        pdf_bytes = f.read()
        
    os.remove(tmp_path)
    return pdf_bytes
