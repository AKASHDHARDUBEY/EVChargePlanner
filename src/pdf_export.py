from fpdf import FPDF
import datetime

class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", 'B', 15)
        self.cell(0, 10, "Intelligent EV Infrastructure Planning Report", border=False, ln=True, align="C")
        self.set_font("Arial", 'I', 10)
        self.cell(0, 10, f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
        self.ln(10)
        
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")
        
    def section_title(self, title):
        self.set_font("Arial", "B", 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 8, title, ln=True, fill=True)
        self.ln(4)
        
    def section_body(self, body):
        self.set_font("Arial", "", 11)
        # Using multi_cell to handle line breaks and long text
        # Clean text
        body = body.replace('\u2018', "'").replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"').replace('\u2013', '-')
        
        # Ensure encoding issues are handled nicely by just replacing unsupported characters
        body = body.encode('latin-1', 'replace').decode('latin-1')
        
        self.multi_cell(0, 6, body)
        self.ln(6)

def generate_planning_report(report_data: dict, output_path="ev_planning_report.pdf"):
    pdf = PDFReport()
    pdf.add_page()
    
    # 1. Demand Summary
    pdf.section_title("1. Charging Demand Summary")
    pdf.section_body(report_data.get("demand_summary", "N/A"))
    
    # 2. High Load Locations
    pdf.section_title("2. High-Load Analysis & Usage Patterns")
    pdf.section_body(report_data.get("high_load_analysis", "N/A"))
    
    # 3. Infrastructure Recommendations
    pdf.section_title("3. Infrastructure Expansion Recommendations")
    pdf.section_body(report_data.get("infrastructure_recommendations", "N/A"))
    
    # 4. Scheduling Insights
    pdf.section_title("4. Scheduling & Load-Balancing Insights")
    pdf.section_body(report_data.get("scheduling_insights", "N/A"))
    
    # 5. Review Feedback
    if report_data.get("review_status"):
        pdf.section_title("5. AI Review Feedback")
        pdf.section_body(report_data.get("review_status", ""))
        
    # 6. References
    pdf.section_title("6. Supporting References & Guidelines")
    pdf.section_body(report_data.get("references", "N/A"))
    
    # Data Warnings
    if report_data.get("data_warnings"):
        pdf.section_title("Data Warnings (Incomplete/Noisy Data)")
        pdf.section_body(report_data.get("data_warnings", ""))
        
    pdf.output(output_path)
    return output_path
