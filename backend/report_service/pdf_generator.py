import os
from jinja2 import Environment, FileSystemLoader
from backend.schemas import ComplianceReport

try:
    from weasyprint import HTML
except (ImportError, OSError, Exception):
    HTML = None # Handled gracefully for dev/hackathon environments without GTK+/Pango dependencies

def generate_pdf_report(report: ComplianceReport, template_dir: str, output_path: str) -> str:
    """
    Renders the compliance report to HTML and converts it to PDF using WeasyPrint.
    Falls back to generating an HTML file if WeasyPrint is not installed.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("report_template.html")
    
    html_out = template.render(report=report)
    
    if HTML:
        # Generate real PDF if WeasyPrint is available
        HTML(string=html_out, base_url=template_dir).write_pdf(output_path)
        return output_path
    else:
        # Fallback for environments lacking WeasyPrint's OS-level dependencies
        html_path = output_path.replace(".pdf", ".html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_out)
        return html_path
