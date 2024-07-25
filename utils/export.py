import pandas as pd
from fpdf import FPDF

def export_to_excel(data, clusters, filename):
    data.to_excel(filename, index=False)

def export_to_pdf(data, clusters, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for i, row in data.iterrows():
        line = ", ".join([str(val) for val in row.values])
        pdf.cell(200, 10, txt=line, ln=True)
    pdf.output(filename)
