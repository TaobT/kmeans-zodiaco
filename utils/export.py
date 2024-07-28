from fpdf import FPDF
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import streamlit as st

def export_to_excel(data, clusters, filename):
    data.to_excel(filename, index=False)
