"""
config.py — File paths for all data files
==========================================
Edit this file once to match your machine's folder structure.
All other scripts import paths from here — no need to touch them.
"""
from pathlib import Path

# Directory containing this script (SH_H_RAG/)
HERE = Path(__file__).parent

# ─── Raw corpus file (Shulchan Arukh text) ────────────────────────────────────
# Default: 4 levels up from this script → study ai/datasete/
# Change this line if your folder structure is different.
TEXT_FILE = HERE.parent / "study ai" / "datasete" / "Shulchan Arukh, Orach Chayim - he - Torat Emet 363.txt"

# ─── Excel file with test questions ───────────────────────────────────────────
# Columns: # | question (שאלה) | answer (תשובה) | source (מקור)
XLSX_PATH = HERE / "שולחן_ערוך_כל_השאלות.xlsx"
