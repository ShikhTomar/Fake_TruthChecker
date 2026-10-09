import pandas as pd
from pathlib import Path
import json

# ============================================================
# 1. PATH TO FACTDRILL FOLDER
# ============================================================

FACTDRILL_FOLDER = (
    Path.home() /
    "Downloads" /
    "TruthLens" /
    "FactDrill"
)

# Find all Excel files
excel_files = sorted(FACTDRILL_FOLDER.glob("*.xlsx"))

print("=" * 70)
print("TRUTHLENS - FACTDRILL DATASET INSPECTION")
print("=" * 70)

print(f"\nFactDrill folder:")
print(FACTDRILL_FOLDER)

print(f"\nExcel files found: {len(excel_files)}")

if len(excel_files) == 0:
    print("\nERROR: No .xlsx files found.")
    print("Check the folder path.")
    exit()

# ============================================================
# 2. INSPECT EVERY FILE
# ============================================================

file_reports = []

for i, file_path in enumerate(excel_files, start=1):

    print("\n" + "-" * 70)
    print(f"[{i}/{len(excel_files)}] {file_path.name}")
    print("-" * 70)

    try:

        # Read Excel workbook information
        excel_file = pd.ExcelFile(file_path)

        sheet_names = excel_file.sheet_names

        print("Sheets:", sheet_names)

        workbook_report = {
            "filename": file_path.name,
            "sheets": sheet_names,
            "sheet_reports": []
        }

        # ----------------------------------------------------
        # Inspect every sheet
        # ----------------------------------------------------

        for sheet_name in sheet_names:

            try:

                df = pd.read_excel(
                    file_path,
                    sheet_name=sheet_name
                )

                print(f"\nSheet: {sheet_name}")
                print(f"Rows: {len(df)}")
                print(f"Columns: {len(df.columns)}")

                print("Column names:")
                print(list(df.columns))

                print("\nFirst 3 rows:")

                print(
                    df.head(3).to_string(index=False)
                )

                sheet_report = {
                    "sheet_name": sheet_name,
                    "rows": len(df),
                    "columns": list(df.columns),
                    "missing_values": df.isnull().sum().to_dict()
                }

                workbook_report["sheet_reports"].append(
                    sheet_report
                )

            except Exception as e:

                print(
                    f"ERROR reading sheet {sheet_name}: {e}"
                )

        file_reports.append(workbook_report)

    except Exception as e:

        print(f"ERROR reading file: {e}")

# ============================================================
# 3. SAVE COMPLETE INSPECTION REPORT
# ============================================================

report_path = (
    FACTDRILL_FOLDER.parent /
    "factdrill_inspection_report.json"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        file_reports,
        f,
        indent=4,
        ensure_ascii=False,
        default=str
    )

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)

print(f"\nReport saved to:")
print(report_path)