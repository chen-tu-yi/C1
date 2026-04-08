import pandas as pd
import sys

def main():
    print("--- 000-2025模組進線日期.xlsx ---")
    try:
        xl_plan = pd.ExcelFile('c:/local_file/專題/000-2025模組進線日期.xlsx')
        print("Sheets:", xl_plan.sheet_names)
        df_plan = pd.read_excel(xl_plan, sheet_name='整機計劃', nrows=5)
        print("Columns in 整機計劃:", df_plan.columns.tolist())
    except Exception as e:
        print("Error reading plan:", e)

    print("\n--- ERP_Table.xlsx ---")
    try:
        xl_erp = pd.ExcelFile('c:/local_file/專題/ERP_Table.xlsx')
        print("Sheets in ERP_Table:", xl_erp.sheet_names)
        for sheet in xl_erp.sheet_names:
            if 'MOCTA' in sheet or 'INV' in sheet:
                df = pd.read_excel(xl_erp, sheet_name=sheet, nrows=5)
                print(f"Columns in {sheet}:", df.columns.tolist())
    except Exception as e:
        print("Error reading ERP_Table:", e)

if __name__ == '__main__':
    main()
