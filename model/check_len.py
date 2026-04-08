import pandas as pd

def check():
    with open('row_count_check.txt', 'w') as f:
        df_in = pd.read_csv('Model缺料物料.csv')
        df_out = pd.read_csv('缺料風險預測報告.csv')
        f.write(f"In: {len(df_in)}\n")
        f.write(f"Out: {len(df_out)}\n")
        
        # Check if df_in has duplicates?
        f.write(f"In unique index: {len(df_in.index.unique())}\n")
        
if __name__ == '__main__':
    check()
