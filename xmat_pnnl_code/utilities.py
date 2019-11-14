import pandas as pd
import xmat_pnnl_code as xcode

def load_data(data='Aus_Steel_Data'):
    if data not in ['9Cr_Data', 'Aus_Steel_Data']:
        raise FileNotFoundError('The requested file does not exist.')
    path = '/'.join(xcode.__path__[0].split('/')[:-1])
    if data == 'Aus_Steel_Data':
        path += '/xmat_pnnl_data/Aus_Steel_data/netl-cwru-aus-alloys.xlsx'
        data = pd.ExcelFile(path)
        df = data.parse('AUS_Alloys')
        legend = data.parse('Legend')['Unnamed: 1']
        alloy_metadata = data.parse('AUS_Alloys LIST')
    if data == '9Cr_Data':
        path += '/xmat_pnnl_data/9Cr_data/netl-9cr-data-book.xlsx'
        data = pd.ExcelFile(path)
        df = data.parse('9%Cr Steel')
        legend = data.parse('Legend')
        alloy_metadata = data.parse('9Cr List')

    df = df[df['CT_RT'].notnull()]
    df.dropna(how='all', inplace=True)
    df.dropna(subset=['ID', 'CT_RT'], inplace=True)
    df = df[df['ID'] != 'ID']
    col_keep = df.columns[~df.isna().all()]
    df = df[col_keep]
    legend = {k: v for k, v in legend.items() if k in col_keep}
    
    return df, legend, alloy_metadata



