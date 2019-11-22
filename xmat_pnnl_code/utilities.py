import pandas as pd
import xmat_pnnl_code as xcode

def load_data(data_file='Aus_Steel_Data'):
    if data_file not in ['9Cr_Data', 'Aus_Steel_Data']:
        raise FileNotFoundError('The requested file does not exist.')
    path = '/'.join(xcode.__path__[0].split('/')[:-1])
    if data_file == 'Aus_Steel_Data':
        path += '/xmat_pnnl_data/Aus_Steel_data/netl-cwru-aus-alloys.xlsx'
        data = pd.ExcelFile(path)
        df = data.parse('AUS_Alloys')
        legend = data.parse('Legend')['Unnamed: 1']
        alloy_metadata = data.parse('AUS_Alloys LIST')
        
    if data_file == '9Cr_Data':
        path += '/xmat_pnnl_data/9Cr_data/netl-9cr-data-book.xlsx'
        data = pd.ExcelFile(path)
        df = data.parse('9%Cr Steel')
        legend = data.parse('Legend')['Unnamed: 1']
        alloy_metadata = data.parse('9Cr List')

    legend = dict(zip(df.columns, legend.values))
    df = df[df['CT_RT'].notnull()]
    df.dropna(how='all', inplace=True)
    df.dropna(subset=['ID', 'CT_RT'], inplace=True)
    df = df[df['ID'] != 'ID']
    col_keep = df.columns[~df.isna().all()]
    df = df[col_keep]
    legend = {k: v for k, v in legend.items() if k in col_keep}
    if data_file == '9Cr_Data':
        df['ID'] = df['ID'].apply(lambda x: '9Cr-' + str(x).zfill(3))
    return df, legend, alloy_metadata

