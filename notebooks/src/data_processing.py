import pandas as pd

def data_processing():
    data = pd.read_csv('data/model.csv', decimal='.')
    data = data.drop(['GHGEmissionsIntensity','Unnamed: 0','ListOfAllPropertyUseTypes','DefaultData',
    'CouncilDistrictCode','Latitude','Longitude','SiteEnergyUseWN(kBtu)','ComplianceStatus','Neighborhood',
    'YearBuilt'], axis=1)
    return data

