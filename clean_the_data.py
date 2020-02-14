'''Data cleaning and tweaking '''

import pandas as pd
import datetime

from functions import convert_str

nyc = pd.read_csv('./data/fhrw-4uyv.csv', parse_dates=['created_date','closed_date'])

nyc.info()

# =============================================================================
# =============================================================================
# # Data Cleaning
# =============================================================================
# =============================================================================

# NAs
nyc.isna().sum(axis=0)

# duplicates
nyc.duplicated().sum(axis=0)

# clean the names of  location_type
nyc.location_type.unique()

# x==x enables one to avoid NaN values while np.nan == np.nan returns False
nyc.loc[:, 'location_type'] = nyc['location_type'].apply(lambda x: convert_str(x) if x==x else x)

nyc.location_type.value_counts(normalize=True, dropna=False)*100

# clean city names
nyc.city.unique()
len(nyc.city.unique())

nyc.loc[:, 'city'] = nyc['city'].apply(lambda x: convert_str(x) if x==x else x)

nyc.city.value_counts(normalize=True, dropna=False)*100
    
# clean borough names
nyc.borough.unique()

nyc.loc[:, 'borough'] = nyc['borough'].apply(lambda x: convert_str(x) if x==x else x)
    

# clean names of the complaint categories
nyc['complaint_type'].unique()

nyc.loc[:, 'complaint_type'] = nyc['complaint_type'] \
                                .apply(lambda x: x.lower().replace(' - ', '/'))

nyc['complaint_type'].value_counts()/len(nyc)*100

# =============================================================================
# # fill in/correct 'city' and 'borough' values
# =============================================================================

city_borough = nyc.groupby(['city','borough'])['city'].count().index

# bild a mapping to recover values
dict_map = dict()
for pair in city_borough:
    
    if pair[0] not in ['Manhattan', 'Brooklyn', 'Bronx', 'Queens', 'Staten Island', 'New York'] and pair[1] != 'Unspecified':
        
        dict_map[pair[0]] = pair[1] 
    
    elif pair[0] in ['Manhattan', 'Brooklyn', 'Bronx', 'Queens', 'Staten Island'] and pair[1] == 'Unspecified':
        
        dict_map[pair[0]] = pair[0] 
        
    elif pair[0] == 'New York'and pair[1] == 'Unspecified':
        
        dict_map[pair[0]] = pair[1] 
        
 
# skip NAs in nyc['city']
filter_city = nyc.city.apply(lambda x: x==x)

# select observations where borough not specified
filter_borough = nyc.borough =='Unspecified'

filter = filter_city & filter_borough

nyc['borough_ext'] = nyc['borough'].copy()
nyc.loc[filter, 'borough_ext'] = nyc.loc[filter, 'city'].map(dict_map, na_action='ignore')

# =============================================================================
# fill in/correct 'incident_zip' and 'borough' values
# recover `borough` values by `incident_zip` values if given
# =============================================================================

index = nyc.groupby(['incident_zip','borough_ext'])['incident_zip'].count().index    

zip_borough = dict()

for i in index:
    if i[1] != 'Unspecified':
        zip_borough[i[0]] = i[1]

filter_borough = nyc.borough_ext =='Unspecified'
filter_zip = nyc.incident_zip.apply(lambda x: x==x)
        
nyc.loc[filter_borough & filter_zip ,'borough_ext'] = nyc.loc[filter_borough & filter_zip,'incident_zip'].map(zip_borough)        


nyc.borough.value_counts(dropna=False)
nyc.borough_ext.value_counts(dropna=False)


# =============================================================================
# Delete complaints that were opened and closed at the same time
# =============================================================================

# How long it takes to close a complaint?
(nyc.closed_date-nyc.created_date).describe()

# most frequent type of complaints
(nyc[nyc.complaint_type == 'heat/hot water'].closed_date-nyc[nyc.complaint_type == 'heat/hot water'].created_date).describe()

# rows where created_date is later than closed_date or created_date = closed_date
# or if it was closed after less than 1 hour

nyc['period'] = (nyc.closed_date-nyc.created_date)

null_duration = nyc[nyc.period <= datetime.timedelta(days=0, 
                                                     seconds=0, 
                                                     microseconds=0, 
                                                     milliseconds=0, 
                                                     minutes=0, 
                                                     hours=1, 
                                                     weeks=0)]

# delete complaints with null duraion
# obviously they were closed automatically by the system
nyc.drop(labels=null_duration.index, axis=0, inplace=True)

# How long it takes to close a complaint?
nyc.groupby('complaint_type')['period'].describe()[['mean', 'std', 'min', 'max']]

# =============================================================================
# Clean address information (keep records with borough and zip info)
# =============================================================================

# For which type of complaints is incident_address not given?
nyc[nyc.incident_address.isna()]['complaint_type'].value_counts()


# How many records in the data are with an address without street name?
nyc.loc[~nyc.incident_address.isna(), 'street_name'].isna().sum()

# How many records are with no borough information ('Unspecified') and no zip given
nyc.loc[nyc.borough_ext =='Unspecified', 'incident_zip'].isna().sum()

# filter (no borough) and (no zip)
(nyc.borough_ext =='Unspecified')&(nyc.incident_zip.isna())

# keep only records with borough and zip information
nyc = nyc[(nyc.borough_ext !='Unspecified')&( ~nyc.incident_zip.isna())]

# =============================================================================
# use data up to end 2019
# =============================================================================

nyc = nyc[nyc.created_date < datetime.datetime(2019,12,31)]

# =============================================================================
# Delete redandant variables
# =============================================================================
nyc.loc[:, 'borough'] = nyc['borough_ext']

del nyc['period']
del nyc['borough_ext']
del nyc['address_type']

nyc.city='New York'

# save to file
nyc.to_csv('./results/nyc_clean.csv', index=False)

