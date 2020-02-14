''' Data exploration to get some insights '''

import pandas as pd
import matplotlib.pyplot as plt

nyc = pd.read_csv('./results/nyc_clean.csv', parse_dates=['created_date','closed_date'])
nyc.info()

nyc.isna().sum(axis=0)

# =============================================================================
# Complaints by its type
# =============================================================================
complaints = nyc['complaint_type'].value_counts()/len(nyc)*100
complaints

plt.style.use('ggplot')

plt.bar(complaints[:10].index, complaints[:10])

plt.title('Complaints by Type')
plt.ylabel('% of complaints')
plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig('./figs/complaints_by_type.pdf')
plt.show()
plt.close()

# =============================================================================
# Complaints by Type across Borough
# =============================================================================

Top4complaints_by_borough = pd.crosstab(nyc.borough, 
                                        nyc.complaint_type)[complaints[:4].index]/len(nyc)*100 

# plot stacked horizontal bar chart
Top4complaints_by_borough.plot.barh(stacked=True)

plt.title('Complaints by Type across Boroughs')
plt.ylabel(None)   
plt.xlabel('% of complaints')
plt.legend(title='Top4 Compl.')
 
plt.tight_layout()
plt.savefig('./figs/top4_complaints_across_borough.pdf')
plt.show()
plt.close()


# =============================================================================
# 'heat/hot water' complaints
# =============================================================================

nyc_hhw = nyc[nyc.complaint_type == 'heat/hot water'].copy()

# =============================================================================
# % of complaints on `heat/hot water` by monthes across borough
# =============================================================================

(pd.crosstab(nyc_hhw.created_date.dt.month, 
             nyc_hhw.borough)/len(nyc_hhw)*100).plot.barh(stacked=True)

plt.title('`heat/hot water` complaints by Months \nacross Boroughs')
plt.ylabel(None)
plt.xlabel('% of complaints')
plt.yticks([i for i in range(0,12)],
           ['Jan', 'Feb', 'Mar',
            'Apr', 'May', 'Jun', 
            'Jul', 'Aug', 'Sep',
            'Oct', 'Nov', 'Dec'])
plt.legend(title=None)
 
plt.tight_layout()
plt.savefig('./figs/hhwater_by_months_across_borough.pdf')
plt.show()
plt.close()

# =============================================================================
# Number of month complaints on `heat/hot water`across borough
# =============================================================================

pd.crosstab(nyc_hhw.created_date.dt.to_period('M'),
            nyc_hhw.borough).plot()

plt.title('`heat/hot water` Complaints in a Month \nacross Boroughs')
plt.xlabel(None)
plt.ylabel('Number of complaints')
plt.legend(title=None)

plt.tight_layout()
plt.savefig('./figs/hhwater_complaints_across_borough.pdf')
plt.show()
plt.close()

# =============================================================================
# number of complaints in a day by an address
# =============================================================================

df = nyc_hhw.groupby([nyc_hhw.created_date.dt.to_period('D'),
                      nyc_hhw.borough, 
                      nyc_hhw.incident_zip, 
                      nyc_hhw.incident_address])['city'].count()

df = df.reset_index()

df.rename(columns={'city':'complaints'}, inplace=True)

# save to file
df.to_csv('./results/nyc_hhw_complaints.csv', index=False)

# =============================================================================
# total number of complaints by an address
# =============================================================================
df.groupby(['borough',
            'incident_zip',
            'incident_address'])['complaints'].sum().sort_values(ascending=False)

