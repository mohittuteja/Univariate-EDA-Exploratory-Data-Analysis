import pandas as pd
import numpy as np
import pickle
pd.set_option('display.max_columns', None)
#pd.set_option('max_colwidth', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
from IPython.display import Markdown, display
def printmd(string, color=None):
    colorstr = "<span style='color:{}'>{}</span>".format(color, string)
    display(Markdown(colorstr))
    
###########################################################

printmd("**Variable Distribution**", color = 'blue')
obs_count = dat.shape[0]
print("#obs in data:", obs_count)
coltypes = dat.dtypes.reset_index()
print('# of vars: ',coltypes.shape[0])
coltypes.columns = ['colname','datatype']
display(pd.DataFrame(coltypes.groupby('datatype').datatype.count()))
coltypes.loc[:,'coltype'] = 'numeric'
coltypes.loc[coltypes['datatype'] == 'object','coltype'] = 'categorical'
coltypes.loc[(coltypes['colname'].str.contains('YEAR')) & (coltypes['datatype']!= 'object'),'coltype'] = 'year_like'
display(pd.DataFrame(coltypes.groupby(['coltype']).colname.count()))

cat_vars = coltypes.loc[coltypes['coltype'] == 'categorical','colname'].tolist()
numeric_vars = coltypes.loc[coltypes['coltype'] == 'numeric','colname'].tolist()
year_like_vars = coltypes.loc[coltypes['coltype'] == 'year_like','colname'].tolist()

display(coltypes.head())

printmd("**Numeric Variable Analysis**", color = 'blue')
############################################################
numeric_vars = dat[numeric_vars].describe(percentiles = [.01,.25, .5, .75, .99]).transpose().drop('std', axis = 1)
#null_vars = numeric_vars.loc[numeric_vars['count']==0,:].reset_index()[['index']]['index'].tolist()
numeric_vars1 = numeric_vars.reset_index()
null_vars = numeric_vars1.loc[numeric_vars1['count']==0,'index'].tolist()
non_null_numeric = numeric_vars.loc[numeric_vars['count'] !=0,:]
non_null_numeric.insert(1, 'pct_nulls', 0)
non_null_numeric.loc[:,'pct_nulls'] = ((1-non_null_numeric.loc[:,'count']/obs_count)*100).round(2)
non_null_numeric.insert(0, 'obs_count', obs_count)

print("# numeric vars:",numeric_vars.shape[0])
print("# of numeric vars with all nulls:",len(null_vars))
print("# remaining vars:",non_null_numeric.shape[0])
print("Numeric vars with all nulls (first 10):",null_vars[:10])
display(non_null_numeric.head(20))

printmd("**Year-Like Variable Analysis**", color = 'blue')
############################################################
year_vars = dat[year_like_vars].describe(percentiles = [.01,.25, .5, .75, .99]).transpose().drop('std', axis = 1).round(4)
year_vars1 = year_vars.reset_index()
yr_null_vars = year_vars1.loc[year_vars1['count']==0,'index'].tolist()
#yr_null_vars = year_vars.loc[year_vars['count']==0,:].reset_index()[['index']]['index'].tolist()
non_null_year_vars = year_vars.loc[year_vars['count'] !=0,:]
non_null_year_vars.insert(1, 'pct_nulls', 0)
non_null_year_vars.loc[:,'pct_nulls'] = ((1-non_null_year_vars.loc[:,'count']/obs_count)*100).round(2)
non_null_year_vars.insert(0, 'obs_count', obs_count)

print("# year-like vars:",year_vars.shape[0])
print("# of year-like vars with all nulls:",len(yr_null_vars))
print("Year-like vars with all nulls (first 10):",yr_null_vars[:10])
print("# remaining vars:",non_null_year_vars.shape[0])
display(non_null_year_vars.head(20))

printmd("**Categorical Variable Analysis**", color = 'blue')
############################################################
categorical_vars = dat[cat_vars].describe(include=[np.object]).transpose()
categorical_vars1 = categorical_vars.reset_index()
cat_null_vars = categorical_vars1.loc[categorical_vars1['count']==0,'index'].tolist()
#cat_null_vars = categorical_vars.loc[categorical_vars['count']==0,:].reset_index()[['index']]['index'].tolist()
print("# categorical vars:",categorical_vars.shape[0])
print("# of categorical vars with all nulls:",len(cat_null_vars))
non_null_cat = categorical_vars.loc[categorical_vars['count'] !=0,:]
non_null_cat.insert(1, 'pct_nulls', 0)
non_null_cat.loc[:,'pct_nulls'] = ((1-non_null_cat.loc[:,'count']/obs_count)*100)
print("# remaining vars:",non_null_cat.shape[0])
print("categorical vars with all nulls (first 10):",cat_null_vars[:10])
sort_var = 'unique'
printmd("***sorted by: " + sort_var + "***")
#display(non_null_cat.sort_values('unique', ascending = False).head(20))

n = 3
cols = list(range(n*2+1))

aggregates = pd.DataFrame(columns = cols)
#display(aggregates.head())

for curr_var in cat_vars:
    #print("variable:", curr_var)
    top_categories = pd.DataFrame(dat[[curr_var]].groupby(curr_var)[curr_var].count().sort_values(0, ascending=False)).head(n).transpose()
    while top_categories.shape[1] < n:
        top_categories.loc[:,'_dummy_placeholder'+str(top_categories.shape[1])] = '_dummy_placeholder'+str(top_categories.shape[1])
    #display(top_categories,top_categories.shape[1])
    test = pd.DataFrame([curr_var] + top_categories.columns.tolist() + top_categories.iloc[0,:].tolist()).transpose()
    aggregates = pd.concat([aggregates,test],ignore_index=True)
    
colnames = ['index']
for k in range(n):
    currval = 'top_val'+str(k+1)
    colnames += [currval]
for k in range(n):
    freq = 'freq'+str(k+1)
    colnames += [freq]
for k in range(n):
    aggregates = aggregates.replace('_dummy_placeholder'+str(k), np.nan)
    
aggregates.columns = colnames
aggregates.iloc[:,-int(np.floor(len(cols)/2)):] = aggregates.iloc[:,-int(np.floor(len(cols)/2)):].fillna(-1).astype('int64')
#display(aggregates.head(), aggregates.shape)
top_categories = non_null_cat.reset_index().merge(aggregates, how = 'inner', \
                        on = ['index']).drop(['top','freq'],axis = 1).set_index('index')
top_categories.insert(0, 'obs_count', obs_count)
display(top_categories.sort_values('unique', ascending = False).head(20))
