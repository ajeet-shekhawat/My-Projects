import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
# plt.rcParams["figure.figsize"] = (5,4)

# import sys
# !{sys.executable} -m pip install imbalanced-learn==0.8.0

import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')
# plt.style.use('default')
# plt.style.use('seaborn-colorblind')

df = pd.read_csv("Attrition Prediction.csv")
df.head()

print(list(df.columns))

df.info()

df.isna().sum()

df.describe(include='all').transpose()

df.hist(figsize=(20,20))
plt.show()

# Removing EmployeeCount and StandardHours as both the variable have one single value in all the rows 
# and EmployeeNumber as its just a id assigned to the employee
df = df.drop(['EmployeeCount','StandardHours','EmployeeNumber','Over18'], axis=1)

# df_colType = df.columns.to_series().groupby(df.dtypes).groups
# df_colType.keys()

# int_value = list(df_colType.items())[0][1]
# cat_value = list(df_colType.items())[1][1]
# print('int Value: ', int_value)
# print(" ")
# print('cat Value: ', cat_value)

catCol=[]
catNumCol = []
numCol = []

for col in df.columns:
    if df[col].dtype=='O':
        catCol.append(col)
    elif df[col].dtype=='int64':
        if df[col].value_counts().count() < 6:
            catNumCol.append(col)
        else: numCol.append(col)

print("catCol:",catCol)
print(" ")
print("catNumCol:",catNumCol)
print(" ")
print("numCol:",numCol)

catCol.remove("Attrition")
df_pp = df.drop(catCol+catNumCol, axis=1)
catCol.insert(0,"Attrition")

sns.pairplot(df_pp,hue='Attrition',corner=True)
plt.show()

# c = sns.countplot(data=df,x='Attrition')
# plt.bar_label(c.containers[0])
# c.get_yaxis().set_visible(False)
# plt.show()

def countCatBar(x,data=df,counter=True):
    Attr_Count = data[x].value_counts(ascending=False)
    ax = sns.countplot(data=df, x=x, order=Attr_Count.index, color='#A66CFF');

    rel_values = round(data[x].value_counts(ascending=False, normalize=True)*100)
#     lbls = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(Attr_Count, rel_values)]
    lbls = [f'{p:.0f}' for p in rel_values]

    ax.bar_label(container=ax.containers[0], labels=lbls)
    ax.get_yaxis().set_visible(False)
    if Attr_Count.count() >= 3 and (counter==True):
        plt.xticks(rotation='vertical')
    plt.title("% people by " + x)

countCatBar(x='Attrition')

def box_hist_plot(df,iCol,dCol):
    
    # Cut the window in 2 parts
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.20, .80)})

    # Add a graph in each part
    sns.boxplot(data=df, x=iCol, y=dCol, ax=ax_box)
    sns.histplot(data=df, x=iCol, hue=dCol, ax=ax_hist,kde=True)
    # Remove x axis name for the boxplot
    ax_box.set(xlabel='')
    plt.show()

def BHDplot(df,iCol,dCol):
    fig = plt.figure(constrained_layout=True, figsize=(8, 4))
    gs = fig.add_gridspec(4, 2)

    ax1 = fig.add_subplot(gs[0, :1])
    ax1 = sns.boxplot(data=df, x=iCol, y=dCol)
    ax1.get_xaxis().set_visible(False)

    ax2 = fig.add_subplot(gs[1:, :1])
    ax2 = sns.histplot(data=df, x=iCol, hue=dCol, kde=True)

    ax3 = fig.add_subplot(gs[0:, -1])
    ax3 = sns.kdeplot(data=df, x=iCol, hue=dCol)
    ax3.yaxis.set_ticks_position("right")
    ax3.yaxis.set_label_position("right")
    plt.show()
    
    print('-----------------------------------------------------------------------------------------------------------')

# All the nemuric columns
for col in numCol:
    BHDplot(df,col,"Attrition")

# All the catNum columns
for col in catNumCol:
    BHDplot(df,col,"Attrition")

df2 = df.copy()

df2['Attrition'] = df2['Attrition'].apply(lambda a: 1 if a=='Yes' else 0)

# function for plotting the count plot features wrt default ratio
def plotUnivariateRatioBar(feature, data,counter=True):
    feature_dimension = data[feature].unique()
    feature_values = []
    for fd in feature_dimension:
        feature_filter = data[data[feature]==fd]
        feature_count = len(feature_filter[feature_filter["Attrition"]==1])
        feature_values.append(round(feature_count*100/feature_filter["Attrition"].count()))
    
    res = pd.DataFrame(zip(feature_dimension, feature_values),columns =['features', 'Attration%'])
    
    splot=sns.barplot(x='features', y='Attration%', data=res, color='#EC994B',
               order=res.sort_values('Attration%',ascending = False).features)
    
    splot.get_yaxis().set_visible(False)
    plt.title("Churn% wrt "+str(feature))
    if (len(feature_dimension) >= 3) and (counter==True):
        plt.xticks(rotation='vertical')
    plt.xlabel(feature, fontsize=16)
    plt.ylabel("Attrition %", fontsize=16)
    plt.bar_label(splot.containers[0])
#     plt.show()

def countAllCat(df,x,y,counter=True):
    Attr_Count = df[df[y]=='Yes'][x].value_counts(ascending=False)
    ax = sns.countplot(data=df[df[y]=='Yes'], x=x,order=Attr_Count.index, color='#F24C4C')

    rel_values = round(df[df[y]=='Yes'][x].value_counts(ascending=False, normalize=True)*100)
#     lbls = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(Attr_Count, rel_values)]
    lbls = [f'{p:.0f}' for p in rel_values]

    ax.bar_label(container=ax.containers[0], labels=lbls)
    ax.get_yaxis().set_visible(False)
    if (Attr_Count.count() >= 3) and (counter==True):
        plt.xticks(rotation='vertical')
    plt.xlabel(x, fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.title("People Churn% by " + x)

for col in catCol:
    fig = plt.figure(constrained_layout=True, figsize=(14, 5))
#     fig.suptitle(col,fontsize=18)
    gs = fig.add_gridspec(1, 14)

    ax1 = fig.add_subplot(gs[0, :4])
    ax1 = countCatBar(x=col)
    
    ax2 = fig.add_subplot(gs[0, 5:9])
    ax2 = countAllCat(df=df,x=col,y='Attrition')

    ax3 = fig.add_subplot(gs[0:, 10:-1])
    ax3 = plotUnivariateRatioBar(col, data=df2)

    plt.show()
    print('\u2500' * 120)

for col in catNumCol:
    fig = plt.figure(constrained_layout=True, figsize=(14, 5))
#     fig.suptitle(col,fontsize=18)
    gs = fig.add_gridspec(1, 14)

    ax1 = fig.add_subplot(gs[0, :4])
    ax1 = countCatBar(x=col,counter=False)
    
    ax2 = fig.add_subplot(gs[0, 5:9])
    ax2 = countAllCat(df=df,x=col,y='Attrition',counter=False)

    ax3 = fig.add_subplot(gs[0:, 10:-1])
    ax3 = plotUnivariateRatioBar(col, data=df2,counter=False)

    plt.show()
    print('\u2500' * 120)

# Calculate correlations
corr = df2.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
# Heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(corr,
            vmax=.5,
            mask=mask,
            annot=True, fmt='.2f',
            linewidths=.2, cmap="YlGnBu")
plt.show()

# # Calculate correlations
# corr = df.corr()
# mask = np.zeros_like(corr)
# mask[np.triu_indices_from(mask)] = True
# # Heatmap
# plt.figure(figsize=(15, 10))
# sns.heatmap(corr,
#             vmax=.5,
#             mask=mask,
#             annot=True, fmt='.2f',
#             linewidths=.2, cmap="YlGnBu")
# plt.show()

plt.rcParams["figure.figsize"] = (6,5)
sns.scatterplot(x="Age",y="MonthlyIncome", data=df, hue='Attrition', size="Attrition",alpha=.6)
# sns.scatterplot(x="Age",y="MonthlyIncome", data=df[df['Attrition']=='Yes'], alpha=.8, s=20)
# sns.scatterplot(x="Age",y="MonthlyIncome", data=df[df['Attrition']=='No'], alpha=1.0, s=10)
plt.show()

sns.displot(data=df, x="Age", y="MonthlyIncome")
sns.displot(data=df, x="Age", y="MonthlyIncome", kind="kde")
plt.show()

sns.stripplot(y="Age",x="Education", hue="Attrition", data=df,size=4,jitter=0.3,
              palette="Set1", dodge=True, alpha=0.8)
plt.show()

sns.stripplot(y="Age",x="JobLevel", hue="Attrition", data=df,size=4,jitter=0.3,
              palette="Set1", dodge=True)
plt.show()

ax = sns.violinplot(y="DistanceFromHome", x="Attrition", data=df2,
                    inner=None, color=".8")
ax = sns.stripplot(y="DistanceFromHome", x="Attrition", data=df2,size=3,jitter=0.15)

plt.rcParams["figure.figsize"] = (8,5)
sns.stripplot(y="Age",x="NumCompaniesWorked", hue="Attrition", data=df, size=4, jitter=0.3, 
              palette="Set1", dodge=True)
plt.show()
# df.Gender

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Create a label encoder object
le = LabelEncoder()

# Label Encoding will be used for columns with 2 or less unique values
le_count = 0
for col in df.columns[1:]:
    if df[col].dtype == 'object':
        if len(list(df[col].unique())) <= 2:
            le.fit(df[col])
            df[col] = le.transform(df[col])
            le_count += 1
print('{} columns were label encoded.'.format(le_count))

# convert rest of categorical variable into dummy
df = pd.get_dummies(df, drop_first=True)

df.head()

df.columns

# import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 5))
HR_col = list(df.columns)
HR_col.remove('Attrition')
for col in HR_col:
    df[col] = df[col].astype(float)
    df[[col]] = scaler.fit_transform(df[[col]])
df['Attrition'] = pd.to_numeric(df['Attrition'], downcast='float')
df.head()









from imblearn.over_sampling import SMOTE

