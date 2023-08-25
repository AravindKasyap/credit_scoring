
import pkg_resources
import pip
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import glob
import math
import toad



def data_split(df, start, end, date_col):
    data = df[(df[date_col] >= start) & (df[date_col] < end)]
    data = data.reset_index(drop=True)
    return data


data = pd.read_csv('PMN_test.csv')

#use the world 'label'
data['label']=data['default.payment.next.month']
data=data.drop(columns=['default.payment.next.month'])

# set an exclude list for the scorecard package Toad
exclude_list = ['ID','label']

# use the ID column to split the train-test data
train = data_split(data,start = 0, end=22500,date_col='ID')
test = data_split(data,start = 22500, end=172792,date_col='ID')


train_selected, drop_lst= toad.selection.select(frame = train,
                                                target=train['label'],
                                                empty = 0.7,
                                                iv = 0.02, corr = 1,
                                                return_drop=True,
                                                exclude=exclude_list)


combiner = toad.transform.Combiner()
combiner.fit(X=train_selected,
             y=train_selected['label'],
             method='chi',
             min_samples = 0.05,
             exclude=exclude_list)

#output binning
bins = combiner.export()

#apply binning
train_selected_bin = combiner.transform(train_selected)
test_bin = combiner.transform(test[train_selected_bin.columns])

features_list = [feat for feat in train_selected_bin.columns if feat not in exclude_list]


t=toad.transform.WOETransformer()

train_woe = t.fit_transform(X=train_selected_bin,
                            y=train_selected_bin['label'],
                            exclude=exclude_list)

test_woe = t.transform(test_bin)

final_data_woe = pd.concat([train_woe,test_woe])

features_use = [feat for feat in final_data_woe.columns if feat not in exclude_list]

#prepare train & test data
x_train = train_woe[features_use]
y_train=train_woe['label']
x_test =test_woe[features_use]
y_test = test_woe['label']

card = toad.ScoreCard(
    combiner = combiner,
    transer = t,
    class_weight = 'balanced',
    C=0.1,
    base_score = 1000,
    base_odds = 35 ,
    pdo = 80,
    rate = 2
)

card.fit(train_woe[features_use], train_woe['label'])

test['CreditScore'] = card.predict(test)
data['CreditScore'] = card.predict(data)

#output the scorecard
final_card_score=card.export()

test.loc[0,:]

card.predict(test)[0]

data = {

    'LIMIT_BAL': [420000.0],
    'EDUCATION': [2.0],
    'AGE': [37.0],
    'PAY_0': [0.0],
    'PAY_2': [0.0],
    'BILL_AMT1': [10000.0],
    'BILL_AMT2': [10000.0],
    'PAY_AMT1': [70000.0],
    'PAY_AMT2': [1846.0],
}

# Create a DataFrame from the data
df = pd.DataFrame(data)
card.predict(df)[0]