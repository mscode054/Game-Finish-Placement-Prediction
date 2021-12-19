from IPython.display import FileLink
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

df_train = pd.read_csv('train_V2.csv')
print(df_train.info())

df_train.dropna(subset=['winPlacePerc'], inplace=True, axis=0)
print(df_train.isnull().sum())

#EDA and Visualization
plt.figure(figsize=(20, 20))
sns.heatmap(df_train.corr(), cmap='Reds', annot=True)
plt.title('Correlation Matrix')
plt.show()

# Encoding categorical data
df_train[['matchType', 'Id', 'groupId', 'matchId']].nunique()


def map(x):
    return 'solo' if ('solo' in x) else 'duo' if ('duo' in x) else 'squad'


df_train['matchType'] = df_train['matchType'].apply(map)
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoder.fit(df_train[['matchType']])
encoder.categories_
encoded_cols = list(encoder.get_feature_names(['matchType']))
print(encoded_cols)
df_train[encoded_cols] = encoder.transform(df_train[['matchType']])

# Training and Validation Sets
col_train = df_train.columns
col_train = ['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 'heals', 'killPlace', 'kills',
             'killStreaks', 'longestKill', 'revives', 'rideDistance', 'walkDistance', 'weaponsAcquired',
             'matchType_duo', 'matchType_solo', 'matchType_squad']

train_inputs, val_inputs, train_targets, val_targets = train_test_split(
    df_train[0:50000][col_train], df_train[0:50000][['winPlacePerc']], test_size=0.2, random_state=0)
print('train_inputs.shape :', train_inputs.shape)
print('val_inputs.shape :', val_inputs.shape)
print('train_targets.shape :', train_targets.shape)
print('val_targets.shape :', val_targets.shape)
train_targets, val_targets = np.ravel(train_targets), np.ravel(val_targets)
print('train_inputs.shape :', train_inputs.shape)
print('val_inputs.shape :', val_inputs.shape)
print('train_targets.shape :', train_targets.shape)
print('val_targets.shape :', val_targets.shape)


# (i) Random Forest Model
RFR = RandomForestRegressor(n_jobs=-1, random_state=0)
RFR.fit(train_inputs, train_targets)

RFR_train_preds = RFR.predict(train_inputs)
RFR_train_rmse = mean_squared_error(
    train_targets, RFR_train_preds, squared=False)

RFR_val_preds = RFR.predict(val_inputs)
RFR_val_rmse = mean_squared_error(val_targets, RFR_val_preds, squared=False)

print('Train RMSE: {}, Validation RMSE: {}'.format(RFR_train_rmse, RFR_val_rmse))
RFR_importance_df = pd.DataFrame({
    'feature': train_inputs.columns,
    'importance': RFR.feature_importances_
}).sort_values('importance', ascending=False)

# Hyperparameter Tuning


def test_params(**params):
    model = RandomForestRegressor(n_jobs=-1, random_state=0, **params)
    model.fit(train_inputs, train_targets)
    train_rmse = mean_squared_error(model.predict(
        train_inputs), train_targets, squared=False)
    val_rmse = mean_squared_error(model.predict(
        val_inputs), val_targets, squared=False)
    print('Train RMSE: {}, Validation RMSE: {}'.format(train_rmse, val_rmse))


test_params(n_estimators=500)
test_params(n_estimators=500, max_depth=20)

# Training
RFR = RandomForestRegressor(
    n_jobs=-1, random_state=0, n_estimators=500, max_depth=20)
RFR.fit(train_inputs, train_targets)
RFR_train_preds = RFR.predict(train_inputs)
RFR_train_rmse = mean_squared_error(
    train_targets, RFR_train_preds, squared=False)

RFR_val_preds = RFR.predict(val_inputs)
RFR_val_rmse = mean_squared_error(val_targets, RFR_val_preds, squared=False)

print('Train RMSE: {}, Validation RMSE: {}'.format(RFR_train_rmse, RFR_val_rmse))
RFR_importance_df = pd.DataFrame({
    'feature': train_inputs.columns,
    'importance': RFR.feature_importances_
}).sort_values('importance', ascending=False)

# Predicting
test_df = pd.read_csv('test_V2.csv')
test_df['matchType'] = test_df['matchType'].apply(map)
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoder.fit(test_df[['matchType']])
encoded_cols = list(encoder.get_feature_names(['matchType']))
test_df[encoded_cols] = encoder.transform(test_df[['matchType']])

test_inputs = test_df[col_train]
test_preds_RFR = RFR.predict(test_inputs)

submission_df = pd.read_csv('sample_submission_V2.csv')
submission_df['winPlacePerc'] = test_preds_RFR
submission_df.to_csv('answer2(i).csv', index=False)

# (ii) Gradient boost regressor
XGB = XGBRegressor(random_state=0, n_jobs=-1, n_estimators=20, max_depth=4)
XGB.fit(train_inputs, train_targets)
XGB_train_preds = XGB.predict(train_inputs)
XGB_train_rmse = mean_squared_error(
    train_targets, XGB_train_preds, squared=False)

XGB_val_preds = XGB.predict(val_inputs)
XGB_val_rmse = mean_squared_error(val_targets, XGB_val_preds, squared=False)

print('Train RMSE: {}, Validation RMSE: {}'.format(XGB_train_rmse, XGB_val_rmse))
XGB_importance_df = pd.DataFrame({
    'feature': train_inputs.columns,
    'importance': XGB.feature_importances_
}).sort_values('importance', ascending=False)

# Hyperparameter tuning


def test_params(**params):
    model = XGBRegressor(n_jobs=-1, random_state=0, **params)
    model.fit(train_inputs, train_targets)
    train_rmse = mean_squared_error(model.predict(
        train_inputs), train_targets, squared=False)
    val_rmse = mean_squared_error(model.predict(
        val_inputs), val_targets, squared=False)
    print('Train RMSE: {}, Validation RMSE: {}'.format(train_rmse, val_rmse))


test_params(n_estimators=100)
test_params(n_estimators=100, max_depth=4)

# Training
XGB = XGBRegressor(random_state=0, n_jobs=-1, n_estimators=100, max_depth=4)
XGB.fit(train_inputs, train_targets)
XGB_train_preds = XGB.predict(train_inputs)
XGB_train_rmse = mean_squared_error(
    train_targets, XGB_train_preds, squared=False)

XGB_val_preds = XGB.predict(val_inputs)
XGB_val_rmse = mean_squared_error(val_targets, XGB_val_preds, squared=False)

print('Train RMSE: {}, Validation RMSE: {}'.format(XGB_train_rmse, XGB_val_rmse))
XGB_importance_df = pd.DataFrame({
    'feature': train_inputs.columns,
    'importance': XGB.feature_importances_
}).sort_values('importance', ascending=False)

# Predicting
test_preds_XGB = XGB.predict(test_inputs)
submission_df = pd.read_csv('sample_submission_V2.csv')
submission_df['winPlacePerc'] = test_preds_XGB
submission_df.to_csv('answer2(ii).csv', index=False)
