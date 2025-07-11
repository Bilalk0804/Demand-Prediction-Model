import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Try to import XGBoost, fallback to RandomForest if not available
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False

# 1. Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 2. Feature Engineering
def create_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    return df

train = create_features(train)
test = create_features(test)

for col in ['store', 'item']:
    le = LabelEncoder()
    le.fit(list(train[col].astype(str)) + list(test[col].astype(str)))
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

features = ['store', 'item', 'year', 'month', 'day', 'dayofweek', 'weekofyear']
X = train[features]
y = train['sales'] if 'sales' in train.columns else train.iloc[:, -1]  # fallback if column name is different
X_test = test[features]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

if xgb_available:
    model = XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, subsample=0.8, random_state=42, n_jobs=-1)
else:
    model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)

model.fit(X_train, y_train)

y_pred_val = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f'Validation RMSE: {rmse:.4f}')

test_preds = model.predict(X_test)

submission = pd.DataFrame({'id': test['id'], 'sales': np.round(test_preds).astype(int)})
submission.to_csv('submission.csv', index=False)
print('submission.csv file created!')

def flag_targeted_offers(test_df, preds, stock_df=None, demand_threshold=20, stock_threshold=100):
    """
    Flags items for targeted offers based on low predicted demand and high stock.
    stock_df: DataFrame with columns ['store', 'item', 'stock'] (if available)
    Returns DataFrame with flagged items.
    """
    flagged = test_df.copy()
    flagged['predicted_sales'] = preds
    if stock_df is not None:
        flagged = flagged.merge(stock_df, on=['store', 'item'], how='left')
        flagged['flag_offer'] = (flagged['predicted_sales'] < demand_threshold) & (flagged['stock'] > stock_threshold)
    else:
        flagged['flag_offer'] = flagged['predicted_sales'] < demand_threshold
    return flagged[flagged['flag_offer']]

'''for data frames and further working'''
# Example usage (uncomment if you have stock_df):
# stock_df = pd.read_csv('stock.csv')
# offers = flag_targeted_offers(test, test_preds, stock_df)
# offers.to_csv('targeted_offers.csv', index=False) 