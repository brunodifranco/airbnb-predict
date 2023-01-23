# Libraries
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# Helper Functions
def to_int_64(df,change_types_list):
    '''
    Changes data type to int64.

    Parameters
    ----------
    df : dataframe on which the function will be applied.

    change_types_list : list of columns on which the function will be applied.

    Returns
    -------
    df : dataframe with new column types.
    '''  
    for i in change_types_list:
        df[f'{i}'] = df[f'{i}'].astype('int64')

    return df

def rows_to_cols(df, variable, original_col):
    '''
    Sets each value in rows from a original column to a new column.

    Parameters
    ----------
    df : dataframe on which the function will be applied.
    variable : variable
    original_col : original column
    
    Returns
    -------
    df : DataFrame 
    '''  
    variable_list = [variable]
    df[variable] = df[original_col].str.contains('|'.join(variable_list)).astype(int)
    df.loc[df[variable] == 1, variable] = df[original_col]
    df[variable] = df[variable].str.rsplit(',').str[-1]    
    return df

def nature_encode(df, col, div_period):
    '''
    Applies a Nature Cyclical Transformation, where each period
    is a combination of sin and cos.
    
    Parameters
    ----------
    df : dataframe on which the function will be applied.

    col : period column on which the function will be applied.

    div_period : amount of periods until the cycle restarts (e.g. month=12, week=7, etc).

    Returns
    -------
    None
    '''  
    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/div_period)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/div_period)
    return None

def top_5_pred(y_pred_proba, member_ids, label_encoder):
    '''
    Getting top 5 predictions for each user.

    Parameters
    ----------
    y_pred_proba : predicted probabilities for each class.

    member_ids : users id list.

    label_encoder: target variable label encoder.

    Returns
    -------
    DataFrame 
    '''
    users_ids = []
    predictions = []
    for i, member_id in enumerate(member_ids):
        users_ids.extend([member_id]*5)
        top_5 = np.argsort(y_pred_proba[i, :])[::-1]
        for j in range(5):
            predictions.append(label_encoder.classes_[top_5[j]])
    return pd.DataFrame({'id': users_ids, 'predicted_country': predictions})

#########################################################################################################################################################################
# Airbnb class
class Airbnb(object):
        def __init__(self):
                self.home_path=''
                self.ss_action_type_unique  = pickle.load(open(self.home_path + 'scalers/ss_action_type_unique.pkl', 'rb')) # Loading action_type_unique StandardScaler
                self.rs_age                 = pickle.load(open(self.home_path + 'scalers/rs_age.pkl', 'rb')) # age RobustScaler
                self.rs_secs_elapsed_median = pickle.load(open(self.home_path + 'scalers/rs_secs_elapsed_median.pkl', 'rb')) # secs_elapsed_median RobustScaler
                self.mm_secs_elapsed_max    = pickle.load(open(self.home_path + 'scalers/mm_secs_elapsed_max.pkl', 'rb')) # secs_elapsed_max MinMaxScaler
                self.mm_secs_elapsed_mean   = pickle.load(open(self.home_path + 'scalers/mm_secs_elapsed_mean.pkl', 'rb')) # secs_elapsed_mean MinMaxScaler
                self.mm_secs_elapsed_sum    = pickle.load(open(self.home_path + 'scalers/mm_secs_elapsed_sum.pkl', 'rb')) # secs_elapsed_sum MinMaxScaler
                self.mm_secs_elapsed_std    = pickle.load(open(self.home_path + 'scalers/mm_secs_elapsed_std.pkl', 'rb')) # secs_elapsed_std MinMaxScaler
                self.mm_amount_of_sessions  = pickle.load(open(self.home_path + 'scalers/mm_amount_of_sessions.pkl', 'rb')) # amount_of_sessions MinMaxScaler

        def get_data(self, conn_url):
            # conn
            conn_url = conn_url
            engine = create_engine(conn_url)

            # queries
            query_users = """
            SELECT * FROM new_users;
            """
            query_sessions = """
            SELECT * FROM sessions_new_users;
            """
            # retrieve data
            df_new_users = pd.read_sql(query_users, con=engine)
            df_sessions_new_users = pd.read_sql(query_sessions, con=engine)

            return df_new_users, df_sessions_new_users

        def data_cleaning(self, df1, df_sessions_new_users):
                '''
                Cleaning data, and separating the original dataset for later use in get_prediction function.
                '''               
                # initial setup           
                new_users_id = df1['id'].to_list() # getting id's 
                df1.drop('date_first_booking', axis=1, inplace=True) # dropping date_first_booking
                df_original = df1.copy() # saving original data (for later display on Streamlit) 
                
                # df_new_users                
                df1['first_affiliate_tracked'].fillna('untracked', inplace=True) # first_affiliate_tracked
                df1['timestamp_first_active'] = pd.to_datetime(df1['timestamp_first_active'], format='%Y%m%d%H%M%S') # timestamp_first_active
                df1['date_account_created'] = pd.to_datetime(df1['date_account_created']) # date_account_created 
                df1['age'].fillna(df1[df1['age'] < 115]['age'].median(), inplace=True) # age
                df1['age'] = df1['age'].astype('int64')
                df1['age'] = df1['age'].apply(lambda x: (2015 - x) if x > 1900 else x)
                df1 = df1[(df1['age'] > 12) & (df1['age'] < 115)]

                # df_sessions_new_users
                df_sessions_new_users['secs_elapsed'].fillna(0, inplace=True) # secs_elapsed
                df_sessions_new_users['action_type'] = np.where(df_sessions_new_users['action_type'].isnull(), df_sessions_new_users['action'], df_sessions_new_users['action_type']) # action_type
                df_sessions_new_users = df_sessions_new_users.drop(['action', 'action_detail', 'device_type'], axis=1).rename(columns={'user_id' : 'id'})

                # Returns the cleaned new users data, cleaned new users df_sessions data, id's and original data
                return df1, df_sessions_new_users, new_users_id, df_original

        def feature_engineering(self, df2, df_sessions_new_users):
                '''
                Creating new features from users and sessions data. Merging them afterwards.
                '''
                # df_new_users 
                # first_active               
                df2['first_active'] = pd.to_datetime(df2['timestamp_first_active'].dt.strftime('%Y-%m-%d'))
                df2['days_active_to_account'] = (df2['date_account_created'] - df2['first_active']).apply(lambda x: x.days)
                df2['day_first_active'] = df2['first_active'].dt.day
                df2['day_of_week_first_active'] = df2['first_active'].dt.dayofweek

                # account created
                df2['month_account_created'] = df2['date_account_created'].dt.month
                df2['week_of_year_account_created'] = df2['date_account_created'].dt.isocalendar().week
                df2 = to_int_64(df2, ['week_of_year_account_created'])

                # drop auxiliary columns
                cols_drop_aux = ['date_account_created', 'timestamp_first_active', 'first_active'] 
                df2 = df2.drop(cols_drop_aux, axis=1)
                
                # df_sessions_new_users
                # secs_elapsed                
                df_ref_sessions = (df_sessions_new_users[['id', 'secs_elapsed']].groupby('id').agg(secs_elapsed_max = ('secs_elapsed', 'max'),
                                                                                                secs_elapsed_mean = ('secs_elapsed', 'mean'),
                                                                                                secs_elapsed_median = ('secs_elapsed', 'median'),
                                                                                                secs_elapsed_sum = ('secs_elapsed', 'sum'),
                                                                                                secs_elapsed_std = ('secs_elapsed', 'std'),
                                                                                                amount_of_sessions = ('secs_elapsed', 'count'))).reset_index()
                df_ref_sessions['secs_elapsed_std'].fillna(0, inplace=True) # fill secs_elapsed_std nan's with zeros

                # most common
                aux = df_sessions_new_users[['id', 'action_type']]
                aux_most_common = aux.groupby('id').value_counts().reset_index()
                df_ref_most_common = aux_most_common.groupby('id').max().reset_index().rename(columns={'action_type' : 'action_type_most_common'}).drop(columns=0) 

                # unique
                df_ref_unique = aux.groupby('id').nunique().reset_index().rename(columns={'action_type' : 'action_type_unique'})                     

                # ratio of unique
                df_ref_unique_ratio = aux.groupby('id').value_counts(normalize=True).reset_index().rename(columns={0 : 'unique_ratio'}) 
                df_ref_unique_ratio['action_type_unique_ratio'] = df_ref_unique_ratio['action_type'] + ',' + df_ref_unique_ratio['unique_ratio'].astype(str) # adding both columns together

                for i in df_ref_unique_ratio['action_type'].unique():
                    rows_to_cols(df_ref_unique_ratio, i, 'action_type_unique_ratio') # this function is available in the helper functions section
                df_ref_unique_ratio = df_ref_unique_ratio.drop(columns=['action_type', 'unique_ratio', 'action_type_unique_ratio']).fillna(0) # fill nan's with zeros and drops auxiliary columns
                df_ref_unique_ratio = df_ref_unique_ratio.add_prefix('action_type_ratio_').rename(columns={'action_type_ratio_id' : 'id'}) # adding action_type_ratio prefix to all columns

                for i in df_ref_unique_ratio.columns.drop('id'):
                    df_ref_unique_ratio[i] = df_ref_unique_ratio[i].astype(float) # convert all to float
                df_ref_unique_ratio = df_ref_unique_ratio.groupby('id').max().reset_index() # .max() so we could get rid of the duplicated id's and still get each correct value

                # merging back to df_sessions_new_users
                df_sessions_new_users = df_sessions_new_users[['id']].drop_duplicates().reset_index(drop=True)
                df_sessions_new_users = pd.merge(df_sessions_new_users, df_ref_most_common, on='id', how='inner')
                df_sessions_new_users = pd.merge(df_sessions_new_users, df_ref_unique, on='id', how='inner')
                df_sessions_new_users = pd.merge(df_sessions_new_users, df_ref_unique_ratio, on='id', how='inner').rename(columns={'action_type_ratio_-unknown-': 'action_type_ratio_unknown'}) # postgresql doesn't accept this character: '-'
                df_sessions_new_users = pd.merge(df_sessions_new_users, df_ref_sessions, on='id', how='inner')

                # Merging df2 and df_sessions_new_users
                df2 = pd.merge(df2, df_sessions_new_users, on='id', how='inner')

                # Returns the transformed data
                return df2

        def data_preparation(self, df3):
            '''
            Scaling, encoding, transforming data. Taking only selected columns.
            '''
            # Scaling
            df3['action_type_unique']  = self.ss_action_type_unique.transform(df3[['action_type_unique']].values)
            df3['age']                 = self.rs_age.transform(df3[['age']].values)
            df3['secs_elapsed_median'] = self.rs_secs_elapsed_median.transform(df3[['secs_elapsed_median']].values)
            df3['secs_elapsed_max']    = self.mm_secs_elapsed_max.transform(df3[['secs_elapsed_max']].values)
            df3['secs_elapsed_mean']   = self.mm_secs_elapsed_mean.transform(df3[['secs_elapsed_mean']].values)
            df3['secs_elapsed_sum']    = self.mm_secs_elapsed_sum.transform(df3[['secs_elapsed_sum']].values)
            df3['secs_elapsed_std']    = self.mm_secs_elapsed_std.transform(df3[['secs_elapsed_std']].values)
            df3['amount_of_sessions']  = self.mm_amount_of_sessions.transform(df3[['amount_of_sessions']].values)

            # Encoding 
            df3 = pd.get_dummies(df3, prefix=['gender'], columns=['gender'])
            df3 = pd.get_dummies(df3, prefix=['signup_method'], columns=['signup_method'])
            df3 = pd.get_dummies(df3, prefix=['first_affiliate_tracked'], columns=['first_affiliate_tracked'])
            df3 = pd.get_dummies(df3, prefix=['first_device_type'], columns=['first_device_type'])

            # Transformation
            cols = {'day_of_week_first_active': 7,    
                    'month_account_created' : 12, 
                    'day_first_active': 30,
                    'week_of_year_account_created': 52}

            for period, cycle in cols.items():
                    nature_encode(df3, period, cycle)

            # Feature Selection
            cols_selected = ['age',
                             'action_type_ratio_booking_request',
                             'secs_elapsed_max',
                             'secs_elapsed_sum',
                             'secs_elapsed_median',
                             'action_type_ratio_message_post',
                             'secs_elapsed_mean',
                             'secs_elapsed_std',
                             'action_type_ratio_submit',
                             'action_type_ratio_data',
                             'action_type_ratio_click',
                             'action_type_ratio_view',
                             'amount_of_sessions',
                             'action_type_ratio_unknown',
                             'day_first_active_cos',
                             'day_first_active_sin',
                             'action_type_ratio_show',
                             'week_of_year_account_created_cos',
                             'week_of_year_account_created_sin',
                             'action_type_unique',
                             'day_of_week_first_active_sin',
                             'gender_-unknown-',
                             'action_type_ratio_lookup',
                             'day_of_week_first_active_cos',
                             'month_account_created_cos',
                             'month_account_created_sin',
                             'signup_method_facebook',
                             'signup_method_basic',
                             'gender_FEMALE',
                             'gender_MALE',
                             'action_type_ratio_campaigns',
                             'first_device_type_Mac Desktop',
                             'first_affiliate_tracked_untracked',
                             'first_device_type_Windows Desktop']

            # Returns the transformed data with selected cols
            return df3[cols_selected]

        def get_prediction(self, model, label_encoder, ids, original_data, df4):
            '''
            Gets the top five predicted countries.
            '''
            # Prediction
            pred = model.predict_proba(df4)
            df_pred = top_5_pred(pred, ids, label_encoder)

            # Get each prediction in order    
            df_pred_1 = df_pred.groupby('id').nth(0).reset_index().rename(columns={'predicted_country' : 'predicted_country_1'})
            df_pred_2 = df_pred.groupby('id').nth(1).reset_index().rename(columns={'predicted_country' : 'predicted_country_2'})
            df_pred_3 = df_pred.groupby('id').nth(2).reset_index().rename(columns={'predicted_country' : 'predicted_country_3'})
            df_pred_4 = df_pred.groupby('id').nth(3).reset_index().rename(columns={'predicted_country' : 'predicted_country_4'})
            df_pred_5 = df_pred.groupby('id').nth(4).reset_index().rename(columns={'predicted_country' : 'predicted_country_5'})

            # Merge predictions to original data for display (in Streamlit)    
            original_data   = pd.merge(original_data, df_pred_1, on='id', how='inner')
            original_data   = pd.merge(original_data, df_pred_2, on='id', how='inner')
            original_data   = pd.merge(original_data, df_pred_3, on='id', how='inner')
            original_data   = pd.merge(original_data, df_pred_4, on='id', how='inner')
            df_get_pred     = pd.merge(original_data, df_pred_5, on='id', how='inner')

            # Final adjustments (it's better looking in Streamlit)
            df_get_pred['timestamp_first_active'] = pd.to_datetime(df_get_pred['timestamp_first_active'], format='%Y%m%d%H%M%S')
            df_get_pred['date_account_created']   = pd.to_datetime(df_get_pred['date_account_created'])

            return df_get_pred

        def adding_to_postgresql(self, conn_url, table, final_data_pred):
            '''
            Adds original data WITH PREDICTIONS in the PostgreSQL Database.
            This is what will be displayed in Streamlit.
            '''
            conn = create_engine(conn_url, echo=False)
            final_data_pred.to_sql(table, con=conn, if_exists='replace', index=False) # Inserting data to table

            return None