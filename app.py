import os
import pickle
from airbnb.airbnb_etl import Airbnb
from flask import Flask, render_template, Response
from waitress import serve

# App
app = Flask(__name__)

# Loading model and encoder
model = pickle.load(open('model/lgbm_airbnb.pkl', 'rb'))
label_encoder = pickle.load(open('scalers/le.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Instantiate Airbnb class
    pipeline = Airbnb()

    # Loading credentials  
    USER = os.getenv('USER')
    PASSWORD = os.getenv('PASSWORD')
    HOST = os.getenv('HOST')
    PORT = os.getenv('PORT')
    DATABASE = os.getenv('DATABASE')     
    
    # Connect to Database
    conn_url = f'postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}'

    # Get Data
    df_new_users, df_sessions_new_users = pipeline.get_data(conn_url)

    # Data Cleaning
    df2, df_sessions_new_users, new_users_id, df_original = pipeline.data_cleaning(df_new_users, df_sessions_new_users)
    
    # Feature Engineering
    df3 = pipeline.feature_engineering(df2, df_sessions_new_users)

    # Data Preparation
    df4 = pipeline.data_preparation(df3)

    # Get Predictions
    df_get_pred = pipeline.get_prediction(model, label_encoder, new_users_id, df_original, df4)

    # Add to PostgreSQL Database
    pipeline.adding_to_postgresql(conn_url, 'data_pred', df_get_pred)
    
    return Response('New data with predictions successfully added in the PostgreSQL Database.', status='200')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    serve(app, host="0.0.0.0", port=port)
