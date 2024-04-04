import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib import rcParams
from joblib import load
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout,Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

def generate_date_list(start_date):
    """
    Generate a list of 12 consecutive dates starting from the given start date in the specified format.
    
    Parameters:
        start_date (str): Start date in 'YYYY-MM-DD' format.
    
    Returns:
        pd.DatetimeIndex: DatetimeIndex containing 12 consecutive dates in the specified format.
    """
    # Convert start date to datetime object
    start_date = pd.to_datetime(start_date)
    
    # Generate 12 consecutive dates starting from the start date
    date_list = pd.date_range(start=start_date, periods=12, freq='W-SUN')
    
    return date_list

def main():

    st.set_option('deprecation.showPyplotGlobalUse', False)

    model = tf.keras.models.load_model('my_model.keras')

    test = pd.read_csv('last_q.csv')
    test['date'] = pd.to_datetime(test['date'])
    test = test.set_index('date')

    scaler = load('scaler.joblib')

    st.title('RevCheck')
    st.subheader("Last quarter's weekly revenue")

    # report performance
    rcParams['figure.figsize'] = 12, 8
    # line plot of observed vs predicted
    plt.plot(test.index,test,label="Last Quarter",color='#2574BF')
    plt.title('Weekly Revenue')
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.legend()
    st.pyplot()

    if st.button('Make Predictions'):

        pred_list_b = []
        last_week = test.iloc[11].name
        pred_start = last_week.strftime('%Y-%m-%d')
        n_input = 12
        n_features = 1

        test_input = np.array(test).reshape(-1,1)
        test_input = scaler.transform(test_input)

        batch = test_input.reshape((1, n_input, n_features))

        for i in range(n_input):   
            pred_list_b.append(model.predict(batch)[0]) 
            batch = np.append(batch[:,1:,:],[[pred_list_b[i]]],axis=1)

        df_predict_bi = pd.DataFrame(scaler.inverse_transform(pred_list_b),
                                index=generate_date_list(pred_start), columns=['Prediction'])

        st.subheader("Next quarter's weekly revenue")
        # report performance
        rcParams['figure.figsize'] = 12, 8
        # line plot of observed vs predicted
        plt.plot(test.index,test,label="Last Quarter",color='#2574BF')
        plt.plot(df_predict_bi.index,df_predict_bi,label="Next Quarter",color='red')
        plt.title('Bidirectional LSTM furniture sales forecasting')
        plt.xlabel('Date')
        plt.ylabel('Weekly Revenue')
        plt.legend()
        st.pyplot()

if __name__ == '__main__':
    main()                            