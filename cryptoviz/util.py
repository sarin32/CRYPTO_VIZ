import base64
import io
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import requests
from keras.models import load_model
from binance.client import Client
from matplotlib import pyplot as plt
from newsapi import NewsApiClient

N_PAST = 28  # Number of past hours we want to use to predict the future.

price_model = load_model('cryptoviz/assets/price.h5')
senti_model = load_model('cryptoviz/assets/sentiment.h5')
scaler = joblib.load('cryptoviz/assets/minmax_scaler.bin')


# method to preprocess data
def preprocess(df):
    n_future = 1  # Number of hours we want to look into the future based on the past hours.
    # Empty lists to be populated using formatted training data
    X = []
    Y = []
    # Reformat input data into a shape: (n_samples x timesteps x n_features)
    for i in range(N_PAST, len(df) - n_future + 1):
        X.append(df[i - N_PAST:i, 0:df.shape[1]])
        Y.append(df[i + n_future - 1:i + n_future, 0])
    X, Y = np.array(X), np.array(Y)
    return X, Y


def getDf():
    df = pd.read_csv('cryptoviz/assets/final_dataset.csv')
    df[['price', 'sentiment']] = scaler.transform(df[['price', 'sentiment']])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def predict_future(length=10):
    df = getDf()
    test_df = np.array(df.iloc[-N_PAST - 1:, 1:])
    prediction = []

    for i in range(length):
        testX, testY = preprocess(test_df)
        price_pred = price_model.predict(testX)
        senti_pred = senti_model.predict(testX)
        data = np.array([[price_pred[0][0], senti_pred[0][0]]])
        test_df = np.append(test_df, data, axis=0)
        test_df = np.delete(test_df, 0, axis=0)

        prediction.append([price_pred[0][0], senti_pred[0][0]])
    prediction = scaler.inverse_transform(prediction)

    dates = pd.date_range(np.array(df.tail(1))[0][0], periods=length + 1, freq='H')
    dates = dates.delete(0)
    dates = pd.DataFrame(dates, columns=['timestamp'])

    prediction = pd.DataFrame(prediction, columns=['price', 'sentiment'])
    prediction = pd.concat([dates, prediction], axis=1, join='inner')
    return prediction


def create_plot(hist_length, pred_length):
    df = getDf()
    prediction = predict_future(pred_length)
    df[['price', 'sentiment']] = scaler.inverse_transform(df[['price', 'sentiment']])

    plt.figure(figsize=(15, 8))
    plt.plot(prediction['timestamp'], prediction['price'], label='Predicted price of BTC', color='red')
    plt.plot(df['timestamp'][-hist_length:], df['price'][-hist_length:], label='Historic price of BTC', color='blue')
    plt.grid(color='grey')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.title('Historic and predicted price of BTC')
    plt.legend()
    plt.style.use('dark_background')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()


def data_extract(sym, start, end):
    client = Client('API_KEY', 'SECRET_KEY')
    if end == "":
        cryptocurrency = client.get_historical_klines(symbol=sym, interval=Client.KLINE_INTERVAL_1HOUR, start_str=start)
    else:
        cryptocurrency = client.get_historical_klines(symbol=sym, interval=Client.KLINE_INTERVAL_1HOUR, start_str=start,
                                                      end_str=end)
    cryptocurrency = pd.DataFrame(cryptocurrency,
                                  columns=['timestamp', 'Open', 'High', 'Low', 'price', 'Volume', 'Close time',
                                           'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
                                           'Taker buy quote asset volume', 'Ignore'])
    return cryptocurrency


def update_dataset():
    df = pd.read_csv('cryptoviz/assets/final_dataset.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    sentiment_data = []
    now = datetime.now()
    sentiment_index = requests.get('http://api.senticrypt.com/v1/history/index.json').json()

    dates = pd.date_range(np.array(df.tail(1))[0][0], end=now, freq='H')
    dates = dates.delete(0)
    print('updating dataset')
    for date in dates:
        # get sentiment data
        x = 'bitcoin-' + date.strftime('%Y-%m-%d_%H') + '.json'
        if x not in sentiment_index:
            break
        text = 'http://api.senticrypt.com/v1/history/bitcoin-' + date.strftime('%Y-%m-%d_%H') + '.json'
        print(text)
        mean = requests.get(text).json()[0]['mean']
        sentiment_data.append([date, mean])
        # print(sentiment_data)

    symbol = 'BTCUSDT'
    start = dates[0].strftime('%Y-%m-%d %H:%M:%S')
    end = dates[-1].strftime('%Y-%m-%d %H:%M:%S')
    btcDf = data_extract(symbol, start, end)
    sDf = pd.DataFrame(sentiment_data, columns=['timestamp', 'sentiment'])
    fDf = pd.concat([sDf['timestamp'], btcDf['price'], sDf['sentiment']], axis=1, join='inner')
    fDf.set_index('timestamp')
    df.set_index('timestamp')
    df = df.append(fDf)
    df.to_csv('cryptoviz/assets/final_dataset.csv', index=False)


def getNews(title):
    newsapi = NewsApiClient(api_key="b0f75ce660c0466a9a98c2478f8abb62")
    topheadlines = newsapi.get_everything(q=title,
                                          language='en')
    articles = topheadlines['articles']
    myArticles = []

    print(articles[0])
    for article in articles:
        d = datetime.fromisoformat(article['publishedAt'][:-1]).astimezone(timezone.utc)
        d= d.strftime('%Y-%m-%d %H:%M:%S')
        art = {
            'title': article['title'],
            'description': article['description'],
            'urlToImage': article['urlToImage'],
            'url': article['url'],
            'publishedAt': d
        }
        myArticles.append(art)

    return myArticles
