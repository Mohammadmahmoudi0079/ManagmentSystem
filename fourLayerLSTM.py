import math
import pandas_datareader as web
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from datetime import datetime

plt.style.use('fivethirtyeight')

stock_symbol = "THYAO.IS"
start_date = "2015-01-01"
end_date = "2023-10-01"
dfyf = yf.download(stock_symbol, start=start_date, end=end_date)
ticker = yf.Ticker(stock_symbol)
stock_info = ticker.info
currency = "TRY"
dfyf['Close']

plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(dfyf['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price Price ('+currency+')',fontsize=18)
plt.show()

from alpha_vantage.foreignexchange import ForeignExchange

def get_closing_prices(symbol, base_currency, quote_currency, api_key, start_date, end_date):
    try:
        fx = ForeignExchange(key=api_key)
        exchange_rates, _ = fx.get_currency_exchange_daily(
            from_symbol=base_currency,
            to_symbol=quote_currency,
            outputsize='full'
        )
        dffx = pd.DataFrame.from_dict(exchange_rates, orient='index')
        dffx.index = pd.to_datetime(dffx.index)
        dffx.sort_index(inplace=True)
        dffx = dffx[(dffx.index >= start_date) & (dffx.index <= end_date)]
        closing_prices = dffx['4. close'].astype(float)
        return closing_prices
    except Exception as e:
        return f"Error: {e}"

api_key = 'DS1HJJG2655YPQ8L'
symbol = 'TRYUSD'
base_currency = 'USD'
quote_currency = 'TRY'
start_date = '2015-01-01'
end_date = '2023-10-01'

dffx = get_closing_prices(symbol, base_currency, quote_currency, api_key, start_date, end_date)
dfyf['Close']= dfyf['Close']/dffx
print(dfyf['Close'])

plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(dfyf['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price Price ('+base_currency+')',fontsize=18)
plt.show()

data = dfyf['Close']
dataset = data.values
dataset=dataset.reshape(-1,1)
training_data_len = math.ceil(len(dataset)*0.8)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
print(scaled_data)
print(data)

sd=dfyf
sd['Close']=scaled_data
plt.figure(figsize=(16,8))
plt.title('Scaled Data')
plt.plot(sd['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Value',fontsize=18)
plt.show()

train_data = scaled_data[0:training_data_len,:]
x_train=[]
y_train=[]

for i in range(60,len(train_data)):
  x_train.append(train_data[i-60:i,0])
  y_train.append(train_data[i,0])
  if i<=60:
    print(x_train)
    print(y_train)

x_train,y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train , (x_train.shape[0],x_train.shape[1],1))
print(x_train.shape)
print(x_train)

from keras.src.engine.sequential import Sequential
model = Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,batch_size=1,epochs=1)

print(x_train.shape)

test_data = scaled_data[training_data_len-60:,:]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60,len(test_data)):
  x_test.append(test_data[i-60:i,0])

x_test = np.array(x_test)

if np.isnan(x_test).any() or np.isinf(x_test).any():
    print("Warning: NaN or Infinite values detected in x_test before prediction.")

predictions = model.predict(x_test)

print(predictions.shape)

pred=data[training_data_len:]
pred['Predictions']=predictions
plt.figure(figsize=(32,16))
plt.title('Predictions 1')
plt.xlabel('date',fontsize=18)
plt.ylabel('Predicted value',fontsize=18)
plt.plot(pred['Predictions'],color='green')

predictions = scaler.inverse_transform(predictions)

pred=data[training_data_len:]
pred['Predictions']=predictions
plt.figure(figsize=(32,16))
plt.title('Predictions 2')
plt.xlabel('date',fontsize=18)
plt.ylabel('Predicted value',fontsize=18)
plt.plot(pred['Predictions'],color='orange')

valid_indices = ~np.isnan(predictions) & ~np.isnan(y_test)
rmse = np.sqrt(np.mean((predictions[valid_indices] - y_test[valid_indices])**2))
print(rmse)

inCase = data
temp= pd.DataFrame()
temp['Close']=data
data=temp

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions']=predictions


plt.figure(figsize=(32,16))
plt.title('Model')
plt.xlabel('date',fontsize=18)
plt.ylabel('Close Price $',fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Predictions'],loc='lower right')
plt.show()

plt.figure(figsize=(32,16))
plt.title('Model')
plt.xlabel('date',fontsize=18)
plt.ylabel('Close Price $',fontsize=18)
plt.plot(valid[['Close','Predictions']])
plt.legend(['Val','Predictions'],loc='lower right')
plt.show()

valid['Close']
vc = list(valid['Close'])
vp= list(valid['Predictions'])
i=0
t=0
f=0
for i in range(len(vc)-1):
  if (vc[i+1]>vc[i]) & (vp[i+1]>vp[i]):
    t+=1
  elif (vc[i+1]<vc[i]) & (vp[i+1]<vp[i]):
    t+=1
  elif (vc[i+1]==vc[i]) & (vp[i+1]==vp[i]):
    t+=1
  else:
    f+=1
  i+=1
print('True: '+str(t)+'    False: '+str(f))

fut_inp = valid['Close']
tmp_inp = list(fut_inp)
fut_inp
lst_output=[]
n_steps=100
i=0
while(i<60):
    
    if(len(tmp_inp)>100):
        fut_inp = np.array(tmp_inp[1:])
        fut_inp=fut_inp.reshape(1,-1)
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        tmp_inp = tmp_inp[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        fut_inp = fut_inp.reshape((1, n_steps,1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i=i+1
    
plt.figure(figsize=(32,16))
plt.title('Next 60 Days')
plt.xlabel('days',fontsize=18)
plt.ylabel('Predicted Close Price $',fontsize=18)
plt.plot(lst_output)
plt.show

valid['Close'].mean()