import logging, sys, os
PATH = '\\'.join(os.getcwd().split('\\')[:-1])
print(PATH)
sys.path.append(PATH)
logger = logging.getLogger("prophet")
handler2 = logging.StreamHandler(sys.stdout)
handler2.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s222222"
)
handler2.setFormatter(formatter)
logger.addHandler(handler2)

# try to fix pystan's override too
logger2 = logging.getLogger("pystan")
logger2.addHandler(handler2)


import yfinance as yf
from tf_toolbox.ds_modules import *
from tf_toolbox.tidy_modules import *
from tf_toolbox.viz_modules import *
from tf_toolbox.parallell_modules import *
from prophet import Prophet

def get_stock_history(ticker = 'PARA', period='1y'):
    """
    tickter is symbol of stock
    period is historical time frame, such as 'max', '1y', '1m'
    """
    msft = yf.Ticker(ticker)
    hist = msft.history(period=period)
    return hist

def plot_hist_stock(ticker = 'PARA', period='1y', price_type='close'):
    plot_df = get_stock_history()
    fig = go.Figure()
    if price_type=='close':
        fig.add_trace(
            go.Scatter(x =plot_df.index, y=plot_df['Close'])
        )
    if price_type=='open':
        fig.add_trace(
            go.Scatter(x =plot_df.index, y=plot_df['Open'])
            )
    fig.update_layout(
        title={
            'text' : f'Daily {price_type.capitalize()} Price For {ticker}, Last {period.upper()}',
            'x':0.5,
            'xanchor': 'center'
        },
        xaxis_title='Date',
        yaxis_title='$')
    return fig

def make_prophet_data(ticker, period):
    df = get_stock_history(ticker= ticker, period=period)
    df['Date'] = df.index
    df = df[['Date', 'Close']]
    df.columns = ['ds', 'y']
    return df


def predict_stock(ticker= 'PARA', period='1y', horizon=90):
    df = make_prophet_data(ticker=ticker, period=period)

    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=90)
    pred = model.predict(future)[['ds', 'yhat_lower', 'yhat_upper', 'yhat']]
    pred_ = pd.merge(pred, df, on='ds', how='left')
    pred_ = pred_.dropna()
    coverage_rate = np.where((pred_.y >= pred_.yhat_lower) & (pred_.y <=pred_.yhat_upper),1,0).mean()
    pred['coverage_rate'] = coverage_rate

    return pred

def plot_stock_forecast(ticker= 'PARA', period='1y', horizon=90):

    df = make_prophet_data(ticker=ticker, period=period)
    pred = predict_stock(ticker= ticker, period=period, horizon=horizon)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x =df['ds'], y=df['y'], name ='actuals'
        )
    )

    fig.add_trace(
        go.Scatter(
            x =pred['ds'], 
            y=pred['yhat'], 
            name='yhat_mean',
            line=dict(
                color='#16A085', 
                width=2)
    ))

    fig.add_trace(
        go.Scatter(
            x =pred['ds'], 
            y=pred['yhat_upper'], 
            name='yhat_hi',
            line=dict(
                color='#16A085', 
                width=1,
                dash='dash')
        )
    )

    fig.add_trace(
        go.Scatter(
            x =pred['ds'], 
            y=pred['yhat_lower'], 
            name='yhat_lo',
            line=dict(
                color='#16A085', 
                width=1,
                dash='dash')
        )
    )

    fig.update_layout(
        title={
            'text' : f'{horizon}-Day Close Price Forecast For {ticker}, Based On Last {period.upper()}',
            'x':0.5,
            'xanchor': 'center'
        },
        xaxis_title='Date',
        yaxis_title='$')
    return fig

def sp500_expectation(i, ticker_list, company_name_list, sector_list, horizon = 90, period='1y'):
    ticker = ticker_list[i]
    company_name = company_name_list[i]
    sector = sector_list[i]
    try:
        hist = get_stock_history(ticker=ticker, period=period)
        current_price = hist['Close'][-1]
        pred = predict_stock(ticker=ticker, horizon=horizon, period=period)
        expected_price = pred['yhat'][len(pred)-1]
        roi = expected_price/current_price-1
        coverage_rate = pred['coverage_rate'][0]

    except:
        current_price = None
        expected_price = None
        roi = None
        coverage_rate = None
    output = {
        'Company': company_name,
        'Symbol': ticker,
        'Industry':sector,
        'Current_Price': current_price,
        'Expected_Price': expected_price,
        'Horizon': horizon,
        'ROI': roi,
        'Coverage': coverage_rate
    }
    return output

def get_projection(file_path='https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv', 
    horizon = 90, n_jobs=-2, period='1y'):
    # get list of stock
    stock_table = pd.read_csv(file_path)
    try:
        stock_table[stock_table.Symbol=='VIAC'].Symbol == 'PARA'
        stock_table[stock_table.Symbol=='VIAC'].Name == 'Paramount Group'
    except:
        pass
    # get iterator lists
    ticker_list = stock_table['Symbol']
    company_name_list = stock_table['Name']
    sector_list = stock_table['Sector']

    # parallel process
    r = Parallel(
        n_jobs=n_jobs, 
        verbose=100)(
            delayed(sp500_expectation)
            (
                i=i, 
                ticker_list=ticker_list,
                company_name_list=company_name_list,
                sector_list=sector_list,
                horizon=horizon,
                period=period
                ) 
            for i in range(len(stock_table))
            )
    # make df
    projection = pd.DataFrame(r)
    # add industry rank
    projection['ind_rank'] = projection.groupby('Industry').ROI.rank(ascending=False)
    return projection