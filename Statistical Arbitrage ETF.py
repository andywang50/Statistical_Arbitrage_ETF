"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
import quantopian.algorithm as algo
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.classifiers.morningstar import Sector
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.filters.morningstar import Q500US, Q1500US
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import statsmodels as sm
from quantopian.pipeline.data import Fundamentals  

def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    # Rebalance every day, 1 hour after market open.
    set_slippage(slippage.FixedSlippage(spread=0.00))
    set_commission(commission.PerShare(cost=0.0, min_trade_cost=0.0))
    context.lookback = 60
    context.leverage = 0.02
    context.day = 1
    #context.ETFs = []
    context.market = [symbol('SPY')]
    context.bo = 1.25
    context.so = 1.25
    context.bc = 0.75
    context.sc = 0.5
    context.stocks = []
    context.initialized = False
    context.holding_book_shares = None
    context.order_hist = {}

    context.xlb = symbol('XLB') #sid(19654) #Materials 101    
    context.xly = symbol('XLY') #sid(19662) #Consumer Discretionary 102
    context.xlf = symbol('XLF') #sid(19656) #Financials 103    
    context.xlre = symbol('IYR') #sid() #Real estate 104
    context.xlp = symbol('XLP') #sid(19659) #Consumer Staples 205
    context.xlv = symbol('XLV') #sid(19661) #Health Care 206
    context.xlu = symbol('XLU') #sid(19660) #Utilities   207
    context.xtl = symbol('IYZ') #sid() #Communication Services 308
    context.xle = symbol('XLE') #sid(19655) #Energy 309
    context.xli = symbol('XLI') #sid(19657) #Industrials 310
    context.xlk = symbol('XLK') #sid(19658) #Technology  311      
    
    context.ETF_lookup = {context.xlb:101, 101:context.xlb,
                         context.xly:102, 102:context.xly,
                         context.xlf:103, 103:context.xlf,
                         context.xlre:104, 104:context.xlre,
                         context.xlp:205, 205: context.xlp,
                         context.xlv:206, 206: context.xlv,
                         context.xlu:207, 207:context.xlu,
                         context.xtl:308, 308:context.xtl,
                         context.xle:309, 309:context.xle,
                         context.xli:310, 310:context.xli,
                         context.xlk:311, 311:context.xlk}
    
    context.ETFs = [context.xlb,
                      context.xly,
                      context.xlf,
                      context.xlre,
                      context.xlp,
                      context.xlv,
                      context.xlu,
                      context.xtl,
                      context.xle,
                      context.xli,
                      context.xlk
                     ]
    
    # context.sector_id = [102,
    #                   205,
    #                   309,
    #                   103,
    #                   #306,
    #                   310,
    #                   101,
    #                   #104,
    #                   311,
    #                   207
    #                  ]

    schedule_function(adjust_for_split,date_rule=date_rules.every_day(),time_rule=time_rules.market_open(minutes=1))
    schedule_function(trade,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_open(minutes=1))
    schedule_function(write_holding_book,
                      date_rule = date_rules.every_day(),
                      time_rule=time_rules.market_open(minutes=100))

    schedule_function(print_holding,date_rule=date_rules.every_day(),time_rule=time_rules.market_open(minutes=120))
       
    # Create our dynamic stock selector.
    algo.attach_pipeline(make_pipeline(), 'pipeline')
    
    
def adjust_for_split(context,data):
    ## Adjust for split
    for sym1 in context.holding_book_shares.index:
        row = context.holding_book_shares.loc[sym1,:]
        #print(row.shape)
        rowsum = row.sum()
        if (rowsum != 0):
            context.holding_book_shares.loc[sym1,:] *= (context.portfolio.positions[sym1]['amount']/rowsum)
            #print(sym1,context.portfolio.positions[sym1]['amount'])

def print_holding(context,data):
    print(context.holding_book_shares)


def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation
    on pipeline can be found here:
    https://www.quantopian.com/help#pipeline-title
    """

    # Base universe set to the QTradableStocksUS
    base_universe = QTradableStocksUS()#Q500US()
    base_universe = (base_universe & Q500US())
    base_universe = (base_universe & Fundamentals.market_cap.latest.top(150))

    # Factor of yesterday's close price.
    #yesterday_close = USEquityPricing.close.latest

    pipe = Pipeline(
        columns={
            #'close': yesterday_close,
            'sector': Sector(),
        },
        screen=base_universe
    )
    return pipe

def handle_data(context, data):
    if not context.initialized:
        context.initialized=True
        pipeline_data = pipeline_output('pipeline')
        #print(pipeline_data)
        context.stocks = list(pipeline_data.index)
        #context.stocks = [symbol('AAPL'),symbol('GE'),symbol('GOOG'),symbol('XOM'),symbol('C'),symbol('JPM'),symbol('IBM'),symbol('FB'),symbol('BABA'),symbol('BIDU'),symbol('TSLA'),symbol('BAC'),symbol('MS'),symbol('MSFT'),symbol('GG'),symbol('TOT'),symbol('AMZN')]#,symbol('TGT'),symbol('GPC'),symbol('HAS'),symbol('ANS'),symbol('IRS'),symbol('ED'),symbol('DUK_PRAC'),symbol('PPE'),symbol('ATT_CL'),symbol('VZA')]        
        context.holding_book_shares = pd.DataFrame(0,columns=context.stocks+['self'],index= context.stocks+context.market+context.ETFs)
        #print(context.holding_book_shares)

    record(leverage=context.account.leverage,
           exposure=context.account.net_leverage)
        
def write_holding_book(context,data):

    for sym1 in context.order_hist:
        for sym2 in context.order_hist[sym1]:
            temp = get_order(context.order_hist[sym1][sym2])
            if temp:
                context.holding_book_shares.loc[sym1,sym2] += temp.filled
            #print(sym1,sym2)
            #print(context.holding_book_shares)
            #print()
    context.order_hist = {} 

    
def trade(context, data):
    pipeline_data = pipeline_output('pipeline')
    sectors = pipeline_data.loc[context.stocks,:]
    #print(pipeline_data)
    prices = data.history(context.stocks,bar_count = context.lookback+2, frequency = '1d', fields = 'price').dropna(axis=1).iloc[0:context.lookback+1,:] ##not including current price (today's open price)
    prices.fillna(method='ffill')
    prices.fillna(method='bfill')
    market_prices = data.history(context.market,bar_count = context.lookback+2,frequency = '1d', fields='price').dropna(axis=1).iloc[0:context.lookback+1,:]##not including current price (today's open price)
    ETF_prices = data.history(context.ETFs,bar_count = context.lookback+2,frequency='1d',fields='price').iloc[0:context.lookback+1,:]
    returns = prices.pct_change().dropna(axis=0)
    market_returns = market_prices.pct_change().dropna(axis=0)
    ETF_returns = ETF_prices.pct_change().dropna(axis=0)
    #E = context.portfolio.portfolio_value
    s,qualified,betas = get_s_score(returns,ETF_returns,market_returns,sectors,context.ETF_lookup,modified_m=True,drift_adjusted=True)
    print(betas)
    for sym in s.index:#context.stocks:
        sj = s.loc[sym]
        beta_j = betas.loc[:,sym]
        temp = sectors.loc[sym].values[0]
        if np.isnan(temp):
            continue
        sector_sym = context.ETF_lookup[temp]   
        #print(sj)
        current_holding_sym = context.holding_book_shares.loc[sym,'self']
        if current_holding_sym == 0:
            if (qualified.loc[sym]) and (sj < -context.bo):   
                orderid_self = order_percent(sym,context.leverage)
                if not sym in context.order_hist:
                    context.order_hist[sym] = {}
                context.order_hist[sym]['self'] = orderid_self
                if beta_j.loc['market'] != 0:
                    orderid_hedge_market = order_percent(context.market[0],(-1.0)*context.leverage*beta_j.loc['market'])       
                if not context.market[0] in context.order_hist:
                        context.order_hist[context.market[0]] = {}
                context.order_hist[context.market[0]][sym] = orderid_hedge_market
                if beta_j.loc['ETF'] != 0:
                    orderid_hedge_factor = order_percent(sector_sym,(-1.0)*context.leverage*beta_j.loc['ETF'])       
                if not sector_sym in context.order_hist:
                        context.order_hist[sector_sym] = {}
                context.order_hist[sector_sym][sym] = orderid_hedge_factor

                context.order_hist[context.market[0]][sym] = orderid_hedge_factor
            elif (qualified.loc[sym]) and (sj > context.so):                                                       
                orderid_self = order_percent(sym,(-1)*context.leverage)
                if not sym in context.order_hist:
                    context.order_hist[sym] = {}
                context.order_hist[sym]['self'] = orderid_self
                if beta_j.loc['market'] != 0:
                    orderid_hedge_market = order_percent(context.market[0],(1.0)*context.leverage*beta_j.loc['market'])       
                if not context.market[0] in context.order_hist:
                        context.order_hist[context.market[0]] = {}
                context.order_hist[context.market[0]][sym] = orderid_hedge_market
                if beta_j.loc['ETF'] != 0:
                    orderid_hedge_factor = order_percent(sector_sym,(1.0)*context.leverage*beta_j.loc['ETF'])       
                if not sector_sym in context.order_hist:
                        context.order_hist[sector_sym] = {}
                context.order_hist[sector_sym][sym] = orderid_hedge_factor                
        else:
            if (((current_holding_sym> 0) and sj>-context.sc) or ((current_holding_sym< 0) and sj<context.bc)):
                #order_target_value(sym,0)
                #order_target_value(context.market[0],0)
                orderid_self = order(sym,context.holding_book_shares.loc[sym,'self']*(-1))
                if not sym in context.order_hist:
                    context.order_hist[sym] = {}
                context.order_hist[sym]['self'] = orderid_self  
                
                orderid_hedge_market = order(context.market[0],context.holding_book_shares.loc[context.market[0],sym]*(-1))       
                if not context.market[0] in context.order_hist:
                        context.order_hist[context.market[0]] = {}
                context.order_hist[context.market[0]][sym] = orderid_hedge_market
                orderid_hedge_factor = order(sector_sym,context.holding_book_shares.loc[sector_sym,sym]*(-1))       
                if not sector_sym in context.order_hist:
                        context.order_hist[sector_sym] = {}
                context.order_hist[sector_sym][sym] = orderid_hedge_factor   

    #print(context.order_hist)                               
    
def get_s_score(returns,ETF_returns,market_returns,sectors,ETF_lookup,modified_m=False,drift_adjusted=False):
    #print(sectors)
    assert market_returns.shape[0] == ETF_returns.shape[0]
    #F1 = pd.concat([market_returns,ETF_returns],axis=1)
    #assert F1.shape[0] == ETF_returns.shape[0]    
    #F = sm.tools.tools.add_constant(F1)
    #betas = np.linalg.solve(np.dot(F.T,F),np.dot(F.T,returns))
    alphas = pd.DataFrame(0,index = ['drift'],columns=returns.columns)
    betas = pd.DataFrame(0,index = ['market','ETF'],columns=returns.columns)
    OU_fits = pd.DataFrame(index=['a','b','var_xi','k','m','sig','sig_eq'],columns = returns.columns)
    for sym in returns.columns:
        #print(sectors.loc[sym].values[0])
        temp = sectors.loc[sym].values[0]
        if np.isnan(temp):
            continue
        sector_sym = ETF_lookup[temp]
        F = pd.concat([market_returns,ETF_returns.loc[:,sector_sym]],axis=1)
        F = sm.tools.tools.add_constant(F)
        beta_j = np.linalg.solve(np.dot(F.T,F),np.dot(F.T,returns.loc[:,sym]))
        alphas.loc[:,sym] = beta_j[0]
        betas.loc[:,sym] = beta_j[1:]        
        resid = returns.loc[:,sym]- np.dot(F,beta_j)
        Xj = resid.cumsum(axis=0)
        Xj_lag = Xj.iloc[0:Xj.shape[0]-1]
        Xj_lag = sm.tools.tools.add_constant(Xj_lag)
        Xj_Response = Xj.iloc[1:]
        OLS_result = np.linalg.solve(np.dot(Xj_lag.T,Xj_lag),np.dot(Xj_lag.T,Xj_Response))
        OU_fits.loc[['a','b'],sym] = OLS_result
        xi = Xj_Response - np.dot(Xj_lag,OLS_result)
        b = OLS_result[1]
        varxi = np.var(xi)
        k = -252*np.log(b)
        m = 1.0*OLS_result[0]/(1-b)
        sig = np.sqrt(varxi*2*k/(1-b*b))
        sig_eq = np.sqrt(varxi/(1-b*b))
        OU_fits.loc[['var_xi'],sym] = varxi
        OU_fits.loc[['k'],sym] = k
        OU_fits.loc[['m'],sym] = m
        OU_fits.loc[['sig'],sym] = sig
        OU_fits.loc[['sig_eq'],sym] = sig_eq
    if modified_m == True:
        m_bar = OU_fits.loc['m'] - np.mean(OU_fits.loc['m'])
        s_score = (-m_bar)/OU_fits.loc['sig_eq']
    else:
        s_score = (-OU_fits.loc['m'])/OU_fits.loc['sig_eq']
    if drift_adjusted == True:
        s_score -= alphas.loc['drift',:]/(OU_fits.loc['k']*OU_fits.loc['sig_eq'])
    b = OU_fits.loc['b']
    qualified = (b<0.9672) * (b>0)
    return s_score,qualified,betas