import plotly.express as px
import pandas as pd
from collections import defaultdict


def int_dict():
    return defaultdict(int)


def double_d():
    return defaultdict(dict)

def plot(title: str, batb_history: list, batl_history: list, time: list = None):
    """Plots the price action of the agent"""
    assert len(batb_history) == len(batl_history)
    if time is None:
        time = list(range(len(batb_history)))
    df1 = pd.DataFrame({'price': batb_history, 'time': time})
    df1['type'] = 'batb'
    df2 = pd.DataFrame({'price': batl_history, 'time': time})
    df2['type'] = 'batl'
    plot_df = pd.concat([df1, df2])

    fig = px.line(plot_df, x='time', y='price', color='type', title=title)

    return fig


def plot_trades(price_plot, trade_times, trade_price, trade_type):
    """Places trade positions onto the market plot"""
    plot_df = pd.DataFrame({'price': trade_price, 'time': trade_times, 'type': trade_type})
    scatter = px.scatter(plot_df, x='time', y='price', color='type')
    for trace in scatter.data:
        price_plot.add_trace(trace)

    return price_plot


def dollars_to_ticks(price: float) -> int:
    """Converts a price in dollars to the level of betfair ticks"""
    if price < 2.:
        ticks = (price - 1) / 0.01
    elif price < 3:
        ticks = 100 + price % 2 / 0.02
    elif price < 4:
        ticks = 150 + price % 3 / 0.05
    elif price < 6:
        ticks = 170 + price % 4 / 0.1
    elif price < 10:
        ticks = 190 + price % 6 / 0.2
    elif price < 20:
        ticks = 210 + price % 10 / 0.5
    elif price < 30:
        ticks = 230 + price % 20 / 1
    elif price < 50:
        ticks = 240 + price % 30 / 2
    elif price < 100:
        ticks = 250 + price % 30 / 5
    else:
        return -1

    return int(round(ticks))


def ticks_to_dollars(ticks: int) -> float:
    """Converts a betfair ticks to price in dollars"""
    if ticks < 100:
        price = 1 + ticks * 0.01
    elif ticks < 150:
        price = 2 + ticks % 100 * 0.02
    elif ticks < 170:
        price = 3 + ticks % 150 * 0.05
    elif ticks < 190:
        price = 4 + ticks % 170 * 0.1
    elif ticks < 210:
        price = 6 + ticks % 190 * 0.2
    elif ticks < 230:
        price = 10 + ticks % 210 * 0.5
    elif ticks < 240:
        price = 20 + ticks % 230 * 1.
    elif ticks < 250:
        price = 30 + ticks % 230 * 2.
    else:
        return -1

    return round(price, 2)


def tick_delta(price, delta):
    """Computes the price given a price and a tick difference"""
    ticks = dollars_to_ticks(price)
    ticks += delta
    ret_price = ticks_to_dollars(ticks)
    return round(ret_price, 2)
