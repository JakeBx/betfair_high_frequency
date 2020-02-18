import betfairlightweight
from collections import defaultdict
import json
import os
import pandas as pd
import datetime
from datetime import datetime as dt
from utils import dollars_to_ticks, plot, plot_trades, tick_delta, int_dict
from strategy import h2oModel
import flumine
import numpy as np


class Agent:
    """
    Agent is loaded to enable active trading according to ML model
    """
    def __init__(self, market_id, trading, recorder, model_path, bet_size=0.03, transform_dict={}, save_dir='./data', verbose=True):
        self.model = h2oModel(model_path=model_path)
        self.market_id = market_id
        self.trading = trading
        self.recorder = recorder
        self.dir = save_dir
        self.bet_size = bet_size
        self.orders = []
        self.back_ladder = defaultdict(dict)
        self.trd_ladder = defaultdict(int_dict)
        self.lay_ladder = defaultdict(dict)
        self.batl_history = []
        self.batb_history = []
        self.flumine_object = None
        self.times = []
        self.times = []
        self.sids = []
        self.tot_vols = []
        self.in_play = []
        self.mean_trd = []
        self.market_defs = []
        self.market_defs_ids = []
        self.trade_direction = defaultdict(list)
        self.trade_price = None
        self.in_market = False
        self.current_direction = None
        self.sid = None
        self.strat_ref = str(market_id) + '_bot'
        self.reverse_dict = {'LAY': 'BACK', 'BACK': 'LAY'}
        self.tick_target = 4
        self.stop_level = 5
        self.open_tps = set()
        self.observed = set()
        self.verbose = verbose
        self.cancelled_tps = set()
        self.batl_v = {}
        self.batb_v = {}
        self.upper_thresh = 0.57  # Never trade up because greater than 1
        self.lower_thresh = 0.0
        for n in range(6):
            self.batl_v[n] = []
            self.batb_v[n] = []
        self.transforms = transform_dict

    def consume_single(self, entry: dict):
        """adds a single race change message to agent state"""
        ts = entry['pt']
        change_sids = []
        for rc in entry['mc']:  # Cycle through race change messages in market entry messages
            if 'marketDefinition' in rc.keys():
                self.market_defs.append(rc['marketDefinition'])
                self.market_defs_ids.append(rc['id'])
            if 'rc' in rc.keys():
                for change in rc['rc']:
                    sid = change['id']
                    if sid == self.sid or self.sid is None:
                        change_sids.append(sid)
                        keys = change.keys()
                        if 'atb' in keys:
                            for prc, vol in change['atb']:
                                self.back_ladder[sid][prc] = vol
                        if 'atl' in keys:
                            for prc, vol in change['atl']:
                                self.lay_ladder[sid][prc] = vol
                        if 'trd' in keys:
                            for prc, vol in change['trd']:
                                self.trd_ladder[sid][prc] += vol

        for sid in change_sids:  # Cycle through the selection ids and build the market book
            cur_batl = min([1000] + [k for k, v in self.lay_ladder[sid].items() if v > 1])
            cur_batb = max([0] + [k for k, v in self.back_ladder[sid].items() if v > 1])
            if cur_batl < 10 and cur_batb > 0:
                tot_vol = 0
                agg_vol = 0
                for p, v in self.trd_ladder[sid].items():
                    agg_vol += float(p) * v
                    tot_vol += v
                if tot_vol > 500:
                    mean_price = agg_vol / tot_vol if tot_vol > 1 else 0
                    self.mean_trd.append(mean_price)
                    self.times.append(ts)
                    self.sids.append(sid)
                    self.batl_history.append(cur_batl)
                    self.batb_history.append(cur_batb)
                    for n in range(6):
                        adj_batl = tick_delta(cur_batl, n)
                        adj_batb = tick_delta(cur_batb, -n)
                        if adj_batl in self.lay_ladder[sid].keys():
                            self.batl_v[n].append(self.lay_ladder[sid][adj_batl])
                        else:
                            self.batl_v[n].append(0)
                        if adj_batb in self.back_ladder[sid].keys():
                            self.batb_v[n].append(self.back_ladder[sid][adj_batb])
                        else:
                            self.batb_v[n].append(0)
                    self.tot_vols.append(tot_vol)

    def init_stream(self, path: str = './secrets/config.json'):
        """targets stream to disk"""
        with open(path, 'r') as fp:
            config = json.load(fp)

        self.flumine_object = flumine.flumine.Flumine(
            recorder=self.recorder,
            settings={'certificate_login': True, 'betfairlightweight': config}
        )

        self.flumine_object.start()

    def poll_directory(self):
        """continues to update agent state as a background process until stopped"""
        with open(os.path.join(self.dir, self.recorder.stream_id, self.market_id)) as f:
            for line in f:
                if line not in self.observed:
                    self.observed.add(line)
                    entry = json.loads(line)
                    self.consume_single(entry)

        if self.sid is None:  # TODO: improve this, there is a better way to find favourite before you poll the directory
            df = self.market_df()
            if df.shape[0]:
                self.sid = int(df[['sid', 'batl']].groupby('sid').last().sort_values('batl').index.values[0])

    def market_df(self):
        """Returns the df of the market data collected"""
        df = pd.DataFrame({'timestamp': self.times,
                             'sid': self.sids,
                             'batl': self.batl_history,
                             'batb': self.batb_history,
                             'batl_v': self.batl_v[0],
                             'batb_v': self.batb_v[0],
                             'batl_v_1': self.batl_v[1],
                             'batb_v_1': self.batb_v[1],
                             'batl_v_2': self.batl_v[2],
                             'batb_v_2': self.batb_v[2],
                             'batl_v_3': self.batl_v[3],
                             'batb_v_3': self.batb_v[3],
                             'batl_v_4': self.batl_v[4],
                             'batb_v_4': self.batb_v[4],
                             'batl_v_5': self.batl_v[5],
                             'batb_v_5': self.batb_v[5],
                             'tot_vol': self.tot_vols,
                             'mean_trd': self.mean_trd,
                             })

        track = self.market_defs[-1]['venue'] # TODO: Change this to dict mapping (or to venue in the model)
        if track in self.transforms.keys():
            track = self.transforms[track]
        df['track'] = track[:4]
        df.track = df.track.replace(self.transforms)
        start = np.datetime64(self.market_defs[-1]['marketTime'])
        df['starttime'] = start
        df = df.groupby(['timestamp', 'sid']).last().reset_index()

        return df

    def assess_bsp(self, verbose=False):
        """Calls the strategy and provides a trade decision under the betfair strike price model"""
        df = self.market_df()
        df = df[df.sid == self.sid]
        decision = self.model.get_predictions(df, bsp=True)
        if verbose:
            print(decision)
        if decision is None:
            return
        elif decision > self.upper_thresh:
            self.trade_price = df.batl.values[-1]
            self.trade_direction[self.sid].append('DOWN')
            self.current_direction = 'DOWN'
        elif decision < self.lower_thresh:
            self.trade_price = df.batb.values[-1]
            self.trade_direction[self.sid].append('UP')
            self.current_direction = 'UP'
        else:
            self.trade_price = None
            self.current_direction = None

    def _trade(self, price: float, side: str, strategy: str, size=0.03, on_close=False, persistence='LAPSE'):
        """Returns instructions dictionary to execute trading on"""
        if on_close:
            order_filter = betfairlightweight.filters.market_on_close_order(size)

            # Define an instructions filter
            instructions_filter = betfairlightweight.filters.place_instruction(
                selection_id=self.sid,
                order_type='MARKET_ON_CLOSE',
                side=side,
                market_on_close_order=order_filter
            )
        # Define a limit order filter
        else:
            order_filter = betfairlightweight.filters.limit_order(
                size=size,
                price=price,
                persistence_type=persistence
            )

            # Define an instructions filter
            instructions_filter = betfairlightweight.filters.place_instruction(
                selection_id=self.sid,
                order_type='LIMIT',
                side=side,
                limit_order=order_filter
            )

        # Place the order
        try:
            self.orders.append(self.trading.betting.place_orders(
                market_id=self.market_id,  # The market id we obtained from before
                customer_strategy_ref=strategy,
                instructions=[instructions_filter]  # This must be a list
            ))
            if self.verbose:
                print(strategy)
                print(instructions_filter)
                print(self.orders[-1].status, self.orders[-1]._data['instructionReports'][0]['betId'])
        except Exception as e:
            print('EXCEPTION: ', e)
            print(instructions_filter)

    def manage_trade(self):
        """monitors and adjusts trades according to what is in the market"""
        if self.in_market:
            self._stop_orders()
        elif self.current_direction is not None:
            if self.current_direction == 'UP':
                self._trade(self.trade_price, "LAY", self.strat_ref)
            elif self.current_direction == 'DOWN':
                self._trade(self.trade_price, "BACK", self.strat_ref)
            if self.orders[-1].status == 'SUCCESS':
                self.in_market = True

    def _stop_orders(self):
        """Coordinates the take profits and stops of orders in the market"""
        current_orders = self.trading.betting.list_current_orders(market_ids=[self.market_id])
        status = set([x.status for x in current_orders.orders])
        if 'EXECUTABLE' not in status:
            # if self.current_direction == 'DOWN':
            #     self.upper_thresh += 0.03
            # else:
            #     self.lower_thresh -= 0.03
            self.in_market = False
            self.current_direction = None


        open_orders = [x for x in current_orders.orders if x.customer_strategy_ref == self.strat_ref]
        current_time = datetime.datetime.utcnow()
        for order in open_orders:
            if order.status == 'EXECUTABLE':
                if order.placed_date + datetime.timedelta(
                        seconds=30) < current_time:  # Cancel order if it does not fill
                    self.poll_directory()
                    self.assess_bsp()
                    if self.current_direction == None: # TODO: account for change in direction
                        print('cancel order')
                        self.trading.betting.cancel_orders(market_id=self.market_id)
                        self.in_market = False
                        self.current_direction = None





    def cashout(self):
        """Cashout all orders"""
        self.trading.betting.cancel_orders(market_id=self.market_id)
        current_orders = self.trading.betting.list_current_orders(market_ids=[self.market_id])
        net_pos = 0
        liab = 0
        for order in current_orders.orders:
            if order.side == 'BACK':
                net_pos += float(order.size_matched)
                liab += order.size_matched * (order.average_price_matched - 1)
            elif order.side == 'LAY':
                net_pos -= float(order.size_matched)
                liab -= order.size_matched * (order.average_price_matched - 1)

        liab = round(liab, 2)
        print(liab)
        if net_pos < 0:
            self._trade(None, 'BACK', 'cashout', size=abs(net_pos), persistence='MARKET_ON_CLOSE', on_close=True)
        if net_pos > 0:
            self._trade(None, 'LAY', 'cashout', size=abs(liab), persistence='MARKET_ON_CLOSE', on_close=True)



