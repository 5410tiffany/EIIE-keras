from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from pgportfolio.marketdata.coinlist import CoinList
import numpy as np
import pandas as pd
from pgportfolio.tools.data import panel_fillna
from pgportfolio.tools.configprocess import parse_time
from pgportfolio.constants import *
import sqlite3
from datetime import datetime
import logging


class HistoryManager:
    # if offline ,the coin_list could be None
    # NOTE: return of the sqlite results is a list of tuples, each tuple is a row
    def __init__(self, coin_number, end, volume_average_days=1, volume_forward=0, online=True):
        self.initialize_db()
        self.__storage_period = FIVE_MINUTES  # keep this as 300
        self._coin_number = coin_number
        self._online = online
        if self._online:
            self._coin_list = CoinList(end, volume_average_days, volume_forward)
        self.__volume_forward = volume_forward
        self.__volume_average_days = volume_average_days
        self.__coins = None

    @property
    def coins(self):
        return self.__coins

    def initialize_db(self):
        print(DATABASE_DIR)
        with sqlite3.connect(DATABASE_DIR) as connection:
            cursor = connection.cursor()
            cursor.execute('CREATE TABLE IF NOT EXISTS History (date INTEGER,'
                           ' coin varchar(20), high FLOAT, low FLOAT,'
                           ' open FLOAT, close FLOAT, volume FLOAT, '
                           ' quoteVolume FLOAT, weightedAverage FLOAT,'
                           'PRIMARY KEY (date, coin));')
            connection.commit()

    def get_global_data_matrix(self, start, end, period=300, features=('close',)):
        """
        :return a numpy ndarray whose axis is [feature, coin, time]
        """
        return self.get_global_panel(start, end, period, features).values


    def get_futures_panel(period=300, features=('close',)):
        futures = pd.read_json('Data_json/FWMTF_json/Futures.json').sort_index()
        features = ('open', 'high', 'low', 'close', 'min_till_expire')
        x = [str(i) for i in futures['date']]
        y = [i for i in futures['time_period']]
        time_index = [i + ' ' + j for i, j in zip(x, y)]
        for i in futures.columns:
            if i not in features:
                futures = futures.drop(i, axis=1)
        futures = futures.transpose()
        futures.columns = time_index
        for feature in features:
            futures[feature] = [i for i in futures[feature]]
            for i in range(len(time_index)):
                if feature != 'min_till_expire':
                    x = futures[feature]
                    futures[feature] = [np.clip(i, 10e-8, 10e8) for i in x]
                else:
                    x = futures[feature]
                    futures[feature] = [i / x.max() for i in futures[feature]]
        return futures

    def get_global_panel(self, start, end, period=300, features=('close',)):

        """
        :param start/end: linux timestamp in seconds
        :param period: time interval of each data access point
        :param features: tuple or list of the feature names
        :return a panel, [feature, coin, time]
        """
        call_8000 = pd.read_json('Data_json/OWMTF_json/CallOption_8000.json').sort_index()
        call_8100 = pd.read_json('Data_json/OWMTF_json/CallOption_8100.json').sort_index()
        call_8200 = pd.read_json('Data_json/OWMTF_json/CallOption_8200.json').sort_index()
        call_8300 = pd.read_json('Data_json/OWMTF_json/CallOption_8300.json').sort_index()
        call_8400 = pd.read_json('Data_json/OWMTF_json/CallOption_8400.json').sort_index()
        call_list = [call_8000, call_8100, call_8200, call_8300, call_8400]

        features = ('open', 'high', 'low', 'close', 'min_till_expire')
        coins = ('8000', '8100', '8200', '8300', '8400')

        x = [str(i) for i in call_8000['date']]
        y = [i for i in call_8000['time_period']]
        time_index = [i + ' ' + j for i, j in zip(x, y)]

        panel = pd.Panel(items=features, major_axis = coins, minor_axis = time_index)
        for feature in features:
            for calljson, strike_price in zip(call_list, coins):
                panel[feature].loc[strike_price] = [i for i in calljson[feature]]
                for i in range(len(time_index)):
                    if np.isnan(panel[feature].loc[strike_price][i]):
                        panel[feature].loc[strike_price][i] = panel[feature].loc[strike_price][i - 1]
                if feature != 'min_till_expire':
                    x = panel[feature].loc[strike_price]
                    panel[feature].loc[strike_price] = [np.clip(i, 10e-8, 10e8) for i in x]
                else:
                    x = panel[feature].loc[strike_price]
                    panel[feature].loc[strike_price] = [i / x.max() for i in panel[feature].loc[strike_price]]

        return panel

    # select top coin_number of coins by volume from start to end
    def select_coins(self, start, end):
        if not self._online:
            logging.info(
                "select coins offline from %s to %s" % (datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M'),
                                                        datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M')))
            connection = sqlite3.connect(DATABASE_DIR)
            try:
                cursor = connection.cursor()
                cursor.execute('SELECT coin,SUM(volume) AS total_volume FROM History WHERE'
                               ' date>=? and date<=? GROUP BY coin'
                               ' ORDER BY total_volume DESC LIMIT ?;',
                               (int(start), int(end), self._coin_number))
                coins_tuples = cursor.fetchall()
                if len(coins_tuples) != self._coin_number:
                    logging.error("the sqlite error happend")
            finally:
                connection.commit()
                connection.close()
            coins = []
            for tuple in coins_tuples:
                coins.append(tuple[0])
        else:
            coins = list(self._coin_list.topNVolume(n=self._coin_number).index)
        logging.debug("Selected coins are: " + str(coins))
        return coins

    def __checkperiod(self, period):
        if period == FIVE_MINUTES:
            return
        elif period == FIFTEEN_MINUTES:
            return
        elif period == HALF_HOUR:
            return
        elif period == TWO_HOUR:
            return
        elif period == FOUR_HOUR:
            return
        elif period == DAY:
            return
        else:
            raise ValueError('peroid has to be 5min, 15min, 30min, 2hr, 4hr, or a day')

    # add new history data into the database
    def update_data(self, start, end, coin):
        connection = sqlite3.connect(DATABASE_DIR)
        try:
            cursor = connection.cursor()
            min_date = cursor.execute('SELECT MIN(date) FROM History WHERE coin=?;', (coin,)).fetchall()[0][0]
            max_date = cursor.execute('SELECT MAX(date) FROM History WHERE coin=?;', (coin,)).fetchall()[0][0]
            if min_date == None or max_date == None:
                self.__fill_data(start, end, coin, cursor)
            else:
                if max_date + 10 * self.__storage_period < end:
                    if not self._online:
                        raise Exception("Have to be online")
                    self.__fill_data(max_date + self.__storage_period, end, coin, cursor)
                if min_date > start and self._online:
                    self.__fill_data(start, min_date - self.__storage_period - 1, coin, cursor)
            # if there is no data
        finally:
            connection.commit()
            connection.close()

    def __fill_data(self, start, end, coin, cursor):
        duration = 7819200  # three months
        bk_start = start
        for bk_end in range(start + duration - 1, end, duration):
            self.__fill_part_data(bk_start, bk_end, coin, cursor)
            bk_start += duration
        if bk_start < end:
            self.__fill_part_data(bk_start, end, coin, cursor)

    def __fill_part_data(self, start, end, coin, cursor):
        chart = self._coin_list.get_chart_until_success(
            pair=self._coin_list.allActiveCoins.at[coin, 'pair'],
            start=start,
            end=end,
            period=self.__storage_period)
        logging.info("fill %s data from %s to %s" % (coin, datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M'),
                                                     datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M')))
        for c in chart:
            if c["date"] > 0:
                if c['weightedAverage'] == 0:
                    weightedAverage = c['close']
                else:
                    weightedAverage = c['weightedAverage']
                # NOTE here the USDT is in reversed order
                if 'reversed_' in coin:
                    cursor.execute('INSERT INTO History VALUES (?,?,?,?,?,?,?,?,?)',
                                   (c['date'], coin, 1.0 / c['low'], 1.0 / c['high'], 1.0 / c['open'],
                                    1.0 / c['close'], c['quoteVolume'], c['volume'],
                                    1.0 / weightedAverage))
                else:
                    cursor.execute('INSERT INTO History VALUES (?,?,?,?,?,?,?,?,?)',
                                   (c['date'], coin, c['high'], c['low'], c['open'],
                                    c['close'], c['volume'], c['quoteVolume'],
                                    weightedAverage))