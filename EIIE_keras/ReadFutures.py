import pandas as pd
import os
import warnings
from datetime import timedelta, datetime
import datetime as dt

warnings.filterwarnings('ignore')


class FutureData:
    def __init__(self):
        self.total_history = pd.DataFrame()
        self.readfromfile()
        
    def get_FutureData(self):
        return self.total_history
        
    def tojson(self):
        self.total_history.to_json(r'Data_json1/Futures.json',orient='records')
        
    def closed_third_wen(self,transcation_date):
        str_date = str(transcation_date)
        int_y = int(str_date[:4])
        int_m = str_date[4:6]
        if int_m.startswith('0'):
            int_m = int(str_date[5:6])
        else:
            int_m = int(str_date[4:6])
            
        day = 21 - (dt.date(int_y, int_m, 1).weekday() + 4) % 7  # weekday函數 禮拜一為0;禮拜日為6
        str(dt.date(int_y, int_m, day))
        if str(transcation_date) < dt.date(int_y, int_m, day).strftime('%Y%m%d'):
            exp = dt.date(int_y, int_m, 21 - (dt.date(int_y, int_m, 1).weekday() + 4) % 7)
        else:
            exp = dt.date(int_y, int_m + 1, 21 - (dt.date(int_y, int_m + 1, 1).weekday() + 4) % 7)
        return exp
    
    def get_duration(self, start_date, start_time, end_date):
        begin_time = datetime(start_date // 10000, (start_date // 100) % 100, start_date % 100,
                              int(start_time[:2]), int(start_time[-2:]), 0)
        expiration_time = datetime(end_date // 10000, (end_date // 100) % 100, end_date % 100,
                                   13, 45, 0)
        time_difference = expiration_time - begin_time
        return time_difference / timedelta(minutes=1)

    def readfromfile(self):
        folder_path = 'Data/FWMTF1'
        folder_content = os.listdir(folder_path)
        folder_content.sort()
        for item in folder_content:
            if os.path.isdir(folder_path + '/' + item):
                print("資料夾：" + item)
                file_list = os.listdir(folder_path + '/' + item)
                file_list.sort()
                print(file_list)
                month_history = pd.DataFrame()
                for f in file_list:
                    if f.startswith('.'):
                        pass
                    else:
                        print(f)
                        data_MTH_ = pd.DataFrame(pd.read_table(folder_path + '/' + item + "/" + f, header=None))
                        # add column name
                        cols_name = ['日期', '商品代號', '第一支腳商品代碼1', '第一支腳買賣別1', '第一支腳成交價格1', '第一支腳成交數量1', '第二支腳商品代碼2', '第二支腳買賣別2',
                                     '第二支腳成交價格2', '第二支腳成交數量2', '買賣別', '成交價格',
                                     '成交數量', '開平倉碼', '搓合標記', '原始成交時間', '單複式碼', '委託方式', '交易與委託報價檔連結代碼']
                        data_MTH_.columns = cols_name
                        # drop unuseful column
                        data_MTH_ = data_MTH_.drop(['第一支腳商品代碼1', '第一支腳買賣別1', '第一支腳成交價格1', '第一支腳成交數量1', '第二支腳商品代碼2', '第二支腳買賣別2',
                                                    '第二支腳成交價格2', '第二支腳成交數量2', '開平倉碼', '搓合標記', '單複式碼', '委託方式', '交易與委託報價檔連結代碼'],
                                                   axis=1)
                        # filter row
                        data_MTH_ = data_MTH_['TXF' == data_MTH_['商品代號'].str[:3]]
                        data_MTH_ = data_MTH_[data_MTH_['成交價格'] > 0]
                        # sort datetime
                        data_MTH_ = data_MTH_.sort_values(by="原始成交時間")
                        data_MTH_ = data_MTH_.reset_index()
                        # select the [open,high,low,close] in 30 mins
                        history_data = []
                        #   data_MTH_[(data_MTH_['原始成交時間']>'%02d:45'%(i)) & (data_MTH_['原始成交時間']<'%02d:15'%(i+1))]
                        #   data_MTH_['原始成交時間'].loc[0]>'08:45:00'
                        for i in range(8, 14, 1):
                            for j in range(0, 60, 5):
                                if j != 55:
                                    time_start = '%02d:%02d' % (i, j)
                                    time_end = '%02d:%02d' % (i, j + 5)
                                    time_period_data = data_MTH_[
                                        (data_MTH_['原始成交時間'] > time_start) & (data_MTH_['原始成交時間'] < time_end)]
                                    if i == 13 and j == 40:
                                        time_start = '%02d:%02d' % (i, j)
                                        time_end = '%02d:%02d' % (i, j + 5)
                                        time_period_data = data_MTH_[(data_MTH_['原始成交時間'] > time_start)]
                                    if  (i == 13 and j > 40):
                                        continue
                                else:
                                    time_start = '%02d:%02d' % (i, j)
                                    time_end = '%02d:%02d' % (i + 1, 0)
                                    time_period_data = data_MTH_[
                                        (data_MTH_['原始成交時間'] > time_start) & (data_MTH_['原始成交時間'] < time_end)]
                                
                                if  (i==8 and j>=45)  or (i==13 and j<=40) or (i>8 and i<13):
                                
                                    date = data_MTH_['日期'][time_period_data.index[0]] if (time_period_data.size !=0) else None
                                    open_price = data_MTH_['成交價格'][time_period_data.index[0]] if (time_period_data.size !=0) else None
                                    high_price = time_period_data['成交價格'].max() if (time_period_data.size !=0) else None
                                    low_price = time_period_data['成交價格'].min()if (time_period_data.size !=0) else None
                                    closed_price = data_MTH_['成交價格'][time_period_data.index[-1]] if (time_period_data.size !=0) else None
                                    volume = time_period_data['成交數量'].sum() if (time_period_data.size !=0) else None
                                    time_period = time_start + '~' + time_end
                                    expiration_day = int(self.closed_third_wen(date).strftime('%Y%m%d'))
                                    expiration_min = self.get_duration(date, time_start, expiration_day)
                                    history_data.append(
                                        [date, time_period, open_price, high_price, low_price, closed_price, volume, expiration_day,
                                         expiration_min])
        
                        history = pd.DataFrame(data=history_data)
                        history.columns = ['date', 'time_period', 'open', 'high', 'low', 'close', 'volume', 'expiration_day',
                                           'min_till_expire']
                        month_history = pd.DataFrame.append(self=month_history, other=history, ignore_index = True)
                        
                month_history = month_history.reset_index()
                self.total_history = self.total_history.append(month_history, ignore_index = True)