# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:47:59 2020

@author: 11101
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


class CallData:
    def __init__(self, Strike_Price='8000'):
        self.Target_Strike_Price = str(Strike_Price)
        self.call_month_check = 'A'
        self.call_data = []
        self.input_columns = ['日期','商品代號','第一支腳商品代碼1','第一支腳買賣別1','第一支腳成交價格1','第一支腳成交數量1',
                '第二支腳商品代碼2','第二支腳買賣別2','第二支腳成交價格2','第二支腳成交數量2','買賣別','成交價格',
                '成交數量','開平倉碼','搓合標記','原始成交時間','單複式碼','交易與委託報價檔連結代碼']
        
        self.cols_name = ['date','time_period','product_id','open','high','low','close','volume','expiration_date','min_till_expire']
        
        self.Call = pd.DataFrame(columns = self.cols_name)
        self.readfromfile()
        
    def get_CallData(self):
        return self.Call
    
    def is_third_wednesday(self, date):
        year = date//10000
        month = (date//100)%100
        day = date%100
        d = datetime(year,month,day)
        return d.weekday() == 2 and 15 <= d.day <= 21  
        
    
    def get_duration(self, date, start_time):
        year = date//10000
        month = (date//100)%100
        day = date%100
        begin_time = datetime(year, month, day,int(start_time[:2]), int(start_time[-2:]),0)
        expiration_time =  self.get_expiration_date(date)
        expiration_time = expiration_time.replace(hour=13, minute=45)
        time_difference = expiration_time - begin_time
        return time_difference / timedelta(minutes = 1)
    def get_expiration_date(self, date):
        year = date//10000
        month = (date//100)%100
        day = date%100
        d = datetime(year,month,day)
        end_date =  21-(datetime(year,month,1).weekday()+4)%7
        expiration_date = datetime(year, month, end_date)
        if d > expiration_date:
            end_date =  21-(datetime(year,month+1,1).weekday()+4)%7
            expiration_date = datetime(year, month+1, end_date)
        return expiration_date

    def tojson(self):
        filename = 'CallOption_'+self.Target_Strike_Price+'.json'
        self.Call.to_json('Data_json/OWMTF_json/'+filename, orient = 'records')

    def readfromfile(self):
        path = 'Data/OWMTF'
        file_content = os.listdir(path)
        file_content.sort()
        for file_dir in file_content:
            file_list = os.listdir(path+'/'+file_dir)
            for file in file_list:
                print(file)
                date = file[-12:-4]
                data_MTH_ = pd.read_table(path+'/'+file_dir+'/'+file, header = None) #把一筆資料讀進來
                data_MTH_.columns = self.input_columns
                #過濾掉沒用的資料
                data_MTH_ = data_MTH_.drop([ '第一支腳商品代碼1','第一支腳買賣別1','第一支腳成交價格1','第一支腳成交數量1',
                                             '第二支腳商品代碼2','第二支腳買賣別2','第二支腳成交價格2','第二支腳成交數量2','買賣別',
                                             '開平倉碼','搓合標記','單複式碼','交易與委託報價檔連結代碼'], axis=1)
                data_MTH_ = data_MTH_[data_MTH_['商品代號'].str[:3]=='TXO']
                data_MTH_ = data_MTH_[data_MTH_['成交價格']>0]
                data_MTH_ = data_MTH_[np.invert([(':' in x) | ('-' in x) | ('/' in x) for x in data_MTH_['商品代號']])]
                strike_price = [x[4:8] for x in data_MTH_['商品代號']]
                data_MTH_['履約價格'] = strike_price
                expiration_day = [x[8] for x in data_MTH_['商品代號']]
                data_MTH_['到期日代號'] = expiration_day
                
                
                #index 0:日期, 1:商品代號, 2:成交價格, 3:成交數量, 4:原始成交時間, 5:履約價格, 6:到期日代號
                call_data = data_MTH_[(data_MTH_['到期日代號'] == self.call_month_check)].to_numpy()
                # print(call_data[0,0],type(call_data[0,0]))
             
                for hour in range(8,14):
                    for minute in range(0,60,5):
                        if hour == 13 and minute >= 40:
                            time_start = '%02d:%02d'%(hour,40)
                            time_end = '%02d:%02d'%(hour,45)
                            call_temp = call_data[(call_data[:,4] >= time_start) &
                                                  (call_data[:,5] == self.Target_Strike_Price)]
     
                        elif minute == 55:
                            time_start = '%02d:%02d'%(hour,minute)
                            time_end = '%02d:%02d'%(hour+1,0)
                            call_temp = call_data[(call_data[:,4] >= time_start) & 
                                                  (call_data[:,4] < time_end) &
                                                  (call_data[:,5] == self.Target_Strike_Price)]
                        else:  
                            time_start = '%02d:%02d'%(hour,minute)
                            time_end = '%02d:%02d'%(hour,minute+5)
                            call_temp = call_data[(call_data[:,4] >= time_start) & 
                                                  (call_data[:,4] < time_end) &
                                                  (call_data[:,5] == self.Target_Strike_Price)]
                        call_temp[:,4].sort()     
                       
                        if hour == 13 and minute>40:
                            continue
                
                       
                        if  (hour==8 and minute>=45)  or (hour==13 and minute<=40) or (hour>8 and hour<13):
                            time_period = time_start + '~' + time_end
                            expiration_min = self.get_duration(int(date), time_start)
                            expiration_date = datetime.date(self.get_expiration_date(int(date))).strftime('%Y%m%d')
                            call_date = date
                            call_open = call_temp[0,2] if (call_temp.size != 0) else None
                            call_close = call_temp[-1,2] if (call_temp.size != 0) else None
                            call_high = call_temp[:,2].max() if (call_temp.size != 0) else None
                            call_low = call_temp[:,2].min() if (call_temp.size != 0) else None
                            call_product_id = call_temp[0,1] if (call_temp.size != 0) else None
                            call_volume = call_temp[:,3].sum() if (call_temp.size != 0) else None
                            
                            temp = pd.Series([call_date,time_period,call_product_id,call_open,call_high,call_low,call_close,call_volume,expiration_date,expiration_min], index = self.cols_name)
                            
                            self.Call = self.Call.append(temp, ignore_index=True)
                       
                            
                            
                # if self.is_third_wednesday(int(call_data[0,0])):
                #     self.call_month_check = chr(ord(self.call_month_check)+1)
