# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 16:06:24 2020

@author: Joukey
"""


import os
os.chdir('D:/EIIE_keras/')

from pgportfolio.learn.tradertrainer import TraderTrainer
from pgportfolio.tools.configprocess import load_config
         
config_2 = load_config(2)
train_dir = "train_package"

s_path = "./" + train_dir + "/" + '2'+ "/netfile"
l_path = "./" + train_dir + "/" + '2' + "/tensorboard"

t_trainer = TraderTrainer(config_2, save_path=s_path, device='gpu')

#%%

import logging

console_level = logging.INFO
logfile_level = logging.DEBUG

logging.basicConfig(filename=l_path.replace("tensorboard","programlog"), level=console_level)
console = logging.StreamHandler()
console.setLevel(console_level)
logging.getLogger().addHandler(console)

t_trainer.train_net(log_file_dir = l_path, index='2')

#%%