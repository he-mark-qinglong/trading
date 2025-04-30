# -*- coding: utf-8 -*-
import logging
import time
import os
import re
from logging.handlers import TimedRotatingFileHandler

#logging.basicConfig(level=logging.INFO)
log = logging.getLogger('mylog')
log.setLevel(logging.INFO)
#log.propagate = False               # 不输出到屏幕
log_fmt = '%(asctime)s,%(levelname)s:%(message)s'
formatter = logging.Formatter(log_fmt)
#创建TimedRotatingFileHandler对象
if not os.path.exists('buildinglogs'):
    os.mkdir('buildinglogs')
log_file_handler = TimedRotatingFileHandler(filename="buildinglogs/mylog", when="D", interval=1, backupCount=1000)
log_file_handler.setLevel(logging.INFO)
log_file_handler.suffix = "%Y-%m-%d.log"
# log_file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")
log_file_handler.setFormatter(formatter)
log.addHandler(log_file_handler)
# log.removeHandler(log_file_handler)

mylog = log

if __name__ == '__main__':
    mylog.error('11111')
