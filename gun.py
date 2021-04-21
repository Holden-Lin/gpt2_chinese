import os
import gevent.monkey
gevent.monkey.patch_all()
# import logging
import multiprocessing

#debug = True
dirs="./logs"
if not os.path.exists(dirs):
    os.makedirs(dirs)
# loglevel = 'info'
bind = '0.0.0.0:5000'
pidfile = r'./logs/gunicorn.pid'
accesslog = r"./logs/micro_access.log"
errorlog = r"./logs/debug.log"

#启动的进程数
workers = multiprocessing.cpu_count() * 2+1
worker_class = 'gevent'
daemon = 'false'

x_forwarded_for_header = 'X-FORWARDED-FOR'
# logging.basicConfig(level=logging.INFO,
#                         format='%(asctime)s %(filename)s[line:%(lineno)d] [%(levelname)s] %(message)s',
#                         datefmt='%a, %d %b %Y %H:%M:%S', filename=r"./logs/micro.log",
#                         filemode='a')