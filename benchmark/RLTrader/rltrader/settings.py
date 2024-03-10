# settings.py

import os
import locale
import platform

# logger name
LOGGER_NAME = 'rltrader'

# path setting
# root path of the project
BASE_DIR = os.environ.get('RLTRADER_BASE', os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))

# locale setting
if 'Lines' in platform.system() or 'Darwin' in platform.system():
    locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
elif 'Windows' in platform.system():
    locale.setlocale(locale.LC_ALL, '')