import ftp_download as ftp
import datetime

# downloading data including price and features
cur_date = datetime.datetime.now().strftime('%Y-%m-%d')
ftp.download_data(cur_date)
ftp.data_processing()
