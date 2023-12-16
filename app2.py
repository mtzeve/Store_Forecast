import streamlit as st
import numpy as np
import pandas as pd
import os
import gc
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px
from sklearn.metrics import mean_squared_error
from math import sqrt
import statsmodels
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import itertools
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.plot import plot_yearly
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation, performance_metrics
from prophet import Prophet
import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import time
import threading


logo_url = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXYAAACHCAMAAAA1OYJfAAAA1VBMVEX///8AAADuOkOtra3S0tLuOkKSkpLh4eH7+/t6enoICAjd3d0NDQ3MzMy1tbUtLS3BwcG7u7v19fWLi4v++PiCgoLu7u6np6dPT09aWlrr6+vf39/V1dX82tzHx8fwTlZqamqcnJxCQkIhISF1dXXwWGE5OTk+Pj6YmJhJSUkWFhZhYWFsbGz96+xVVVUlJSXuMDswMDD85OX2qq3+8fHuKDP4t7rwaW760dL0i47zfYLtIi31mJz6ysz5wMP3qKvvQU3yc3j0nJ/ybXTwU1vsFyPye4DuUJJUAAAX4ElEQVR4nO1dB3viOBO2ANNNKKE5NAMhlIQ4hWRJZXPZ/f8/6VMZyZIsyEIgue/W7z3PLW4qr0aj0WikWNbX4vbi4vr64osz/atx/fzycne6Wjw+xo6/uyx/CZ6flsvT2GLh2wTdiPZD49h6+d19X9m+b8cYItoPjOt2e+k/EhGPcc4pItoPh+Obm3t/ATIuWCc/Imk/FG6fft/7gV4Bwm34J6L9ALi4eMNSrlCuIqJ97zh+fuk+bqA8ov0AaL89xBbYQlTUeUT7YXF9v/J9WZdHtB8c7TdfKJfNOiaifW94xqOowrm9gfyI9v3g6aHrr+M4ov0wuH2NEWsRBtKI9i9B+07TLhHth8ft2+liC+0S0b4XLLuBwRgpma/B9Q1zAGywWQz3cev4i1VE+464eOn+sXqh4y1pHX/x6L/f393cpb+7+P+nuL73GZMb2ZZk3PcXv1b3L+12tJC6M47/sf0YZf0jfU5l3Le7p/fL6+8u9f850thQ/0DQxVwVc959eHqNKP80npd8JN1IO1boPtbkNy8/ovFzD7iLBet063WMTeX8+uL2u4v730A7tlhDNdcq9P9+7P7h7rvL+p/BxctH60ZUnS+6r3eRmO8Nt/e+bdTpgaRjzv1lOxpB94g3e4N3l2oXH0+Gfnx3Mf9bOH7YpGCIoC/sn0/fXcr/Gp5/b1zIsLGk3z3/ZQo94Rw6h7tVSK3TGRNYkdhCf/3rJv0nl+jAOTwt7FiIdj5Ttf3uz7+Oc8ty0VHtsDk8GZ2NNuPej739lcNor5E6aPo/lovAWNHVjL94ah80938rvKF70PSfuwvbQDq958d+7tXjki4TaDfJrcBBn+aQPzPd5OkZPjW4+0O30mk53/CHtVpaT1rPaF0t/wRsYmobvF9Ypz/tc2JUS3hXjd4s06g4rpUD8l2nPqxmeo1Oi9Wilpw3MerJ3jDhtODLYZPcHHvDXtHJifROEpnzTHUyi+fEaxg9rzo7l1+jSDfjynUu3hgNZ70M+biG3y2QHObzSlCqXqY66yegVG4FP27w4liJYQNfVj6hhF4kH68dk6MbN+j09uvN9mNsq4fmJVLSVuFsPr6ivKTjE1TAA5dbQtMq1LCEEGpaNad+hAZx1tFzxQlCo5rVKjXQVRLS845G5HFujC7jbOzLJfGnFeukU8ijkSfn7VZRRrqs1S9RJUF+dc6z9V4T3yG5XpZYOulSY1Bo0aJkq5BAAT/PJkRdcE71z7Bu0i1U0n89rGP26ef7oru1ws9dogZXL+UBQrTQTfgXVwShPjw9Qoyj2hChAtyr4mrSLwuEWAw3g87B0Chk0Qz0MP6CSrWLSRpLmWNS+4H8k4R5q5RxxjN4ZQL3zhGCpJ08aoKMz0m7CHm3xoNtCZBw92irnNswkK5+GqdGxxcvq4Xvx+zTbWnPYWGRLkeE7vQZQif8TgIzw+T9HGi3HNw6IGAV3gJuk/5yJ+hI8BhHqMF+VflrJwgdBbmVewjagyDdDFi3SIPPIH9IZCJYx3cv0YyVKtfHpRmIPOud7eov4+XRZDXascX9s+n14+cliVUi69Xb0u7K9bZIK2C+PYSqYlwqD/lVj9NuNcK0W3WEkvTTSpDaFUIl+kPQnhpB92C5z5pSVrjDXElFqRElQ2nv8QyGwahf4R8mCkVENR0UY3faQ25eJu1+7M40krafHoic0/F3a9px6SeyBVOu5qyTCcqWgluFKRrQThzQ3jTQHieEY82AJPuuyGVb0I57A0qK58kC/oqrMwv/VDgrcGmn/6b6ymP3Ej5M1El7icbenfaQXoeRdGmyGW95bDtrmi1pz/UDNc2AB9cSly9AA+ob0D5ZRzvWQyPpSzcLsh1I+xXKBr3rjHYIYAx3FKROP+mLnPa4MgwQdchKg2lPS7pqZ9rvHsUIGuwojfn3BtKvb/jGGT7ibkl7KYumejnLM4TO5Rv4+oz0Z0F7aYqmwEBAe5KI8UiWZawncLcYkh+CdiyYZ+LxOE6fwNjSU5SIAKd9rKgvOtJQ1wymHSsrwfuutD//0rU6cXl1X8JDaftmpa/0bUs7FqGjhHavdoQUq44wgwgfnHYXi38dFLKgPYXNypSFx2LFMB9rtHeyXNsTEBMJD+nwCeaxaCgi0E4yreqlos/rLHdIeEfan1chV4BtP96Hlfrx8jTsiN+WdjJM6ZM6N69V0Atox/dbVaxiXP6RoL1BVS2mvSV/WoCWqlJSSpkBugyoLdIWOQOVlprILRKA0z7RSuXKtFspYkYSAdqN9hDrVNTD9svtcuEbHPE70J7R722i/Wg+IJOmoKUw7Z7ruh66bJCh1EA7l/Y+bhg1tyodEvnEYFva4wrtVmqKh5XcjrQfr3xFqZOf/jJE5Y9XtqHg07T/kZLBFuSlkPYcrr40CmPaZ8lKteIxRbFByZSsItbAg4CV1qBewMCvMGv9AyUz36RkLGo1kRd2ov1poTtg/NWN/tL122qxJlZmW9qxNIaH1F54SG0GQypVo6KpKqol1NeG1EagZPBrOWy2NISxUqmWCIq4Vc+hVSsGLxYfUpPakIpLNaXPgfZylXamXWh/W2hk2ot/QgbMQxck3RROvYMBqfbsQoq0xUT2a8xBDGFIxRPNoFk02qvKJJRq3ALcJ/+W8sAhRnkE/aKQRwPSjEndgMxRmjntBc2AxE1I1/ZaYvZF3ASV+Pa0v+pnBfin+tp0+sbeuKK9w3SpIVtt7jxnpRoKl2SKT1uBWzKONDvRaLdU6orcn8ItGU/YLVZSKLIeGH95IFJkrEyXiCkjKaFCFk3oxKwUTAPwG/np1uus16pWJ7a6bsC8LNcFh+1IO5nsybzViXMAC1ZVtEX5nM9ZZ0A7cRfkoY/otHuKBkacyKpk8ORhGBBjbwahqzLNG8l+rPKMOwdYB8E/MspYzoowD2ivzZHWcn+Ce1+iHf/7eK/Z6tfL2Af7Ibd3heHKTCXjg/rA0pVgxk5e8Fht+bzQKl8Jgqr6LLeKAnmrC/9Llc9mSE9pEiktBLPZMgI5HiuusHNGd0vMmauS+6h1ydugIbmUaqPtaf9NlvAC3e7HNK/68Q/f/yiufXvardYZuuT6sJTNM9djcsp1RQpNuac2KwggxhvTHljwZ+rUsoLOYMCtowE0Se5MEEo+jWPCZmqnYI1Qz6MxpFZugpQPhQfScrJ8IEoIYysttxTRalvSfmeDNwtY72qsX//zKEIG9km7VfP6KFlMJBLFylGlOGMe31YFVTq5EyeJeuxGudMhdnfHobRgvYDmnVZinCXmX0JZ3YzPsnEnl+tUphnoRh1iZKA6VS41Mo/vd4ao4UDDujThJk24M8wOCp1EouNdZuIVPHCfEDcX8hwm2YkKqnZaKaeIeoz/moPNTk/qrR7abki9PoUjphini3eNv9dTWb+s3Sumf/ZHSBULJcdxCiWycsNlt1PoVJL4f3Bd6xSIuQfXZXxRKCXgnqMuKpdxcpkKvs1vFNlrjJ0c/Y3vFOErl9qRhSLLyMG/nESxgLUOWdTL0U9LHa7UO4VixpNKRZIuSDOPdGmrBe7jex8CYij5i1dNrd+bj+cJduzZNEBssdwm048QWtT+c9R2//QjfKJUIfx4FNFehEktRv118VGgNeH8/ffb/gr0V+CazYDYJMj2Vdbb5iglIe/ku0VsefN3xs18BkvuiiFiu1JZf373P9gQaT/aP9rRTqWt8SIsFFv3Nx6/+Ju255G+0Y1i23fC8ZIflIkH0wdFV1yfCjeNkXnfv49i23fEa6C7NVOkfbpRwfiPry9/YczvnuBDtDoeGX8rD667ZqsR4qwflxHnu+MBtvkSDSPfP17+2jAf9d/3aqP/dbjo8tmpyvrF0nzgHe0Xi9VNtC3vU3jz2cwUsy4bgcc/zd5GYmT6q8hI/yTa72Cq+KpeX/li7q+rl9jh1cvnDp75ZIz5V4CZMWSWJAvw7al+QhL8tBenT7tszCsnNkILTq6zFdGyo77VkdiUH7UCN0ku0elUGo1+3HESJ0qarpYY+/IktA8prb3i5PCnrZMPPDGpnFNUPf8F08osx22MbQqzVy/y3XuThiHLfP7rbuqlVhkhlGVA+UmDoJ8nXlVyF82Vlx3Up+3gNqf8G/wOGiSlisQnPD2UTQInqXrlCF1V4/F4gQRt9b2S9EFqdiS+yPcbjQlZCz8663meutUg7fWRnOukZJXQtD/0xgTJCkZyrP/2SGrKEndugDYEur8umD2o+mFOfZ1w+p+9Prj9Q5TdxBliiLuAVCrVmjHqZTf1yYiveNbcKhJIpRTxcd0xu19MMdbTQ5xB32GhS+UU8ZRPG1K6NdcdQVKZGs0+0adXo6oi8mmXRGfwN12XBWMQNJrz804ud1Jhl6Ucxnmzyd9WAjqacmhxCDxwXWb9OBZaRmIa5/TPWTYCONRWAsiSPQtnYSjRECR+leAEGBzZJJJpAN7uNEn8SIl0odkNFUovWVqCEGh1NajJopsJKKjPvky3Dsik0md5SZq9qRr71LlCahCDijYdOO2FzPqFWcMsTu8+6+6qGGm34riMI1Fxj73EL8mCNUHfQHv5HOUhMYdE3fY09ijvI1nlFjTaRZm0LxNXLFO2InWlbNiwDLRbDpbuM6GtoA5rtQzd2m77shgf3xuc67b9aAxu3w5xSYRknDSkcJkj9pIgqzRdW4dak8dVx8lLvdCwxziVInFyLC1p2TnN3hmowWnpc0ltkHBHdXgO0251plKkFfShUKAh4EeX+hCVpbiVSdb90314GQtk6RNlDau8V2jOlcEApFs8a7Ib4/BXqUtY6a6TF0LxZRZdgMYPAnnPqV2JYmbqTRl6kxlUrqa2jbQTXSnat8GymVhm3FB3jLxafauPpsy4fF2TwJa4Wkc7JhvErYaYuE+Fmm6xOhj2YqVY4Dv0Ii/8guXSRsyLtHL5MO3QFGp0mkx7Sw52pTDRjrnmSZShDmsG1dsHwrEvrcXdagdqMBf8w74c6kdraS9eQpTc+aSOpCoTMD1rGFM9popqgzCXHEyZ9zltJtqhf6miKdOeQXp4rJH2FuIGbvUM6hCKIqdo03Ps3oORMv0Q1jD+49veVo7W044pZGW/jLeYuTEXNDt6O3CgEa05M+BM4boYbGTgJrWR9plB88i0T0I62ki7Ve+DopxnLDY2XOkWEsUzmaHK5x1fP4a3XJt2z+yKDbRbTEeMkTAuBI/ulVlTQmgQk6zRmkgJMEBB7Iy0xz+gfRQiz0x7DsaHOAKrWA9ZYzjukqVTSYG0TXHW+zyjx0B7S6ErdTWmMbi0zmLjP6vDVI/6H9ItqDU251k3FW8xFQLzAiPtiQ9oV/fHE5hpt5iTojzEc+4UaC6TTwHLti+5Hdvv+uR0h13VGxGmvdxUlEOLbgeGyaSY6HSYptBsGXfEt9ARmHZiEKTZHI15G8y0Vz+gPYw1tPM6EJMKjBkD7TdYuCWLnU6T7OBoJNwmN3uOBzDQPlVon1C1AVomsAPYlKmvOk8KtAvXevTZmVGLEsDoxkTWSHt/v7RnJkReQBoMpvspCf8KLu8WaoiAr3jH9oIw7SllKEyB0LIiX4r7nklTVtkmlqmsRAwoDqSuYqI9bSJod9rTUIesoTEpyLaM4Or5UUg5W2taGne6fwph2ocK7b0565TgXBL1YuY3aspdNpGnvcFZK1WA2oS+cEQTM9EOo5/alXanvQADqyf3Mgk/VvZjIM9t8rd8bBH2ay+WB1gqCNGOaZBoT02hnuDyC+aGs7CmdJghD+OhYQ7L0ZRoNdCeYyOJNiTvTvsRlBpmYaFu+Ob7/wRmytKXZd23DxLPqNPujBR7O5mHHy4bkIQbkisCaY5eHtE90+nmGqEKwJQ/c3aFaU+xrBqq02V32uO8emUw3fWpFiY6UCNv0uY7sn3mMAckM9qhIOVWnejlgPZysycKz6gSyjzNhr1pkFSOfQgP8nrlJGQ2SXsCWNcp3JX29Exsro+bBWK5CA78OhZGDF3z0LfP7AvMVZEsleqeVxmyK2lbdKBfO2ymGpxvAh7boKOcN6l9WWY8bjo8Z7ie9tSY5RM+Z2pX2jtBEWusRbWDDLCVHmw5vRchStQPfKg4Ukb01Rlf5VFoL/f6gX5lyvxKdH111oMB8/3ydGfac5UGm/42a+FhbFfaCyhIS85Y4Nnuit+v/GxTSvzPDXX4HGQlU/N6Ku0J2RFf0NoEzO8j/kodztjYTdoHl0GzGyeSu9Luyp+A61SdyN39Evr7+tQPgjMWxnNj9oMjVVMkFWbnI1kuWJEb4hpWO7imPAcPTZoJ7HTDQaQG2judksdXdvvG3UY70h5Xlqlg6UB54+6XoPduIexG+1f7gAcla7Rbnky7WsmhVuT0HBQCvWrx07ngttHnBGAWRciScWGoZSdj6NiRdjV6AIxbpV3vhGvxh/hbSvjfg8bYhez2TEB7BVdExElk88BJUA2P3WAc1UV7gCNhgwE5C0s7ezDmasbw0W60F5Q6ZGE5cia9cfsuXI8QTU0Xmg4bZBeiPZXltJf7o3FSBgRTBO+yOlBb3g2Oc4RZqrbqJiHNDIorwywVmowd9aBiN9obR0odxmz0upKSvxAMv4gNpzH7wKGNYefAhNMe10Z8MHungQZoBmonF7ycYL6P9T4ZsEXZLFSlPc31THjxbSfaW4OKeqPDkpeWG6/FLoxTMU36dego3jDtJ0B7rac7jRIDTYxhtk3cALOgehBKtJ72jR5I0Pth/+xOtI/1OuRgLSC4I2i/4S4Bf3XwXQIGxy/QnghXkbmogqWdMhP3M/KRpMoZrdM1a3o8mQkcT6vRnoO15iPdct+F9kS48cHNFgyqFzBXuv3JjMeYf2C9brHzXDTa03AA8ji8Qh1av2ChWvmCVZGjK1LMElxnypyMZIELOQd4+JfuStuFdi9cB1gXDrTMBTgfn2GmZPuH3yewPk7GQsnQTLHM+Awm1w74FqyJIlVsYOytsdzZnCULT8OuMNBB+tlZO9Cenp6Fy8C8zlfBA1ZNiO21bf8h9MnesZ72jGlRDhRvIEJs9e0sk1dTYK+tWV5idgwfM8O0uzyUaH14UhhG2j1TjwNjSe8YPx7B5/gVO5EY7YazP2rZpmHhP6FrGVjb1l0BTFE0LRNYGj1+aYgKSxhDiTbSnjbSftQ/MbwLjcovwWo/ZXveffNJ1XvGWtqT5goijaTyld4QDEykjL5fako0RR9vhWnnw54aaLOR9riJ9o557tBXxR1oZrt9/a/54+CsuGElgzu60TECWibou2Bn6yqUBYqaxJ3mKE1XWgaFBEsoEOuk5LSGds9E+9AYDchNd7VJ2GEDvm9Oft/Q4444POVA5AAp9n7gD2PyXw37aakHJ2y7U9bltHMmhVIG3qXvTyabaB+rEkxRWFMHyHEul/n2lK1shM56PAzM2wrIArV5tpOa6uJOTPeBSZtUiP7S5ohWnNxUDrUCf7JKZ5o7CQKRhKntmhXaXpj29FCWDgkua0BlrfaOLWt80bEBJ6DndGk9X+dSccE5G1gZZG07Y1pbT7dI9YaaAJLqmnZzaGZLeabLO5jzM6NV2oIhRo5WqKwb02FJlRz8LPDq04Be4/t7R4p7uLVZRVVTBAFa/IMBrz6WnVBMHkdlSv56SZAdlsmsxtuQJ9hU9XIRuhWasJ7o8WuDk0ywrrtHB8bYzhb3pWZFUm1ykLK9WlONvSLncIki6DkdGNUcmK6gpKMZ3i2nJ30xhOO56oobVUXZG2JpzrB9jVgd9aqydy1XlEuAqk5HorQu7uc7jiCdvScnUu44fTmVXhE/bcVhPWNedNRmdp2M9PI5VLFN7JjHL/krykWvHg9Qr9dBLOv8dt1TJxvpYl35wmMVSoXDb2U4hbg36jca/Xk9ro3caglIEWSKlAfKe548FNXURPBTnEtB1KEeVztySq21x4pE7JjFFw2n+8LHJ8/RvZcH/muDn0F6gRX7e3RkwxeDhLOvogOnvhoPfsz/57sL8feha9uP0YF2X43jbuyrJkoRAtzYdjc67OvL8eT7e9rlG2ELLBenkfH45bj9HWn2b8DzKhZp9q/Hy2Nks38DXn7964+P+w8i/baKpkpfj9sD7QqLsBEX+p+aifAVuPgd2THfgOvIaP8OXEd/NuY7cLv/gxwiRIgQIUA0V/oW7P1kpAgRIkSQ8D94Kegw23sJUwAAAABJRU5ErkJggg=='  # Replace with the actual URL of your logo image
st.image(logo_url, width=400)
#load train data
@st.cache_data
def load_train_data():

    file_path = 'E:\BDS\Sales_forecast\\train.csv'

    train_df = pd.read_csv(file_path)
    
    return train_df


train_df = load_train_data()

#selection criteria for training and test data
selected_family = st.sidebar.selectbox("Select Product Family", train_df['family'].unique().tolist(), index=0)
selected_store = st.sidebar.slider("store number", 1, 54)

if not selected_family:
    st.warning("Please select a product family from the sidebar ⚠️")
    st.stop()
if not selected_store:
    st.warning("Please select a store from the sidebar ⚠️")
    st.stop()

#filter train and test
filtered_train = train_df[(train_df['family'] == selected_family) & (train_df['store_nbr'] == selected_store)]

#load and clean holiday data
@st.cache_data
def load_holiday_data():
    file_path = 'E:\BDS\Sales_forecast\holidays_events.csv'
    hol_df = pd.read_csv(file_path)
    hol_df["date"] = pd.to_datetime(hol_df.date)
    tr1 = hol_df[(hol_df.type == "Holiday") & (hol_df.transferred == True)].drop("transferred", axis = 1).reset_index(drop = True)
    tr2 = hol_df[(hol_df.type == "Transfer")].drop("transferred", axis = 1).reset_index(drop = True)
    tr = pd.concat([tr1,tr2], axis = 1)
    tr = tr.iloc[:, [5,1,2,3,4]]
    hol_df = hol_df[(hol_df.transferred == False) & (hol_df.type != "Transfer")].drop("transferred", axis = 1)
    hol_df = pd.concat([hol_df, tr]).reset_index(drop=True)
    hol_df["description"] = hol_df["description"].str.replace("-", "").str.replace("+", "").str.replace('\d+', "")
    hol_df["type"] = np.where(hol_df["type"] == "Additional", "Holiday", hol_df["type"])
    hol_df["description"] = hol_df["description"].str.replace("Puente ", "")
    hol_df["type"] = np.where(hol_df["type"] == "Bridge", "Holiday", hol_df["type"])
    hol_df = hol_df[hol_df.type != "Work Day"]
    hol_df = hol_df[hol_df['locale'] == "National"].drop(['type', 'locale_name', 'locale'], axis=1)
    hol_df.rename(columns={"date": "ds", "description": "holiday"}, inplace = True)

    return hol_df

hol_df = load_holiday_data()

#aggregate train
df = filtered_train.groupby(['date'])[['sales','onpromotion']].sum().reset_index()

df.rename(columns={"date": "ds", "sales": "y"}, inplace = True)



#def tuning():
#    param_grid = {
#    # tuning those parameters can potentially improve the performance of our model
#    'changepoint_prior_scale': [0.001, 0.1],
#    #'seasonality_prior_scale': [0.01, 1.0],
#    #'holidays_prior_scale': [0.01, 0.1],
#    'seasonality_mode': ['additive', 'multiplicative'],
#    }
#
## Generate all combinations of parameters
#    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
#    rmses = []
#    all_params
#    for params in all_params:
#        m = Prophet(**params).fit(df)
#        df_cv = cross_validation(m, initial = '365.5 days', period = '30 days', horizon='30 days')
#        df_p = performance_metrics(df_cv, rolling_window=1)
#        rmses.append(df_p['rmse'].values[0])
#
#    # Find the best parameters
#    tuning_results = pd.DataFrame(all_params)
#    tuning_results['rmse'] = rmses
#
#    best_params = all_params[np.argmin(rmses)]
#    return best_params
#best_params = tuning()
#best_params_df = pd.DataFrame([best_params])
#cp_prior_scale = best_params_df['changepoint_prior_scale'].iloc[0]
#season_mo = best_params_df['seasonality_mode'].iloc[0]

m = Prophet(holidays=hol_df,weekly_seasonality=False,changepoint_prior_scale=0.1,seasonality_mode='multiplicative')
m.add_seasonality(name='monthly', period=30.5, fourier_order=5, prior_scale=15)
m.add_regressor('onpromotion')
m.fit(df)
future = m.make_future_dataframe(periods=365)
no_of_promotions = st.sidebar.slider("Pick number of promotional items", 0, 100)
future['onpromotion'] = no_of_promotions
forecast = m.predict(future)

#whole model
st.title('Sales Forecast :chart_with_upwards_trend:')
st.markdown('Here is an interactive Plotly plot showcasing the forecasted values produced by the Prophet model. Utilizing the model instance (m) and the forecasted DataFrame (forecast) as inputs, this dynamic visualization allows users to explore and analyze the predicted trends in an interactive and engaging manner.')

st.plotly_chart(plot_plotly(m, forecast),use_container_width=True)
#forecast components
st.title('Forecast components')
with st.expander("See explanation"):
     st.write('Trend: The trend component represents the overall direction of the time series. It can show the general upward or downward movement in sales over time.')
     st.write('Seasonality (Monthly): The monthly seasonality component captures recurring patterns with a period of approximately one month (30.5 days). The Fourier terms control the shape and complexity of this seasonal pattern.')
     st.write('Holidays: This component visualizes the impact of holidays on the time series.')
     st.write('Onpromotion Regressor: This component illustrates how the onpromotion variable influences the time series in a proportional (multiplicative) manner')


st.plotly_chart(plot_components_plotly(m, forecast),use_container_width=True)

#performance evaluation metrics
st.title('Performance Evaluation')

df_cv = cross_validation(m, initial = '1000 days', period = '30 days', horizon='30 days')
df_p1 = performance_metrics(df_cv)
df_p1['horizon'] = df_p1['horizon'].astype(str)


col1, col2 = st.columns(2)

with col1:
    rmse = round(df_p1['rmse'].mean(),2)
    st.metric(label="RMSE", value=rmse)
with col2:
    mean_mdape = round(df_p1['mdape'].mean(),2)
    st.metric(label="mdape", value=mean_mdape)

with st.expander("See explanation"):
     st.write("RMSE of: {}".format(rmse), "would mean, on average, the model's predictions are off by approximately: {}".format(rmse), "units of sales.")
     st.write("MdAPE at: {}".format(mean_mdape), "signifies that half of the model's predictions for that time horizon deviate by less than: {}".format(mean_mdape*100), "% from the actual sales")

#col1, col2 = st.columns(2)
#with col1:
col1, col2 = st.columns([1, 1])  # Equal width for both columns

with col1:
    rmse_chart = px.bar(df_p1, x='horizon', y='rmse', labels={'horizon': 'Horizon', 'rmse': 'RMSE'}, title='RMSE by Horizon')
    st.plotly_chart(rmse_chart, use_container_width=True)  # Adjust chart width to match column width

with col2:
    mdape_chart = px.bar(df_p1, x='horizon', y='mdape', labels={'horizon': 'Horizon', 'mdape': 'MDAPE'}, title='MDAPE by Horizon')
    mdape_chart.update_traces(marker_color='#ff4b4b')
    st.plotly_chart(mdape_chart, use_container_width=True)  # Adjust chart width to match column width

