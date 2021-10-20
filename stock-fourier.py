from gradio import inputs
from numpy import fft
from numpy.core.fromnumeric import size
from numpy.core.numeric import zeros_like
from numpy.lib.function_base import place
import tushare as ts
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
def fourier(ts_code,place,start_date='20060101',end_date='20211011'):
    #ts_code = '000001.SZ' #第一号股票，平安
    #ts_code = '601985.SH' 
    ts_code=str(ts_code)+'.'+str(place[0])
    print(ts_code)
    #start_date = '20060101'
    #end_date = '20211011'
    fig=plt.figure()
    pro = ts.pro_api(token="cea582af6838f321b8357217c4690b5350e5b8eb98d978dc3b6dd66c")
    df = pro.daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date)
    use_cols = ['close', 'open', 'high', 'pre_close', 'vol', 'amount']
    total=df['ts_code'].shape[0]
    price=[]
    for i in range(total):
        price.append(df['close'][total-i-1])
    #data_FT = df[['Date', 'GS']]
    data_FT=price
    close_fft = np.fft.fft(np.asarray(price))
    fft_df = pd.DataFrame({'fft':close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    plt.figure(figsize=(14, 7), dpi=100)
    fft_list = np.asarray(fft_df['fft'].tolist())
    for num_ in [3, 6, 9,100]:
        fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0
        print(np.nonzero(fft_list_m10))
        for i in range(len(fft_list_m10)):
            if fft_list_m10[i]!=0 :
                print(i,fft_list_m10[i])
        print("-------")
        fore=np.copy(fft_list[:num_-1])
        back=np.copy(fft_list[-(num_-1):])
        zeroes=np.zeros(len(fft_list_m10)+30)
        final=np.concatenate((fore,zeroes,back))
        print()
        plt.plot(np.fft.ifft(fft_list_m10,1000), label='Fourier transform with {} components'.format(num_))
    plt.plot(price, label='Real')
    plt.xlabel('Days')
    plt.ylabel('CNY')
    plt.title('Figure 3: Goldman Sachs (close) stock prices & Fourier transforms')
    plt.legend()
    #plt.savefig("D:\\a.png")
   # figa__ = cv2.imread(f'D:\a.png') 
    #print(type(figa__))
    plt.show()
    return plt.gcf()
#place = gr.inputs.CheckboxGroup(["SZ", "SH"])
fourier("000001",['SZ'],"20210101")
#gr.Interface(fn=fourier,inputs=["text",place,"text","text"],outputs="plot").launch(share=False)
