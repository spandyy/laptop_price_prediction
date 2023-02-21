import streamlit as st
import pickle 
import numpy as np
#impoorting the model
pipe=pickle.load(open("pipe.pkl", "rb"))
df=pickle.load(open("df.pkl", "rb"))

st.title("LAPTOP PRICE PREDICTOR🐷")

#making form for taking inputs
#brand
company=st.selectbox("Brand", df["Company"].unique())

#type of laptop
type=st.selectbox("Type", df["TypeName"].unique())

#RAM
RAM=st.selectbox("RAM in GB", [2,4,6,8,12,16,24,32,64])#[ 8, 16,  4,  2, 12,  6, 32, 24, 64]

#weight
weight=st.number_input("Weight of the laptop")

#touchsreen
touchscreen=st.selectbox("Touchscreen", ["No", "Yes"] )#df["Touchscreen"].unique()

#Ips
ips=st.selectbox("IPS", ["No", "Yes"] )

#Screen sizze
screen_size=st.number_input("Screen size")

#resolution
resolution=st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900',
'3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',df['Cpu_brand'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['Gpu_brand'].unique())

os = st.selectbox('OS',df['3_OPS'].unique())

if st.button("Predict Price"):
    #input query point
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])

    ppi = ((X_res**2) + (Y_res**2))**0.5/int(screen_size)

    query=np.array([company,type,RAM,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
    print(query)
    query=query.reshape(1,12)
    st.title(int(np.exp(pipe.predict(query))))