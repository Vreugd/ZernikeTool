# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 08:08:21 2022

@author: Jan de Vreugd
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import gamma, factorial
import matplotlib.tri as mtri
from scipy.interpolate import griddata
from matplotlib import cm
import streamlit as st
import plotly.graph_objects as go
from matplotlib.ticker import LinearLocator
import os


def main():
    st.set_page_config(layout="wide")
    with st.sidebar:
        st.title('Zernike Decomposition Tool')
        st.write('info: jan.devreugd@tno.nl')
        uploaded_file = st.file_uploader("Select a datafile:")
        
    if uploaded_file is not None:
                
        filename,file_extension = os.path.splitext(uploaded_file.name)

        if file_extension == '.xlsx':
            df = pd.read_excel(uploaded_file)
        if file_extension ==  '.txt':
            df = pd.read_csv(uploaded_file, sep = '\s+', header = None)            
        
        with st.sidebar:
            units = st.radio('data units:',('meters', 'millimeters'))
            if units == 'meters':
                UnitFactor = 1E9
            else:
                UnitFactor = 1E6
        
        shapeFile = df.shape
        with st.sidebar:
            st.write('the datafile contains ' + str(shapeFile[0]) + ' datapoints')
            st.write('the datafile contains ' + str(shapeFile[1]) + ' columns')
                    
            with st.container():    
                col1,col2,col3 = st.columns(3)
                with col1:
                    values = list(range(1,shapeFile[1]+1))
                    if shapeFile[1] == 7:
                        v = 2
                    elif shapeFile[1] == 6:
                        v = 1
                    elif shapeFile[1] == 4:
                        v = 2
                    elif shapeFile[1] == 3:
                        v = 1   
                    else:
                        v = 1
                    default_ix = values.index(v)
                    columnx = st.selectbox('x-column:',values,index = default_ix)
                    
                with col2:
                    values = list(range(1,shapeFile[1]+1))
                    if shapeFile[1] == 7:
                        v = 3
                    elif shapeFile[1] == 6:
                        v = 2
                    elif shapeFile[1] == 4:
                        v = 3
                    elif shapeFile[1] == 3:
                        v = 2
                    else:
                        v = 2
                        
                    default_iy = values.index(v)
                    columny = st.selectbox('y-column:',values,index = default_iy)

                with col3:
                    values = list(range(1,shapeFile[1]+1))
                    if shapeFile[1] == 7:
                        v = 7
                    elif shapeFile[1] == 6:
                        v = 3
                    elif shapeFile[1] == 4:
                        v = 4
                    elif shapeFile[1] == 3:
                        v = 3
                    else: 
                        v = 3
                        
                    default_iz = values.index(v)
                    columnz = st.selectbox('z-column:',values,index = default_iz)

              
            mm = list(range(3,15))
            NN = []
            for i in range(len(mm)-1):
                NN.append(sum(range(mm[i+1])))
                
            default_NN = NN.index(6)
            N_Zernikes = st.selectbox('# Zernike terms ',NN,index = default_NN)    
             
            index = NN.index(N_Zernikes)  
            m_max = mm[index]    
   
 
        x = df.iloc[:,columnx-1].to_numpy()
        x = x.reshape((len(x)))
        x = x - np.mean(x)        
        y = df.iloc[:,columny-1].to_numpy()
        y = y.reshape((len(y)))
        y = y - np.mean(x)
        dz = df.iloc[:,columnz-1].to_numpy()
        dz = dz.reshape((len(dz)))
        SFE_dz = np.std(dz)*UnitFactor
        PV_dz = np.max(dz)-np.min(dz)*UnitFactor
               
        Xjan = np.linspace(min(x),max(x),100)
        Yjan = np.linspace(min(y),max(y),100)
        xi,yi = np.meshgrid(Xjan,Yjan) # Needs to be checked!!
        z_grid = griddata((x,y),dz,(xi,yi),method='cubic')
        
        R = np.sqrt(x**2 + y**2)
        phi = np.arctan2(x,y)
        rho = R/np.max(R)
        #rho = rho.to_numpy()
        
        A = [[0,0]]
        #m_max = 4
        for i in range(1,m_max):
            for j in range(-i,i+1,2):
                A.append([j,i])
        mnlist = [['Z[' + str(A[0][0]) + ']' +'[' + str(A[0][1]) + ']']]        
        for i in range(1,len(A)):
            mnlist.append(['Z[' + str(A[i][0]) + ']' +'[' + str(A[i][1]) + ']'])
        
        ZernikeInfluenceFunctions = np.zeros([len(x),len(A)])
        for i in range(len(A)):
            
            m = A[i][0]
            n = A[i][1]
            k_inf = int(((n-abs(m))/2))
        
            Zs = np.zeros([len(x),k_inf+1])
            
            if abs(m)-n == 0:
                #print('boe')
                k = 0
                F1 = np.math.factorial(n-k)
                F2 = np.math.factorial(k)
                F3 = np.math.factorial(int((n+abs(m))/2) - k )
                F4 = np.math.factorial(int((n-abs(m))/2) - k )
                Zs = (-1)**k*F1/(F2*F3*F4)*rho**(n-2*k)
            else:
                
                for k in range(int((n-abs(m))/2)+1):
                    F1 = np.math.factorial(n-k)
                    F2 = np.math.factorial(k)
                    F3 = np.math.factorial(int((n+abs(m))/2) - k )
                    F4 = np.math.factorial(int((n-abs(m))/2) - k )
                    Ri = (-1)**k*F1/(F2*F3*F4)*rho**(n-2*k)
                    Zs[:,k] = Ri  
                Zs = np.sum(Zs,axis=1)
            
            if m >= 0:    
                Zs = Zs.reshape(len(Zs))*np.cos(abs(m)*phi)
            else:
                Zs = Zs.reshape(len(Zs))*np.sin(abs(m)*phi)
                
            ZernikeInfluenceFunctions[:,i] = Zs   
        
        Xlinear = np.linalg.lstsq(ZernikeInfluenceFunctions,dz,rcond=None)[0] 
        
        Zernikes = Xlinear*ZernikeInfluenceFunctions
        
        dzPTT = dz-Zernikes[:,0]-Zernikes[:,1]-Zernikes[:,2]
        SFE_dzPTT = np.std(dzPTT) * UnitFactor
        PV_dzPTT = (np.max(dzPTT) - np.min(dzPTT)) * UnitFactor
        
        dzPTTF = dzPTT-Zernikes[:,4] 
        SFE_dzPTTF = np.std(dzPTTF) * UnitFactor
        PV_dzPTTF = (np.max(dzPTTF) - np.min(dzPTTF)) * UnitFactor
        
        dzPTTg = griddata((x,y),dzPTT,(xi,yi),method='cubic')
        dzPTTFg = griddata((x,y),dzPTTF,(xi,yi),method='cubic')
                
        
        W = 450
        H = 450
                
        col1, col2,col3 = st.columns(3)
        
        with col1:
            fig = go.Figure(go.Surface(x=xi,y=yi,z=z_grid,colorscale='jet'))
            fig.update_layout(title='original data:' + '<br>' + 
                              'SFE = ' + str(np.round(SFE_dz,2)) + 'nm' + '<br>' + 
                              'PV = ' + str(np.round(PV_dz,2)) + 'nm', autosize=False,width=W, height=H,title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig2 = go.Figure(go.Surface(x=xi,y=yi,z=dzPTTg,colorscale='jet'))
            fig2.update_layout(title='original data minus Piston, Tip and Tilt:' + '<br>' + 
                               'SFE = ' + str(np.round(SFE_dzPTT,2)) + 'nm' + '<br>' + 
                               'PV = ' + str(np.round(PV_dzPTT,2)) + 'nm', autosize=False,width=W, height=H,title_x=0.5)
            st.plotly_chart(fig2, use_container_width=True)    
        
        with col3:
            fig3 = go.Figure(go.Surface(x=xi,y=yi,z=dzPTTFg,colorscale='jet'))
            fig3.update_layout(title='original data minus Piston, Tip, Tilt and Focus:' + '<br>' +
                               'SFE = ' + str(np.round(SFE_dzPTTF,2)) + 'nm' + '<br>' + 
                               'PV = ' + str(np.round(PV_dzPTTF,2)) + 'nm', autosize=False,width=W, height=H,title_x=0.5)
            st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("""---""")
        
        SFEs = np.round(np.std(Zernikes,axis=0) * UnitFactor,2)
        PVs = np.round((np.max(Zernikes,axis=0) - np.min(Zernikes,axis=0)) * UnitFactor,2)
        
        fig2 = go.Figure(data=[go.Table(header=dict(values=['Zernike Mode:', 'PV [nm]', 'SFE [nm RMS]']),
                 cells=dict(values=[mnlist, SFEs,PVs]))
                     ])
        #fig2.update_traces(cells_font=dict(size = 15))
        fig2.update_layout(width=300,height = len(A)*25 + 400)
        fig2.update_layout(margin=dict(l=0, r=0, t=50, b=250))
        
        with st.expander("Table with Zernike polynomials:"):
            st.plotly_chart(fig2)
            
        col1, col2,col3,col4,col5,col6 = st.columns(6)
        H = [col1,col2,col3,col4,col5,col6]
        
        
        for j in range(len(A)):
            i = np.argsort(SFEs)[-1-j]
            plt.figure(i+1)
            Zjan = griddata((x,y),ZernikeInfluenceFunctions[:,i],(xi,yi),method='cubic')
            fig,ax = plt.subplots(figsize=(6,3))
            pc = ax.pcolormesh(xi,yi,Zjan,cmap=cm.jet)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title('Zernike Mode: '+ 'n=' + str(A[i][1]) + ' m=' + str(A[i][0]) + 
                         '\nPV = ' + str(PVs[i]) + ' nm' +
                         '\nSFE = ' + str(SFEs[i]) + ' nm RMS'
                         )
            with H[j%6]:
                st.pyplot(fig)

main()               



