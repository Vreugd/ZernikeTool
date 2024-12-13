# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:06:36 2024

@author: vreugdjd
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
from matplotlib import cm
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import os
from scipy.interpolate import interp1d
from scipy.spatial import Delaunay
import plotly.figure_factory as ff
import matplotlib.tri as tri

def funcSphere(R, Rc, offset):
    C = 1/Rc
    
def funcASphere(R,Rc,k,offset):
    asphereZ = R**2 / ( Rc * ( 1 + np.sqrt( 1 - (1+k) * (R/Rc)**2  ) ) ) + offset
    return asphereZ


def dataread(uploaded_file):
        if uploaded_file != 'TestFile_FEM.txt' and uploaded_file != 'TestFile_CMM.txt':
            filename,file_extension = os.path.splitext(uploaded_file.name)
            if file_extension == '.xlsx':
                df = pd.read_excel(uploaded_file)
            if file_extension ==  '.txt':
                df = pd.read_csv(uploaded_file, sep = '\s+', header = None)
            if file_extension == '.csv':
                 df= pd.read_csv(uploaded_file, sep = ',', header = None)
            shapeFile = df.shape    
            return df, shapeFile
        elif uploaded_file ==  'TestFile_FEM.txt':
            df = pd.read_csv('TestFile_FEM.txt', sep = '\s+', header = None)
            shapeFile = df.shape    
            return df, shapeFile
        elif uploaded_file ==  'TestFile_CMM.txt':
            df = pd.read_csv('TestFile_CMM.txt', sep = '\s+', header = None)
            shapeFile = df.shape    
            return df, shapeFile
        
def dataselection(data, shapeFile):
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
            
            x = data.iloc[:,columnx-1].to_numpy()
            x = x.reshape((len(x)))
            x = x - np.mean(x)        
            y = data.iloc[:,columny-1].to_numpy()
            y = y.reshape((len(y)))
            y = y - np.mean(y)
            dz = data.iloc[:,columnz-1].to_numpy()
            dz = dz.reshape((len(dz)))
            
            R = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y,x)
            rho = R/np.max(R)
            
                    
            triang = tri.Triangulation(x, y)
            triangles = triang.triangles
            
            return x, y, dz, R, phi, rho, triangles
        
def dataselectionMasked(x, y, dz, R, phi, rho, triangles, MaskRi):
    pos = np.where(R>MaskRi)
    x = x[pos]
    y = y[pos]
    dz = dz[pos]
    R = R[pos]
    phi = phi[pos]
    rho = rho[pos]
    
    triang = tri.Triangulation(x, y)
    mask = np.hypot(x[triang.triangles].mean(axis=1),
                              y[triang.triangles].mean(axis=1)) < MaskRi

    triangles = triang.triangles[~mask]
    return x, y, dz, R, phi, rho, triangles
            
        
def TipTilt(x,y,dz):
    A = np.ones((len(x),3))
    A[:,0] = 1.*A[:,0]
    A[:,1] = A[:,1] * x 
    A[:,2] = A[:,2] * y 
    
    #A[:,3] = np.sqrt(A[:,1]**2 + A[:,2]**2)
    
    Xlinear = np.dot(np.linalg.pinv(A),dz)
    Xlinear = np.linalg.lstsq(A,dz,rcond=None)[0] 
    Fit = np.sum(Xlinear*A,axis = 1)
    ddz = dz - Fit
    return ddz, Xlinear 


def sagsign(R,dz):
    Ri = np.linspace(np.min(R),np.max(R),100)
    f = interp1d(R, dz)
    dzi = f(Ri)
    ddz = np.diff(dzi)/np.diff(Ri)
    sign = np.sign(sum(ddz))
    return sign

def SFE_calc(dz,UnitFactor):
    SFE  = np.round(np.std(dz)  * UnitFactor,2)
    return SFE

def funcSphere(R, Rc, offset):
    C = 1/Rc
    return C*R**2/(1+np.sqrt(1-C**2*R**2)) + offset


def PV_calc(dz,UnitFactor):
    PV  = np.round( (np.max(dz)-np.min(dz)) * UnitFactor, 2)
    return PV    

def plotlyfunc(x,y,dz,mesh,UnitFactor,title): 

    SFE = str(SFE_calc(dz, UnitFactor))
    PV = str(PV_calc(dz, UnitFactor))
    
    fig = ff.create_trisurf(x=x, y=y, z=dz,
                          simplices=mesh,
                          aspectratio=dict(x=1.0, y=1.0, z=0.2),
                          colormap='Rainbow',
                          plot_edges = False,
                          backgroundcolor='rgb(255, 255, 255)',
                          gridcolor='rgb(0, 0, 0)',
                          title = title + '<br>' + 
                          'PV = ' + PV + 'nm' + '<br>' + 
                          'SFE = ' + SFE + 'nm RMS',
                          )
    fig.update_layout(
        title={
            'x':0.5,
            'xanchor': 'center'
            }
        )
    st.plotly_chart(fig, use_container_width=True)

def plotly_function(x,y,title):
    
    fsize = 18
    
    fig=go.Figure()
    fig = fig.add_trace(go.Scatter(x=x,y=y,mode = 'markers'))
    fig.update_layout(title_text=title, title_x=0.5,title_font_size=22)
    fig.update_layout(xaxis = dict(tickfont = dict(size=fsize)))
    fig.update_xaxes(title_font_size=fsize)
    fig.update_layout(yaxis = dict(tickfont = dict(size=fsize)))
    fig.update_yaxes(title_font_size=fsize)
    fig.update_layout(width=1000, height=1000)
    fig = fig.update_xaxes(title_text = 'X-coordinates')
    fig = fig.update_yaxes(title_text = 'Y-coordinates')
    fig.update_layout(legend = dict(font = dict(size = fsize, color = "black")))
    st.plotly_chart(fig,use_container_width=False)
    

def ZernikeTerms():
    mm = list(range(2,16))
    NN = []
    for i in range(len(mm)-1):
        NN.append(sum(range(mm[i+1])))   
    return NN, mm

    

def ZernikeDecomposition(rho,phi,m_max,N_Zernikes,dz,UnitFactor):
    A = [[0,0]]

    for i in range(1,m_max):
        for j in range(-i,i+1,2):
            A.append([j,i])
    mnlist = ['Z[' + str(A[0][0]) + ']' +'[' + str(A[0][1]) + ']']        
    for i in range(1,len(A)):
        mnlist.append('Z[' + str(A[i][0]) + ']' +'[' + str(A[i][1]) + ']')
    
    ZernikeInfluenceFunctions = np.zeros([len(rho),len(A)])
    for i in range(len(A)):
        
        m = A[i][0]
        n = A[i][1]
        k_inf = int(((n-abs(m))/2))
    
        Zs = np.zeros([len(rho),k_inf+1])
        
        if abs(m)-n == 0:
            k = 0
            F1 = math.factorial(n-k)
            F2 = math.factorial(k)
            F3 = math.factorial(int((n+abs(m))/2) - k )
            F4 = math.factorial(int((n-abs(m))/2) - k )
            Zs = (-1)**k*F1/(F2*F3*F4)*rho**(n-2*k)
        else:
            
            for k in range(int((n-abs(m))/2)+1):
                F1 = math.factorial(n-k)
                F2 = math.factorial(k)
                F3 = math.factorial(int((n+abs(m))/2) - k )
                F4 = math.factorial(int((n-abs(m))/2) - k )
                Ri = (-1)**k*F1/(F2*F3*F4)*rho**(n-2*k)
                Zs[:,k] = Ri  
            Zs = np.sum(Zs,axis=1)
        
        if m >= 0:    
            Zs = Zs.reshape(len(Zs))*np.cos(abs(m)*phi)
        else:
            Zs = Zs.reshape(len(Zs))*np.sin(abs(m)*phi)
            
        ZernikeInfluenceFunctions[:,i] = Zs
    
    if N_Zernikes == 4:
        ZernikeInfluenceFunctionsN = np.delete(ZernikeInfluenceFunctions, [3, 5], axis=1 )    
        A = A[:-2]
        ZernikeInfluenceFunctions = ZernikeInfluenceFunctionsN 
        
    #Xlinear = np.linalg.lstsq(ZernikeInfluenceFunctions,dz,rcond=None)[0] 
    Xlinear = np.dot(np.linalg.pinv(ZernikeInfluenceFunctions),dz)
    Zernikes = Xlinear*ZernikeInfluenceFunctions
    SFEs = np.round(np.std(Zernikes,axis=0) * UnitFactor,3)
    PVs = np.round((np.max(Zernikes,axis=0) - np.min(Zernikes,axis=0)) * UnitFactor,3)
    
    return Zernikes, ZernikeInfluenceFunctions, Xlinear, m, A, SFEs, PVs, mnlist


def ZernikeNamesFunc():
    ZernikeNames = [' Piston',' Tip',' Tilt',' Astigmatism 1', ' Defocus',' Astigmatism 2',' Trefoil 1',
                    ' Coma 1', ' Coma 2',' Trefoil 2',' ', ' ', ' Spherical Aberration']
    for i in range(1000):
        ZernikeNames.append(' ')
    return ZernikeNames        

def PistonTipTiltTableFunc(Xlinear, PTT):
    PistonTable = [str(np.format_float_scientific(PTT[0],precision=4))]
    
    TipTiltTable = [' ', str(np.format_float_scientific(PTT[1],precision=4)),str(np.format_float_scientific(PTT[2],precision=4))]

    for i in range(1,len(Xlinear)+4):
        PistonTable.append(' ')
    for i in range(3,len(Xlinear)+4):
        TipTiltTable.append(' ')
    return PistonTable, TipTiltTable

def ZernikeTableFunc(mnlist, ZernikeNames):
    ZernikeTable = []
    ZernikeNames = ZernikeNamesFunc()
    
    for i in range(len(mnlist)):
        ZernikeTable.append(str(mnlist[i])+ZernikeNames[i])
    ZernikeTable.append(' ')
    ZernikeTable.append('Original data:')
    ZernikeTable.append('Quadratic Sum Zernike Terms:')
    ZernikeTable.append('Residual error:')
        
    return ZernikeTable       