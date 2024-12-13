# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:08:43 2024

@author: vreugdjd
"""

import streamlit as st
import numpy as np
from scipy.optimize import curve_fit
import ZernikeFunctions as ZF
import matplotlib.pyplot as plt
import pandas as pd

SphereFit_opt = False
Asphere_M = False
ZernikeDecomposition_opt = False


st.set_page_config(layout="wide")
    
with st.sidebar:
    st.title('Zernike Decomposition Tool')
    st.write('info: jan.devreugd@tno.nl')
    
    uploaded_file = st.file_uploader("Select a datafile:")
    
    if uploaded_file is None:
        Testdat_opt = st.checkbox('Use test data')
          
        if Testdat_opt == True:
            Testdataset = st.radio('select data set', ('FE-data', 'Measurement Data'))
            if Testdataset == 'FE-data':
                uploaded_file = 'TestFile_FEM.txt'
            else:
                uploaded_file = 'TestFile_CMM.txt'
                
    if uploaded_file is not None:
        data, shapeFile = ZF.dataread(uploaded_file)

        st.write(' \# data points = ' + str(shapeFile[0]) + ', # columns = ' + str(shapeFile[1]) )
        units = st.radio('data units:', ('meters', 'millimeters'))
            
        if units == 'meters':
            UnitFactor = 1E9
        else:
            UnitFactor = 1E6

        x, y, dz, R, phi, rho, triangles = ZF.dataselection(data, shapeFile)       
        dzPTT, PTT = ZF.TipTilt(x, y, dz)
        
        Maskopt = st.checkbox('mask data')   
        if Maskopt == True:
            MaskRiS = st.number_input('Inner Radius Mask:',value = 0.1*np.max(R), step = 0.001, format="%.6f")
            x, y, dz, R, phi, rho, triangles = ZF.dataselectionMasked(x, y, dz, R, phi, rho, triangles, MaskRiS)
            dzPTT, PTT = ZF.TipTilt(x, y, dz)
          
                            
with st.expander('Plot original data + Piston Tip Tilt removal:', expanded=True): 
    if uploaded_file is not None:

        col1, col2 = st.columns(2)
        with col1:
            ZF.plotlyfunc(x, y, dz, triangles, UnitFactor, 'orginal data')
            SFE = str(ZF.SFE_calc(dz, UnitFactor))
            PV = str(ZF.PV_calc(dz, UnitFactor))
            fig, ax = plt.subplots(figsize=(6, 3))
            pc = ax.tripcolor(x, y, triangles, dz, linewidth=0.0, antialiased=False, cmap=plt.cm.jet, shading='gouraud')
            ax.set_aspect('equal', adjustable='box')
            ax.set_title('SFE' +
                         '\nPV = ' + PV + ' nm' +
                         '\nSFE = ' + SFE + ' nm RMS'
                         )
            fig.colorbar(pc)
            st.pyplot(fig)
        with col2:
            ZF.plotlyfunc(x, y, dzPTT, triangles, UnitFactor, 'orignal data - {piston, tip, tilt}')
            SFE = str(ZF.SFE_calc(dzPTT, UnitFactor))
            PV = str(ZF.PV_calc(dzPTT, UnitFactor))
            fig, ax = plt.subplots(figsize=(6, 3))
            pc = ax.tripcolor(x, y, triangles, dzPTT, linewidth=0.0, antialiased=False, cmap=plt.cm.jet, shading='gouraud')
            ax.set_aspect('equal', adjustable='box')
            ax.set_title('SFE ex. PTT' +
                         '\nPV = ' + PV + ' nm' +
                         '\nSFE = ' + SFE + ' nm RMS'
                         )
            fig.colorbar(pc)
            st.pyplot(fig)
         
        with st.sidebar:
            SphereFit_opt = st.checkbox('Calculate best fitting sphere and asphere')
            Asphere_M = st.checkbox('Subtract Asphere shape from original data')
            ZernikeDecomposition_opt = st.checkbox('Zernike decompostion')
            if Asphere_M == True:
                Radius_User = st.number_input('Radius of Curvature:',value = np.max(R)*2,step = 0.001,format="%.6f")
                Kappa_User = st.number_input('Conical constant:',step = 0.0001,format="%.6f")
                
            if ZernikeDecomposition_opt == True:
                NN, mm = ZF.ZernikeTerms()
                NN.append(4)
                NN.sort()
                default_NN = NN.index(6)
                
                mm.append(3)
                mm.sort()
                
                N_Zernikes = st.selectbox('# Zernike terms: ', NN, index = default_NN)
                index = NN.index(N_Zernikes)  
                m_max = mm[index]
                # SortZernikes_opt = st.checkbox('Sort Zernikes',True)
                if (SphereFit_opt==False) and (Asphere_M == False):
                    ZernikeOption = 'Original Data'
                if (SphereFit_opt==True) and (Asphere_M == False):
                    ZernikeOption = st.selectbox('Zernike Decomposition on:',('Original Data', 'Original data - Best Fit Sphere','Original Data - Best Fit A-Sphere'))                
                if (Asphere_M==True) and (SphereFit_opt==False):
                    ZernikeOption = st.selectbox('Zernike Decomposition on:',('Original Data', 'Original data - UserDefined A-Sphere')) 
                if (SphereFit_opt==True) and (Asphere_M==True):
                    ZernikeOption = st.selectbox('Zernike Decomposition on:',('Original Data', 'Original data - Best Fit Sphere','Original Data - Best Fit A-Sphere','Original data - UserDefined A-Sphere'))    
                        
                
if SphereFit_opt == True:   
    initial_guess = [ZF.sagsign(R,dz)*np.max(R)*10, -1]
    parsS, pcovS = curve_fit(ZF.funcSphere, R, dzPTT, p0=initial_guess)
    fitSphere = ZF.funcSphere(R,parsS[0],parsS[1])
    dzSphFit =  dzPTT-fitSphere
    
    initial_guess = [ZF.sagsign(R,dz)*np.max(R)*10, 0., -1]
    parsAS, pcovAS = curve_fit(ZF.funcASphere, R, dzPTT, p0=initial_guess)
    fitASphere = ZF.funcASphere(R,parsAS[0],parsAS[1],parsAS[2])
    dzASphFit =  dzPTT-fitASphere
            
    with st.expander('Best fitting (A-) Sphere Removed'):
        col1, col2 = st.columns(2)
        with col1:
            ZF.plotlyfunc(x, y, dzSphFit, triangles, UnitFactor, 
                          'Original Data - Sphere' +  '<br>' + f'best fitting Sphere Radius = {parsS[0]:.4f} ' + units)
        with col2:
            ZF.plotlyfunc(x, y, dzASphFit, triangles, UnitFactor, 
                          'Original Data - ASphere' + '<br>' + f'best fitting Asphere Radius = {parsAS[0]:.4f}' + units + '<br>'
                          f'best fitting conical constant = {parsAS[1]:.3f}')
            
if Asphere_M == True:
    with st.expander('Asphere Removed'):
        col1, col2 = st.columns(2)
        with col1:
            ZF.plotlyfunc(x, y, dzPTT, triangles, UnitFactor, 
                          'Original Data - {piston, tip, tilt}')
        with col2:
            ASphereM = ZF.funcASphere(R,Radius_User,Kappa_User,0)
            ZF.plotlyfunc(x, y, dzPTT-ASphereM , triangles, UnitFactor, 
                          'Original Data - {Asphere}' + '<br>' + 
                          f'selected Asphere radius = {Radius_User:.4f} [m]' + '<br>' + 
                          f'selected conical constant = {Kappa_User:.4f}')
            
if ZernikeDecomposition_opt:
    if ZernikeOption == 'Original Data':
        data4Zernike = dz
    if ZernikeOption == 'Original data - Best Fit Sphere':
        data4Zernike = dzSphFit
    if ZernikeOption == 'Original Data - Best Fit A-Sphere':
        data4Zernike = dzASphFit
    if ZernikeOption == 'Original data - UserDefined A-Sphere':
        data4Zernike = dzPTT-ASphereM
    
    Zernikes, ZernikeInfluenceFunctions, Xlinear, m, ZernikeModeNames, SFEs, PVs, mnlist = ZF.ZernikeDecomposition(rho, phi, m_max, N_Zernikes, data4Zernike, UnitFactor)
    ZernikeNames = ZF.ZernikeNamesFunc()
    ZernikeTable = ZF.ZernikeTableFunc(mnlist, ZernikeNames)
    if N_Zernikes == 4:
        ZernikeTable.pop(3)
        ZernikeTable.pop(4)
        
        ZernikeNames.pop(3)
        ZernikeNames.pop(4)

    ZernikesSum = np.sum(Zernikes,axis = 1)
    ZernikeDelta = data4Zernike - ZernikesSum    

    with st.expander('Selected data minus summation of Zernikes'):
        ZernikesSum = np.sum(Zernikes,axis = 1)
        ZernikeDelta = data4Zernike - ZernikesSum
        
        col1, col2 = st.columns(2)
        with col1:
            #plotlyfunc(x,y,xi,yi,data4Zernike,UnitFactor, ZernikeOption)
            ZF.plotlyfunc(x,y,data4Zernike,triangles,UnitFactor, ZernikeOption)
        with col2:    
            #plotlyfunc(x,y,xi,yi,ZernikeDelta,UnitFactor, '(' + ZernikeOption + ')' + ' minus Zernikes:') 
            ZF.plotlyfunc(x,y,ZernikeDelta,triangles,UnitFactor,  '(' + ZernikeOption + ')' + ' minus Zernikes:')
    
    with st.expander('Zernike decompostion plots, sorted'):
        col1, col2,col3,col4,col5,col6 = st.columns(6)
        H = [col1,col2,col3,col4,col5,col6]
        
        for j in range(len(ZernikeModeNames)):
            i = np.argsort(SFEs)[-1-j]
            plt.figure(i+1)
            
            fig,ax = plt.subplots(figsize=(6,3))
            pc = ax.tripcolor(x, y, triangles, ZernikeInfluenceFunctions[:,i], linewidth=0.0, antialiased=False, cmap=plt.cm.jet, shading='gouraud')
            ax.set_aspect('equal', adjustable='box')
            ax.set_title('Zernike Mode: '+ ZernikeNames[i]  + '\n ' + 
                         'n=' + str(ZernikeModeNames[i][1]) + ' m=' + str(ZernikeModeNames[i][0]) +
                         '\nPV = ' + str(PVs[i]) + ' nm' +
                         '\nSFE = ' + str(SFEs[i]) + ' nm RMS'
                         )
            with H[j%6]:
                st.pyplot(fig) 
                

                
    with st.expander('Zernike Table'):
        PistonTable, TipTiltTable = ZF.PistonTipTiltTableFunc(Xlinear,PTT)
        SFEColumn = SFEs
        SFEColumn = np.append(SFEColumn, ' ')
        SFEColumn = np.append(SFEColumn,  str(np.round(np.std(dz)*UnitFactor,3))    )
        SFEColumn = np.append(SFEColumn,  np.round(np.sum(np.sqrt(np.sum(SFEs**2))),3) )
        SFEColumn = np.append(SFEColumn,  str(np.round(np.std(ZernikeDelta)*UnitFactor,3))    )
        
        PVs = np.append(PVs, ' ')
        PVs = np.append(PVs, str(np.round((np.max(dz) - np.min(dz))*UnitFactor , 3)) )
        PVs = np.append(PVs, ' ' )
        PVs = np.append(PVs, str(np.round((np.max(ZernikeDelta) - np.min(ZernikeDelta))*UnitFactor , 3)) )
        
        if units == 'meters':
            dfTable = pd.DataFrame({'Zernike Mode:' : ZernikeTable, 'PV [nm]' : PVs, 'SFE [nm RMS]:' : SFEColumn, 'Piston [m]:' : PistonTable, 'Tip Tilt angle [rad]:' : TipTiltTable}) 
        elif units == 'millimeters':
            dfTable = pd.DataFrame({'Zernike Mode:' : ZernikeTable, 'PV [nm]' : PVs, 'SFE [nm RMS]:' : SFEColumn, 'Piston [mm]:' : PistonTable, 'Tip Tilt angle [rad]:' : TipTiltTable}) 
        st.write(dfTable) 

        #st.table(dfTable.style)

if uploaded_file is not None:
    with st.expander('X and Y locations of all datapoints'):
        ZF.plotly_function(x,y,'data coordinates')