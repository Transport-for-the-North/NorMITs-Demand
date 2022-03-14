# -*- coding: utf-8 -*-
"""
Created on Thu May 20 13:14:17 2021

@author: Skyrim
"""

import os
from os import path
import subprocess
from io import StringIO
import pandas as pd
import re

import os
import pandas as pd

import normits_demand as nd

# TODO: Class wrapper
# TODO: Search for Saturn instance
# TODO: Classify saturn instance
# TODO: Move functionality to normits_demand.matrices.ufm_converter

class CsvToUFM:

    _default_csv_path = r'J:\IPBA\Highway Model\3.Future_Scenarios\EFS FTS Matrices\iter3g2 (With elasticity)'
    _default_sat_exes = r'C:\Program Files (x86)\Atkins\SATURN\XEXES 11.5.05J Beta MC N4'

    def __init__(self):
        # TODO: Init
        print('')

    @staticmethod
    def build_io_paths(self,
                       wd: nd.PathLike = _default_csv_path,
                       sat_exes: nd.PathLike = _default_sat_exes):
        return wd, sat_exes

    def satwCSV2UFM(satEXES, csvFile):
        toWrite = []
        toWrite.append(
            'Path = C:\WINDOWS\System32;C:\WINDOWS\System;C:\WINDOWS\Command;"' + satEXES + '";"' + satEXES + '\\BATS"')
        toWrite.append('SETLOCAL')
        toWrite.append('@echo off')
        toWrite.append('cls')
        toWrite.append('if exist "' + csvFile + '.key" del "' + csvFile + '.key"')
        toWrite.append('if exist "' + csvFile + '.ufm" del "' + csvFile + '.ufm"')
        toWrite.append('echo.           1   >' + csvFile + '.key')
        toWrite.append('echo.' + csvFile + '_MX.csv >>' + csvFile + '.key')
        #    toWrite.append('echo.Y             >>'+csvFile+'.key') #Not used in Saturn version 11.5.05
        toWrite.append('echo.           2  >>' + csvFile + '.key')
        toWrite.append('echo.           7  >>' + csvFile + '.key')
        toWrite.append('echo.           1  >>' + csvFile + '.key')
        toWrite.append('echo.          14  >>' + csvFile + '.key')
        toWrite.append('echo.           1  >>' + csvFile + '.key')
        toWrite.append('echo.' + csvFile + '.ufm >>' + csvFile + '.key')
        toWrite.append('echo.' + csvFile + '   >>' + csvFile + '.key')
        toWrite.append('echo.           0  >>' + csvFile + '.key')
        toWrite.append('echo.           0  >>' + csvFile + '.key')
        toWrite.append('echo.Y             >>' + csvFile + '.key')
        toWrite.append('call MXX I KEY ' + csvFile + ' vdu vdu')
        toWrite.append('if exist "' + csvFile + '.key" del "' + csvFile + '.key"')
        toWrite.append('if exist "' + csvFile + '_MX.csv" del "' + csvFile + '_MX.csv"')
        toWrite.append('if exist "' + csvFile + '.LPX" del "' + csvFile + '.LPX"')
        toWrite.append('if exist *.vdu del *.vdu')
        toWrite.append('if exist MX*.log del MX*.log')
        toWrite.append('ENDLOCAL')
        toWrite.append('GOTO :EOF')
        with open(csvFile + '.bat', 'w') as bat:
            for line in toWrite:
                print(line, file=bat)
        #    procSingle(['call "'+csvFile+'.bat"'])
        procSingle(['call "' + csvFile + '.bat"', 'del "' + csvFile + '.bat"'])

    def satwCSV2UFM_11407(satEXES, csvFile):
        toWrite = []
        toWrite.append(
            'Path = C:\WINDOWS\System32;C:\WINDOWS\System;C:\WINDOWS\Command;"' + satEXES + '";"' + satEXES + '\\BATS"')
        toWrite.append('SETLOCAL')
        toWrite.append('@echo off')
        toWrite.append('cls')
        toWrite.append('if exist "' + csvFile + '.key" del "' + csvFile + '.key"')
        toWrite.append('if exist "' + csvFile + '.ufm" del "' + csvFile + '.ufm"')
        toWrite.append('echo.           1   >' + csvFile + '.key')
        toWrite.append('echo.' + csvFile + '_MX.csv >>' + csvFile + '.key')
        toWrite.append('echo.Y             >>' + csvFile + '.key')
        toWrite.append('echo.           2  >>' + csvFile + '.key')
        toWrite.append('echo.           7  >>' + csvFile + '.key')
        toWrite.append('echo.           1  >>' + csvFile + '.key')
        toWrite.append('echo.          14  >>' + csvFile + '.key')
        toWrite.append('echo.           1  >>' + csvFile + '.key')
        toWrite.append('echo.' + csvFile + '.ufm >>' + csvFile + '.key')
        toWrite.append('echo.' + csvFile + '   >>' + csvFile + '.key')
        toWrite.append('echo.           0  >>' + csvFile + '.key')
        toWrite.append('echo.           0  >>' + csvFile + '.key')
        toWrite.append('echo.Y             >>' + csvFile + '.key')
        toWrite.append('call MXX I KEY ' + csvFile + ' vdu vdu')
        toWrite.append('if exist "' + csvFile + '.key" del "' + csvFile + '.key"')
        toWrite.append('if exist "' + csvFile + '_MX.csv" del "' + csvFile + '_MX.csv"')
        toWrite.append('if exist "' + csvFile + '.LPX" del "' + csvFile + '.LPX"')
        toWrite.append('if exist *.vdu del *.vdu')
        toWrite.append('if exist MX*.log del MX*.log')
        toWrite.append('ENDLOCAL')
        toWrite.append('GOTO :EOF')
        with open(csvFile + '.bat', 'w') as bat:
            for line in toWrite:
                print(line, file=bat)
        #    procSingle(['call "'+csvFile+'.bat"'])
        procSingle(['call "' + csvFile + '.bat"', 'del "' + csvFile + '.bat"'])

    def satwCSV2UFM_square(satEXES, csvFile):
        toWrite = []
        toWrite.append(
            'Path = C:\WINDOWS\System32;C:\WINDOWS\System;C:\WINDOWS\Command;"' + satEXES + '";"' + satEXES + '\\BATS"')
        toWrite.append('SETLOCAL')
        toWrite.append('@echo off')
        toWrite.append('cls')
        toWrite.append('if exist "' + csvFile + '.key" del "' + csvFile + '.key"')
        toWrite.append('if exist "' + csvFile + '.ufm" del "' + csvFile + '.ufm"')
        toWrite.append('echo.           1   >' + csvFile + '.key')
        toWrite.append('echo.' + csvFile + '_MX.csv >>' + csvFile + '.key')
        #    toWrite.append('echo.Y             >>'+csvFile+'.key')
        toWrite.append('echo.           1  >>' + csvFile + '.key')
        toWrite.append('echo.           5  >>' + csvFile + '.key')
        toWrite.append('echo.           0  >>' + csvFile + '.key')
        toWrite.append('echo.          14  >>' + csvFile + '.key')
        toWrite.append('echo.           1  >>' + csvFile + '.key')
        toWrite.append('echo.' + csvFile + '.ufm >>' + csvFile + '.key')
        toWrite.append('echo.' + csvFile + '   >>' + csvFile + '.key')
        toWrite.append('echo.           0  >>' + csvFile + '.key')
        toWrite.append('echo.           0  >>' + csvFile + '.key')
        toWrite.append('echo.Y             >>' + csvFile + '.key')
        toWrite.append('call MX I KEY ' + csvFile + ' vdu vdu')
        toWrite.append('if exist "' + csvFile + '.key" del "' + csvFile + '.key"')
        toWrite.append('if exist "' + csvFile + '_MX.csv" del "' + csvFile + '_MX.csv"')
        toWrite.append('if exist "' + csvFile + '.LPX" del "' + csvFile + '.LPX"')
        toWrite.append('if exist *.vdu del *.vdu')
        toWrite.append('if exist MX*.log del MX*.log')
        toWrite.append('ENDLOCAL')
        toWrite.append('GOTO :EOF')
        with open(csvFile + '.bat', 'w') as bat:
            for line in toWrite:
                print(line, file=bat)
        procSingle(['call "' + csvFile + '.bat"', 'del "' + csvFile + '.bat"'])

    def satwCSV2UFM_square_11407(satEXES, csvFile):
        toWrite = []
        toWrite.append(
            'Path = C:\WINDOWS\System32;C:\WINDOWS\System;C:\WINDOWS\Command;"' + satEXES + '";"' + satEXES + '\\BATS"')
        toWrite.append('SETLOCAL')
        toWrite.append('@echo off')
        toWrite.append('cls')
        toWrite.append('if exist "' + csvFile + '.key" del "' + csvFile + '.key"')
        toWrite.append('if exist "' + csvFile + '.ufm" del "' + csvFile + '.ufm"')
        toWrite.append('echo.           1   >' + csvFile + '.key')
        toWrite.append('echo.' + csvFile + '_MX.csv >>' + csvFile + '.key')
        toWrite.append('echo.Y             >>' + csvFile + '.key')
        toWrite.append('echo.           1  >>' + csvFile + '.key')
        toWrite.append('echo.           5  >>' + csvFile + '.key')
        toWrite.append('echo.           0  >>' + csvFile + '.key')
        toWrite.append('echo.          14  >>' + csvFile + '.key')
        toWrite.append('echo.           1  >>' + csvFile + '.key')
        toWrite.append('echo.' + csvFile + '.ufm >>' + csvFile + '.key')
        toWrite.append('echo.' + csvFile + '   >>' + csvFile + '.key')
        toWrite.append('echo.           0  >>' + csvFile + '.key')
        toWrite.append('echo.           0  >>' + csvFile + '.key')
        toWrite.append('echo.Y             >>' + csvFile + '.key')
        toWrite.append('call MX I KEY ' + csvFile + ' vdu vdu')
        toWrite.append('if exist "' + csvFile + '.key" del "' + csvFile + '.key"')
        toWrite.append('if exist "' + csvFile + '_MX.csv" del "' + csvFile + '_MX.csv"')
        toWrite.append('if exist "' + csvFile + '.LPX" del "' + csvFile + '.LPX"')
        toWrite.append('if exist *.vdu del *.vdu')
        toWrite.append('if exist MX*.log del MX*.log')
        toWrite.append('ENDLOCAL')
        toWrite.append('GOTO :EOF')
        with open(csvFile + '.bat', 'w') as bat:
            for line in toWrite:
                print(line, file=bat)
        procSingle(['call "' + csvFile + '.bat"', 'del "' + csvFile + '.bat"'])

    def CSV2UFM(self,
                wd=r'C:\Users\DataAnalytics\Documents\Python_Scripts\Saturn\Example',
                satEXES=r'C:\Program Files (x86)\Atkins\SATURN\XEXES 11.5.05J Beta MC N4',
                csvFile=[]):
        """
        Describe params here
        """
        os.chdir(wd)

        for file in csvFile:
            # satwCSV2UFM(satEXES,file,Square)
            if os.path.isfile(file + ".csv") == False:
                print('The CSV file does not exist in this location')
            else:
                File = pd.read_csv(file + '.csv', header=None, low_memory=False)
                if type(File.at[0, 0]) == str or pd.isna(File.iloc[0, 0]):
                    print('Contains column headings')
                    File = File[1:]
                    File.to_csv(file + '_MX.csv', sep=',', index=False, header=None,
                                float_format='%.6f')
                else:
                    print('Does not contain column headings')
                    File.to_csv(file + '_MX.csv', sep=',', index=False, header=None,
                                float_format='%.6f')

                if len(File.columns) == 3:
                    print('  .. Long matrix (TUBA 2) - Convert to UFM...')
                    if satEXES == r'C:\Program Files (x86)\Atkins\SATURN\XEXES 11.4.07H MC N4':
                        satwCSV2UFM_11407(satEXES, file)
                    else:
                        satwCSV2UFM(satEXES, file)

                elif len(File.columns) == len(File.index) + 1:
                    print('  .. Square matrix with row zone names - Convert to UFM...')
                    if satEXES == r'C:\Program Files (x86)\Atkins\SATURN\XEXES 11.4.07H MC N4':
                        satwCSV2UFM_square_11407(satEXES, file)
                    else:
                        satwCSV2UFM_square(satEXES, file)
                #                 # Rename col:
                #
                #                 File = File.rename(columns ={0 : 'o_zone'})
                #                 # Convert to long format
                #                 File = pd.melt(File,
                #                        id_vars=['o_zone'],
                #                        var_name='d_zone',
                #                        value_name='demand',
                #                        col_level=0)
                #
                #                 File = File.sort_values(['o_zone', 'd_zone']).reset_index(drop=True)
                #                 File.to_csv(file+'_MX.csv',sep=',', index = False, header = None, float_format='%.10f')
                #                 satwCSV2UFM(satEXES,file)

                elif len(File.columns) == len(File.index):
                    print("Add zone names along the LHS of the square matrix m8")

                else:
                    print("Neither square nor long style matrix format m8")

        return ('Done')

    def main(self):
        wd, sat_exes = self.build_io_paths()

        ###########
        # SLA_Matrix_OD:
        ###########

        #TODO
        # potentially loop over time periods?

        # Run SLA to produce OD Matrix. UFO created and used in process.

        sat.CSV2UFM(wd,
                    satEXES,
                    csvFile = ['od_business_yr2050_m3_tp1',
                               'od_business_yr2050_m3_tp2',
                               'od_business_yr2050_m3_tp3',
                               'od_commute_yr2050_m3_tp1',
                               'od_commute_yr2050_m3_tp2',
                               'od_commute_yr2050_m3_tp3',
                               'od_other_yr2050_m3_tp1',
                               'od_other_yr2050_m3_tp2',
                               'od_other_yr2050_m3_tp3']
                                )