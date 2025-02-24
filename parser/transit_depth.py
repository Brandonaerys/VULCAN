# reads .vul data and plots a zeroth order of transit depth via cross-section times mixing ratio

# sample usage:python transit_depth.py GasDwarf.vul H2O,CH4,CO,N2,H2,CO2,NH3,H2S,HCN,CS2 GasDwarf_incomplete 1e-4 1e-1 300 5e3 1e4
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.legend as lg
import pickle
import os, sys
import pandas as pd

from mixing_ratios import mixing_ratios

# inputs: vul_data as .vul file, plot spec as single comma-separated string, min pressure as float, max pressure as float, temperature as float, plot_name as string
# outputs: array of species labels, array of mixing ratios
def transit_depth(vul_data,spec,plot_name,min_pressure_bar,max_pressure_bar,temp,min_wavenumber,max_wavenumber,mixing_plot_save=False,plot_save=True):


    data_dir = 'cross_section_data'

    spec, mix = mixing_ratios(vul_data, spec, f'{plot_name}', min_pressure_bar, max_pressure_bar=max_pressure_bar, use_range=True, plot_save=mixing_plot_save)


    # tex labels for plotting
    tex_labels = {'H':'H','H2':'H$_2$','O':'O','OH':'OH','H2O':'H$_2$O','CH':'CH','C':'C','CH2':'CH$_2$','CH3':'CH$_3$','CH4':'CH$_4$','HCO':'HCO','H2CO':'H$_2$CO', 'C4H2':'C$_4$H$_2$',\
    'C2':'C$_2$','C2H2':'C$_2$H$_2$','C2H3':'C$_2$H$_3$','C2H':'C$_2$H','CO':'CO','CO2':'CO$_2$','He':'He','O2':'O$_2$','CH3OH':'CH$_3$OH','C2H4':'C$_2$H$_4$','C2H5':'C$_2$H$_5$','C2H6':'C$_2$H$_6$','CH3O': 'CH$_3$O'\
    ,'CH2OH':'CH$_2$OH','N2':'N$_2$','NH3':'NH$_3$', 'NO2':'NO$_2$','HCN':'HCN','NO':'NO', 'NO2':'NO$_2$', 'H2S':'H$_2$S', 'CS2':'CS$_2$'}

    # wavenumbers (in cm^-1) corresponding to wavelengths of 1-10 microns
    # separation of 10 cm^-1
    # wavenumbers = np.arange(1e3, 1e4+1, 10)

    wavenumbers = np.arange(min_wavenumber, max_wavenumber, 10)
    df = pd.DataFrame({'wavenumber': wavenumbers})
    df['wavenumber'] = df['wavenumber'].astype(float)


    for i in range(len(spec)):
        sp = spec[i]
        try:
            if sp in tex_labels:
                label = tex_labels[sp]
            else:
                label = sp

            filename = f'{data_dir}/{spec[i]}/{int(temp)}.csv'
            cross = pd.read_csv(filename, sep=' ', header=None, names=['wavenumber', 'cross_section'], comment='#')
            cross = cross.astype(float)
            cross['cross_section'] *= mix[i]
            df = df.merge(cross, on='wavenumber', how='left')
            df.rename(columns={'cross_section': label}, inplace=True)




        except FileNotFoundError:
            print(f'{spec[i]} cross-section data not available at {temp}K')
            # print(f'attempted filename: {filename}')
        except Exception as e:
            print(f"Error for {sp}: {e}")



    if plot_save:
        plot_dir = '../parser_output/transit_depths'

        if not os.path.exists(plot_dir):
            print( 'Directory ' , plot_dir,  " created.")
            os.makedirs(plot_dir)


        plt.figure(figsize=(10, 6))

        for column in df.columns[1:]:
            plt.plot(df['wavenumber'], df[column], label=column, linewidth=1)

        df['max_value'] = df.iloc[:, 1:].max(axis=1)
        # plt.plot(df['wavenumber'], df['max_value'], color='black', linestyle='--', linewidth=1, label='Max')

        plt.xlabel('Wavenumber (cm$^{-1}$)')
        plt.ylabel('$\sigma n$')
        plt.title('Transit depth via mixing ratio times cross-section')
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(plot_dir, plot_name + '.png'))
        plt.savefig(os.path.join(plot_dir, plot_name + '.eps'))


        plt.show()


        return df




if __name__ == '__main__':
    use_range = True
    vul_data = sys.argv[1]
    spec = sys.argv[2]
    plot_name = sys.argv[3]
    min_pressure_bar = float(sys.argv[4])
    max_pressure_bar = float(sys.argv[5])
    temp = float(sys.argv[6])
    min_wavenumber = float(sys.argv[7])
    max_wavenumber = float(sys.argv[8])
    transit_depth(vul_data,spec,plot_name,min_pressure_bar,max_pressure_bar,temp,min_wavenumber,max_wavenumber,mixing_plot_save=False,plot_save=True)
