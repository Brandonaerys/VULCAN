'''
This script reads VULCAN output (.vul) files using pickle and plot the species volumn mixing ratios as a function of pressure, with the initial abundances (typically equilibrium) shown in dashed lines.
Plots are saved in the folder assigned in vulcan_cfg.py, with the default plot_dir = 'plot/'.
'''

import sys
sys.path.insert(0, '../') # including the upper level of directory for the path of modules

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.legend as lg
import vulcan_cfg
try: from PIL import Image
except ImportError:
    try: import Image
    except: vulcan_cfg.use_PIL = False
import os, sys
import pickle

# swtich for plot
if '-h' in sys.argv: use_height = True
else: use_height = False

def parser(vul_data,plot_spec,plot_pressure,plot_name):
    plot_dir = '../' + 'parser_output'
    # Checking if the plot folder exsists
    if not os.path.exists(plot_dir):
        print( 'Directory ' , plot_dir,  " created.")
        os.mkdir(plot_dir)

    # taking user input species and splitting into separate strings and then converting the list to a tuple
    plot_spec = tuple(plot_spec.split(','))
    nspec = len(plot_spec)

    # These are the "Tableau 20" colors as RGB.
    tableau20 = [(31, 119, 180),(255, 127, 14),(44, 160, 44),(214, 39, 40),(148, 103, 189),(140, 86, 75), (227, 119, 194),(127, 127, 127),(188, 189, 34),(23, 190, 207),\
    (174, 199, 232),(255, 187, 120),(152, 223, 138),(255, 152, 150),(197, 176, 213),(196, 156, 148),(247, 182, 210),(199, 199, 199),(219, 219, 141),(158, 218, 229)]
    #


    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    # tex labels for plotting
    tex_labels = {'H':'H','H2':'H$_2$','O':'O','OH':'OH','H2O':'H$_2$O','CH':'CH','C':'C','CH2':'CH$_2$','CH3':'CH$_3$','CH4':'CH$_4$','HCO':'HCO','H2CO':'H$_2$CO', 'C4H2':'C$_4$H$_2$',\
    'C2':'C$_2$','C2H2':'C$_2$H$_2$','C2H3':'C$_2$H$_3$','C2H':'C$_2$H','CO':'CO','CO2':'CO$_2$','He':'He','O2':'O$_2$','CH3OH':'CH$_3$OH','C2H4':'C$_2$H$_4$','C2H5':'C$_2$H$_5$','C2H6':'C$_2$H$_6$','CH3O': 'CH$_3$O'\
    ,'CH2OH':'CH$_2$OH','N2':'N$_2$','NH3':'NH$_3$', 'NO2':'NO$_2$','HCN':'HCN','NO':'NO', 'NO2':'NO$_2$' }

    with open(vul_data, 'rb') as handle:
      data = pickle.load(handle)

    vulcan_spec = data['variable']['species']
    pressure_array = data['atm']['pco'] / 1e6  # Convert to bar

    # Find the closest pressure index
    closest_idx = np.argmin(np.abs(pressure_array - plot_pressure))

    print(f"Plotting data at closest pressure: {pressure_array[closest_idx]:.3e} bar")

    # Prepare data for bar chart
    mixing_ratios = []
    species_labels = []

    for sp in plot_spec:
        if sp in vulcan_spec:
            mixing_ratios.append(data['variable']['ymix'][closest_idx, vulcan_spec.index(sp)])
            species_labels.append(sp)
        else:
            print(f"Warning: {sp} not found in the dataset.")

    # Plot bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(species_labels, mixing_ratios, log=True)

    plt.xlabel("Species")
    plt.ylabel("Mixing Ratio")
    plt.title(f"Mixing Ratios at {pressure_array[closest_idx]:.3e} bar")
    plt.xticks(rotation=45, ha="right")
    plt.yscale("log")  # Log scale for better visualization

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, plot_name + '.png'))
    plt.savefig(os.path.join(plot_dir, plot_name + '.eps'))

    if vulcan_cfg.use_PIL:
        Image.open(os.path.join(plot_dir, plot_name + '.png')).show()
    else:
        plt.show()





if __name__ == '__main__':
    vul_data = sys.argv[1]
    plot_spec = sys.argv[2]
    plot_pressure = float(sys.argv[3])
    plot_name = sys.argv[4]
    parser(vul_data,plot_spec,plot_pressure,plot_name)
