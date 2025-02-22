# reads .vul data and plots a zeroth-order transit depth via mixingratio times cross section

# sample usage:
# python mixing_ratios.py ../output/GasDwarf.vul H2O,CH4,CO,N2,H2,CO2,NH3,H2S,HCN,CS2 GasDwarf_incomplete 1e-4 1e-1 -r
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



# inputs: vul_data as .vul file, plot spec as single comma-separated string, pressures as list of min/max pressure, plot_name as string

def mixing_ratios(vul_data,spec,plot_name,min_pressure_bar,max_pressure_bar=1,use_range=True, plot_save=True):



    plot_dir = '../' + 'parser_output'
    # Checking if the plot folder exsists
    if not os.path.exists(plot_dir):
        print( 'Directory ' , plot_dir,  " created.")
        os.mkdir(plot_dir)

    # taking user input species and splitting into separate strings and then converting the list to a tuple
    spec = tuple(spec.split(','))
    nspec = len(spec)

    # These are the "Tableau 20" colors as RGB.
    tableau20 = [(31, 119, 180),(255, 127, 14),(44, 160, 44),(214, 39, 40),(148, 103, 189),(140, 86, 75), (227, 119, 194),(127, 127, 127),(188, 189, 34),(23, 190, 207),\
    (174, 199, 232),(255, 187, 120),(152, 223, 138),(255, 152, 150),(197, 176, 213),(196, 156, 148),(247, 182, 210),(199, 199, 199),(219, 219, 141),(158, 218, 229)]
    #


    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    # # tex labels for plotting
    # tex_labels = {'H':'H','H2':'H$_2$','O':'O','OH':'OH','H2O':'H$_2$O','CH':'CH','C':'C','CH2':'CH$_2$','CH3':'CH$_3$','CH4':'CH$_4$','HCO':'HCO','H2CO':'H$_2$CO', 'C4H2':'C$_4$H$_2$',\
    # 'C2':'C$_2$','C2H2':'C$_2$H$_2$','C2H3':'C$_2$H$_3$','C2H':'C$_2$H','CO':'CO','CO2':'CO$_2$','He':'He','O2':'O$_2$','CH3OH':'CH$_3$OH','C2H4':'C$_2$H$_4$','C2H5':'C$_2$H$_5$','C2H6':'C$_2$H$_6$','CH3O': 'CH$_3$O'\
    # ,'CH2OH':'CH$_2$OH','N2':'N$_2$','NH3':'NH$_3$', 'NO2':'NO$_2$','HCN':'HCN','NO':'NO', 'NO2':'NO$_2$' }

    with open(vul_data, 'rb') as handle:
      data = pickle.load(handle)

    vulcan_spec = data['variable']['species']
    pressure_array = data['atm']['pco'] / 1e6  # Convert to bar


    if use_range:
        pressure_mask = (pressure_array >= min_pressure_bar) & (pressure_array <= max_pressure_bar)
        selected_indices = np.where(pressure_mask)[0]

        if len(selected_indices) == 0:
            print(f"Error: No data points found in the pressure range {min_pressure_bar} - {max_pressure_bar} bar.")
            sys.exit(1)
        print(f"Plotting data averaged over pressure range: {min_pressure_bar:.3e} - {max_pressure_bar:.3e} bar")
    else:
        # Find the closest pressure index
        closest_idx = np.argmin(np.abs(pressure_array - min_pressure_bar))
        selected_indices = [closest_idx]

        print(f"Plotting data at closest pressure: {pressure_array[closest_idx]:.3e} bar")

    # Prepare data for bar chart
    mixing_ratios = []
    species_labels = []

    for sp in spec:
        if sp in vulcan_spec:
            avg_ratio = np.mean(data['variable']['ymix'][selected_indices, vulcan_spec.index(sp)])
            mixing_ratios.append(avg_ratio)
            species_labels.append(sp)
        else:
            print(f"Warning: {sp} not found in the dataset.")

    # Plot bar chart
    if plot_save:
        plt.figure(figsize=(10, 6))
        plt.bar(species_labels, mixing_ratios, log=True)

        plt.xlabel("Species")
        plt.ylabel("Mixing Ratio" if not use_range else "Averaged Mixing Ratio")
        plt.title(f"Mixing Ratios at {pressure_array[closest_idx]:.3e} bar" if not use_range else f"Averaged Mixing Ratios between {min_pressure_bar:.3e} - {max_pressure_bar:.3e} bar")
        plt.xticks(rotation=45, ha="right")
        plt.yscale("log")  # Log scale for better visualization

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, plot_name + '.png'))
        plt.savefig(os.path.join(plot_dir, plot_name + '.eps'))

        if vulcan_cfg.use_PIL:
            Image.open(os.path.join(plot_dir, plot_name + '.png')).show()
        else:
            plt.show()


        return species_labels, mixing_ratios




if __name__ == '__main__':
    if '-r' in sys.argv:
        use_range = True
        vul_data = sys.argv[1]
        spec = sys.argv[2]
        plot_name = sys.argv[3]
        min_pressure = float(sys.argv[4])
        max_pressure = float(sys.argv[5])
        mixing_ratios(vul_data,spec,plot_name,min_pressure,max_pressure_bar=max_pressure,use_range=use_range)
    else:
        use_range = False
        vul_data = sys.argv[1]
        spec = sys.argv[2]
        plot_name = sys.argv[3]
        pressure = float(sys.argv[4])
        mixing_ratios(vul_data,spec,plot_name,pressure,use_range=use_range)
