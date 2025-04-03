# reads .vul data and plots mixing ratios of given species at a specified pressure, or the averaged mixing ratios in a specified pressure range

# sample usage:
# python mixing_ratios.py GasDwarf.vul H2O,CH4,CO,N2,H2,CO2,NH3,H2S,HCN,CS2 GasDwarf_incomplete 1e-4 1e-1
# -s option for single value of pressure

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.legend as lg
import pickle
import os, sys

# inputs: vul_data as .vul file, plot spec as single comma-separated string, min pressure as float, max pressure as float, plot_name as string
# outputs: array of species labels, array of mixing ratios
def mixing_ratios(vul_data,spec,plot_name,min_pressure_bar,max_pressure_bar=1,use_range=True, plot_save=True):


    # taking user input species and splitting into separate strings and then converting the list to a tuple
    spec = tuple(spec.split(','))
    nspec = len(spec)


    # tex labels for plotting
    tex_labels = {'H':'H','H2':'H$_2$','O':'O','OH':'OH','H2O':'H$_2$O','CH':'CH','C':'C','CH2':'CH$_2$','CH3':'CH$_3$','CH4':'CH$_4$','HCO':'HCO','H2CO':'H$_2$CO', 'C4H2':'C$_4$H$_2$',\
    'C2':'C$_2$','C2H2':'C$_2$H$_2$','C2H3':'C$_2$H$_3$','C2H':'C$_2$H','CO':'CO','CO2':'CO$_2$','He':'He','O2':'O$_2$','CH3OH':'CH$_3$OH','C2H4':'C$_2$H$_4$','C2H5':'C$_2$H$_5$','C2H6':'C$_2$H$_6$','CH3O': 'CH$_3$O'\
    ,'CH2OH':'CH$_2$OH','N2':'N$_2$','NH3':'NH$_3$', 'NO2':'NO$_2$','HCN':'HCN','NO':'NO', 'NO2':'NO$_2$', 'H2S':'H$_2$S', 'CS2':'CS$_2$'}

    vul_file = f'../output/{vul_data}'
    with open(vul_file, 'rb') as handle:
      data = pickle.load(handle)

    vulcan_spec = data['variable']['species']
    pressure_array = data['atm']['pco'] / 1e6  # Convert to bar


    if use_range:
        pressure_mask = (pressure_array >= min_pressure_bar) & (pressure_array <= max_pressure_bar)
        selected_indices = np.where(pressure_mask)[0]

        if len(selected_indices) == 0:
            print(f"Error: No data points found in the pressure range {min_pressure_bar} - {max_pressure_bar} bar.")
            sys.exit(1)
        print(f"data averaged over pressure range: {min_pressure_bar:.3e} - {max_pressure_bar:.3e} bar")
    else:
        # Find the closest pressure index
        closest_idx = np.argmin(np.abs(pressure_array - min_pressure_bar))
        selected_indices = [closest_idx]

        print(f"data at closest pressure: {pressure_array[closest_idx]:.3e} bar")

    # Prepare data for bar chart
    mixing_ratios = []
    species_labels = []

    for sp in spec:
        if sp in vulcan_spec:
            avg_ratio = np.mean(data['variable']['ymix'][selected_indices, vulcan_spec.index(sp)])
            mixing_ratios.append(avg_ratio)
            if sp in tex_labels:
                species_labels.append(tex_labels[sp])
            else:
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

        plot_dir = '../parser_output/mixing_ratios'
        # Checking if the plot folder exsists
        if not os.path.exists(plot_dir):
            print( 'Directory ' , plot_dir,  " created.")
            os.makedirs(plot_dir)

        plt.savefig(os.path.join(plot_dir, plot_name + '.png'))


        plt.show(block=False)


    return spec, mixing_ratios




if __name__ == '__main__':
    if '-s' in sys.argv:
        use_range = False
        vul_data = sys.argv[2]
        spec = sys.argv[3]
        plot_name = sys.argv[4]
        pressure = float(sys.argv[5])
        mixing_ratios(vul_data,spec,plot_name,pressure,use_range=use_range)
    else:
        use_range = True
        vul_data = sys.argv[1]
        spec = sys.argv[2]
        plot_name = sys.argv[3]
        min_pressure_bar = float(sys.argv[4])
        max_pressure_bar = float(sys.argv[5])
        mixing_ratios(vul_data,spec,plot_name,min_pressure_bar,max_pressure_bar=max_pressure_bar,use_range=use_range)
