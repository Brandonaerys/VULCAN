# simple and barebones script to generate plots comparing three planet types

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.legend as lg
import pickle
import os, sys
import pandas as pd

from transit_depth import transit_depth


files = ['GasDwarf_50_50.vul', 'Hycean_50_50.vul', 'MiniNep_50_50.vul']
labels = ['Gas Dwarf', 'Hycean', 'MiniNep']

spec = 'H2O,CH4,CO,CO2,NH3,H2S,HCN'
plot_name = 'throwaway'
min_pressure_bar = 1e-4
max_pressure_bar = 1e-1
temp = 300
min_wavenumber = 1e3
max_wavenumber = 1e4

plt.figure(figsize=(10, 6))

for i in range(len(files)):
    out = transit_depth(files[i],spec,plot_name,min_pressure_bar,max_pressure_bar,temp,min_wavenumber,max_wavenumber,mixing_plot_save=False,plot_save=False,log=True)
    plt.plot(out['wavelength'], out['max_value'], label=labels[i])

plt.xlabel('Wavelength ($\mu m$)')
plt.ylabel('$log (\sigma n)$')
plt.title('Comparison of transit depths via $\sigma n$')
plt.legend()
plt.grid(True)


savepath = '../parser_output/comparisons/'
if not os.path.exists(savepath):
    print( 'Directory ' , savepath,  " created.")
    os.makedirs(savepath)

plt.tight_layout(pad=0)

plt.savefig(f'{savepath}three_50_50_maxonly')

plt.show()
