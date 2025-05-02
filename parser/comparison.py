import numpy as np
import matplotlib.pyplot as plt
import matplotlib.legend as lg
import pickle
import os, sys
import pandas as pd

from transit_depth import transit_depth

def sigma_n_compare(vul_data_1,vul_data_2,log_threshold=1.0, planet1_name='Planet 1', planet2_name='Planet 2', plot=True):
    # !!!defining parameters for transit_depth df - change here instead of arguments
    spec = 'H2O,CH4,CO,CO2,NH3,H2S,HCN' # ,H2' # ,CS2,N2'
    min_pressure_bar = 1e-4
    max_pressure_bar = 1e-1
    temp = 300
    # wavenumber corresponding to 1-10 microns
    min_wavenumber = 1e3
    max_wavenumber = 1e4
    # no need plot name as plot_save=False
    plot_name = 'throwaway'





    df1 = transit_depth(vul_data_1,spec,plot_name,min_pressure_bar,max_pressure_bar,temp,min_wavenumber,max_wavenumber,mixing_plot_save=False,plot_save=False,log=True)
    df2 = transit_depth(vul_data_2,spec,plot_name,min_pressure_bar,max_pressure_bar,temp,min_wavenumber,max_wavenumber,mixing_plot_save=False,plot_save=False,log=True)



    wavelength = df1['wavelength']

    epsilon = 1e-12
    abs_diff = np.abs(df1['max_value'] - df2['max_value'])
    significant_mask = abs_diff > log_threshold



    # Collect dominant species info for significant differences
    dominant_species_list = []
    for idx in df1.index[significant_mask]:
        p1_val = df1.loc[idx, 'max_value']
        p2_val = df2.loc[idx, 'max_value']

        if p1_val > p2_val:
            dominant_df = df1
            higher_planet = planet1_name
        else:
            dominant_df = df2
            higher_planet = planet2_name

        species = dominant_df.drop(columns=['wavelength', 'max_value']).iloc[idx].idxmax()

        dominant_species_list.append({
            'wavelength': wavelength[idx],
            'relative_difference': abs_diff[idx],
            'higher_transit_planet': higher_planet,
            'dominant_species': species
        })

    df_diff = pd.DataFrame(dominant_species_list)

    # Group adjacent rows by same species and planet
    grouped_regions = []
    if not df_diff.empty:
        df_diff.sort_values('wavelength', inplace=True)
        df_diff.reset_index(drop=True, inplace=True)

        current_group = {
            'start': df_diff.loc[0, 'wavelength'],
            'end': df_diff.loc[0, 'wavelength'],
            'species': df_diff.loc[0, 'dominant_species'],
            'planet': df_diff.loc[0, 'higher_transit_planet'],
            'max_abs_diff': df_diff.loc[0, 'relative_difference']
        }

        for i in range(1, len(df_diff)):
            curr = df_diff.loc[i]
            prev = df_diff.loc[i - 1]

            contiguous = np.isclose(curr['wavelength'], prev['wavelength'], atol=0.1)
            same_species = curr['dominant_species'] == prev['dominant_species']
            same_planet = curr['higher_transit_planet'] == prev['higher_transit_planet']

            if contiguous and same_species and same_planet:
                current_group['end'] = curr['wavelength']
                current_group['max_abs_diff'] = max(current_group['max_abs_diff'], curr['relative_difference'])
            else:
                grouped_regions.append(current_group.copy())
                current_group = {
                    'start': curr['wavelength'],
                    'end': curr['wavelength'],
                    'species': curr['dominant_species'],
                    'planet': curr['higher_transit_planet'],
                    'max_abs_diff': curr['relative_difference']
                }

        # Add final group
        grouped_regions.append(current_group.copy())

    grouped_df = pd.DataFrame(grouped_regions)
    print(grouped_df.to_string())

    # Optional plot
    if plot:
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df1['wavelength'], df1['max_value'], label=planet1_name, color='blue', linewidth=1)
        ax.plot(df2['wavelength'], df2['max_value'], label=planet2_name, color='orange', linewidth=1)

        # Generate unique colors for each (planet, species) pair
        from itertools import cycle
        import seaborn as sns

        unique_categories = grouped_df[['planet', 'species']].drop_duplicates()
        category_labels = unique_categories.apply(lambda row: f"{row['planet']} - {row['species']}", axis=1)
        color_palette = sns.color_palette("husl", len(category_labels))  # You can change to 'tab10' or others

        color_map = {
            label: color for label, color in zip(category_labels, color_palette)
        }

        # Add shaded regions with category-specific colors
        for _, row in grouped_df.iterrows():
            label = f"{row['planet']} - {row['species']}"
            ax.axvspan(row['start'], row['end'], color=color_map[label], alpha=0.3, label=label)

        # Prevent duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        unique_legend = dict(zip(labels, handles))
        ax.legend(unique_legend.values(), unique_legend.keys(), loc='upper right')

        ax.set_xlabel("Wavelength (μm)")
        ax.set_ylabel("Transit Depth Metric (log scale)")
        ax.set_title("Comparison of Transit Depth Metrics")
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    return grouped_df


if __name__ == '__main__':
    vul_data_1 = 'GasDwarf_30_100.vul'
    vul_data_2 = 'GasDwarf_200_200.vul'
    planet1_name = vul_data_1
    planet2_name = vul_data_2
    log_threshold = 1.0
    sigma_n_compare(vul_data_1,vul_data_2,log_threshold=log_threshold, planet1_name=planet1_name, planet2_name=planet2_name, plot=True)
