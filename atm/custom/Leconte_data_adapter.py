import numpy as np
import os

# dyne/cm2 to bar
def dynetobar(dyne):
    return dyne*1e-6

# bar to dyne/cm2
def bartodyne(bar):
    return bar*1e6

# kzz to P profile as in Rigby et al
# takes pressure in dyne/cm2
def kzz_profile(pressure):
    default = 1e6
    if pressure > 5e5:
        return default
    else:
        value = np.minimum(1e10, 5.6e4/np.sqrt(pressure/1e6))
        return value

# fixed 300K for K218b
def temp_profile(pressure):
    return 300

def modify_data(input_filename, output_filename):
    if not os.path.exists(input_filename):
        print(f"File '{input_filename}' not found")
        return

    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        lines = infile.readlines()

        # Write header
        outfile.write('# PT profile from Leconte 2024 (https://arxiv.org/pdf/2401.06608)\n')
        outfile.write('# Kzz profile from Rigby 2024\n')
        outfile.write("#(dyne/cm2)\t(K)\t(cm2/s)\n")
        outfile.write('Pressure\tTemp\tKzz\n')


        # Process data
        for line in lines[2:]:
            parts = line.split()


            pressure = bartodyne(float(parts[0]))
            temp = float(parts[1])
            kzz = kzz_profile(pressure)


            outfile.write(f"{pressure:.3E}\t{temp:.3E}\t{kzz:.3E}\n")

if __name__ == "__main__":
    input_filename = "Leconte_base_data.txt"
    output_filename = "atm_Leconte_K218b.txt"
    modify_data(input_filename, output_filename)
    print(f'{output_filename} generated')
