import numpy as np
import os

# dyne/cm2 to bar
def dynetobar(dyne):
    return dyne*1e-6

# bar to dyne/cm2
def bartodyne(bar):
    return bar*1e6

# kzz to P profile as in Rigby et al
def kzz_profile(pressure):
    default = 1e6
    if pressure > 5e5:
        return f"{default:.2E}"
    else:
        value = np.minimum(1e10, 5.6e4/np.sqrt(pressure/1e6))
        return f'{value:.2E}'

# fixed 300K for K218b
def temp_profile(pressure):
    return 300

def generate_data(filename, min_pressure, max_pressure, pressure_levels):

    if os.path.exists(filename):
        print(f"File '{filename}' already exists. Terminating without overwriting.")
        return

    # pressure separations as in VULCAN sample config
    # pressures = [
    #     1.013E+06, 7.889E+05, 6.144E+05, 4.785E+05, 3.727E+05, 2.902E+05,
    #     2.260E+05, 1.760E+05, 1.371E+05, 1.068E+05, 8.315E+04, 6.476E+04,
    #     5.043E+04, 3.928E+04, 3.059E+04, 2.382E+04, 1.855E+04, 1.445E+04,
    #     1.125E+04, 8.764E+03, 6.826E+03, 5.316E+03, 4.140E+03, 3.224E+03,
    #     2.511E+03, 1.956E+03, 1.523E+03, 1.186E+03, 9.237E+02, 7.194E+02,
    #     5.603E+02, 4.363E+02, 3.398E+02, 2.647E+02, 2.061E+02, 1.605E+02,
    #     1.250E+02, 9.740E+01, 7.580E+01, 5.910E+01, 4.600E+01, 3.580E+01,
    #     2.790E+01, 2.170E+01, 1.690E+01, 1.320E+01, 1.030E+01, 7.990E+00,
    #     6.220E+00, 4.850E+00, 3.780E+00, 2.940E+00, 2.290E+00, 1.780E+00,
    #     1.390E+00, 1.080E+00, 8.420E-01, 6.560E-01, 5.110E-01, 3.980E-01,
    #     3.100E-01, 2.410E-01, 1.880E-01, 1.460E-01, 1.140E-01, 8.880E-02,
    #     6.910E-02, 5.380E-02, 4.190E-02, 3.270E-02, 2.540E-02
    # ]

    pressures = np.logspace(np.log10(min_pressure), np.log10(max_pressure),pressure_levels)

    with open(filename, 'w') as file:
        file.write("# (dyne/cm2) (K)     (cm2/s)\n")
        file.write("Pressure\tTemp\tKzz\n")

        for pressure in pressures:
            temp = temp_profile(pressure)
            kzz = kzz_profile(pressure)
            file.write(f"{pressure:.3E}\t{temp}\t{kzz}\n")

if __name__ == "__main__":
    filename = 'atm_GasDwarf.txt'
    min_pressure_bar = 1e-11
    max_pressure_bar = 1
    pressure_levels = 100
    generate_data(filename, bartodyne(min_pressure_bar), bartodyne(max_pressure_bar), pressure_levels)
    print(f"File {filename} generated successfully.")
