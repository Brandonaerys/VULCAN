import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from transit_depth import transit_depth

seed = 20


specs = 'H2O,CH4,CO,CO2,NH3,H2S,HCN'


plot_name = 'throwaway'
min_pressure_bar = 1e-4
max_pressure_bar = 1e-1
temp = 300
min_wavenumber = 1e3
max_wavenumber = 1e4

types = ['GasDwarf','Hycean','miniNep']
mets = [30,50,75,100,125,150,175,200]
COs = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0]

tex_labels = {'H':'H','H2':'H$_2$','O':'O','OH':'OH','H2O':'H$_2$O','CH':'CH','C':'C','CH2':'CH$_2$','CH3':'CH$_3$','CH4':'CH$_4$','HCO':'HCO','H2CO':'H$_2$CO', 'C4H2':'C$_4$H$_2$',\
'C2':'C$_2$','C2H2':'C$_2$H$_2$','C2H3':'C$_2$H$_3$','C2H':'C$_2$H','CO':'CO','CO2':'CO$_2$','He':'He','O2':'O$_2$','CH3OH':'CH$_3$OH','C2H4':'C$_2$H$_4$','C2H5':'C$_2$H$_5$','C2H6':'C$_2$H$_6$','CH3O': 'CH$_3$O'\
,'CH2OH':'CH$_2$OH','N2':'N$_2$','NH3':'NH$_3$', 'NO2':'NO$_2$','HCN':'HCN','NO':'NO', 'NO2':'NO$_2$', 'H2S':'H$_2$S', 'CS2':'CS$_2$'}
spec_list = [tex_labels.get(name,name) for name in specs.split(',')]


dfs = []
labels = []
with np.errstate(divide='ignore'):
    for type in types:
        for met in mets:
            for CO in COs:
                vul_data_name = f'{type}_{int(met)}_{int(CO*100)}.vul'
                try:
                    df = transit_depth(vul_data_name,specs,plot_name,min_pressure_bar,max_pressure_bar,temp,min_wavenumber,max_wavenumber,mixing_plot_save=False,plot_save=False,log=True)
                    dfs.append(df)
                    labels.append(type)
                    ref_wavelengths = df_filtered[['wavelength']].sort_values(by='wavelength').values.flatten()
                except Exception as e:
                    pass
                    # print(vul_data_name, 'error occured,', e)




dfs_train, dfs_test, y_train, y_test = train_test_split(dfs, labels, test_size=0.4, random_state=seed)

X_test = []
# just normal max for test data
for df in dfs_test:
    df_filtered = df[['wavelength', 'max_value']].copy()
    df_filtered.sort_values(by='wavelength', inplace=True)
    X_test.append(df_filtered[['max_value']].values.flatten())
    ref_wavelengths = df_filtered[['wavelength']].sort_values(by='wavelength').values.flatten()

X_test = np.array(X_test)
# optional add noise to test data
gaussian_sd = 20.0
X_test = X_test + np.random.normal(0, gaussian_sd, size=X_test.shape)



# one run for each species
for sp in spec_list:
    X_train = []
    for df in dfs_train:
        df_removed = df.drop([sp, 'max_value'], axis=1)
        df_removed['removed_max'] = df_removed.drop(columns='wavelength').max(axis=1)
        df_removed.sort_values(by='wavelength', inplace=True)
        X_train.append(df_removed[['removed_max']].values.flatten())


    X_train = np.array(X_train)


    # train
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, max_features='sqrt', random_state=seed)
    clf.fit(X_train, y_train)


    # eval
    y_pred = clf.predict(X_test)
    print(f'--------------- Report for {sp} removed -------------------------------------------')
    print(classification_report(y_test, y_pred))

    # fetaure importance
    importances = clf.feature_importances_

    plt.figure(figsize=(10, 4))
    plt.plot(ref_wavelengths, importances, marker='o', linewidth=1)
    plt.title(f'Feature Importance by Wavelength; {sp} removed')
    plt.xlabel("Wavelength")
    plt.ylabel("Importance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)

    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix; {sp} removed')
    plt.tight_layout()
    plt.show()

    del clf
