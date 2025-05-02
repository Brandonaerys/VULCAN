import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from transit_depth import transit_depth

seed = 20


spec = 'H2O,CH4,CO,CO2,NH3,H2S,HCN'
plot_name = 'throwaway'
min_pressure_bar = 1e-4
max_pressure_bar = 1e-1
temp = 300
min_wavenumber = 1e3
max_wavenumber = 1e4

types = ['GasDwarf', 'Hycean','miniNep']
mets = [30,50,75,100,125,150,175,200]
COs = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0]

dfs = []
labels = []
with np.errstate(divide='ignore'):
    for type in types:
        for met in mets:
            for CO in COs:
                vul_data_name = f'{type}_{int(met)}_{int(CO*100)}.vul'
                try:
                    df = transit_depth(vul_data_name,spec,plot_name,min_pressure_bar,max_pressure_bar,temp,min_wavenumber,max_wavenumber,mixing_plot_save=False,plot_save=False,log=True)
                    df_filtered = df_filtered = df[['wavelength', 'max_value']].copy()
                    df_filtered.sort_values(by='wavelength', inplace=True)

                    dfs.append(df_filtered[['max_value']].values.flatten())
                    labels.append(type)
                    ref_wavelengths = df_filtered[['wavelength']].sort_values(by='wavelength').values.flatten()
                except Exception as e:
                    pass
                    # print(vul_data_name, 'error occured,', e)



X = np.array(dfs)
y = np.array(labels)

# print(len(y))
# print(X)
# print(y)
# exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=seed)

# train
clf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, max_features='sqrt', random_state=seed)
clf.fit(X_train, y_train)

# eval
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# fetaure importance
importances = clf.feature_importances_

plt.figure(figsize=(10, 4))
plt.plot(ref_wavelengths, importances, marker='o', linewidth=1)
plt.title("Feature Importance by Wavelength")
plt.xlabel("Wavelength")
plt.ylabel("Importance")
plt.grid(True)
plt.tight_layout()
plt.show()

# confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)

disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# joblib save model
joblib.dump(clf, 'random_forest_model.joblib')
