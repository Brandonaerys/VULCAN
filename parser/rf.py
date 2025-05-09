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

types = ['GasDwarf', 'Hycean','MiniNep']
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
                    df = transit_depth(vul_data_name,specs,plot_name,min_pressure_bar,max_pressure_bar,temp,min_wavenumber,max_wavenumber,mixing_plot_save=False,plot_save=False,log=True)
                    df_filtered = df[['wavelength', 'max_value']].copy()
                    df_filtered.sort_values(by='wavelength', inplace=True)
                    dfs.append(df_filtered[['max_value']].values.flatten())
                    labels.append(type)
                    ref_wavelengths = df_filtered[['wavelength']].sort_values(by='wavelength').values.flatten()
                except Exception as e:
                    pass
                    # print(vul_data_name, 'error occured,', e)

print(len(labels))
# exit()


X = np.array(dfs)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)


# train
clf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, max_features='sqrt', random_state=seed)


# intentionally overfitted classifier
# clf = RandomForestClassifier(n_estimators=1, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, bootstrap=False, random_state=seed)


clf.fit(X_train, y_train)

# optional add noise to test data
# gaussian_sd = 10
# X_test = X_test + np.random.normal(0, gaussian_sd, size=X_test.shape)

# optional add pre-log noise (note values very unstable)
# e^(-45) = 2.86e-20
# e^(-100) = 3.72e-44
prelog_gaussian_sd = 1e-30
X_test = np.log(np.maximum(np.exp(X_test) + np.random.normal(0, prelog_gaussian_sd, size=X_test.shape), 1e-50))


# eval
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))



# confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)

disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.tight_layout(pad=0)
plt.show()


# fetaure importance with full model instead
del clf
clf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, max_features='sqrt', random_state=seed)
clf.fit(X,y)
importances = clf.feature_importances_

plt.figure(figsize=(10, 4))
plt.plot(ref_wavelengths, importances, marker='o', linewidth=1)
plt.title("Feature Importance by Wavelength")
plt.xlabel('Wavelength (μm)')
plt.ylabel("Importance")
plt.grid(True)
plt.tight_layout(pad=0)
plt.show()


# joblib save model
joblib.dump(clf, 'random_forest_model.joblib')
