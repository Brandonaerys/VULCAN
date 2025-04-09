import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from transit_depth import transit_depth

def rf_train(file_lists):

    X = []  # Feature vectors
    y = []  # Labels
    planet_names = []

    for category, file_list in file_lists.items():
        for filepath in file_list:
            try:
                plot_name = os.path.splitext(os.path.basename(filepath))[0]

                # Load planet-specific vul_data (adjust to your data format)
                vul_data = pd.read_csv(filepath)

                df = transit_depth(vul_data, spec, plot_name,
                                   min_pressure_bar, max_pressure_bar,
                                   temp,
                                   min_wavenumber, max_wavenumber,
                                   mixing_plot_save=False,
                                   plot_save=False,
                                   log=log)

                if 'max_value' not in df.columns:
                    print(f"Skipping {filepath} — no 'max_value' column.")
                    continue

                feature_vector = df['max_value'].values
                X.append(feature_vector)
                y.append(category)
                planet_names.append(plot_name)

            except Exception as e:
                print(f"Error processing {filepath}: {e}")

    # Convert to arrays
    X = np.array(X)
    y = np.array(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

    # Create the pipeline (no preprocessing steps yet)
    pipeline = Pipeline([
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train the classifier within the pipeline
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save the pipeline
    joblib.dump(pipeline, 'planet_classifier_pipeline.joblib')

    return pipeline, X, y, planet_names
