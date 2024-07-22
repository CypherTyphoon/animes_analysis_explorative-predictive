import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Funktion zum Laden von Daten mit Caching
@st.cache_data
def load_data():
    file_paths = {
        'animes': 'animes.csv',
        'profiles': 'profiles.csv',
        'reviews': 'reviews.csv'
    }
    
    data = {}
    for key, file_name in file_paths.items():
        if os.path.isfile(file_name):
            data[key] = pd.read_csv(file_name)
        else:
            st.error(f"Datei fehlt: {file_name}")
            return None, None, None
    return data['animes'], data['profiles'], data['reviews']

# Liste von möglichen Datumsformaten
DATE_FORMATS = [
    '%b %d, %Y',
    '%b %Y',
    '%Y',
    '%b, %Y',
    '%d %b, %Y',
]

# Funktion zum Parsen von Datumsinformationen in 'aired'
def parse_dates(aired_string):
    aired_string = aired_string.strip()
    
    def try_formats(date_str, formats):
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return pd.NaT
    
    if ' to ' in aired_string:
        start_date_str, end_date_str = aired_string.split(' to ')
        start_date = try_formats(start_date_str.strip(), DATE_FORMATS)
        end_date = try_formats(end_date_str.strip(), DATE_FORMATS)
    elif ',' in aired_string:
        if aired_string.count(',') == 1:
            start_date = try_formats(aired_string.strip(), ['%b, %Y'])
            end_date = pd.NaT
        else:
            start_date = try_formats(aired_string.strip(), ['%Y'])
            end_date = pd.NaT
    elif 'Not available' in aired_string:
        start_date = pd.NaT
        end_date = pd.NaT
    else:
        start_date = try_formats(aired_string, DATE_FORMATS)
        end_date = start_date
    
    return pd.Series([start_date, end_date], index=['start_date', 'end_date'])

# Funktion zur Bereinigung der Geschlechterdaten
def clean_gender(data):
    valid_genders = ['Male', 'Female', 'Non-Binary']
    data['gender'] = data['gender'].apply(lambda x: x if x in valid_genders else 'Unbekannt')
    return data

# Funktion zur Bereinigung der Birthday-Daten
def calculate_age(birthday):
    if pd.isna(birthday):
        return 16  # Setzt das Alter auf 16, wenn das Geburtsdatum fehlt

    if isinstance(birthday, pd.Timestamp):
        today = pd.Timestamp(datetime.now())
        age = today.year - birthday.year - ((today.month, today.day) < (birthday.month, birthday.day))
        return age

    if isinstance(birthday, str):
        if not birthday.strip():
            return 16  # Setzt das Alter auf 16, wenn das Geburtsdatum fehlt
        try:
            birth_date = datetime.strptime(birthday, "%b %d, %Y")  # Format z.B. "Oct 2, 1994"
        except ValueError:
            try:
                birth_date = datetime.strptime(birthday, "%b %d")  # Format z.B. "Sep 5"
            except ValueError:
                return 16  # Setzt das Alter auf 16, wenn das Datum nicht analysiert werden kann
        today = datetime.today()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return max(age, 16)  # Setzt das Alter auf mindestens 16 Jahre

    return 16  # Falls birthday ein unerwarteter Typ ist

# Funktion zur Bereinigung der Geburtstag-Daten
def clean_birthday(data):
    # Funktion zum Füllen fehlender Geburtstage
    def fill_missing_birthday(birthday):
        if pd.isna(birthday) or not birthday.strip():
            return "Jan 1, 2008"  # Ein Beispiel für ein Datum, das mindestens 16 Jahre alt ist
        return birthday

    # Anwendung der Funktionen auf die Daten
    data['birthday'] = data['birthday'].apply(fill_missing_birthday)
    
    # Konvertierung der Geburtstagsspalte in ein Datetime-Format
    data['birthday'] = pd.to_datetime(data['birthday'], errors='coerce', format="%b %d, %Y")
    
    # Berechnung des Alters
    data['age'] = data['birthday'].apply(calculate_age)
    
    # Optional: Entfernen der Geburtstagsspalte, wenn nicht mehr benötigt
    data = data.drop(columns=['birthday'])
    
    return data

# Funktion zur Aufbereitung der 'aired' Spalte im Animes DataFrame
def preprocess_animes(animes_df):
    animes_df[['start_date', 'end_date']] = animes_df['aired'].apply(parse_dates)
    animes_df['start_date'] = pd.to_datetime(animes_df['start_date'], errors='coerce')
    animes_df['end_date'] = pd.to_datetime(animes_df['end_date'], errors='coerce')
    
    animes_df['start_year'] = animes_df['start_date'].dt.year
    animes_df['start_month'] = animes_df['start_date'].dt.month
    animes_df['start_day'] = animes_df['start_date'].dt.day
    
    animes_df['end_year'] = animes_df['end_date'].dt.year
    animes_df['end_month'] = animes_df['end_date'].dt.month
    animes_df['end_day'] = animes_df['end_date'].dt.day

    return animes_df

# Funktion zum Aufteilen der Genres in separate Spalten
def split_genres(animes_df):
    animes_df['genre'] = animes_df['genre'].apply(literal_eval)
    all_genres = set()
    for genres in animes_df['genre']:
        all_genres.update(genres)
    
    for genre in all_genres:
        animes_df[f'genre_{genre}'] = animes_df['genre'].apply(lambda x: 1 if genre in x else 0)
    
    return animes_df

# Funktion zur Verknüpfung von Animes und Reviews
def merge_animes_reviews(animes, reviews):
    merged_anime_reviews = pd.merge(animes, reviews, left_on='uid', right_on='anime_uid', how='left')
    return merged_anime_reviews

# Funktion zur Verknüpfung von Profiles und Reviews
def merge_profiles_reviews(profiles, reviews):
    merged_profile_reviews = pd.merge(profiles, reviews, left_on='profile', right_on='profile', how='left')
    return merged_profile_reviews

# Funktion zur Verknüpfung aller Daten
def merge_all_data(animes, profiles, reviews):
    anime_reviews = merge_animes_reviews(animes, reviews)
    all_data = merge_profiles_reviews(profiles, anime_reviews)
    return all_data

# Funktion zur Erstellung von Grafiken für Stakeholder-Fragen
def stakeholder_analysis(data, regenerate, save_graphs):
    st.header("Stakeholder-Bereich")
    
    # Ordner zum Speichern der Grafiken
    output_dir = "stakeholder_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Funktion zum Speichern und Laden von Grafiken
    def save_plot(fig, filename):
        fig.savefig(os.path.join(output_dir, filename))

    def load_plot(filename):
        img_path = os.path.join(output_dir, filename)
        return img_path if os.path.exists(img_path) else None

    
    # Frage 1: Wie haben sich die Genres auf die ausgewählte Metrik ausgewirkt?
    st.subheader("Wie haben sich die Genres auf die ausgewählte Metrik ausgewirkt?")

    # Auswahl der Metrik durch den Benutzer
    metrics = ['ranked', 'popularity', 'members', 'score_x', 'score_y']
    metric = st.selectbox("Metrik auswählen", metrics)

    # Überprüfen, ob die ausgewählte Metrik in den Daten vorhanden ist
    if metric not in data.columns:
        st.error(f"Die Spalte '{metric}' ist nicht in den Daten vorhanden.")
        return

    # Finden der Genre-Spalten
    genre_cols = [col for col in data.columns if col.startswith('genre_')]

    # Überprüfen, ob Genre-Spalten vorhanden sind
    if not genre_cols:
        st.error("Keine Genre-Spalten gefunden. Überprüfen Sie die Spaltennamen.")
        return

    # Erstellen und Laden der Grafik, falls notwendig
    if regenerate or not load_plot(f"genre_scores_{metric}.png"):
        try:
        # Berechnung der Durchschnittswerte für die ausgewählte Metrik nach Genre
            genre_scores = data[genre_cols + [metric]].groupby(genre_cols).mean()
            genre_scores = genre_scores.reset_index().melt(id_vars=metric, var_name='genre', value_name='value')
        
        # Optionales Filtern der Daten (wenn nötig)
            genre_scores = genre_scores[genre_scores['value'] == 1]

        # Erstellen der Grafik
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.barplot(x='genre', y=metric, data=genre_scores, ax=ax)
            ax.set_title(f'Durchschnittlicher {metric} nach Genre')
            plt.xticks(rotation=90)
        
        # Speichern der Grafik
            save_plot(fig, f"genre_scores_{metric}.png")
        except Exception as e:
            st.error(f"Fehler bei der Erstellung der Genre-Grafik für {metric}: {e}")

    # Anzeigen der Grafik
    st.image(load_plot(f"genre_scores_{metric}.png"))

    # Frage 2: Wie hat sich die Länge der Episoden auf den Beliebtheitsscore ausgewirkt?
    st.subheader("Wie hat sich die Länge der Episoden auf den Beliebtheitsscore ausgewirkt?")
    if regenerate or not load_plot("episodes_vs_score.png"):
        try:
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.scatterplot(x='episodes', y='ranked', data=data, ax=ax)
            ax.set_title('Beliebtheitsscore nach Episodenlänge')
            save_plot(fig, "episodes_vs_score.png")
        except Exception as e:
            st.error(f"Fehler bei der Erstellung der Episodenlänge-Grafik: {e}")
    st.image(load_plot("episodes_vs_score.png"))

    # Frage 3: Gibt es demographische Unterschiede in der Beliebtheit?
    st.subheader("Gibt es demographische Unterschiede in der Beliebtheit?")
    if regenerate or not load_plot("gender_vs_score.png"):
        try:
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.boxplot(x='gender', y='ranked', data=data, ax=ax)
            ax.set_title('Beliebtheitsscore nach Geschlecht')
            save_plot(fig, "gender_vs_score.png")
        except Exception as e:
            st.error(f"Fehler bei der Erstellung der Geschlecht-Grafik: {e}")
    st.image(load_plot("gender_vs_score.png"))

    # Frage 4: Welche anderen Parameter hatten Einfluss auf den Beliebtheitsscore?
    st.subheader("Welche anderen Parameter hatten Einfluss auf den Beliebtheitsscore?")
    numerical_columns = data.select_dtypes(include=['number']).columns
    numerical_columns = numerical_columns[numerical_columns != 'ranked']
    if regenerate or not load_plot("correlation_matrix.png"):
        try:
            correlation_matrix = data[numerical_columns].corr()
            fig, ax = plt.subplots(figsize=(14, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            ax.set_title('Korrelationsmatrix der Parameter')
            save_plot(fig, "correlation_matrix.png")
        except Exception as e:
            st.error(f"Fehler bei der Erstellung der Korrelationsmatrix-Grafik: {e}")
    st.image(load_plot("correlation_matrix.png"))

    # Falls der save_graphs Button geklickt wurde, alle Grafiken erneut speichern
    if save_graphs:
        try:
            genre_scores = data[genre_cols + ['ranked']].groupby(genre_cols).mean()
            genre_scores = genre_scores.reset_index().melt(id_vars='ranked', var_name='genre', value_name='value')
            genre_scores = genre_scores[genre_scores['value'] == 1]
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.barplot(x='genre', y='ranked', data=genre_scores, ax=ax)
            ax.set_title('Durchschnittlicher Beliebtheitsscore nach Genre')
            plt.xticks(rotation=90)
            save_plot(fig, "genre_scores_ranked.png")
        except Exception as e:
            st.error(f"Fehler beim Speichern der Genre-Grafik für ranked: {e}")

        try:
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.scatterplot(x='episodes', y='ranked', data=data, ax=ax)
            ax.set_title('Beliebtheitsscore nach Episodenlänge')
            save_plot(fig, "episodes_vs_ranked.png")
        except Exception as e:
            st.error(f"Fehler beim Speichern der Episodenlänge-Grafik für ranked: {e}")

        try:
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.boxplot(x='gender', y='ranked', data=data, ax=ax)
            ax.set_title('Beliebtheitsscore nach Geschlecht')
            save_plot(fig, "gender_vs_ranked.png")
        except Exception as e:
            st.error(f"Fehler beim Speichern der Geschlecht-Grafik für ranked: {e}")

        try:
            correlation_matrix = data[numerical_columns].corr()
            fig, ax = plt.subplots(figsize=(14, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            ax.set_title('Korrelationsmatrix der Parameter')
            save_plot(fig, "correlation_matrix.png")
        except Exception as e:
            st.error(f"Fehler beim Speichern der Korrelationsmatrix-Grafik: {e}")


# Funktion zur Durchführung einer deskriptiven Analyse
def descriptive_analysis(data, data_name):
    st.subheader(f"Deskriptive Analyse der {data_name}-Daten")
    st.write(data.describe())

# Funktion zur Durchführung einer explorativen Analyse
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def exploratory_analysis(data, chart_type, fixed_x_var, var_choice_y):
    st.subheader("Explorative Analyse")
    
    if chart_type == "Histogramm":
        if var_choice_y:
            plt.figure(figsize=(10, 6))
            sns.histplot(data[var_choice_y], kde=True)
            plt.title(f'Histogramm von {var_choice_y}')
            st.pyplot(plt)
        else:
            st.warning("Bitte wählen Sie eine Variable für das Histogramm aus.")
    
    elif chart_type == "Boxplot":
        if var_choice_y:
            plt.figure(figsize=(10, 6))
            sns.boxplot(y=data[var_choice_y])
            plt.title(f'Boxplot von {var_choice_y}')
            st.pyplot(plt)
        else:
            st.warning("Bitte wählen Sie eine Variable für den Boxplot aus.")
    
    elif chart_type == "Scatterplot":
        if fixed_x_var and var_choice_y:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=data[fixed_x_var], y=data[var_choice_y])
            plt.title(f'Scatterplot von {fixed_x_var} und {var_choice_y}')
            st.pyplot(plt)
        else:
            st.warning("Bitte wählen Sie eine X-Variable und eine Y-Variable für den Scatterplot aus.")
    
    elif chart_type == "Heatmap":
        if len(var_choice_y) > 1:
            plt.figure(figsize=(14, 10))
            corr = data[var_choice_y].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Korrelationsmatrix')
            st.pyplot(plt)
        else:
            st.warning("Bitte wählen Sie mindestens zwei Variablen für die Heatmap aus.")
    
    elif chart_type == "Linienplot":
        if fixed_x_var and var_choice_y:
            plt.figure(figsize=(10, 6))
            for var in var_choice_y:
                plt.plot(data[fixed_x_var], data[var], label=var)
            plt.title(f'Linienplot von {fixed_x_var} und {", ".join(var_choice_y)}')
            plt.legend()
            st.pyplot(plt)
        else:
            st.warning("Bitte wählen Sie eine X-Variable und eine Y-Variable für den Linienplot aus.")
    
    else:
        st.error("Unbekannter Diagrammtyp. Bitte wählen Sie einen gültigen Diagrammtyp aus.")

# Funktion zur Darstellung fehlender Werte
def plot_missing_values(data, data_name):
    st.subheader(f"Fehlende Werte in den {data_name}-Daten")
    missing_values = data.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing_values.index, y=missing_values.values)
        plt.title('Fehlende Werte')
        plt.xticks(rotation=90)
        st.pyplot(plt)
    else:
        st.write("Keine fehlenden Werte gefunden.")

# Funktion für Random Forest
def machine_learning(data):
    st.header("Machine Learning")
    
    # Auswahl der Ziel- und Feature-Variablen
    target = 'ranked'
    features = data.select_dtypes(include=['number']).columns.tolist()
    features.remove(target)
    
    # Entfernen spezifischer Spalten
    columns_to_remove = ['score_x', 'score_y']  # Liste der zu entfernenden Spalten
    for col in columns_to_remove:
        if col in features:
            features.remove(col)
    
    X = data[features].fillna(0)
    y = data[target].fillna(0)
    
    # Aufteilen der Daten in Trainings- und Testdaten
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Modell erstellen und trainieren
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Vorhersagen und Fehlerberechnung
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R² (Bestimmtheitsmaß): {r2}")
    
    # Wichtigste Features anzeigen
    feature_importances = pd.Series(model.feature_importances_, index=features)
    st.write(feature_importances.sort_values(ascending=False).head(10))

#Funktion Lineare Regression
def linear_regression_analysis(data):
    st.header("Lineare Regression")

    # Ziel- und Feature-Variablen
    target = 'ranked'
    features = data.select_dtypes(include=['number']).columns.tolist()
    
    # Entfernen des Ziel-Features aus den Features
    if target in features:
        features.remove(target)
    
    # Entfernen spezifischer Spalten
    columns_to_remove = ['uid_x']  # Liste der zu entfernenden Spalten
    for col in columns_to_remove:
        if col in features:
            features.remove(col)
    
    # Datenaufbereitung
    X = data[features].fillna(0)
    y = data[target].fillna(0)
    
    # Aufteilen der Daten in Trainings- und Testdaten
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Pipeline für Standardisierung und Regressionsmodell
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    # Modell erstellen und trainieren
    pipeline.fit(X_train, y_train)
    
    # Vorhersagen und Fehlerberechnung
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R² (Bestimmtheitsmaß): {r2}")
    
    # Wichtigste Features anzeigen
    coef_df = pd.DataFrame(pipeline.named_steps['regressor'].coef_, index=features, columns=['Coefficient'])
    st.write(coef_df.sort_values(by='Coefficient', ascending=False))

# Hauptfunktion der Streamlit-App
def main():
    st.sidebar.title("Nutzerrollen")
    role = st.sidebar.selectbox("Wählen Sie Ihre Rolle", ["Stakeholder", "Data-Analyst", "Entwickler"])

    # Daten laden mit Caching
    animes, profiles, reviews = load_data()
    if animes is None or profiles is None or reviews is None:
        return

    # Bereinigen der Geburtsdaten
    profiles = clean_birthday(profiles)
    
    # Daten vorbereiten
    animes = preprocess_animes(animes)
    animes = split_genres(animes)
    profiles = clean_gender(profiles)
    all_data = merge_all_data(animes, profiles, reviews)

    if role == "Stakeholder":
        st.title("Stakeholder-Bereich")
        regenerate = st.button("Grafiken neu generieren")
        save_graphs = st.button("Grafiken speichern")
        stakeholder_analysis(all_data, regenerate, save_graphs)
    
    elif role == "Data-Analyst":
        st.title("Data-Analyst-Bereich")
        st.write("Hier können weitere Analysen durchgeführt und neue Grafiken erstellt werden.")
        
        # Deskriptive Analyse
        st.sidebar.subheader("Deskriptive Analyse")
        if st.sidebar.checkbox("Deskriptive Analyse anzeigen"):
            descriptive_analysis(all_data, "alle Daten")
        
        # Explorative Analyse
        st.sidebar.subheader("Explorative Analyse")
        chart_type = st.sidebar.selectbox("Diagrammtyp auswählen", ["Histogramm", "Boxplot", "Scatterplot", "Heatmap", "Linienplot"])

        if chart_type in ["Boxplot", "Scatterplot", "Linienplot"]:
            var_choice_y = st.sidebar.selectbox("Wählen Sie die Y-Achse", all_data.columns.tolist())
            fixed_x_var = st.sidebar.selectbox("Wählen Sie die X-Achse (für Scatterplot und Linienplot)", all_data.columns.tolist()) if chart_type in ["Scatterplot", "Linienplot"] else None
        elif chart_type == "Heatmap":
            var_choice_y = st.sidebar.multiselect("Wählen Sie die Variablen", all_data.select_dtypes(include=['number']).columns.tolist())
            fixed_x_var = None
        else:
            var_choice_y = st.sidebar.selectbox("Wählen Sie die Variable", all_data.columns.tolist()) if chart_type == "Histogramm" else []
            fixed_x_var = None

        if st.sidebar.button("Plot erstellen"):
            exploratory_analysis(all_data, chart_type, fixed_x_var, var_choice_y)

        # Fehlende Werte
        st.sidebar.subheader("Fehlende Werte")
        if st.sidebar.checkbox("Fehlende Werte anzeigen"):
            plot_missing_values(all_data, "alle Daten")

        # Machine Learning
        st.sidebar.subheader("Machine Learning")
        if st.sidebar.button("Random Forest Regressions Analyse durchführen"):
            machine_learning(all_data)
        
        # Lineare Regression
        st.sidebar.subheader("Lineare Regression")
        if st.sidebar.button("Lineare Regressions Analyse durchführen"):
            linear_regression_analysis(all_data)

    elif role == "Entwickler":
        st.title("Entwickler-Bereich")
        st.write("Hier können neue Daten hinzugefügt und die bestehenden Daten bearbeitet werden.")
        st.write("Upload- und Datenbearbeitungsoptionen kommen hier.")

if __name__ == "__main__":
    main()