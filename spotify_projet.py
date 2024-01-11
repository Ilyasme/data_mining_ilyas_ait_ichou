# Import des bibliothèques nécessaires
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Charger les données
data = pd.read_csv("C:/Users/aitic/Desktop/sidi/2eme_annes/DATA_mining/mini_projet_2/spotify-2023.csv", encoding='latin1')

data['in_shazam_charts'] = data['in_shazam_charts'].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
data['in_spotify_charts'] = data['in_spotify_charts'].astype(float)
data['streams'] = pd.to_numeric(data['streams'], errors='coerce')
data = data.dropna(subset=['streams'])


# Exploration initiale
print(data.head())
print(data.describe())

# Analyse des tendances audio
audio_features = ['bpm', 'danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%']
sns.pairplot(data, vars=audio_features)
plt.show()

# Comparaison des plateformes
platforms = ['in_spotify_charts', 'in_apple_charts', 'in_deezer_charts', 'in_shazam_charts']
platform_corr = data[platforms].corr()
sns.heatmap(platform_corr, annot=True, cmap="coolwarm")
plt.show()

# Analyse de l'impact des artistes
artist_impact = data.groupby('artist_count')['streams'].mean()
artist_impact.plot(kind='bar')
plt.xlabel('Nombre d\'artistes')
plt.ylabel('Streams moyens')
plt.show()

# Analyse temporelle
monthly_trends = data.groupby('released_month')['streams'].mean()
monthly_trends.plot(kind='line', marker='o')
plt.xlabel('Mois de sortie')
plt.ylabel('Streams moyens')
plt.show()

# Modélisation de prédiction de popularité (exemple simple)
X = data[audio_features]
y = data['streams']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Score de prédiction :', model.score(X_test, y_test))
print('Erreur absolue moyenne :', metrics.mean_absolute_error(y_test, y_pred))


