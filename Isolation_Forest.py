import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# 1. Importar los datos
df_new = pd.read_csv('tracking_data11.csv')

# 2. Filtrar IDs que han aparecido al menos 2 segundos (60 frames para un video de 30fps)
ids_to_consider = df_new['Track_ID'].value_counts()
ids_to_consider = ids_to_consider[ids_to_consider >= 120].index
df_new_filtered = df_new[df_new['Track_ID'].isin(ids_to_consider)]

# 3. Calcular las estadísticas agrupadas para el dataframe filtrado
grouped_new_filtered = df_new_filtered.groupby('Track_ID').agg({
    'X': 'std',
    'Y': 'std',
    'Class': 'mean'
})
grouped_new_filtered['avg_std_deviation'] = (grouped_new_filtered['X'] + grouped_new_filtered['Y']) / 2

# 4. Extraer las características y estandarizarlas
features_new_filtered = grouped_new_filtered[['Class', 'avg_std_deviation']]
scaler = StandardScaler()
features_new_filtered_standardized = scaler.fit_transform(features_new_filtered)

# 5. Aplicar Isolation Forest
iso_forest = IsolationForest(contamination=0.2)
anomalies_filtered = iso_forest.fit_predict(features_new_filtered_standardized)
anomalies_filtered = [-1 if a == -1 else 1 for a in anomalies_filtered]

# 6. Visualizar los resultados
plt.figure(figsize=(12, 7))
plt.scatter(features_new_filtered_standardized[:, 0], features_new_filtered_standardized[:, 1], c=anomalies_filtered, cmap='viridis', s=50)
for i, txt in enumerate(grouped_new_filtered.index):
    plt.annotate(txt, (features_new_filtered_standardized[i, 0], features_new_filtered_standardized[i, 1]), fontsize=9)
plt.xlabel('Standardized Average Class')
plt.ylabel('Standardized Average Standard Deviation')
plt.title('Isolation Forest Result on Filtered Data with ID labels')
plt.grid(True)
plt.show()
