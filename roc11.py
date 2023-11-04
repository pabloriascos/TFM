import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, auc

# 1. Importar los datos
df_new21 = pd.read_csv('tracking_data11.csv')

# Filtrar IDs que han aparecido al menos 2 segundos (60 frames para un video de 30fps)
ids_to_consider21 = df_new21['Track_ID'].value_counts()
ids_to_consider21 = ids_to_consider21[ids_to_consider21 >= 120].index
df_new_filtered21 = df_new21[df_new21['Track_ID'].isin(ids_to_consider21)]

# Calcular las estadísticas agrupadas para el dataframe filtrado
grouped_new_filtered21 = df_new_filtered21.groupby('Track_ID').agg({
    'X': 'std',
    'Y': 'std',
    'Class': 'mean'
})
grouped_new_filtered21['avg_std_deviation'] = (grouped_new_filtered21['X'] + grouped_new_filtered21['Y']) / 2

# Extraer las características y estandarizarlas
features_new_filtered21 = grouped_new_filtered21[['Class', 'avg_std_deviation']]
scaler = StandardScaler()
features_new_filtered_standardized21 = scaler.fit_transform(features_new_filtered21)

# 2. Aplicar Isolation Forest con la configuración ajustada
iso_forest_best = IsolationForest(contamination=0.19)
iso_forest_best.fit(features_new_filtered_standardized21)  # Ajuste del modelo
predicted_scores_new_gt = iso_forest_best.decision_function(features_new_filtered_standardized21)

# 3. Definir el "ground truth" para IDs 8, 54 y 37 y calcular la curva ROC
true_labels_new_gt_837 = [1 if id in [4, 8, 16] else 0 for id in grouped_new_filtered21.index]
fpr_new_gt_837, tpr_new_gt_837, thresholds_new_gt_837 = roc_curve(true_labels_new_gt_837, -predicted_scores_new_gt)
roc_auc_new_gt_837 = auc(fpr_new_gt_837, tpr_new_gt_837)

# 4. Visualizar la curva ROC
plt.figure(figsize=(10, 6))
plt.plot(fpr_new_gt_837, tpr_new_gt_837, color='blue', label='ROC curve (area = %0.2f)' % roc_auc_new_gt_837)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve ')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

