
# LINK GITHUB PARA VISUALIZAÇÃO GRÁFICOS: https://github.com/luisabrmatos/Alg.IA.Clusterizacao

import pandas as pd

# Carregar o arquivo CSV
df = pd.read_csv("archive/Country-data.csv")

# Exibir as primeiras linhas do dataset
print(df.head())

# Exibir informações sobre o dataset
print(df.info())

# Exibir estatísticas descritivas
print(df.describe())
print(df.columns)

print()

# Quantidade de países no dataset
print("Número de países:", df.shape[0])

print()

# Variáveis e análise de tarefas de clusterização
# Verificação de valores nulos
print("Valores Nulos em cada coluna:\n", df.isnull().sum())

# Normalização de Dados
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_normalizado = scaler.fit_transform(df.select_dtypes(include=[float, int]))

# Visualização da faixa dinâmica
import matplotlib.pyplot as plt
import seaborn as sns

# Criar uma lista com os nomes das colunas numéricas
colunas_numericas = df.select_dtypes(include=[float, int]).columns

# Plotar um boxplot para cada variável numérica
plt.figure(figsize=(15, 10))
for i, coluna in enumerate(colunas_numericas, 1):
    plt.subplot(4, 3, i)
    sns.boxplot(y=df[coluna])
    plt.title(f'Boxplot - {coluna}')

plt.savefig("boxplot_country_data_variables.png")

plt.tight_layout()
plt.show()

# CLUSTERIZAÇÃO

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Clusterização K-médias

# Clusterização K-means com diferentes números de clusters
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Testar de 2 a 5 clusters
for n_clusters in range(2, 6):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(df_normalizado)
    df[f'KMeans_Cluster_{n_clusters}'] = kmeans.labels_

    # Aplicação de PCA para visualização
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(df_normalizado)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=df[f'KMeans_Cluster_{n_clusters}'], palette='viridis')
    plt.title(f'Clusters K-Means com {n_clusters} clusters')
    plt.show()

for n_clusters in range(2, 6):
    print(f"Análise para {n_clusters} clusters (K-Means):")
    numeric_cols = df.select_dtypes(include=['number'])
    print(numeric_cols.groupby(f'KMeans_Cluster_{n_clusters}').mean())
    print("\n")

# Clusterização Hierárquica

from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np

# Verificar o formato do df_normalizado
print(type(df_normalizado))

# Converter o DataFrame em NumPy array (se necessário)
if not isinstance(df_normalizado, np.ndarray):
    df_normalizado_array = np.array(df_normalizado)
else:
    df_normalizado_array = df_normalizado

# Calcular a ligação hierárquica usando o método 'ward'
linked = linkage(df_normalizado_array, method='ward')

# Plotar o dendrograma
plt.figure(figsize=(15, 10))  # Ajustar o tamanho do gráfico
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Dendrograma - Clusterização Hierárquica')
plt.xlabel('Países')
plt.ylabel('Distância')
plt.show()

# SUBSTITUIÇÃO DO K-MÉDIAS POR K-MEDOIDES

from sklearn.metrics import pairwise_distances
import numpy as np

# Número de clusters
n_clusters = 3

# Inicializando os medoides aleatoriamente
np.random.seed(42)
medoids = np.random.choice(range(len(df_normalizado)), n_clusters, replace=False)

for iteration in range(10):  # Número máximo de iterações
    clusters = {i: [] for i in range(n_clusters)}
    
    # Atribuir cada ponto ao medoide mais próximo
    for idx, point in enumerate(df_normalizado):
        distances = [np.linalg.norm(point - df_normalizado[medoid]) for medoid in medoids]
        cluster_id = np.argmin(distances)
        clusters[cluster_id].append(idx)
    
    # Recalcular os medoides
    new_medoids = []
    for cluster_id, points in clusters.items():
        cluster_points = df_normalizado[points]
        medoid = np.argmin(pairwise_distances(cluster_points, cluster_points).sum(axis=1))
        new_medoids.append(points[medoid])
    
    # Verificar convergência
    if np.array_equal(medoids, new_medoids):
        print(f"Convergência atingida na iteração {iteration + 1}")
        break
    medoids = new_medoids

# Resultados finais
print("Medoides finais:", medoids)
print("Clusters finais:", clusters)

plt.savefig('Resultados/Cluster_Kmedias.png')  # Salvar gráficos K-médias
plt.savefig('Resultados/Cluster_Hierarquico.png')  # Salvar clusters hierárquicos
plt.savefig('Resultados/Dendograma_ClusterHierarquico.png')  # Salvar dendrograma




