import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_parquet('data/dados_clones.parquet')

df = df.drop(columns=['General Jedi encarregado'])

le = LabelEncoder()
for col in df.select_dtypes(include=['str']).columns:
    df[col] = le.fit_transform(df[col])

caracteristicas = ['Massa(em kilos)', 'Estatura(cm)', 'Distância Ombro a ombro',
                   'Tamanho do crânio', 'Tamanho dos pés', 'Tempo de existência(em meses)']

X = df[caracteristicas]
y = df['Status ']

model = tree.DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X, y)

class_names = [str(c) for c in model.classes_]

plt.figure(figsize=(20, 10))
tree.plot_tree(model,
               feature_names=caracteristicas,
               class_names=class_names,
               filled=True,
               rounded=True,
               fontsize=10)
plt.title("Árvore de Decisão - Defeitos nos Clones")
plt.tight_layout()
plt.show()