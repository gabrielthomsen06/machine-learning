import pandas as pd

df = pd.read_excel('data/dados_frutas.xlsx')

from sklearn import tree
import matplotlib.pyplot as plt

arvore = tree.DecisionTreeClassifier()

y = df['Fruta']

caracteristicas = ['Arredondada', 'Suculenta', 'Vermelha']
x = df[caracteristicas]

arvore.fit(x, y)

plt.figure(figsize=(12, 8))  

tree.plot_tree(arvore,
               feature_names=caracteristicas,
               class_names=arvore.classes_,
               filled=True)

plt.show()  