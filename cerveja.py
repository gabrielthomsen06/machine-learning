import pandas as pd

df = pd.read_excel('data/dados_cerveja.xlsx')

print(df.head())

features = ['temperatura', 'copo', 'espuma', 'cor', 'classe']
target = 'classe' 

X = df[features]
y = df[target]

from sklearn import tree

model = tree.DecisionTreeClassifier()
model.fit(X=X, y=y)
