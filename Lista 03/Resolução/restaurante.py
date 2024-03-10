import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Carregando dados do arquivo CSV
file_path = r'C:\Users\larin\Desktop\PUC\IA\Bases de dados\Lista 02\restaurantev2.csv'
df = pd.read_csv(file_path, encoding='latin1', delimiter=';')

# Ajustando o código para usar os nomes reais das colunas
X = df.drop(['Exemplo', 'conc'], axis=1)

# Selecionando o alvo
y = df['conc']

# Aplicando One-Hot Encoding para variáveis categóricas
categorical_features = ['Alternativo', 'Bar', 'Sex/Sab', 'fome', 'Cliente', 'Preço', 'Chuva', 'Res', 'Tipo', 'Tempo']
column_transformer = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough'
)

X_encoded = column_transformer.fit_transform(X)

# Criando modelos de árvore de decisão
id3_tree = DecisionTreeClassifier(criterion='entropy') # Entropia
c45_tree = DecisionTreeClassifier(criterion='entropy') # Entropia
cart_tree = DecisionTreeClassifier(criterion='gini') # Índice de Gini

# Treinando os modelos
id3_tree.fit(X_encoded, y)
c45_tree.fit(X_encoded, y)
cart_tree.fit(X_encoded, y)

# Exibindo as árvores
print("Árvore ID3:")
print(export_text(id3_tree, feature_names=column_transformer.get_feature_names_out()))

print("\nÁrvore C4.5:")
print(export_text(c45_tree, feature_names=column_transformer.get_feature_names_out()))

print("\nÁrvore CART:")
print(export_text(cart_tree, feature_names=column_transformer.get_feature_names_out()))
