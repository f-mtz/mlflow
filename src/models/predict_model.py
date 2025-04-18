import mlflow
logged_model = 'runs:/b1f6288bab6145d7a7f0783a42e52e70/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd

# copiando a base de dados sem a variavel target
# df.drop('preco', axis=1).to_csv('data/processed/casas_X.csv', index=False)

data = pd.read_csv('../../data/processed/casas_X.csv')
predicted = loaded_model.predict(data)

data['predicted'] = predicted
data.to_csv('precos.csv')

