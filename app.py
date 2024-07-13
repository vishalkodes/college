from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import bz2 

app = Flask(__name__)

# Load and preprocess your CSV data
data = pd.read_csv(r"Drug prescription Dataset.csv")

# Encode Categorical Variables
def encode_categorical(data):
    text_features = ['disease', 'gender', 'severity']
    data = pd.get_dummies(data, columns=text_features)
    return data

data = encode_categorical(data)

# Label Encode the final target column
le = LabelEncoder()
data['drug'] = le.fit_transform(data['drug'])
labels = le.classes_.tolist()

# Split the dataset in training and testing set
y = data['drug']
X = data.drop(['drug'], axis=1)
col_order = X.columns

# Load the compressed model from the file
with bz2.BZ2File('model.pbz2', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_gender = request.form['gender']
        user_severity = request.form['severity']
        user_disease = request.form['disease']
        user_age = int(request.form['age'])

        user_df = pd.DataFrame({'age': [user_age], 'disease': [user_disease], 'gender': [user_gender], 'severity': [user_severity]})
        user_df = encode_categorical(user_df)
        columns_to_add = [col for col in data.columns if col not in user_df.columns]
        columns_to_add.remove('drug')
        for x in columns_to_add:
            user_df[x] = 0
        user_df = user_df[col_order]
        user_pred = model.predict(user_df)
        prediction = labels[user_pred[0]]

        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
