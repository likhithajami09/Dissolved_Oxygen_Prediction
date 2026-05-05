#import require python classes and packages
import pandas as pd
import numpy as np
import lightgbm as lgb
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM #class for LSTM training
import os
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers import Bidirectional,GRU
from Attention import attention #importing attention layer
from keras.layers import SimpleRNN
from sklearn.metrics import mean_absolute_error
from math import sqrt
from flask import *
import random
from auth_utils import *
import matplotlib
matplotlib.use("Agg")



mse = []
rmse = []
mae = []
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt','csv'}
MAX_UPLOAD_SIZE_MB = 512  # Maximum upload size in megabytes

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE_MB * 1024 * 1024  # Set max content length

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/login')
def login():
    return render_template('login.html')


@app.route("/signup")
def signup_route():
    return signup()

@app.route("/signin")
def signin_route():
    return signin()

@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/load_dataset')
def load_dataset():
    global dataset
    # Load the dataset and fill missing values with 0
    try:
        dataset = pd.read_csv("Dataset/Water_Quality_Testing.csv")
        dataset.fillna(0, inplace=True)
        
        # Convert the dataset to HTML for easy display
        dataset_html = dataset.to_html(classes='table table-striped')
        message = "Dataset loaded successfully."
    except FileNotFoundError:
        dataset_html = "<p>Dataset file not found. Please ensure the file path is correct.</p>"
        message = "Error: Could not load dataset."
    except Exception as e:
        dataset_html = f"<p>An error occurred: {str(e)}</p>"
        message = "An error occurred while loading the dataset."

    # Render the template and display the dataset and message
    return render_template("dataset.html", dataset_html=dataset_html, message=message)

@app.route("/preprocess")
def preprocess_and_split_data():
    global sc,sc1, X_train, X_test, y_train, y_test
    Y = dataset.values[:,1:2]
    Y = dataset['Dissolved Oxygen (mg/L)'].ravel()
    Y1 = dataset.values[:,4:5]
    dataset.drop(['Sample ID','Dissolved Oxygen (mg/L)'], axis = 1,inplace=True)
    columns = dataset.columns.tolist()
    X = dataset.values
    #now create and apply LIGHtGBM object on water dataset
    clf = lgb.LGBMRegressor(num_leaves=100,learning_rate=0.2,boosting_type='gbdt',n_estimators=5)
    clf.fit(X, Y)
    #calculate importance
    feature_importances = (clf.feature_importances_ / sum(clf.feature_importances_)) * 100
    results = pd.DataFrame({'Features': columns, 'Importances': feature_importances})
    results.sort_values(by='Importances', inplace=True)
    print()
    print("Temperature having less importance value so it will be removed out")
    print(results)
    columns = results['Features'].tolist()
    importance = results['Importances'].tolist()
    for i in range(len(columns)):
        if importance[i] < 20:
            dataset.drop([columns[i]], axis = 1,inplace=True)
    X = dataset.values
    sc = MinMaxScaler(feature_range = (0, 1))
    sc1 = MinMaxScaler(feature_range = (0, 1))
    X = sc.fit_transform(X)
    Y = sc1.fit_transform(Y1)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
    print("Total records found in dataset = "+str(X.shape[0]))
    print("Total features found in dataset after LIGHTGBM selection : "+str(X.shape[1]))
    print("80% dataset for training : "+str(X_train.shape[0]))
    print("20% dataset for testing  : "+str(X_test.shape[0]))
    
    total_size = X.shape[0]
    training_size = X_train.shape[0]
    testing_size = X_test.shape[0]
    
    return render_template("preprocess.html", 
                           total_size=total_size,
                           training_size=training_size, 
                           testing_size=testing_size,
                           train_percentage=80, 
                           test_percentage=20)

def calculateMetrics(algorithm, predict, test_labels):
    predict = sc1.inverse_transform(predict)
    test_label = sc1.inverse_transform(test_labels)
    predict = predict.ravel()
    test_label = test_label.ravel()
    mse_value = round(mean_squared_error(test_label, predict),2)
    rmse_value = round(sqrt(mse_value),2)
    mae_value = round(mean_absolute_error(test_label, predict),2)
    mse.append(mse_value)
    rmse.append(rmse_value)
    mae.append(mae_value)
    print(algorithm+" MSE  : "+str(mse_value))
    print(algorithm+" RMSE : "+str(rmse_value))
    print(algorithm+" MAE  : "+str(mae_value))
    
    return mse_value,rmse_value,mae_value



@app.route('/existing_alg')
def existing_algorithm():
    lstm = Sequential()
    lstm.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
    lstm.add(Dropout(0.2))
    lstm.add(LSTM(units = 50, return_sequences = True))
    lstm.add(Dropout(0.2))
    lstm.add(LSTM(units = 50, return_sequences = True))
    lstm.add(Dropout(0.2))
    lstm.add(LSTM(units = 50))
    lstm.add(Dropout(0.2))
    lstm.add(Dense(units = 1))
    lstm.compile(optimizer = 'adam', loss = 'mean_squared_error')
    if os.path.exists('model/lstm_weights.hdf5') == False:
        model_check_point = ModelCheckpoint(filepath='model/lstm_weights.hdf5', verbose = 1, save_best_only = True)
        lstm.fit(X_train, y_train, epochs = 250, batch_size = 4, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    else:
        lstm = load_model('model/lstm_weights.hdf5')
    predict = lstm.predict(X_test)
    
    # Get accuracy metrics from calculateMetrics
    mse_value, rmse_value, mae_value = calculateMetrics("Existing LSTM:", predict, y_test)
    
    # Pass metrics to the template
    return render_template("existing_alg.html", 
                           mse_value=mse_value, 
                           rmse_value=rmse_value, 
                           mae_value=mae_value,  
                           )


@app.route('/proposed_alg')
def proposed_algorithm():
    bisru = Sequential()
    bisru.add(Bidirectional(SimpleRNN(units = 50, input_shape = (X_train.shape[1], X_train.shape[2]), return_sequences=True)))
    bisru.add(Dropout(0.4))
    bisru.add(Bidirectional(SimpleRNN(units = 50, return_sequences = True)))
    bisru.add(Dropout(0.4))
    bisru.add(Bidirectional(SimpleRNN(units = 50, return_sequences = True)))
    bisru.add(Dropout(0.4))
    bisru.add(attention(return_sequences=True,name='attention')) # define Attention layer
    bisru.add(Bidirectional(SimpleRNN(units = 50)))
    bisru.add(Dropout(0.4))
    bisru.add(Dense(units = 1))
    bisru.compile(optimizer = 'adam', loss = 'mean_squared_error')
    if os.path.exists('model/bisru_weights.hdf5') == False:
        model_check_point = ModelCheckpoint(filepath='model/bisru_weights.hdf5', verbose = 1, save_best_only = True)
        bisru.fit(X_train, y_train, epochs = 250, batch_size = 4, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    else:
        bisru = load_model('model/bisru_weights.hdf5')
    predict = bisru.predict(X_test)    
    # Get accuracy metrics from calculateMetrics
    mse_value, rmse_value, mae_value = calculateMetrics("Proposed LightGBM-BISRU-Attention:", predict, y_test)
    
    # Pass metrics to the template
    return render_template("proposed_alg.html", 
                           mse_value=mse_value, 
                           rmse_value=rmse_value, 
                           mae_value=mae_value,  
                           )

@app.route('/extension_alg')
def extension_algorithm():
    global extension
    #now train extension ensemble model by combining all 3 different algorithms as a single model
    extension = Sequential()
    #adding BILSTM 
    extension.add(Bidirectional(LSTM(units = 50, input_shape = (X_train.shape[1], X_train.shape[2]), return_sequences=True)))
    extension.add(Dropout(0.5))
    #adding BIGRU as ensemble
    extension.add(Bidirectional(GRU(units = 50, return_sequences = True)))
    extension.add(Dropout(0.5))
    #adding BISRU as ensemble
    extension.add(Bidirectional(SimpleRNN(units = 50, return_sequences = True)))
    extension.add(Dropout(0.5))
    extension.add(attention(return_sequences=True,name='attention')) # define Attention layer
    extension.add(Bidirectional(SimpleRNN(units = 50)))
    extension.add(Dropout(0.5))
    extension.add(Dense(units = 1))
    extension.compile(optimizer = 'adam', loss = 'mean_squared_error')
    if os.path.exists('model/extension_weights.hdf5') == False:
        model_check_point = ModelCheckpoint(filepath='model/extension_weights.hdf5', verbose = 1, save_best_only = True)
        extension.fit(X, Y, epochs = 250, batch_size = 4, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    else:
        extension = load_model('model/extension_weights.hdf5', custom_objects={'attention': attention})
    predict = extension.predict(X_test)
    # Get accuracy metrics from calculateMetrics
    mse_value, rmse_value, mae_value = calculateMetrics("Extension Ensemble LightGBM-BISRU-Attention:", predict, y_test)
    
    # Pass metrics to the template
    return render_template("extension_alg.html", 
                           mse_value=mse_value, 
                           rmse_value=rmse_value, 
                           mae_value=mae_value,  
                           )


@app.route('/graph')
def display_graph():
    df = pd.DataFrame([['LightGBM-LSTM','MSE',mse[0]],['LightGBM-LSTM','RMSE',rmse[0]],['LightGBM-LSTM','MAE',mae[0]],
                    ['Propose LightGBM-BISRU-Attention','MSE',mse[1]],['Propose LightGBM-BISRU-Attention','RMSE',rmse[1]],['Propose LightGBM-BISRU-Attention','MAE',mae[1]],
                    ['Extension Ensemble LightGBM-BISRU-Attention','MSE',mse[2]],['Extension Ensemble LightGBM-BISRU-Attention','RMSE',rmse[2]],['Extension Ensemble LightGBM-BISRU-Attention','MAE',mae[2]],
                    ],columns=['Parameters','Algorithms','Value'])

    # Create the bar graph
    fig, ax = plt.subplots(figsize=(5, 3))  # Increase the figure size (10x6 inches)
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', ax=ax)

    # Set the title and labels
    ax.set_title("Algorithms Performance Comparison", fontsize=6)
    ax.set_xlabel("Metrics", fontsize=4)
    ax.set_ylabel("Values", fontsize=4)
    
    # Adjust tick labels for better readability
    plt.xticks(rotation=45, ha="right", fontsize=6)
    plt.yticks(fontsize=4)

    # Move the legend outside the plot
    plt.legend(title='Algorithms', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=4)

    # Save the graph as an image with high DPI for clarity
    graph_path = os.path.join(app.static_folder, 'graph.png')
    plt.tight_layout()  # Ensure everything fits well
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')  # Save the plot with 300 DPI for higher resolution
    plt.close()  # Close the plot to free memory

    # Render the HTML template and pass the image path
    return render_template("graph.html", graph_url='/static/graph.png')



@app.route('/predict')
def upload():
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def upload_file():
    global df,extension_model
    if 'testdata' not in request.files:
        message = 'No file selected'
        return render_template('predict.html', message=message)

    dataset = request.files['testdata']

    if dataset.filename == '':
        message = 'No selected file'
        return render_template('upload.html', message=message)

    if dataset and allowed_file(dataset.filename):
        filename = secure_filename(dataset.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        dataset.save(filepath)
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            testData = pd.read_csv(filepath)#reading test data
            testData = testData.values
            test = sc.transform(testData)
            test = np.reshape(test, (test.shape[0], test.shape[1], 1))
            extension = load_model('model/extension_weights.hdf5', custom_objects={'attention': attention})
            predict = extension.predict(test)#predict with extension object
            predict = sc1.inverse_transform(predict)
            predict = predict.ravel()
            for i in range(len(testData)):
                print("Test Data : "+str(testData[i])+" =====> Predicted Oxygen : "+str(predict[i])+"\n")

            results = []
            # Collect the prediction results
            for i in range(len(predict)):
                # Formatting the test data to make it readable
                formatted_test_data = ", ".join([str(value) for value in testData[i]])

                # Appending to the results list
                results.append({
                    'test_data': formatted_test_data, 
                    'predicted_performance': str(predict[i])
                })

            return render_template('predict.html', results=results)
        except Exception as e:
            message = f"Error processing file: {e}"
        return render_template('predict.html', message=message)
    else:
        message = 'Allowed file types: .csv'
        return render_template('predict.html', message=message)


@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)