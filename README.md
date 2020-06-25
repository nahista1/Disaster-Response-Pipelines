# Disaster Response Pipeline Project

### Summary:
This is a web application written in python to automatically classify text messages into different categories of disaster management. Furthermore, two visualizations from the data set are available as examples. The project consists of 3 areas.

ETL pipeline (process_data.py) to cleans data and store data for further processing
ML pipeline (train_classifier.py) to train and save a classifier model
Flask web app (run.py) for interactive classification based on trained model of arbitary text data and visualisations of some data
### Requierments:
You need following python packages:

-  flask
- plotly
- sqlalchemy
- pandas
- numpy
- sklearn
- nltk

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/




  [mylink](https://view6914b2f4-3001.udacity-student-workspaces.com)
