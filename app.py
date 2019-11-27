# Library Imports
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Read csv file into a dataframe
dataframe = pd.read_csv('combined_data.csv')

# Data reformatting and cleansing
dataframe = dataframe.dropna()
dataframe = dataframe[dataframe['Father\'s Highest Grade Level'] != 'Unknown']
dataframe = dataframe[dataframe['Mother\'s Highest Grade Level'] != 'Unknown']

# dataframe['Year'] = dataframe['cohort'].str.extract(r'^(\d{4})', expand=False).astype(int)
dataframe = dataframe.drop(['cohort', 'StudentID', 'cohort term'] , axis=1)
dataframe = dataframe.drop(['Marital Status', 'Housing'], axis=1) # Columns with one hot encoding

# Map to linear scale
def grade_level_linear(x):
    if x == 'Middle School':
        return 0
    elif x == 'High School':
        return 1
    elif x == 'College':
        return 2
    else:
        return -1

dataframe['Father\'s Highest Grade Level'] = dataframe.apply(lambda x: grade_level_linear(x['Father\'s Highest Grade Level']), axis=1)
dataframe['Mother\'s Highest Grade Level'] = dataframe.apply(lambda x: grade_level_linear(x['Mother\'s Highest Grade Level']), axis=1)

# One Hot Encoding
# dataframe = pd.get_dummies(dataframe)

# # Set 'Dropout' as the label to be predicted
# labels = np.array(dataframe['Dropout'])
# # Drop the label from the original dataframe
# features = dataframe.drop('Dropout', axis=1)
# feature_list = list(features.columns)
# features = np.array(features)

# # # Check for uniques
# # for x in feature_list:
# #     if len(dataframe[x].unique()) < 10:
# #         print(x, dataframe[x].unique())
# #     else: print(x, len(dataframe[x].unique()))

# # Train_Test_Split
# train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25)

# # Random Forest
# classifier = RandomForestClassifier(n_estimators = 100, min_samples_leaf=10, min_samples_split=10)
# classifier.fit(train_features, train_labels)
# y_pred = classifier.predict(test_features)

# print("Random forest accuracy:", accuracy_score(test_labels, y_pred))

# # How important is each feature?
# feature_importance = pd.DataFrame(classifier.feature_importances_, index = feature_list, columns = ['importance']).sort_values('importance', ascending=False)
# print(feature_importance)
# # Most important features
selected_features = dataframe[['Total Debt', 'Grants', 'Loans', 'Adjusted Gross Income', 'Parent Adjusted Gross Income', 'CumGPA', 'Dropout']]

# Set 'Dropout' as the label to be predicted
labels = np.array(selected_features['Dropout'])

# Drop the label from the original dataframe
features = selected_features.drop('Dropout', axis=1)
feature_list = list(features.columns)
features = np.array(features)

# Train_Test_Split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25)

# Random Forest with only top scoring features
top_clf = RandomForestClassifier(n_estimators = 100, min_samples_leaf=10, min_samples_split=10)
top_clf.fit(train_features, train_labels)
y_pred = top_clf.predict(test_features)

print("High Score Accuracy:", accuracy_score(test_labels, y_pred))

### DASH CODING STARTS HERE

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Predicting Student Dropout Using Random Forest'),


    html.Div([
        html.P("Cumulative GPA:"),
        dcc.Slider(
            id='gpa',
            min=0, max=4, step=0.01, value=3,
            marks={i: '{}'.format(i) for i in range(5)}
        ),
        html.Br(),
        html.P("Total Debt"),
        dcc.Input(id='debt', type='number', style={"margin-bottom":"10px"}),
        html.Br(),
        html.P("Loan"),
        dcc.Input(id='loan', type='number', style={"margin-bottom":"10px"}),
        html.Br(),
        html.P("Grants"),
        dcc.Input(id='grant', type='number', style={"margin-bottom":"10px"}),
        html.Br(),
        html.P("Adjusted Gross Income"),
        dcc.Input(id='agi', type='number', style={"margin-bottom":"10px"}),
        html.Br(),
        html.P("Parent Adjusted Gross Income"),
        dcc.Input(id='pagi', type='number', style={"margin-bottom":"10px"}),
        html.Br(),
        
        html.Button("Submit", id='submit-button')

    ], style={"float": "left", "width": "40%"}),

    html.Div(id='result_of_prediction', children='Enter the values and press the button to predict.',
    style={"float": "left", "width": "50%", "margin-left": "1%"})
])
@app.callback(
    Output('result_of_prediction', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('gpa', 'value'),
    State('debt', 'value'),
    State('loan', 'value'),
    State('grant', 'value'),
    State('agi', 'value'),
    State('pagi', 'value')]
)
def predict(*args):
    values = list(args)
    if(values[0] is None):
        return html.P('Enter the values and press the button to predict.')
    values.pop(0)
    text_out = ""
    if None in values:
        text_out = "A value has been left empty."
    else:        
        prediction = top_clf.predict([values])
        if prediction[0] == 0:
            text_out = "With these values, you will drop out."
        else:
            text_out = "With these values, you are safe."
    return html.P(text_out)

### DASH CODING ENDS HERE
if __name__ == '__main__':
    app.run_server(debug=True)
