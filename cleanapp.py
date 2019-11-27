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
dataframe = dataframe.drop(['cohort', 'StudentID', 'cohort term'] , axis=1)
dataframe = dataframe.drop(['Marital Status', 'Housing'], axis=1)

# Choose the top scoring features based on feature importance
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
y_pred = top_clf.predict(test_features) # to be used in confusion matrix
y_true = test_labels # to be used in confusion matrix

# accuracy
acc = accuracy_score(test_labels, y_pred) * 100

print("Random Forest accuracy:", accuracy_score(test_labels, y_pred))

### DASH CODING STARTS HERE

external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(className="container p-5", children=[
    html.H1(children='Predicting Student Dropout Using Random Forest'),


    html.Div([
        html.Span("Cumulative GPA:"), html.Span(id='current-gpa-selected'),
        html.Br(),
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
        html.P("Accuracy: {}%".format(acc)),
        html.Button("Submit", id='submit-button', className="btn btn-success")

    ], style={"float": "left", "width": "40%"}),

    html.Div(id='result_of_prediction', children='Enter the values and press the button to predict.',
    style={"float": "left", "width": "50%", "margin-left": "5%", "margin-top": "5%"})
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

@app.callback(
    Output('current-gpa-selected', 'children'),
    [Input('gpa', 'value')]
)
def update_output(value):
    return ' {}'.format(value)

### DASH CODING ENDS HERE
if __name__ == '__main__':
    app.run_server(debug=True)
