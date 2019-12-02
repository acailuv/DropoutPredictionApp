# Student Dropout Prediction App
Predict whether a student will be dropped out when goinf through university using 6 most influential features. The dataset provided in this repository is modified from Kaggle.

Since we do not have access to the problem, it is not impossible that we might have missed some extra information that might affect overall predictions. This means that if something is a little bit odd in this prediction app, we cannot say for sure why or how the artificial intelligence predict so. I hope you understand this handicap, enjoy :).

## Team Members
- Leon Chrisdion
- Liang Cai
- Sebastian Aldi
- Sutedja The Ho Ping

## Requirements
- Python Version 3.6 or higher. [(Download Python here)](https://www.python.org/downloads/)
	- Python 3.6 or higher should include pip if you correctly tick the checkbox that says "Add to PATH".
	- Refer to the tutorial below if could not install libraries through pip even though you have 	installed Python.
- Install pip. [(Tutorial how to Install pip)](https://www.makeuseof.com/tag/install-pip-for-python/)
- Install dash. 
	- Open your command prompt or console.
	- Type `pip install dash`.
	- Wait until finished.
- Install numpy.
	- Open your command prompt or console.
	- Type `pip install numpy`.
	- Wait until finished.
- Install pandas.
	- Open your command prompt or console.
	- Type `pip install pandas`.
	- Wait until finished.
- Install matplotlib.
	- Open your command prompt or console.
	- Type `pip install matplotlib`.
	- Wait until finished.
- Install plotly.
	- Open your command prompt or console.
	- Type `pip install plotly`.
	- Wait until finished.
- Install sklearn.
	- Open your command prompt or console.
	- Type `pip install sklearn`.
	- Wait until finished.
- Download the dataset needed. [(Download dataset here)](https://gofile.io/?c=fEgkoS)
	- Put the `.csv` in the same folder as the python script. (`cleanapp.py`)

## Setting Up
Double click `cleanapp.py` or type `python cleanapp.py` in Command Prompt, then browse `localhost:8050`

## The Application
When you open up the browser, this page will show up by default:

There are sliders at the top to input the Cumulative GPA, then the next column you have :
* `Total Debt` : to input your total debt.
* `Loan` : to input your loan.
* `Grants` : to input your grants from any sources.
* `Adjusted Gross Income` : to input your income if you have any income
* `Parent Adjusted Gross Income` : to input your parent's income

We have `~98%` accuracy. Note that this value vary each execution.

Confusion Matrix Pict

According to the data that can be visibly found within the confusion matrix, we can see that there is a significant difference between the wrong guessings (blue areas means low concentration od occurences) and the correct guessings (red/beige colored means high concentration of occurences) and this means that the Random Forest can correctly guess the correct result with high accuracy 

Correlation Pict

According to the Correlation data, shown that the red areas mean that the correlation is a positive correlation, meaning that if the x axis' value is high, the y axis' value will be high also. If the colour is blue then it is a negative correlation, meaning that if the x axis' value is high, then the y axis' value will be the opposite from it.

You're safe example
## The Script
-----
This section will explain the script and the libraries that used, and how the machine work and how the script projects the data in form of graph via Dash.

### **Part One: Essential Libraries**

The libraries needed for this script to run. We will explain each library's role.
First off: `dash`, `dash_core_components`, and `dash_html_component`.
* `dash` is used to create web based application.
* `dash_core_components` is used to create graphs within the appliation. 
* `dash_html_components` is used as a source of html tags like `<p>`, `<div>`, and many other things. The tags are called like this: `html.P()` in the script. Where html is the 'alias' of the `dash_html_components`. Because the library is imported like this: `import dash_html_components as html`,`html` is what we meant by 'alias'.

Next, `matplotlib.pyplot` is used for the graph objects, then `numpy` and `pandas` are used to execute the mathematical operations. Finally, `sklearn` is used to do machine learning and for confusion matrix and accuracy `(from sklearn.metrics import confusion_matrix, accuracy_score)`, and train test split `(from sklearn.model_selection import train_test_split)`.

### **Part Two: How It Works**

First, it will reading the CSV File into a dataframe.

csv pict

Second, we do reformat and cleansing data frame.

dataframe pict

Third, after doing reformat and cleansing the data frame, we choose the top scoring features based on feature importance from dataframe like : [`Total Debt`, `Grants`, `Loans`, `Adjusted Gross Income`, `Parent Adjusted Gross Income`, `CumGPA`, `Dropout`], then we set the 'Drop out' as the label so it can be predicted by this function `labels = np.array(selected_features['Dropout'])`, after that we drop the label from the original dataframe.

feature pic

Fourth, we train the data, after training the data, we doing testing for the data, and then we split the data, some of the data become subsets where the subsets(selected at random) are trained and tested.

test pic

Fifth, random forest classifier that has a decision tree inside.

randomforest pic
![Decision Tree Illustration](https://miro.medium.com/max/2612/0*f_qQPFpdofWGLQqc.png)

Finally, we print out the accuracy by plugging in the labels marked by `test_labels` variable and the predictions marked by `y_pred` variable into a function specifically designed for calculating accuracy which is `accuracy_score()`.

Next comes the interesting part. The Application itself.

app_initialize

In here we declare the stylesheet we want to impor into our application and initialize the app by making a `Dash` object.

Then, we need to declare the layout so that the form will be rendered in the application.

layout_declare

Next, we need to insert HTML element such as `<p>` or `<h1>` into our application

header_dataset_links

With that done, the these lines of codes are the commands for rendering the prediction form

prediction_form

Lastly, the graphs that shows additional data such as correlation between features and confusion matrix that shows the amount of inaccuracy visually. That can be achieved by typing in the following code:

graph_code

We also need to declare callback functions to react accordingly to user input. This can be achieved using the following code:

callback

## Dataset Handling
The dataset is very messy, that is why we need to fill in the gaps accordingly. We decided to remove broken data from the dataset and then fill it some of itself to improve accuracy.

We use [One Hot Encoding](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f) to symbolize character values in such a way that certain string is symbolized by a number. Refer to the illustration below to see One Hot Encoding in action:

![One Hot Encoding](https://hackernoon.com/photos/4HK5qyMbWfetPhAavzyTZrEb90N2-3o23tie)