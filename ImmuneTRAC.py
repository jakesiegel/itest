# Another example chaining Bokeh's to Flask.

from flask import Flask, render_template, redirect, request, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import seaborn as sns
import xgboost
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from bokeh.sampledata.iris import flowers
from bokeh.plotting import figure
from bokeh.embed import components

df = pd.read_csv('https://data.boston.gov/dataset/c8b8ef8c-dd31-4e4e-bf19-af7e4e0d7f36/resource/29e74884-a777-4242-9fcc-c30aaaf3fb10/download/economic-indicators.csv',
                 parse_dates=[['Year', 'Month']])
length = len(df)

dbname = 'TCR_db'
username = 'jakesiegel'
pw = 'Berkeley'
# engine to connect to the db
# make sure PostGRES is running
engine = create_engine('postgresql://%s:%s@localhost/%s'%(username,pw,dbname))
print 'postgresql://%s:%s@localhost/%s'%(username,pw,dbname)
print engine.url
# create a database (if it doesn't exist)
if not database_exists(engine.url):
    create_database(engine.url)
print(database_exists(engine.url))
print engine.url

con = None
con = psycopg2.connect(database = dbname, user = username, host='localhost', password=pw)

def make_plot():
	# generic call to PDA via sql query to get the data
	sql_query = """
	SELECT * FROM key WHERE condition = 'cont_t2' or condition = 'TherC_t2';
	"""
	TC_t1_key = pd.read_sql_query(sql_query,con)
	TC_t1_list=TC_t1_key['ID'].tolist()
	y_all = pd.get_dummies(TC_t1_key.loc[:,'condition']).iloc[:,0]
	y_all = y_all.values
# 	print y_all

	sql_query = """
	SELECT * FROM scaled_sorted_data_table;
	"""
	x_all = pd.read_sql_query(sql_query,con)
	x_all.set_index('cdr3',inplace=True)
	x_all = x_all.loc[:,TC_t1_list].transpose()
	x_all.head()
	x_all = x_all.values

	# print np.any(np.isnan(x_all))
	# print np.any(np.isfinite(x_all))

	pca = PCA(n_components=6)
	X_r = pca.fit(x_all).transform(x_all)

	# Percentage of variance explained for each components
	print('explained variance ratio (first n components): \n %s'
		% str(pca.explained_variance_ratio_))

# 	colormap = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
# 	colors = [colormap[x] for x in flowers['species']]
# 
# 	p = figure(title = "Iris Morphology")
# 	p.xaxis.axis_label = 'Petal Length'
# 	p.yaxis.axis_label = 'Petal Width'
# 
# 	p.circle(flowers["petal_length"], flowers["petal_width"],
# 		color=colors, fill_alpha=0.2, size=10)
# 
# 	return p
	p = figure(title = "PCA scatter plot")
	colors = ['navy', 'darkorange']
	lw = 2

	for color, i, in zip(colors, [0, 1]):
	    p.circle(X_r[y_all == i, 0], X_r[y_all == i, 1], color=color, fill_alpha=.5, size=10)
	p.xaxis.axis_label = 'Principle Component 1'
	p.yaxis.axis_label = 'Principle Component 2'

	return p
# 	return

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
	if request.method=='POST':
		return (url_for('PCApage'))
	return render_template('index.html')

@app.route('/PCA', methods=['GET','POST'])
def PCApage(*args):
    figure = make_plot()
    fig_script, fig_div = components(figure)
    return render_template('data.html', fig_script=fig_script, fig_div=fig_div)


if __name__ == '__main__':
    app.run(debug=True)
