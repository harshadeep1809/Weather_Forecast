from keras.models import load_model
model = load_model(r'model.h5',compile=True)

import keras
import pandas as pd
dd = pd.read_csv("https://raw.githubusercontent.comharshadeep1809/Time_Series_Weather_Forecast/main/ValFeatures.csv")
y = pd.read_csv("https://raw.githubusercontent.com/harshadeep1809/Time_Series_Weather_Forecast/main/ValLabels.csv")
import numpy as np
batch_size = 256

dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    np.array(dd),
     np.array(y),
    sequence_length=(720/6),
    sampling_rate=6,
    batch_size=batch_size)
    
import datetime
first_time = datetime.datetime.now()

later_time = datetime.datetime.strptime('01.07.2017 00:00:00', '%d.%m.%Y %H:%M:%S')

difference =  first_time - later_time
stps = difference.total_seconds()//3600

import pandas as pd
from dash.dependencies import Input, Output
import plotly.express as px
import dash_table as dt

import pandas as pd
from dash.dependencies import Input, Output
import plotly.express as px


import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import base64




import urllib.request
img = urllib.request.urlretrieve("https://image.freepik.com/free-vector/family-wearing-face-masks_52683-38547.jpg", "gender.jpg")


encoded_image = base64.b64encode(open(img[0], 'rb').read())
tab1 = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

img2 = urllib.request.urlretrieve("https://static01.nyt.com/images/2014/12/11/technology/personaltech/11machin-illo/11machin-illo-articleLarge-v3.jpg?quality=75&auto=webp&disable=upscale", "mask.jpg")
encoded_image2 = base64.b64encode(open(img2[0], 'rb').read())
tab2 = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


body = html.Div([
    dbc.Row([
               dbc.Col(html.Div(dbc.Alert("** The model is trained on Jena Climate dataset recorded by the Max Planck Institute for Biogeochemistry. The predictions are for next hours from 26.12.2016 12:00.", color="info",style={'height':'90px','font-size':16,'font-style':'italic','fontWeight': 'bold','font-family':"Arial"}))),
               
                dbc.Col(dcc.Input(id="input1", type="text", placeholder="Enter the number of hours to be predicted", style={'marginRight':'10px','width':'450px', 'height':50}),)
              ],className="mt-2"),
        dbc.Row([
            dbc.Col([
                     
                     dbc.Row([dbc.Col(html.Div(id="grp1")), dbc.Col(html.Div([
    html.Img(src='data:image/jpg;base64,{}'.format(encoded_image2.decode()), 
             style={'height': '300px','height': '500px',"margin-left": "20px","margin-right":'10-px'})]))])], className="mt-2")])])

    
tab1.layout = html.Div([body])

#adding all the options like dropdowns that should be there in the app

body1 = html.Div([
    
               
    dbc.Row([dbc.Col(html.Div(dbc.Card([dbc.CardHeader("Pressure (mpbar)",style={
                    'color':'black','font-weight': 'bold'
                }),dbc.CardBody(dcc.Input(id="input2", type="text", style={'marginRight':'10px','width':'450px', 'height':50}))],
                                       color='warning',style={'height':'13vh','width':500,'margin-left':30})))
              ,
             dbc.Col(html.Div(dbc.Card([dbc.CardHeader("Temperature (deg C)",style={
                    'color':'white','font-weight': 'bold'
                }),dbc.CardBody(dcc.Input(id="input3", type="text", style={'marginRight':'10px','width':'450px', 'height':50}))],
                                       color='secondary',style={'height':'13vh','width':500,'margin-left':30}))),
             
            dbc.Col(html.Div(dbc.Card([dbc.CardHeader("Saturation Pressure (mpbar)",style={
                    'color':'white','font-weight': 'bold'
                }),dbc.CardBody(dcc.Input(id="input4", type="text", style={'marginRight':'10px','width':'450px', 'height':50}))],
                    color='secondary',style={'height':'13vh','width':500,'margin-left':30})))],className="mt-2"),
    
    dbc.Row([dbc.Col(html.Div(dbc.Card([dbc.CardHeader("Vapor pressure deficit",style={
                    'color':'white','font-weight': 'bold'
                }),dbc.CardBody(dcc.Input(id="input5", type="text", style={'marginRight':'10px','width':'450px', 'height':50}))],
                                       color='secondary',style={'height':'13vh','width':500,'margin-left':30}))),
             
            dbc.Col(html.Div(dbc.Card([dbc.CardHeader("Specific Humidity",style={
                    'color':'white','font-weight': 'bold'
                }),dbc.CardBody(dcc.Input(id="input6", type="text", style={'marginRight':'10px','width':'450px', 'height':50}))],
                                       color='secondary',style={'height':'13vh','width':500,'margin-left':30}))),
              
             dbc.Col(html.Div(dbc.Card([dbc.CardHeader("Airtight",style={
                    'color':'white','font-weight': 'bold'
                }),dbc.CardBody(dcc.Input(id="input7", type="text", style={'marginRight':'10px','width':'450px', 'height':50}))],
                                       color='secondary',style={'height':'13vh','width':500,'margin-left':30}))),
              
             
           
            ],className="mt-2"),
    
    dbc.Row([dbc.Col(html.Div(dbc.Card([dbc.CardHeader("Wind speed (m/s)",style={
                    'color':'black','font-weight': 'bold'
                }),dbc.CardBody(dcc.Input(id="input8", type="text", style={'marginRight':'10px','width':'450px', 'height':50}))],color='warning',style={'height':'13vh','width':500,'margin-left':30})))],className="mt-2"),
    
  
   
      
   
   
    dbc.Row([dbc.Col(dbc.Col(html.Div(id="grp2")))],className="mt-2"),
    


      ])
       
 

tab2.layout = html.Div([body1])


import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import base64



import urllib.request
img = urllib.request.urlretrieve("https://raw.githubusercontent.com/mllover5901/dat/main/gender-equality.jpg", "gender.jpg")


encoded_image = base64.b64encode(open(img[0], 'rb').read())

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


app.config['suppress_callback_exceptions'] = True
server = app.server

app.layout = html.Div([
    html.H1('Weather Forecasting'),
    dcc.Tabs(id="tabs-example", value='tab-1-example', children=[
        dcc.Tab(label='Weather Forecasting', value='tab-1-example',style={'color':'white','font-size':25}),
        dcc.Tab(label='Anomaly Detection', value='tab-2-example',style={'color':'white','font-size':25})
      
    ],colors={
            "border": "white",
            "primary": "black",
            "background": "black"}),
    html.Div(id='tabs-content-example')
])



@app.callback(Output('tabs-content-example', 'children'),
              [Input('tabs-example', 'value')])
def render_content(tab):
    if tab == 'tab-1-example':
        return tab1.layout
    elif tab == 'tab-2-example':
        return tab2.layout
@app.callback(Output('grp1', 'children'),
                                  [Input('input1', 'value')])
def update_figure1(area):
    
    
    
            import keras
            import pandas as pd
            dd = pd.read_csv("https://raw.githubusercontent.com/Nibs007/LSTM/main/ValFeatures.csv")
            y = pd.read_csv("https://raw.githubusercontent.com/Nibs007/LSTM/main/ValLabels.csv")
            import numpy as np
            batch_size = 256

            dataset_val = keras.preprocessing.timeseries_dataset_from_array(
                np.array(dd),
                 np.array(y),
                sequence_length=(720/6),
                sampling_rate=6,
                batch_size=batch_size)
    
            import datetime
           
            if area!= None:
                stps = int(area)
                
                pred=[]
                p=[]
                from tqdm import tqdm
                
                #The trained model above will now be able to make predictions for 5 sets of values from validation set.
                for x, y in tqdm(dataset_val.take(int(stps))):
                       m = ((model.predict(x)[0])*8.635 + 9.25)[0]
                       p.append(round(m,1))
     
                p1 =[round(float(i),1) for i in p]
                pred.append(p1)
            else:
                stps=1
                pred=['No Input']

            
            dd = pd.DataFrame(pred)
            #dd.columns =['Predicted Temperature For next 10 hours']

            data = dd.to_dict('rows')
            columns =   [{"name": 'Prediction (deg C) for Next '+ str(i+1) + ' hour', "id": str(i),} for i in range(0,stps)]
            return dt.DataTable(data=data,columns=columns,style_header={ 'whiteSpace': 'normal','height': 'auto','backgroundColor': 'rgb(30, 30, 30)','color':'white','font_size':18},
                style_table={'overflowX': 'auto'},
                style_cell={'backgroundColor': 'white',
                    'color': 'black','font_size':18,'height':100,'fontWeight': 'bold',
                    # all three widths are needed
                    'minWidth': '300px', 'width': '300px', 'maxWidth': '300px',
                    'overflow': 'hidden','border': '1px solid grey',"margin-left": "20px","margin-left": "40px",
                    'textOverflow': 'ellipsis','textAlign': 'center','whiteSpace': 'normal'
       
                })
   
@app.callback(Output('grp2', 'children'),
                                  [Input('input2', 'value'),Input('input3', 'value'),Input('input4', 'value'),
                                  Input('input5', 'value'),Input('input6', 'value'),Input('input7', 'value'),
                                  Input('input8', 'value')])
def update_figure1(a1,a2,a3,a4,a5,a6,a7):
             from keras.models import load_model
             model = load_model('AnomalyDetection.h5', compile = True)
            
             import pandas as pd
             meanst = pd.read_csv("https://raw.githubusercontent.com/Nibs007/LSTM/main/MeanStd_Anomaly.csv")
             if a1!= None and a2!= None and a3!=None and a4!= None and a5!= None and a6!= None and a7!=None:
                x = [a1,a2,a3,a4,a5,a6,a7]
                x = [float(i) for i in x]
                x1 = pd.DataFrame([x])

                def normalize(data):
                    data_mean =meanst['Mean'].values.tolist()
                    data_std = meanst['Std Dev'].values.tolist()
                    return (data - data_mean) / data_std

                p = normalize(x1)
                import numpy as np
                X_test = np.array(p).reshape(p.shape[0], 1, p.shape[1])
                X_pred = model.predict(X_test)
                X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
                X_pred = pd.DataFrame(X_pred)
                #X_pred.index = val_data.index

                scored = pd.DataFrame()
                Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
                scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtest), axis = 1)
                scored['Threshold'] = 0.38
                def mak(col1,col2):
                    if col1> col2:
                        return 'Anomaly Detected'
                    elif col1<=col2:
                        return 'Not an Anomaly'
                scored['Anomaly'] = scored.apply(lambda e: mak(e['Loss_mae'],e['Threshold']),axis=1)
                scored = scored['Anomaly'].values.tolist()


           
             else:
                
                scored=['No Input']

             mod=[]
             for it in scored:
                    if scored=='false':
                        mod.append('It is not an Anomaly')
                    elif scored == 'true':
                        mod.append('It is an Anomaly')
             dd = pd.DataFrame(scored)
             

             data = dd.to_dict('rows')
             columns =   [{"name": 'Is it an Anamoly', "id": str(i),} for i in range(0,len(dd.columns))]
             return dt.DataTable(data=data,columns=columns,style_header={ 'whiteSpace': 'normal','height': 'auto','backgroundColor': 'rgb(30, 30, 30)','color':'white','font_size':25},
                style_table={'overflowX': 'auto'},
                style_cell={'backgroundColor': 'white',
                    'color': 'black','font_size':18,'height':100,'fontWeight': 'bold',
                    # all three widths are needed
                    'minWidth': '300px', 'width': '300px', 'maxWidth': '300px',
                    'overflow': 'hidden','border': '1px solid grey',"margin-left": "20px","margin-left": "40px",
                    'textOverflow': 'ellipsis','textAlign': 'center','whiteSpace': 'normal'
       
                })
  
if __name__ == '__main__':
    app.server.run(debug=True, threaded=True)
