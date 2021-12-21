import dash
from dash.dependencies import Input, Output, State
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from scipy.optimize import minimize, linprog
import pandas as pd
import numpy as np
import math

app = dash.Dash(__name__)

#params = ['Customer-1', 'Customer-2', 'Customer-3', 'Customer-4', 'Customer-5']

customers = ['Customer-1', 'Customer-2', 'Customer-3', 'Customer-4', 'Customer-5']
warehouses = ['Warehouse-1','Warehouse-2','Warehouse-3','Warehouse-4']

costs_fw = [0.5, 0.5, 1, 0.2, 1.5, 0.3, 0.5, 0.2]
costs_fc = [1.75, 2.25, 1.50, 2.00, 1.50,  2.00, 2.50, 2.50, 1.50, 1.00]
costs_wc = [1.5, 1.5, 0.5, 1.5, 3.0,
            1.5, 0.5, 0.5, 1.0, 0.5,
            1.0, 1.5, 2.0, 2.0, 0.5,
            2.5, 1.5, 0.2, 1.5, 0.5]

costs = [1.75, 2.25, 1.50, 2.00, 1.50, 
         2.00, 2.50, 2.50, 1.50, 1.00]

app.layout = html.Div([

  dcc.Tabs([

    dcc.Tab(label='Transportation problem 1', 
            style={'width': '50%','font-size': '130%','height': '30%'},
            children=[

                html.H2('Transportation problem 1'),
                html.H5('Minimize the costs of shipping goods from factories to customers, while not exceeding the supply available from each factory and meeting the demand of each customer.'),
                html.P('Example:',style={'textAlign': 'left','font-weight': 'bold'}),
                html.Img(src='static/problem1.png',style={'width':'45%'}),
                html.Br(),
                html.H3('Cost of shipping ($ per product) and destinations',
                        style={'textAlign': 'center','font-weight': 'bold'}),

                dash_table.DataTable(
                    id='table-editing-simple',
                    columns=(
                        [{'id': 'Factory', 'name': 'Factory'}] +
                        [{'id': p, 'name': p} for p in customers]
                    ),
                    data=[
                        dict(Factory=i+1, **{param: costs[i*5+ix] for ix,param in enumerate(customers)})
                        for i in range(0, 2)
                    ],
                    editable=True
                ),

                html.H5('The table above is interactive. You may put your numbers and press "Solve problem" button.'),
                html.Button('Solve problem', id='submit-val1', n_clicks=0,
                            style={'background-color':'lightblue','font-weight': 'bold'}),
                html.Br(),

                html.Div(id='table-editing-simple-output'),
                html.Div([html.P(id = "tot_cost1",
                        style={'textAlign': 'left','font-size': '100%', 'font-weight': 'bold'},
                        children=["init"])
                        ])
    ]),

    dcc.Tab(label='Transportation problem 2', 
            style={'width': '50%','font-size': '130%','height': '30%'},
            children=[

                html.H2('Transportation problem 2'),
                html.H5('''Minimize the costs of shipping goods from factories to warehouses and customers, and		
            warehouses to customers, while not exceeding the supply available from each factory or		
            the capacity of each warehouse, and meeting the demand from each customer.'''),
                html.P('Example:',style={'textAlign': 'left','font-weight': 'bold'}),
                html.Img(src='static/problem2.png',style={'width':'45%'}),
                html.Br(),
                html.Br(),
                html.H3('Cost of shipping ($ per product) and destinations',
                        style={'textAlign': 'center','font-weight': 'bold'}),

                html.P('Costs from Factories to Warehouses',style={'textAlign': 'center','font-weight': 'bold'}),
                dash_table.DataTable(
                    id='table-editing-simple1',
                    columns=(
                        [{'id': 'Factory', 'name': 'Factory'}] +
                        [{'id': p, 'name': p} for p in warehouses]
                    ),
                    data=[
                        dict(Factory=i+1, **{param: costs_fw[i*4+ix] for ix,param in enumerate(warehouses)})
                        for i in range(0, 2)
                    ],
                    editable=True
                ),

                html.P('Costs from Factories to Customers',style={'textAlign': 'center','font-weight': 'bold'}),
                dash_table.DataTable(
                    id='table-editing-simple2',
                    columns=(
                        [{'id': 'Factory', 'name': 'Factory'}] +
                        [{'id': p, 'name': p} for p in customers]
                    ),
                    data=[
                        dict(Factory=i+1, **{param: costs_fc[i*5+ix] for ix,param in enumerate(customers)})
                        for i in range(0, 2)
                    ],
                    editable=True
                ),

                html.P('Costs from Warehouses to Customers',style={'textAlign': 'center','font-weight': 'bold'}), 
                dash_table.DataTable(
                    id='table-editing-simple3',
                    columns=(
                        [{'id': 'Warehouse', 'name': 'Warehouse'}] +
                        [{'id': p, 'name': p} for p in customers]
                    ),
                    data=[
                        dict(Warehouse=i+1, **{param: costs_wc[i*5+ix] for ix,param in enumerate(customers)})
                        for i in range(0, 4)
                    ],
                    editable=True
                ),

                html.H5('The tables above are interactive. You may put your numbers and press "Solve problem" button.'),
                html.Button('Solve problem', id='submit-val2', n_clicks=0, 
                            style={'background-color':'lightblue','font-weight': 'bold'}),
                html.Br(),

                html.Div(id='table-editing-simple-output1'),
                html.Div(id='table-editing-simple-output2'),
                html.Div(id='table-editing-simple-output3'),
                html.Div([html.P(id = "tot_cost2",
                        style={'textAlign': 'left','font-size': '100%', 'font-weight': 'bold'},
                        children=["init"])
                        ])
            ])
    ])
])

ncl1 = 0
@app.callback(
    [Output('table-editing-simple-output', 'children'),
     Output('tot_cost1', 'children')],
    [Input('submit-val1', 'n_clicks'), 
     Input('table-editing-simple', 'data'),
     Input('table-editing-simple', 'columns')])     
def display_output1(n_clicks, rows, columns):
    global ncl1

    df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    new_costs = df.values[:,1:].flatten()
    res = solver1(new_costs)

    sol = np.array([int(round(x,0)) for x in res.x])
    cust_min = np.array([30000, 23000, 15000, 32000, 16000])
    sf = [cust_min[ix]/sol.reshape(2,5)[:,ix].sum() for ix in range(5)]
    #print(sf)

    total1 = int(res.x[:5].sum())
    total2 = int(res.x[5:].sum())

    sol = np.insert(sol,5,total1)
    sol = np.append(sol,total2)

    tot_cust = sol.reshape(2,6).sum(axis=0) 

    params_t = customers + ['Total']

    if n_clicks > ncl1:
        ncl1 = n_clicks
        return html.Div([
            html.Hr(),
            html.H3('Optimized number of products shipped',
            style={'textAlign': 'center','font-weight': 'bold'}),
            html.Br(),
            html.Div('From Factories to Customers',
                     style={'textAlign': 'center','font-size': '100%','font-weight': 'bold'}),
            html.Hr(),
            dash_table.DataTable(
                columns=(
                    [{'id': 'Factory', 'name': 'Factory'}] +
                    [{'id': p, 'name': p} for p in params_t] 
                ),
                data=[
                    dict(Factory=i+1, **{param: sol[i*6+ix] for ix,param in enumerate(params_t)})
                    for i in [0,1]
                ] +
                [dict(Factory='Total', **{param: tot_cust[ix] for ix,param in enumerate(params_t)})],
                editable=False
                )
            ]), 'Total cost of shipping: {:.2f}'.format(res.fun)
    else:
        return html.Div([
        ]),''

def solver1(new_costs):
    #Objective:         
    fobj = new_costs

    #Constraints:
    A_ub = [list(np.ones(5)) + list(np.zeros(5)),
            list(np.zeros(5)) + list(np.ones(5)),
        [-1,0,0,0,0, -1,0,0,0,0],
        [0,-1,0,0,0, 0,-1,0,0,0],
        [0,0,-1,0,0, 0,0,-1,0,0],
        [0,0,0,-1,0, 0,0,0,-1,0],
        [0,0,0,0,-1, 0,0,0,0,-1]
        ]

    b_ub = [60000, 60000, 
           -30000, -23000, -15000, -32000, -16000]

    #Bounds
    Max_prod = [None]*10
    Min_prod = np.zeros(10)
    bnds = zip(Min_prod,Max_prod)

    #Solve
    res = linprog(fobj, A_ub=A_ub, b_ub=b_ub, bounds=tuple(bnds), 
                        options={'maxiter':1000,"disp": False, "tol":1.e-9})
    return res


ncl2 = 0
@app.callback(
    [Output('table-editing-simple-output1', 'children'),
     Output('table-editing-simple-output2', 'children'),
     Output('table-editing-simple-output3', 'children'),
     Output('tot_cost2', 'children')],
    [Input('submit-val2', 'n_clicks'), 
     Input('table-editing-simple1', 'data'),
     Input('table-editing-simple1', 'columns'),
     Input('table-editing-simple2', 'data'),
     Input('table-editing-simple2', 'columns'),
     Input('table-editing-simple3', 'data'),
     Input('table-editing-simple3', 'columns')
     ])     
def display_output2(n_clicks, rows1, columns1, rows2, columns2, rows3, columns3):
    global ncl2

    df_fw = pd.DataFrame(rows1, columns=[c['name'] for c in columns1])
    df_fc = pd.DataFrame(rows2, columns=[c['name'] for c in columns2])
    df_wc = pd.DataFrame(rows3, columns=[c['name'] for c in columns3])

    new_costs_fw = df_fw.values[:,1:].flatten().tolist() 
    new_costs_fc = df_fc.values[:,1:].flatten().tolist() 
    new_costs_wc = df_wc.values[:,1:].flatten().tolist() 

    new_costs = new_costs_fw + new_costs_fc + new_costs_wc
    res = solver2(new_costs)

    sol = np.array([int(round(x,0)) for x in res.x])
    
    #Solution for Table-1
    sol1 = sol[:8]
    total1_1 = int(sol1[:4].sum())
    total1_2 = int(sol1[4:8].sum())
    sol1 = np.insert(sol1,4,total1_1)
    sol1 = np.append(sol1,total1_2)
    tot_wh = sol1.reshape(2,5).sum(axis=0) 

    #Solution for Table-2
    sol2 = sol[8:18]
    total2_1 = int(sol[8:13].sum())
    total2_2 = int(sol[13:18].sum())
    sol2 = np.insert(sol2,5,total2_1)
    sol2 = np.append(sol2,total2_2)
    tot_cust_f = sol2.reshape(2,6).sum(axis=0) 

    #Solution for Table-3
    sol3 = sol[18:]
    total3_1 = int(sol[18:23].sum())
    total3_2 = int(sol[23:28].sum())
    total3_3 = int(sol[28:33].sum())
    total3_4 = int(sol[33:].sum())
    print(sol3)
    sol3 = np.insert(sol3,5,total3_1)
    sol3 = np.insert(sol3,10+1,total3_2)
    sol3 = np.insert(sol3,15+2,total3_3)
    sol3 = np.append(sol3,total3_4)
    print(len(sol3),list(sol3))
    print(sol3.reshape(4,6))
    tot_cust_wh = sol3.reshape(4,6).sum(axis=0) 

    warehouses_t = warehouses + ['Total']
    customers_t = customers + ['Total']

    if n_clicks > ncl2:
        ncl2 = n_clicks
        return html.Div([
            html.Hr(),
            html.H3('Optimized number of products shipped',
            style={'textAlign': 'center','font-weight': 'bold'}),
            #html.Br(),
            #html.Div('Solution-1: ',style={'font-size': '100%','font-weight': 'bold'}),
            html.P('From Factories to Warehouses',style={'textAlign': 'center','font-weight': 'bold'}),
            html.Hr(),
            dash_table.DataTable(
                columns=(
                    [{'id': 'Factory', 'name': 'Factory'}] +
                    [{'id': p, 'name': p} for p in warehouses_t] 
                ),
                data=[
                    dict(Factory=i+1, **{param: sol1[i*5+ix] for ix,param in enumerate(warehouses_t)})
                    for i in [0,1]
                ] +
                [dict(Factory='Total', **{param: tot_wh[ix] for ix,param in enumerate(warehouses_t)})],
                editable=False
                )
            ]), html.Div([
            html.Br(),
            html.P('From Factories to Customers',style={'textAlign': 'center','font-weight': 'bold'}),
            dash_table.DataTable(
                columns=(
                    [{'id': 'Factory', 'name': 'Factory'}] +
                    [{'id': p, 'name': p} for p in customers_t] 
                ),
                data=[
                    dict(Factory=i+1, **{param: sol2[i*6+ix] for ix,param in enumerate(customers_t)})
                    for i in [0,1]
                ] +
                [dict(Factory='Total', **{param: tot_cust_f[ix] for ix,param in enumerate(customers_t)})],
                editable=False
                )
            ]), html.Div([
            html.Br(),
            html.P('From Warehouses to Customers',style={'textAlign': 'center','font-weight': 'bold'}),
            dash_table.DataTable(
                columns=(
                    [{'id': 'Warehouse', 'name': 'Warehouse'}] +
                    [{'id': p, 'name': p} for p in customers_t] 
                ),
                data=[
                    dict(Warehouse=i+1, **{param: sol3[i*6+ix] for ix,param in enumerate(customers_t)})
                    for i in [0,1,2,3]
                ] +
                [dict(Warehouse='Total', **{param: tot_cust_wh[ix] for ix,param in enumerate(customers_t)})],
                editable=False
                )
            ]), 'Total cost of shipping: {:.2f}'.format(res.fun)
    else:
        return html.Div([]), html.Div([]), html.Div([]), ''

def solver2(new_costs):
    #Objective:         
    fobj = new_costs

    # x1,..., x8  : factories -> WHs
    # x9,..., x18 : factories -> WHs
    # x19,...,x23 : WH1 -> customers
    # x24,...,x28 : WH2 -> customers
    # x29,...,x33 : WH3 -> customers
    # x34,...,x38 : WH4 -> customers

    n_fact = 2
    n_cust = 5
    n_wh = 4

    z = np.zeros(n_cust-1) 
    n_to_c = []
    for i in range(n_cust):
        n_to_c.append(list(np.insert(z,i,-1)))  

    z = np.zeros(n_wh-1) 
    n_to_wh = []
    for i in range(n_wh):
        n_to_wh.append(list(np.insert(z,i,1))) 

    z = np.zeros(15) 
    n_from_wh = []
    for i in range(n_wh):
        n_from_wh.append(list(np.insert(z,i*n_cust,-1*np.ones(n_cust))))

    #Constraints:
    ### total destinations from each factory (to customers and warehouse)
    n_tot1 = n_cust + n_wh
    ### total sources to each customer (from factories and warehouse)
    n_tot2 = n_fact + n_wh
    ### total sources to each warehouse (2 factories)
    n_tot3 = n_fact

    #Constraints:
    A_ub = [
        # Total_from_factory <= Factory_capacity
        list( np.ones(4)) + list(np.zeros(4)) + list( np.ones(5)) + list(np.zeros(5)) + list(np.zeros(20)),
        list( np.zeros(4)) + list(np.ones(4)) + list( np.zeros(5)) + list(np.ones(5)) + list(np.zeros(20)),
        # Total_to_customer >= Demand
        list(np.zeros(8)) + n_to_c[0]*n_tot2,
        list(np.zeros(8)) + n_to_c[1]*n_tot2,
        list(np.zeros(8)) + n_to_c[2]*n_tot2,
        list(np.zeros(8)) + n_to_c[3]*n_tot2,
        list(np.zeros(8)) + n_to_c[4]*n_tot2,
        # # Total_to_warehouse <= Warehouse_capacity
        n_to_wh[0]*n_tot3 + list(np.zeros(30)),
        n_to_wh[1]*n_tot3 + list(np.zeros(30)),
        n_to_wh[2]*n_tot3 + list(np.zeros(30)),
        n_to_wh[3]*n_tot3 + list(np.zeros(30))
    ]

    b_ub = [60000, 60000, 
           -30000, -23000, -15000, -32000, -16000,
            45000,  20000,  30000,  15000
           ]

    A_eq = [
            # Total_to_warehouse = Total_from_warehouse
            n_to_wh[0]*n_fact + list(np.zeros(10)) + n_from_wh[0],
            n_to_wh[1]*n_fact + list(np.zeros(10)) + n_from_wh[1],
            n_to_wh[2]*n_fact + list(np.zeros(10)) + n_from_wh[2],
            n_to_wh[3]*n_fact + list(np.zeros(10)) + n_from_wh[3]
    ]

    b_eq = [0,0,0,0]

    #Bounds
    # Factory_to_warehouse >= 0	 	 	 	 	 
    # Factory_to_customer >= 0	 	 	 	 	 
    # Warehouse_to_customer >= 0
    n_bnds = n_fact * n_tot1 + n_wh * n_cust # == n_vars
    Max_prod = [None]*n_bnds
    Min_prod = np.zeros(n_bnds)
    bnds = zip(Min_prod,Max_prod)

    #Solve
    res = linprog(fobj, A_ub=A_ub, b_ub=b_ub, 
                        A_eq=A_eq, b_eq=b_eq,
                        bounds=tuple(bnds), 
                        options={'maxiter':1000,"disp": True, "tol":1.e-9})
    
    #Results:
    print('Total cost: {:.2f}'.format(res.fun))
    print('Shipments from Factory 1 to WH: {}, Total: {}'.format(res.x[:4].round(0), res.x[:4].sum().round(0)))  
    print('Shipments from Factory 2 to WH: {}, Total: {}'.format(res.x[4:8].round(0), res.x[4:8].sum().round(0)))  
    print('Shipments from Factory 1 to Cust: {}, Total: {}'.format(res.x[8:13].round(0), res.x[8:13].sum().round(0)))  
    print('Shipments from Factory 2 to Cust: {}, Total: {}'.format(res.x[13:18].round(0), res.x[13:18].sum().round(0)))  
    print('Shipments from WH 1 to Cust: {}, Total: {}'.format(res.x[18:23].round(0), res.x[18:23].sum().round(0)))
    print('Shipments from WH 2 to Cust: {}, Total: {}'.format(res.x[23:28].round(0), res.x[23:28].sum().round(0)))
    print('Shipments from WH 3 to Cust: {}, Total: {}'.format(res.x[28:33].round(0), res.x[28:33].sum().round(0)))
    print('Shipments from WH 4 to Cust: {}, Total: {}'.format(res.x[33:].round(0), res.x[33:].sum().round(0)))


    return res


server = app.server

if __name__ == '__main__':
    app.run_server(host="0.0.0.0", debug=False)

