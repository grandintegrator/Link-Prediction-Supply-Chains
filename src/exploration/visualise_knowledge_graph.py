import dash
import dash_cytoscape as cyto
import dash_html_components as html
import pandas as pd

# from exploration.visualise_graph import VisualiseGraph
# from exploration import VisualiseGraph
#
# from ingestion.dataloader import SupplyKnowledgeGraphDataset
# loader = SupplyKnowledgeGraphDataset()
# pair_frame = loader.triplets
#
# pair_frame.to_parquet('../data/02_intermediate/triplets.parquet',
#                       engine='pyarrow', compression='gzip')

pair_frame = pd.read_parquet('../../data/02_intermediate/triplets.parquet')

app = dash.Dash(__name__)

# graph_class = VisualiseGraph()
# pair_frame = (
#     graph_class.create_multi_pair_frame(sample_portion=2,
#                                         product_product_n=10)
# ).reset_index(drop=True)

# Cosmetics - removing the UPPER CASES
# pair_frame['subjects'] = pair_frame['subjects'].str.capitalize()
# pair_frame['objects'] = pair_frame['objects'].str.capitalize()

# Create a nice set of companies, products, and capabilities to sample from.


companies = [#'Jiangsu Yuhua Automobile Parts',
             # 'Danyang Boliang Lamps Factory',
             # 'Bill Forge',
             # 'Varta',
             'Mitsubishi Motors Europe',
             'Activline',
             'Fehrer',
             'Ebm-Papst St. Georgen']

capabilities = ['Machining',
                'Assembly',
                'Stamping']

cond = (
    pair_frame['src'].isin(companies)
    # pair_frame['src'].isin(capabilities)
    # pair_frame['src'].isin(products)
)
pair_frame = pair_frame.loc[cond]
# products = pair_frame.loc[pair_frame['relation_type'] == 'capability_produces']
# pair_frame = pair_frame.loc[~(pair_frame['relation_type'] == 'capability_produces')]
# pair_frame = pair_frame.sample(n=10)
# pair_frame = pd.concat([pair_frame, products.sample(n=20)], axis=0)
# pair_frame = pair_frame.sample(n=100, random_state=1)

################################################################################
# CREATE NODE ELEMENTS FOR DASH APP
################################################################################
node_list_unique = \
    list(set(list(pair_frame['src']) + list(pair_frame['dst'])))

# Create node dictionary from dataframe
node_dict_list = []
for node in node_list_unique:
    try:
        node_type = list(
            pair_frame.loc[pair_frame['src'] == node, 'src_type'].head(1)
        )[0]
    except IndexError:
        node_type = list(
            pair_frame.loc[pair_frame['dst'] == node, 'dst_type'].head(1)
        )[0]

    if node_type == 'company':
        class_type = 'black'
    elif node_type == 'product':
        class_type = 'blue triangle'
    elif node_type == 'capability':
        class_type = 'orange square'
    elif node_type == 'country':
        class_type = 'yellow diamond'
    elif node_type == 'certification':
        class_type = 'black hexagon'

    # class_type = 'blue triangle' if node_type == 'Process' else 'black'
    node_dict_row = {'data': {'id': node},
                     'label': node,
                     'classes': class_type}
    node_dict_list.append(node_dict_row)

################################################################################
# CREATE EDGE ELEMENTS FOR DASH APP
################################################################################
pair_frame = pair_frame.reset_index(drop=True)
edge_dict_list = []
for row in range(pair_frame.shape[0]):
    # Create edge dictionary from the row class
    row_edge_dict = {'data': {'source': pair_frame.loc[row, 'src'],
                              'target': pair_frame.loc[row, 'dst'],
                              'relation_type': pair_frame.loc[row,
                                                              'relation_type']}}
    edge_dict_list.append(row_edge_dict)

app.layout = html.Div([
    cyto.Cytoscape(
        id='cytoscape-automotive_network',
        layout={'name': 'breadthfirst'},  # cose - physics
        style={'width': '100%', 'height': '1000px', 'font-size': '24'},
        elements=edge_dict_list + node_dict_list,
        stylesheet=[
            # Edge selectors
            {
                'selector': 'edge',
                'style': {
                    'label': 'data(relation_type)', 'font-size': '24'}
            },
            # Group selectors
            {
                'selector': 'node',
                'style': {
                    'content': 'data(id)',
                    'width': '80%',
                    'height': '80%',
                    'font-size': '24'
                }
            },
            # Class selectors
            {
                'selector': '.blue',
                'style': {
                    'background-color': 'blue',
                    'line-color': 'black'
                }
            },
            {
                'selector': '.orange',
                'style': {
                    'background-color': 'orange',
                    'shape': 'square',
                    'font-size': '24'
                }
            },
            {
                'selector': '.triangle',
                'style': {
                    'shape': 'triangle',
                    'font-size': '24'
                }
            },
            {
                'selector': '.diamond',
                'style': {
                    'shape': 'diamond',
                    'background-color': 'yellow',
                    'font-size': '24'
                }
            },
            {
                'selector': '.hexagon',
                'style': {
                    'shape': 'hexagon',
                    'background-color': 'red',
                    'font-size': '24'
                }
            }
        ]
    )
])

if __name__ == '__main__':
    app.run_server(debug=True,
                   port=8052)
