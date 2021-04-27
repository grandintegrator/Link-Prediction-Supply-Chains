import dash
import dash_cytoscape as cyto
import dash_html_components as html

from src.exploration.visualise_graph import VisualiseGraph

# from exploration import VisualiseGraph

app = dash.Dash(__name__)

graph_class = VisualiseGraph()
pair_frame = (
    graph_class.create_multi_pair_frame(sample_portion=2,
                                        product_product_n=10)
).reset_index(drop=True)

# Cosmetics - removing the UPPER CASES
pair_frame['subjects'] = pair_frame['subjects'].str.capitalize()
pair_frame['objects'] = pair_frame['objects'].str.capitalize()

################################################################################
# CREATE NODE ELEMENTS FOR DASH APP
################################################################################
node_list_unique = \
    list(set(list(pair_frame['subjects']) + list(pair_frame['objects'])))

# Create node dictionary from dataframe
node_dict_list = []
for node in node_list_unique:
    try:
        node_type = list(
            pair_frame.loc[pair_frame['subjects'] == node, 'subject_type'].head(1)
        )[0]
    except IndexError:
        node_type = list(
            pair_frame.loc[pair_frame['objects'] == node, 'object_type'].head(1)
        )[0]
    class_type = 'blue triangle' if node_type == 'Process' else 'black'
    node_dict_row = {'data': {'id': node},
                     'label': node,
                     'classes': class_type}
    node_dict_list.append(node_dict_row)

################################################################################
# CREATE EDGE ELEMENTS FOR DASH APP
################################################################################
edge_dict_list = []
for row in range(pair_frame.shape[0]):
    # Create edge dictionary from the row class
    row_edge_dict = {'data': {'source': pair_frame.loc[row, 'subjects'],
                              'target': pair_frame.loc[row, 'objects'],
                              'relation_type': pair_frame.loc[row,
                                                              'relation_type']}}
    edge_dict_list.append(row_edge_dict)

app.layout = html.Div([
    cyto.Cytoscape(
        id='cytoscape-automotive_network',
        layout={'name': 'cose'},  # breadthfirst - physics
        style={'width': '100%', 'height': '1000px'},
        elements=edge_dict_list + node_dict_list,
        stylesheet=[
            # Edge selectors
            {
                'selector': 'edge',
                'style': {
                    'label': 'data(relation_type)'}
            },
            # Group selectors
            {
                'selector': 'node',
                'style': {
                    'content': 'data(id)',
                    'width': '80%',
                    'height': '80%'
                }
            },
            # Class selectors
            {
                'selector': '.blue',
                'style': {
                    'background-color': 'blue',
                    'line-color': 'red'
                }
            },
            {
                'selector': '.triangle',
                'style': {
                    'shape': 'triangle'
                }
            }
        ]
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
