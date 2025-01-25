import random
import os
import subprocess
from langchain_core.runnables.graph import MermaidDrawMethod
import sys

def display_graph(graph):
    # Code to visulaize the graph
    mermaid_png = graph.get_graph(xray=1).draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    
    # Create an output folder if it doesn't exist
    current_folder = os.getcwd()
    output_folder = current_folder + "/graphs/"
    os.makedirs(output_folder, exist_ok=True)

    filename = os.path.join(output_folder, f"graph_{random.randint(1, 100000)}.png")
    with open(filename, 'wb') as f:
        f.write(mermaid_png)
    
    print(filename)
        
    if sys.platform.startswith('darwini'):
        subprocess.call(('open', filename))
    elif sys.platform.startswith('linux'):
        subprocess.call(('xdg-open', filename))
    elif sys.platform.startswith('win'):
        os.startfile(filename)
        