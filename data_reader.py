import numpy as np
import os



class Edge:
    out_going_domain = set()
    domain_dict = dict()
    def __init__(self):

        return
def read_idpc(file_name):
 NUMBER_OF_NODE = 0
 NUMBER_OF_DOMAIN = 0
 SOURCE_NODE = 0
 TERMINAL_NODE = 0
 LIST_EDGE = []
 with open(file_name) as f:
    lines = f.read()
    lines = lines.split(sep = '\n')
    NUMBER_OF_NODE, NUMBER_OF_DOMAIN = [int(i) for i in lines[0].split(' ')]
    SOURCE_NODE, TERMINAL_NODE = [int(i) for i in lines[1].split(" ")]
    for i in range(NUMBER_OF_NODE+1):
        LIST_EDGE.append(Edge)
    for i in range(2,len(lines)):

        u,v,w,d  = [int(k) for k in lines[i].split(' ')]

        LIST_EDGE[u].out_going_domain.add(d)
        on_node_now = LIST_EDGE[u]
        destiny = [v,w]
        if(d in on_node_now.domain_dict):
            if(destiny[0] not in on_node_now.domain_dict[d]):
                on_node_now.domain_dict[d][destiny[0]] = destiny[1]
            else:
                if(destiny[1] < on_node_now.domain_dict[d][destiny[0]]):
                    on_node_now.domain_dict[d][destiny[0]] = destiny[1]
        else:
            on_node_now.domain_dict[d] = dict()
            on_node_now.domain_dict[d][destiny[0]] = destiny[1]
 return NUMBER_OF_NODE,NUMBER_OF_DOMAIN,SOURCE_NODE,TERMINAL_NODE,LIST_EDGE



