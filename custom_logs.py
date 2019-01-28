import tensorflow as tf
import sys
import os
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

def main(path):
    graphs = {}
    tags = ['test_acc','test_memory_real_acc']
    fig = plt.figure(figsize=(15,12))
    for root,dirlst,filelst in os.walk(path):
        events=None
        for f in filelst:
            if f[:6]=='events':
                events=f
                break
        if events is None:
            continue
        d = {}
        for tag in tags:
            d[tag]=[]
        try:
            for e in tf.train.summary_iterator(os.path.join(root,events)):
                for v in e.summary.value:
                    if v.tag in tags:
                        d[v.tag].append(v.simple_value)
        except:
            continue
        for key in d:
            d[key]=np.array(d[key])
        graphs[root]=d
    for i,g in enumerate(graphs.keys()):
        subplt = fig.add_subplot(len(graphs),1,i+1)
        subplt.set_title(g)
        subplt.plot(graphs[g]['test_memory_real_acc']/graphs[g]['test_acc'])
        subplt.axhline(1,color='red')
    fig.set_size_inches(15, 5*len(graphs), forward=True)    
    fig.savefig('custom_logs.png')


main(sys.argv[1])