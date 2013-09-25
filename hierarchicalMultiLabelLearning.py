import sys
from optparse import OptionParser
from sklearn.metrics import precision_recall_fscore_support
import operator
from sklearn.svm import SVR
from sklearn.datasets import load_svmlight_file
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import RidgeCV, Ridge
import string


labelDict = {}
labelIndex = {}



def createLabelDict(hierarchy):
    """ Creates the label dictionary from hierarchy file. Each line in the
    file is <nodeID> <childNodeID1> <childNodeID2> ... The labelDict is a
    dict with each entry keyed by the node id and containing list of
    parents and children. Leave children list blank if no children. """
    global labelDict

    for line in open(hierarchy):
        line = line.strip().split()
        line = [x.strip() for x in line]
        item = line[0]
        if item not in labelDict:
            labelDict[item] = {'parents':[], 'children': []}
        if len(line) < 2: continue
        labelDict[item]['children'] = list(set(labelDict[item]['children'] + line[1:]))
        for child in line[1:]:
            if child not in labelDict:
                labelDict[child] = {'parents':[item], 'children':[]}
            else:
                if item not in labelDict[child]['parents'] : labelDict[child]['parents'].append(item)

def createDummyRoot():
    """ Creates dummy root in labelDict for a hierarchy that is not rooted, i.e. one with multiple top level nodes. """
    global labelDict
    root = {'parents':[], 'children':[]}
    for label in labelDict:
        if len(labelDict[label]['parents']) == 0:
            root['children'].append(label)
            labelDict[label]['parents'].append('ROOT')
    labelDict['ROOT'] = root
    
def preliminaries(hierarchy):

    # Create labelDict, a dictionary to represent the hierarchy with children and parent information
    createLabelDict(hierarchy)

    global labelDict
    global labelIndex
   
    # Create a dummy root to take care of cases where tree/DAG is not rooted
    createDummyRoot()

    # TO DO: Test if the hierarchy has any cycles: if cycles exist, exit##########################################################################################################################################################################################################################################
    
    # Create label index from labelDict
    labelIndex = {}
    labelIndex['ROOT'] = 0
    idx = 1
    for item in labelDict:
        if item == 'ROOT': continue
        labelIndex[item] = idx
        idx += 1

def getPathToRoot(label):
    """ Returns all the nodes present in all the paths from node 'label' to root. """
    path = [label]
    if len(labelDict[label]['parents'])==0:
        return path
    for item in labelDict[label]['parents']:
        path += getPathToRoot(item)
    return list(set(path))


def getProjectionMatrixKPCA(dim=50):
    """ Kernel PCA : see paper for detailed description"""
    # Create an X for the hierarchy
    X = np.zeros((len(labelDict), len(labelDict)))
    for item in labelDict:
        pars = getPathToRoot(item)
        for par in pars:
            X[labelIndex[item]][labelIndex[par]] = 1
    kpca = KernelPCA(n_components=dim, fit_inverse_transform=True)
    X_kpca = kpca.fit(X)
    return kpca, kpca.alphas_
    
def getProjectionMatrixPCA(Y, dim=50):
    pca = PCA(n_components=dim)
    pca.fit(Y)
    return pca, pca.components_

def simpleTraining(X,Y,X_test, regressor='ridge'):
    # X and Y are numpy arrays
    print 'Input shape: ', X.shape, Y.shape
    Y_train = Y
    dim = Y.shape[1]
    regressors = []
    for i in range(dim):
        reg = Ridge() if regressor=='ridge' else SVR()
        y = [x[i] for x in Y_train]
        reg.fit(X, y)
        regressors.append(reg)
        print 'at regressor: ', i

    Z_pred = []
    for reg in regressors:
        Z_pred.append(reg.predict(X_test))
    print 'prediction shapes:' , len(Z_pred), len(Z_pred[0])
    Y_pred = np.array(Z_pred)
    return None, regressors, Y_pred.transpose()

def training(X,Y,X_test, pca='kpca', regressor='ridge', dim=50):
    # X and Y are numpy arrays
    print 'Input data and label shape: ', X.shape, Y.shape

    if pca == 'nopca': return simpleTraining(X, Y, X_test, regressor)

    model, P = getProjectionMatrixPCA(Y, dim) if pca=='pca' else getProjectionMatrixKPCA(dim)
    Y_train = np.dot(Y, P) if pca=='kpca' else np.dot(Y,P.transpose())


    regressors = []
    for i in range(dim):
        print 'at regressor number: ', i
        reg = Ridge() if regressor=='ridge' else SVR()
        y = [x[i] for x in Y_train]
        reg.fit(X, y)
        regressors.append(reg)

    Z_pred = []
    for reg in regressors:
        Z_pred.append(reg.predict(X_test))
    print 'prediction shapes:' , len(Z_pred), len(Z_pred[0])
    Z_pred = np.array(Z_pred)
    Y_pred = np.dot(P, Z_pred).transpose() if pca=='kpca' else np.dot(Z_pred.transpose(), P)
    return model, regressors, Y_pred


def condense(S, S_tilde, graph, sorted_snv, psi):
    """ Condensation step in the algorithm. Condenses parents and children according to OR or AND tree in a greedy way."""

    # Create new entry in graph for the condensed node
    item = tuple(sorted(S + S_tilde))
    graph[item] = {}
    graph[item]['parents'] = graph[S_tilde]['parents'] #list(set(graph[S]['parents']+graph[S_tilde]['parents']))
    graph[item]['children'] = list(set(graph[S]['children']+graph[S_tilde]['children']))
    if S in graph[item]['children']: graph[item]['children'].remove(S)
    if S_tilde in graph[item]['children']: graph[item]['children'].remove(S_tilde)
    graph[item]['snv'] = (graph[S]['snv']+graph[S_tilde]['snv'])/2
    
    # Remove old S_tilde from sorted_snv and create new sorted_snv
    sorted_snv.remove((S_tilde, graph[S_tilde]))
    sorted_snv = [(x,y) for x,y in sorted_snv if y['snv']<=graph[item]['snv']] + [(item, graph[item])]+[(x,y) for x,y in sorted_snv if y['snv']>graph[item]['snv']]

    # Update the parent and child nodes to reflect the merge
    for parent in graph[item]['parents']:
        if S in graph[parent]['children']: graph[parent]['children'].remove(S)
        if S_tilde in graph[parent]['children']: graph[parent]['children'].remove(S_tilde)
        graph[parent]['children'].append(item)
    for child in graph[item]['children']:
        if S in graph[child]['parents']: graph[child]['parents'].remove(S)
        if S_tilde in graph[child]['parents']:graph[child]['parents'].remove(S_tilde)
        graph[child]['parents'].append(item)
    for parent in graph[S]['parents']:
        if S in graph[parent]['children']: graph[parent]['children'].remove(S)
    
    # Update psi with new node
    psi[item] = 0
    return graph, sorted_snv, psi
    
    

def CSSAG(L, y_pred, prop='and'):
    """ CSSAG algorithm described in the paper. """
    psi = {}
    for item in labelDict:
        psi[(item,)] = 0
    psi[('ROOT',)] = 1

    T = 1
    graph = {}
    for item in labelDict:
        graph[(item,)] = {}
        graph[(item,)]['parents'] = [(x,) for x in labelDict[item]['parents']]
        graph[(item,)]['children'] = [(x,) for x in labelDict[item]['children']]
        graph[(item,)]['snv'] = y_pred[labelIndex[item]]
    # Sorted in the increasing order so that pop function outputs the largest SNV
    sorted_snv = sorted(graph.items(), cmp=lambda x,y:cmp(x['snv'], y['snv']), key=operator.itemgetter(1))
    sorted_snv.remove((('ROOT',), graph[('ROOT',)]))

    while T < L:
        S = sorted_snv.pop()[0]
        
        if prop=='and':
            ################ AND Routine ###################################
            cond = True
            for parent in graph[S]['parents']:
                if psi[parent] != 1:
                    cond = False
                    break
            if cond == True:
                psi[S] = min(1, (L-T)/len(S))
                T += len(S)
            else:
                unassigned_parents = [x[0] for x in sorted_snv if (x[0] in graph[S]['parents'] and psi[x[0]] == 0)]
                S_tilde = unassigned_parents[0]

                # Condense S and S_tilde
                graph, sorted_snv, psi = condense(S, S_tilde, graph, sorted_snv, psi)
        else:
            ################### OR Routine ##################################
            cond = False
            for parent in graph[S]['parents']:
                if psi[parent] == 1:
                    cond = True
                    break
            if cond == True:
                psi[S] = min(1, (L-T)/len(S))
                T += len(S)
                unassigned_nodes = [x for x,y in sorted_snv if (psi[x] == 0 and len(set(x).intersection(set(S)))>0)]
                # For each S_ in the list, delete S_ and form new supernode for each node in S_\S
                for S_ in unassigned_nodes:
                    sorted_snv.remove((S_, graph[S_]))
                    for node in list(set(S_).difference(set(S))):
                        while (node,) in graph: node += '_' #('_',) # Replication may create two nodes with same name, hence this
                        node = (node,) # Convert to tuple as graph has tuples only
                        graph[node] = {}
                        graph[node]['parents'] = graph[S_]['parents']
                        graph[node]['children'] = graph[S_]['children']
                        graph[node]['snv'] = graph[S_]['snv']
                        # Update the parent and child nodes to reflect the merge
                        for parent in graph[node]['parents']:
                            if S_ in graph[parent]['children']: graph[parent]['children'].remove(S_)
                            graph[parent]['children'].append(node)
                        for child in graph[node]['children']:
                            if S_ in graph[child]['parents']: graph[child]['parents'].remove(S_)
                            graph[child]['parents'].append(node)
                        sorted_snv = [(x,y) for x,y in sorted_snv if y['snv']<=graph[node]['snv']] + [(node, graph[node])]+[(x,y) for x,y in sorted_snv if y['snv']>graph[node]['snv']]
                        psi[node] = 0

            else:
                unassigned_parents = [x for x,y in sorted_snv if (x in graph[S]['parents'] and psi[x] == 0)]
                for S_tilde in unassigned_parents:
                    graph, sorted_snv, psi = condense(S, S_tilde, graph, sorted_snv, psi)
    result_labels = []
    for item in psi:
        if psi[item] > 0:
            result_labels += list(item)

    if 'ROOT' in result_labels: result_labels.remove('ROOT')
    return result_labels

def populateY(X, Y, labelFile):
    """ Label file and original train file are input separately due to limitation of svmlight format. This function uses the labelFile to populate the training and testing vectors corresponding to svmlight input files. """
    idx = 0
    for line in open(labelFile):
        labels = line.strip().split(',')
        for label in labels:
            label = label.strip()
            Y[idx][labelIndex[label]] = 1
        idx += 1
    print 'reached to idx: ', idx
    if idx != X.shape[0]: print 'Possible Errorrrrrrrrrrrrrrrrrrrrrrrrrr....\n\n\n'
    return Y
    
def prepareDataset(trainfile, trainlabels, testfile, testlabels):
    """ Prepare train and test sets from input files """
    X, y_train = load_svmlight_file(trainfile)
    X_test, y_test = load_svmlight_file(testfile, n_features=X.shape[1])

    Y = np.zeros([X.shape[0],len(labelIndex)])
    Y_test = np.zeros([X_test.shape[0],len(labelIndex)])

    Y = populateY(X, Y, trainlabels)
    Y_test = populateY(X_test, Y_test, testlabels)

    return X, Y, X_test, Y_test

def pres_rec_f1(Y_true, Y_preds_list):
    """ Calculates micro, macro and sample averaged metrics for the classification task"""
    pres_sample, rec_sample, f1_sample, pres_micro, rec_micro, f1_micro, pres_macro, rec_macro, f1_macro = ([],[],[],[],[],[],[],[],[])
    for n in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        p, r, f, s = precision_recall_fscore_support(Y_true, Y_preds_list[n], average='samples')
        pres_sample.append(p)
        rec_sample.append(r)
        f1_sample.append(f)
        p, r, f, s = precision_recall_fscore_support(Y_true, Y_preds_list[n], average='micro')
        pres_micro.append(p)
        rec_micro.append(r)
        f1_micro.append(f)
        p, r, f, s = precision_recall_fscore_support(Y_true, Y_preds_list[n], average='macro')
        pres_macro.append(p)
        rec_macro.append(r)
        f1_macro.append(f)
    data_reg = {}
    data_reg['sample'] = (pres_sample, rec_sample, f1_sample)
    data_reg['micro'] = (pres_micro, rec_micro, f1_micro)
    data_reg['macro'] = (pres_macro, rec_macro, f1_macro)
    return data_reg


if __name__ == '__main__':

    # Set up input parsing
    parser = OptionParser()
    parser.add_option('-i', '--hierarchy', dest='hierarchy', help='the file with the label hierarchy information: Each line in the file is <nodeID> <childNodeID1> <childNodeID2>... <childNodeIDn>', type='str')
    parser.add_option('-e', '--test', dest='testfile', help='the file for testing in svmlight format', type='str')
    parser.add_option('-a', '--train', dest='trainfile', help='the file for training in svmlight format', type='str')
    parser.add_option('-x', '--test-labels', dest='testlabels', help='the test labels file: comma separated labels per line', type='str')
    parser.add_option('-y', '--train-labels', dest='trainlabels', help='the train labels file: comma separated labels per line', type='str')
    parser.add_option('-d', '--dim', dest='dim', type='int', help='number of dimensions in the PCA')
    parser.add_option('-t', '--tree', dest='tree', type='str', help='selest AND or OR tree: takes input "and" or "or"')
    parser.add_option('-r', '--regressor', dest='regressor', type='str', help='type of regressor to use: "ridge" or "svr"')
    parser.add_option('-p', '--pca', dest='pca', type='str', help='type of pca: "pca"/"kpca"(kernel PCA)/"nopca"(a regressor is trained on each label)')

    (options, args) = parser.parse_args()


    pca = options.pca if options.pca != None else 'kpca'
    regressor = options.regressor if options.regressor != None else 'ridge'
    dim = options.dim if options.dim != None else 50
    tree = options.tree if options.tree != None else 'or'

    # Prepare label dict and others...
    preliminaries(options.hierarchy)
    
    # Prepare dataset    
    X, Y, X_test, Y_test = prepareDataset(options.trainfile, options.trainlabels, options.testfile, options.testlabels)

    # Train and get prediction
    print 'training model.......pca: ', pca, 'regressor: ', regressor, 'dim: ', dim, 'tree: ', tree
    PCA, regressors, Y_pred = training(X, Y, X_test, pca, regressor, dim) #simpleTraining(X, Y, X_test, pca, regressor, dim)
    
    # First just test the regressors without employing CSSAG
    print 'testing the regressor model...'
    Y_preds_list = {5:[], 10:[], 15:[], 20:[], 25:[], 30:[], 35:[], 40:[], 45:[], 50:[]}
    for row in Y_pred:
        y = {}
        for i in range(len(row)):
            y[i] = row[i]
        sorted_y = sorted(y.items(), key=operator.itemgetter(1), reverse=True)
        for n in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
            Y_preds_list[n].append([x for x,y in sorted_y[:n]])

    Y_true = []
    for row in Y_test:
        y = []
        for i in range(len(row)):
            if row[i] == 1: y.append(i)
        Y_true.append(y)

    # Get precision, recall and f1 scores
    data_reg = pres_rec_f1(Y_true, Y_preds_list)
    print "report without using the greedy tree step: "
    print data_reg

    # Get final Y_pred using CSSAG algo
    print 'starting with CSSAG ......'
    Y_preds_list = {5:[], 10:[], 15:[], 20:[], 25:[], 30:[], 35:[], 40:[], 45:[], 50:[]}
    for n in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        idx = 1
        for y_pred in Y_pred:
            print 'at idx: ', idx, 'for n: ', n
            result = CSSAG(n, y_pred, tree)
            result = [labelIndex[x.strip('_')] for x in result if x != 'ROOT']
            Y_preds_list[n].append(result)
            idx += 1

    # Get precision, recall and f1 scores
    data_cssag = pres_rec_f1(Y_true, Y_preds_list)
    print 'results after CSSAG algorithm: '
    print data_cssag

    print 'trained model.......pca: ', pca, 'regressor: ', regressor, 'dim: ', dim, 'tree: ', tree
