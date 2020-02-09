from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix


def metrics(y_test, pred):
    '''
    Evaluate the trained model.
    '''
    acc = accuracy_score(y_test,pred)
    precision = precision_score(y_test, pred, average='macro')
    recall = recall_score(y_test, pred, average='macro')
    f1 = f1_score(y_test, pred, average='macro')
    print('acc: {}, macro pre: {}, macro rec: {}, macro f1: {}'.format(acc, precision, recall, f1))
    confuse_mat = confusion_matrix(y_test, pred)
    print('confuse matrix: \n {}'.format(confuse_mat))