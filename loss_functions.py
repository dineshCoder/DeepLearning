# Implement various loss functions.
import numpy as np

def logistic_loss_function(y_true, y_pred,
                           Weight_vector,
                           regularization_degree = 0,
                           regularization_constant = 0):
    """
    This calculates the logistic loss of the function.
    :param y_true:
    :param y_predicted:
    :return:
    """
    assert isinstance(y_true, np.ndarray), 'y_true has to be a numpy array.'
    assert isinstance(y_pred, np.ndarray), 'y_pred has to be a numpy array.'
    temp = -1*y_true*np.log(y_pred)-(1-y_true)*np.log(1-y_pred)
    result = np.sum(temp)
    if regularization_degree:
        weight_sum = np.sum(np.absolute(Weight_vector)**regularization_degree)
        result = result + regularization_constant*weight_sum
    return result/len(y_true)

def svm_loss(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    assert isinstance(y_true, np.ndarray), 'y_true has to be a numpy array.'
    assert isinstance(y_pred, np.ndarray), 'y_pred has to be a numpy array.'



def soft_max_loss_function(y_true, y_pred):
    pass


def reg_loss(y_true, y_pred):
    """
    This function implements a regression loss function.
    :param y_true: True label values.
    :param y_pred: Values returned from our hypothesis function.
    :return: a scalar loss value.
    """
    assert isinstance(y_true, np.ndarray), 'y_true has to be a numpy array.'
    assert isinstance(y_pred, np.ndarray), 'y_pred has to be a numpy array.'
    # TO-DO (assert shape compatibility of the 2 arrays.)
    temp = np.sqrt(np.sum((y_true - y_pred)**2))
    return temp/(2*len(y_true))

if __name__ == '__main__':
    y_true = np.array([1,1,1,1])
    y_pred = np.array([0.5,0.5,0.5,0.5])
    # print(logistic_loss_function(y_true,y_pred))
    print(np.log(y_true))
    print(1-y_pred)
    print(np.sum(np.absolute(y_pred)**4))