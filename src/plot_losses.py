import pickle
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    with open('train_loss_500kdps_nondup_01drop_80e.pickle', 'rb') as f:
        train_1 = pickle.load(f)

    with open('val_loss_500kdps_nondup_01drop_80e.pickle', 'rb') as f:
        val_1 = pickle.load(f)

    # with open('train_loss_100kdps_nondup_02drop.pickle', 'rb') as f:
    #     train_2 = pickle.load(f)
    #
    # with open('val_loss_100kdps_nondup_02drop.pickle', 'rb') as f:
    #     val_2 = pickle.load(f)
    #
    # with open('train_loss_100kdps_nondup_03drop.pickle', 'rb') as f:
    #     train_3 = pickle.load(f)
    #
    # with open('val_loss_100kdps_nondup_03drop.pickle', 'rb') as f:
    #     val_3 = pickle.load(f)
    #
    # with open('train_loss_100kdps_nondup_04drop.pickle', 'rb') as f:
    #     train_4 = pickle.load(f)
    #
    # with open('val_loss_100kdps_nondup_04drop.pickle', 'rb') as f:
    #     val_4 = pickle.load(f)

    x_1 = np.arange(0, len(train_1), 1)
    # x_2 = np.arange(0, len(train_2), 1)
    # x_3 = np.arange(0, len(train_3), 1)
    # x_4 = np.arange(0, len(train_4), 1)

    plt.plot(x_1,train_1,'r',ls=':', label='Training set - 0.1 dropout')
    plt.plot(x_1,val_1,'r', label='Validation set - 0.1 dropout')
    plt.ylim(3.5,9)
    # plt.plot(x_2, train_2, 'b', ls=':', label='Training set - 0.2 dropout')
    # plt.plot(x_2, val_2, 'b', label='Validation set - 0.2 dropout')
    # plt.plot(x_1, train_3, 'g', ls=':', label='Training set - 0.3 dropout')
    # plt.plot(x_1, val_3, 'g', label='Validation set - 0.3 dropout')
    # plt.plot(x_2, train_4, 'y', ls=':', label='Training set - 0.4 dropout')
    # plt.plot(x_2, val_4, 'y', label='Validation set - 0.4 dropout')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Transformer train/val loss - 500k data points')
    plt.legend()
    
    plt.show()