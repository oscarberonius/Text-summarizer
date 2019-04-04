import pickle
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    with open('train_loss_300kdps_gd.pickle', 'rb') as f:
        train = pickle.load(f)

    with open('val_loss_300kdps_gd.pickle', 'rb') as f:
        val = pickle.load(f)

    x = np.arange(0, len(train), 1)
    plt.plot(x,train,'r', label='Training set')
    plt.plot(x,val,'b', label='Validation set')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Transformer train/val loss - 300k data points')
    plt.legend()
    
    plt.show()