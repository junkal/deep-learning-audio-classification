import os
import matplotlib.pyplot as plt

def plot_training_graph(history):
    plt.plot(history.epoch, history.history['accuracy'])
    plt.plot(history.epoch, history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.legend(['train', 'validation'])
    plt.xlabel('No. of Epochs')
    plt.ylabel('Accuracy')
    plt.figure()

    plt.plot(history.epoch, history.history['loss'])
    plt.plot(history.epoch, history.history['val_loss'])
    plt.title('Model Loss')
    plt.legend(['train', 'validation'])
    plt.xlabel('No. of Epochs')
    plt.ylabel('Loss')
    plt.show()

def clear_folder(folder_name):
    for filename in os.listdir(folder_name):
        os.remove(os.path.join(folder_name, filename))
