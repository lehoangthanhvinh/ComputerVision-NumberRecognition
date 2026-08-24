import gzip, pickle
import matplotlib.pyplot as plt

def read_data(path):
    with gzip.open(path, 'rb') as mnist_pickle:
        MNIST = pickle.load(mnist_pickle, encoding='latin1')

    return MNIST

def get_datapoint(dataset, idx):
    return (dataset[0][idx], dataset[1][idx])

def draw_image(img):
    if (len(img) != 784):
        print('Error, unmatched img size, require (784, ) shape img')
        return False
    plt.figure(figsize=(5, 5))
    plt.imshow(img.reshape(28, 28))
    plt.show()
    return True

def save_image(img, path):
    if (len(img) != 784):
        print('Error, unmatched img size, require (784, ) shape img')
        return False

    plt.imshow(img.reshape(28, 28))
    plt.savefig(path)
    
def show_datapoint(dataset, idx):
    print('Label: ', dataset[1][idx])
    print('Image:')
    draw_image(dataset[0][idx])
    return True

def classify_dataset(dataset):
    clasified = {}
    for i in range(len(dataset[0])):
        feature, label = get_datapoint(dataset, i)
        if str(label) in clasified.keys():
            clasified[str(label)].append(feature)
        else:
            clasified[str(label)] = [feature]
    return clasified

def get_average_features(features):
    res = features[0]
    for feature in features:
        res += feature
    return res / len(features)