import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import urllib.request
import urllib.error
import os


class PreProcess:
    TRAINING_AMOUNT = 30000

    def __init__(self):
        self.training_labels = set()
        self.training_classes = None
        self.testing_classes = None

    def load_data(self):
        # load data
        print('Loading data...')
        training_data = pd.read_csv('data/train.csv', delimiter=',')
        x, y = training_data.iloc[:self.TRAINING_AMOUNT, 1].values, training_data.iloc[:self.TRAINING_AMOUNT, 2].values

        # splitting data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        return x_train, x_test, y_train, y_test

    def clean_data(self, x, y, type):
        cleaned_urls, cleaned_ids = [], []
        for url, id in zip(x, y):
            if url != 'None' and id != 'None' and self.valid_url(url):
                cleaned_urls.append(url)
                cleaned_ids.append(id)
        self.create_cleaned_data(cleaned_ids, cleaned_urls, type)
        if type == 'training':
            self.training_classes = len(cleaned_urls)
        elif type == 'testing':
            self.testing_classes = len(cleaned_urls)
        return cleaned_urls, cleaned_ids

    def valid_url(self, url):
        try:
            urllib.request.urlopen(url)
        except urllib.error.HTTPError:
            return False
        except urllib.error.URLError:
            return False
        else:
            return True

    def create_cleaned_data(self, ids, urls, name):
        f = open("cleaned_data_{}.txt".format(name), "w+")
        for id, url in zip(ids, urls):
            f.write(id + "\t" + url + "\n")
        f.close()

    def download_images(self):
        # Download pictures
        print('Downloading training images...')
        self.fetch_images('training')
        print('Downloading testing images...')
        self.fetch_images('testing')

    def fetch_images(self, type):
        lost = []
        if type == 'training':
            lost = [n for n in range(len(self.training_classes))]
        elif type == 'testing':
            lost = [n for n in range(len(self.testing_classes))]

        while len(lost) != 0:
            lost = self.download_img(lost, type)

    def download_img(self, miss, type):
        miss_index = []
        f = open("cleaned_data_{}.txt".format(type), "r")
        lines = f.readlines()
        if type == 'training':
            for i in miss:
                label, url = lines[i].split()
                self.create_directory(label)
                self.training_labels.add(label)
                try:
                    urllib.request.urlretrieve(url, "pictures/training/" + str(label) + "/" + str(i + 1) + ".jpg")
                except urllib.error.URLError:
                    miss_index.append(i)
        elif type == 'testing':
            for i in miss:
                label, url = lines[i].split()
                if label in self.training_labels:
                    try:
                        urllib.request.urlretrieve(url, "pictures/testing/" + str(label) + "/" + str(i + 1) + ".jpg")
                    except urllib.error.URLError:
                        miss_index.append(i)
        f.close()
        return miss_index

    def create_directory(self, label):
        path_train = 'pictures/training/' + str(label)
        path_test = 'pictures/testing/' + str(label)
        try:
            os.mkdir(path_train)
        except OSError:
            pass

        try:
            os.mkdir(path_test)
        except OSError:
            pass

    def image_preprocess(self):
        print('Preprocess images...')
        # Image Preprocess
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest")

        training_set = train_datagen.flow_from_directory(
            'pictures/training',
            target_size=(96, 96),
            batch_size=32,
            class_mode='categorical')

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        test_set = test_datagen.flow_from_directory(
            'pictures/testing',
            target_size=(96, 96),
            batch_size=32,
            class_mode='categorical')

        return training_set, test_set, (96, 96, 3)
