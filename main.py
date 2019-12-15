from pre_process import PreProcess
from cnn import CNN


if __name__ == '__main__':
    pre_process = PreProcess()

    # # load data
    # X_train, X_test, y_train, y_test = pre_process.load_data()
    #
    # # clean data
    # print('Cleaning training data...')
    # train_urls, train_ids = pre_process.clean_data(X_train, y_train, 'training')
    # print('Cleaning testing data...')
    # test_urls, test_ids = pre_process.clean_data(X_test, y_test, 'testing')
    #
    # # Download pictures
    # pre_process.download_images()

    training_data, testing_data, shape = pre_process.image_preprocess()

    cnn = CNN(
        epochs=25,
        input_shape=shape,
        classes=4742
    )

    cnn.build_model()

    cnn.fit(training_data, testing_data)

    cnn.draw_curves()



