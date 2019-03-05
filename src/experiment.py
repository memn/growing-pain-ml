from sklearn import svm

from src.prepare_data import split

if __name__ == '__main__':
    clf = svm.SVC(gamma=0.001, C=100.)
    train, test = split()
    train_data, train_target = train
    test_data, test_target = test

    clf.fit(train_data, train_target)
    print(clf.predict(test_data))
