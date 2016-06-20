# coding: UTF-8

import os
import pickle
from gensim import corpora, matutils


def make_data():
    """
    以下のラベルに分類する
    0:芥川 竜之介
    1:有島 武郎
    2:夏目 漱石
    3:新渡戸 稲造
    4:野村 胡堂
    5:長谷川 時雨
    6:堀 辰雄
    7:太宰 治
    """
    train_label = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                   1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                   2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                   3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
                   4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
                   5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
                   6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
                   7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]


    with open(os.path.dirname(os.path.abspath(__file__))+"/novel_mecab_dump.pickle", "rb") as file:
        novel_dict = pickle.load(file)
    
    def filename_to_label(string):
        """
        ファイルネームを受け取ってラベルを返す
        """
        if "芥川 竜之介" in string:
            return 0
        elif "有島 武郎" in string:
            return 1
        elif "夏目 漱石" in string:
            return 2
        elif "新渡戸 稲造" in string:
            return 3
        elif "野村 胡堂" in string:
            return 4
        elif "長谷川 時雨" in string:
            return 5
        elif "堀 辰雄" in string:
            return 6
        elif "太宰 治" in string:
            return 7
        else:
            print(string)
            return 10

    label_novel_list = list(map( lambda x:(filename_to_label(x[0]), 
                                           list(map(lambda y:y[0], #品詞タグを除去
                                                    x[1]))), 
                                 novel_dict.items()))



    novel_dict = {} #GCを働かせるために空辞書にする

    dictionary = corpora.Dictionary(list(map(lambda x: x[1], label_novel_list)))
    label_vec_list = list(map( lambda x:(x[0], 
                                         list(matutils.corpus2dense([dictionary.doc2bow(x[1])], 
                                                                    num_terms=len(dictionary)).T[0])),
                               label_novel_list))

    dictionary = {} #GCを働かせるために空辞書にする


    label_train_data = []
    for count in range(8): #訓練用をそれぞれ20取る
        append_list = list(filter(lambda x: x[0] == count,
                                  label_vec_list))[:20]
        if(len(append_list)!=20):
            print("error")
            print(count)
        label_train_data.extend(append_list)

    for data in label_train_data: #テスト用から訓練用に入れたものを取り去る
        label_vec_list.remove(data)
    

    train_data  = list(map( lambda x:x[1], label_train_data))
    train_label = train_label
    test_data   = list(map( lambda x:x[1], label_vec_list))
    test_label  = list(map( lambda x:x[0], label_vec_list))

    return (train_data, train_label, test_data, test_label )






def test_svm( train_data, train_label, test_data, test_label ):
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.grid_search import GridSearchCV


    #パラメータ設定

    #学習
    """
    #SVM(だめ）
    C = 1.
    kernel = 'rbf'
    gamma  = 0.01


    estimator = SVC(C=C, kernel=kernel,gamma=gamma)
    classifier = OneVsRestClassifier(estimator)
    
    """

    """
    #ランダムフォレスト（良い）
    classifier = RandomForestClassifier(n_estimators=150, max_features=4000, oob_score=True)
    """

    """
    #グリッドサーチ
    tuned_parameters = [{'n_estimators': [10, 40, 70, 100,130,160,190], 'max_features': ['auto', 'sqrt', 'log2', 1000,2000,4000,8000], 'oob_score':[True, False]}]
    classifier = GridSearchCV(RandomForestClassifier(), tuned_parameters, scoring='accuracy', n_jobs=1)
    """
    #最良パラメータらしい
    classifier = RandomForestClassifier(bootstrap=True, class_weight=None, 
                                        criterion='gini', max_depth=None, 
                                        max_features=4000, max_leaf_nodes=None,
                                        min_samples_leaf=1, min_samples_split=2,
                                        min_weight_fraction_leaf=0.0,
                                        n_estimators=100, n_jobs=4,
                                        oob_score=True, random_state=None,
                                        verbose=0, warm_start=False)


    classifier.fit(train_data, train_label)



    #予測
    pred_label = classifier.predict(test_data)
    print(pred_label)
    print(test_label)
    print(classifier.score(test_data, test_label))


if __name__ == '__main__':
    train_data, train_label, test_data, test_label = make_data()
    test_svm(train_data, train_label, test_data, test_label)
