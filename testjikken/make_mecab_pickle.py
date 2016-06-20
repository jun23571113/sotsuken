# coding: UTF-8

import os
import pickle
from textcook import TextCook
from multiprocessing import Pool

hinshi_list = ["接頭詞", "名詞", "動詞", "形容詞", "副詞", "接続詞", "助詞", "助動詞", "感動詞", "記号", "フィラー", "その他", "未知語"]


#並列化オン    
t = TextCook()
def tok_wrapper(arg):
    """
    並列化するにはグローバル関数が必要なので
    """
    return_tuple = tuple(t.tokenize( arg, hinshi_list, f= 
                                     lambda x: (x[0], hinshi_list[hinshi_list.index(x[1].split(",")[0])])))

    return return_tuple
#ここまで


def main():
    """
    novel_dict = {"novel_name":[[word, feature],[word, feature],,,]}
    みたいなのを作り、pickle化
    idf計算用のリスト、tf計算用のリストどちらにも対応できる。このpickleファイルができればmecabは叩かなくても良くなるようにする
    """
    #strのリストを作成
    str_list = []
    filename_list = []
    
    for filename in os.listdir(os.path.dirname(os.path.abspath(__file__))+'/idf/'):
        with open(os.path.dirname(os.path.abspath(__file__))+"/idf/"+filename, "r") as file:
            file_str = file.read()
            filename_list.append(filename)
            str_list.append(file_str)

    """
    #並列化オフ
    t = TextCook()
    func = (lambda x:
            (x[0], hinshi_list[hinshi_list.index(x[1].split(",")[0])]))
    tok_list = list(map( lambda arg:
                        tuple(t.tokenize( arg, hinshi_list, 
                                          f=func)),
                        str_list))
    """
    #並列化オン
    p = Pool(2)
    tok_list = list(p.map(tok_wrapper, str_list))

    #ここまで
    print("fuga")
    
    novel_dict = {}
    for (novel_name, novel_str) in zip(filename_list, tok_list):
        novel_dict[novel_name] = novel_str
    
    
    with open(os.path.dirname(os.path.abspath(__file__))+"/novel_mecab_dump.pickle", "wb") as file:
        pickle.dump(novel_dict, file)


if __name__ == '__main__':
    main()
