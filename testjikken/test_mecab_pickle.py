# coding: UTF-8

import os
import pickle


hinshi_list = ["接頭詞", "名詞", "動詞", "形容詞", "副詞", "接続詞", "助詞", "助動詞", "感動詞", "記号", "フィラー", "その他", "未知語"]


def main():
    """
    pickleのテストをしたいだけ
    """
    with open(os.path.dirname(os.path.abspath(__file__))+"/novel_mecab_dump.pickle", "rb") as file:
        novel_dict = pickle.load(file)
    print(len(novel_dict[list(novel_dict.keys())[0]]))
    print(list(novel_dict.keys()))
    print(novel_dict[list(novel_dict.keys())[0]])
    


if __name__ == '__main__':
    main()
