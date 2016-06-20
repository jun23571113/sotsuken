import os
import pickle
from textcook import TextCook
from multiprocessing import Pool

hinshi_list = ["接頭詞", "名詞", "動詞", "形容詞", "副詞", "接続詞", "助詞", "助動詞", "感動詞", "記号", "フィラー", "その他", "未知語"]


#並列化用のなにか
t = TextCook()
def parse_wrapper(arg):
    """
    並列化用のなにか
    """
    return t.parse(arg)


def treenize(tree_tuple):
    """
    treenizeという辛い処理をする
    
    """




def main():
    """
    cabochaで構文解析する->固有の単語情報を消し飛ばして裸の構文木にする
    """
    str_list = []
    filename_list = []
    
    for filename in os.listdir(os.path.dirname(os.path.abspath(__file__))+'/idf/'):
        with open(os.path.dirname(os.path.abspath(__file__))+"/idf/"+filename, "r") as file:
            file_str = file.read()
            filename_list.append(filename)
            str_list.append(file_str)
    

    splited_str_list = map(t.split_sentence, str_list)
    print("hoge")
    p = Pool(4)
    lst = list(map( lambda x: p.map( parse_wrapper, x ), 
                    splited_str_list))
    print(lst[0][:10])
    
    with open(os.path.dirname(os.path.abspath(__file__))+"/novel_cabocha_dump.pickle", "wb") as file:
        pickle.dump(lst, file)


if __name__ == '__main__':
    main()


