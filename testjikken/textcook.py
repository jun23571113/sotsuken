# coding: UTF-8


import os
import gc
import re
import MeCab
import CaboCha
from functools import reduce
from sklearn.feature_extraction.text import TfidfVectorizer
from multiprocessing import Pool

class TextCook:
    """
    簡単なテキスト処理を定義しておく（tokenize,tfidfなど）
    """
    def __init__(self):
        self.tagger = MeCab.Tagger("")
        self.tagger.parse("")
        self.parser = CaboCha.Parser("")
        self.vectorizer = TfidfVectorizer(use_idf=True)

    def tokenize(self, text, feature_list, f=(lambda x:x.surface)):
        """
        文を受け取ってfeature_lsitに入っている品詞に合致する単語のリストを返す
        node->文の変換処理はfの関数で変更できるようにしておく（無指定だとsurfaceを返す）
        """
        encoded_text = text
        tagger = self.tagger
        
        parsed_list = list( map( lambda x: x.split(), 
                                 filter( lambda x: x != "EOS" and  x != "",
                                         tagger.parse(encoded_text).split("\n"))))
        matched_list = list(filter( lambda x: x[1].split(",")[0] in feature_list, parsed_list))

        #fでmapして返す
        return_list = list(map(f, matched_list))
        return return_list


    def parse(self, text):
        """
        cabochaでtextをパースして返す。textは単文であることが保証されているとする
        返り値はタプルのリスト
        例：[((surface,feature), (surface,feature), link), (...), ...]
        """
        encoded_text = text
        tree = self.parser.parse(encoded_text)

        str_list = []
        for i in range(tree.chunk_size()):
            chunk = tree.chunk(i)
            chunk_list = []
            for j in range(chunk.token_pos, chunk.token_pos + chunk.token_size):
                chunk_list.append((tree.token(j).surface, tree.token(j).feature))
            chunk_list.append(chunk.link)
            str_list.append(tuple(chunk_list))

        return tuple(str_list)

    def tfidf(self, wakached_list, for_idf_list ):
        """
        形態素解析でリスト化された文を受け取って単語ごとにtfidf値を算出し、返す
        内部でリスト化されたものをわざわざスペース区切りの分かち書きにする処理をする（そうやってTfidfVectorizerに渡すのが手っ取り早い）
        wakachied_list:処理対象の文のリスト（tfidfがほしいやつ）
        for_idf_list:idf算出のための大量の文のリスト（こちらもリスト化前提）
        """
        vectorizer = self.vectrizer
        lst = [" ".join(wakached_list), " ".join(for_idf_list)]

        tfidf_result = vectorizer.fit_transform(lst)
        tfidf_names  = vectorizer.get_feature_names()

        return filter(lambda x: x[1]>0.05, list(zip(tfidf_names, tfidf_result.toarray()[0])))
           
    def split_sentence(self, text):
        """
        文を。で区切ってリストにする。セリフとかは手に負えないので、「や」の入っている文は消し飛ばすことでクリーニングする
        """
        pattern = r"。|」$|」\n|\n"
        repattern = re.compile(pattern)
        lst = repattern.split(text)
        return list( map( lambda x: re.sub(r"　","",x), 
                          filter( lambda x: "「" not in x and "」" not in x and x != "", lst )))

def tok_wrapper(arg):
    # メソッドfをクラスメソッドとして呼び出す関数
    t = TextCook()
    return t.tokenize(arg, ["名詞"])

def testfunc():
    test = TextCook()
    tok1 = test.tokenize("米国などでだまし取られた約３億円を日本国内の口座から不正に引き出したとして、大阪府警と兵庫県警などの合同捜査本部が、ナイジェリア人と日本人の計９人を組織犯罪処罰法違反（犯罪収益の隠匿）と詐欺の疑いで逮捕していたことが１１日、府警への取材で分かった。グループが関係する数十の口座に海外から計約１５億円が送金されており、国際犯罪組織によるマネーロンダリング（資金洗浄）とみて、米連邦捜査局（ＦＢＩ）などと連携して捜査している。", ["名詞"])

    file_str_list = []
    
    for filename in os.listdir(os.path.dirname(os.path.abspath(__file__))+'/idf/'):
        with open(os.path.dirname(os.path.abspath(__file__))+"/idf/"+filename, "r") as file:
            file_str = file.read()
            file_str_list.append(file_str)
        
    #tok3 = reduce( lambda a, b: a + b, map(lambda x:test.tokenize(x, ["助詞"]),file_str_list))
    p = Pool(4)
    tok3 = reduce( lambda a, b: a + b, p.map(tok_wrapper,file_str_list))


    for x in range(0,10):
        print(tok3[x])
    print(len(tok3))

    """
    file_str = reduce(lambda a,b: a+b,file_str_list)
    print("hoge")
    tok3 = test.tokenize(file_str, ["名詞"])
    print("hoge")
    """

    lst = test.tfidf(tok1,tok3)
    sorted_lst = sorted(lst, key=lambda x:x[1])
    print(sorted_lst)

def testfunc2():
    t = TextCook()
    print(t.split_sentence("吾輩は猫である。「ほげふが」\n名前はまだない。「にゃあ」と鳴いて生まれた。"))


if __name__ == '__main__':
    testfunc2()
