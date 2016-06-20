# coding: UTF-8

import os
import lxml.html
from functools import reduce
import urllib.request
import re
import time
import zipfile
import subprocess

class AozoraCook:
    """
    青空文庫からファイルを落としてきて最終的には整形されたlinuxフォーマットのテキストファイルにする
    """

    url_root = "http://www.aozora.gr.jp/cards/" #作品カードのページにとりあえず飛ばせるように

    def get_zipfile(self, author_number, novel_number, zip_name):
        """
        zipファイルを落としてきてfiles/zip_name.zipに置く
        """
        try:
            response = urllib.request.urlopen( self.url_root +
                                               str(author_number).zfill(6) + "/" +
                                               "card" + str(novel_number) +
                                               ".html"
            )
            html = response.read()
            root = lxml.html.fromstring(html)
        
            lst =  root.xpath("//td/a")
            filename = list(filter(lambda x: "zip" in x, map( lambda x:x.text_content(), lst)))[0]
            zipfile_url = self.url_root + str(author_number).zfill(6) + "/files/" + filename
            urllib.request.urlretrieve(zipfile_url, "files/" + zip_name)
        except:
            print("http error")
    
    def csv_read(self, csv_path, f = lambda x: '"芥川 竜之介"' in x, g = lambda x: x ):
        """
        CSVを読んでfの条件に合致した行のリストをgの条件式で返す
        """
        with open(csv_path, "r") as file:
            file_str = file.read()
        csv_lst = list(map(lambda x:x.split(","), file_str.split("\n")))
        return list( map( g, ( filter( f, csv_lst ))))

    def unzip(self, zipfile_name, output_filename):
        """
        一つのzipファイルの名前を受け取ってoutput_filenameに書き出す
        全ファイルに対して処理したいとかならそれ用の処理は自前でやる
        """
        zip_file = zipfile.ZipFile(zipfile_name, "r")
        for filename in zip_file.namelist():
                unzip_file = open(output_filename, "wb")
                unzip_file.write(zip_file.read(filename))
                unzip_file.close()
        zip_file.close()

    def txt_clean( self, txt_filename ):
        """
        zip解凍直後の.txtに以下の処理をする
        ・文字コードをlinux用にする
        ・青空文庫の訳のわからない色々やルビを削除して純粋な小説だけの文章にする
        """
        #文字コード変換
        cmd = "nkf -Lu -w --overwrite" + ' "' + txt_filename + '"'
        subprocess.call(cmd, shell=True)

        #fileopen
        with open( txt_filename, "r" ) as file:
            file_str = file.read()

        #ルビなどを削除
        tmp = re.sub( u"《[^》]*》", "", file_str )
        file_str = tmp
        tmp = re.sub( u"［[^］]*］", "", file_str )
        file_str = tmp

        tmp = re.sub( u"｜", "", file_str )
        file_str = tmp

        #冒頭を削除
        itr = list(re.finditer(r"-", file_str))
        if [] == itr:
            pass
        else:
            print(itr[-1].end())
            file_str = file_str[itr[-1].end():]

        print(txt_filename)

        #底本：以下を削除
        print(len(file_str))
        itr = list(re.finditer(u"底本：", file_str))
        if [] == itr:
            pass
        else:
            n = itr[0].start()
            file_str = file_str[:n]
        print(len(file_str))

        #filewrite
        with open( txt_filename, "w" ) as file:
            file.write(file_str)

def testfunc1():
    test = AozoraCook()
    test.get_zipfile(35,2295, "渡り鳥")

def testfunc2():
    def func_f(x):
        return ('新字新仮名' in x and
                (('"芥川 竜之介"' in x) or 
                 ('"有島 武郎"' in x) or
                 ('"夏目 漱石"' in x) or
                 ('"中原 中也"' in x) or
                 ('"新渡戸 稲造"' in x) or
                 ('"野村 胡堂"' in x) or
                 ('"長谷川 時雨"' in x) or
                 ('"堀 辰雄"' in x) or
                 ('"牧野 信一"' in x) or
                 ('"太宰 治"' in x)
                ))

    test = AozoraCook()
    lst = test.csv_read("list_person_all_utf8.csv",
                        f = func_f,
                        g = lambda x: [x[1], x[3], x[0],x[2]])
    for x in lst:
        print(x)
    print(len(lst))

def testfunc3():
    """
    それなりに大量の文章を取ってくる処理の実装例
    zipを取ってくるところまで例として作ってある。unzipとテキスト処理は別
    """

    def func_f(x):
        return ('新字新仮名' in x and
                (('"芥川 竜之介"' in x) or 
                 ('"有島 武郎"' in x) or
                 ('"夏目 漱石"' in x) or
                 ('"中原 中也"' in x) or
                 ('"新渡戸 稲造"' in x) or
                 ('"野村 胡堂"' in x) or
                 ('"長谷川 時雨"' in x) or
                 ('"堀 辰雄"' in x) or
                 ('"牧野 信一"' in x) or
                 ('"太宰 治"' in x)
                ))

    test = AozoraCook()
    lst = test.csv_read("list_person_all_utf8.csv",
                        f = func_f,
                        g = lambda x: [x[1], x[3], x[0],x[2]])
    
    i = 0
    for x in lst:
        test.get_zipfile( int(x[2]), int(x[3]), re.sub('"', '', x[0]) + "_" + re.sub('"', '', x[1]) + ".zip" )
        i += 1
        print(str(i) + ": " + re.sub('"', '', x[0]) + "_" + re.sub('"', '', x[1]) + ".zip")
        time.sleep(1.0)



def testfunc4():
    """
    unzipする
    """
    test = AozoraCook()
    zip_lst = filter(lambda x: "zip" in x, os.listdir(os.path.dirname(os.path.abspath(__file__))+'/files/'))
    for x in zip_lst:
        test.unzip( "files/"+x, "files/"+x[:-4]+".txt")

def testfunc5():
    """
    cleanする
    """
    test = AozoraCook()        
    txt_lst = filter(lambda x: ".txt" in x, os.listdir(os.path.dirname(os.path.abspath(__file__))+'/files/'))
    for x in txt_lst:
        test.txt_clean("files/" + x)


if __name__ == '__main__':
    testfunc5()
