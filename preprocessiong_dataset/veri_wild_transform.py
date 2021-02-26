import os
import numpy as np

train_file = './train_test_split/train_list.txt'
test_small = './train_test_split/test_3000_id.txt'
query_small = './train_test_split/test_3000_id_query.txt'
test_middle = './train_test_split/test_5000_id.txt'
query_middle = './train_test_split/test_5000_id_query.txt'
test_large = './train_test_split/test_10000_id.txt'
query_large = './train_test_split/test_10000_id_query.txt'

input_img_path = './images/'
output_train_img_path =  './bounding_box_train/'
output_test_img_path =  './bounding_box_test/'
output_query_img_path = './query/'



output_test_middle_img_path =  './test_middle/'
output_query_middle_img_path = './query_middle/'

output_test_small_img_path =  './test_small/'
output_query_small_img_path = './query_small/'

def file_parsing(filepath,output_path,output_list = False):
    files = open(filepath,'r')
    file_list =[]
    for _file in files:
        _parse = _file.split(' ')
        _fpath = _parse[0].split('/')[1]
        _vid = '%d'%(int(_parse[1]))
        _cid = '%d'%(int(_parse[2]))
        transformed_file = str(_vid).zfill(5)+'_c'+_cid+'_'+_fpath
        transformed_path =output_path + transformed_file
        path_from = input_img_path+_parse[0]
        _command = 'cp %s %s'%(path_from,transformed_path)
        print(_command)
        os.system(_command)
        file_list.append(transformed_file)
    if output_list==True:
        return file_list


def duplication_check(query_list,test_list):
    for _qf in query_list:
        if _qf in test_list:
            print(_qf+' is duplicated')
            _command = 'rm %s'%(output_query_img_path+query_list)
            print(_command)
            os.system(_command)

if __name__ =='__main__':
    #for Training
    print('Preparing Trainig dataaset for Veri-Wild')
    file_parsing(train_file,output_train_img_path)
    #for Test

    print('Preparing large test dataaset for Veri-Wild')
    test_file_list = file_parsing(test_large,output_test_img_path,output_list=True)
    #for Test Query
    print('Preparing large query dataaset for Veri-Wild')
    query_file_list = file_parsing(query_large,output_query_img_path,output_list=True)


    print('Preparing middle test dataaset for Veri-Wild')
    test_file_list = file_parsing(test_middle,output_test_middle_img_path,output_list=False)
    #for Test Query
    print('Preparing middle query dataaset for Veri-Wild')
    query_file_list = file_parsing(query_middle,output_query_middle_img_path,output_list=False)

    
    print('Preparing small test dataaset for Veri-Wild')
    test_file_list = file_parsing(test_small,output_test_small_img_path,output_list=False)
    #for Test Query
    print('Preparing small query dataaset for Veri-Wild')
    query_file_list = file_parsing(query_small,output_query_small_img_path,output_list=False)

    #Duplicated file check
    #duplication_check(query_file_list,test_file_list)
