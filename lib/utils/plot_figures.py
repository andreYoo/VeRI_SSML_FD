import os
import sys
import os
import pdb
def utils_for_fig3(path_mem,query_id,ss_id,graph_id, positive_id,epoch):
    data_path = './data/duke/bounding_box_train_camstyle/'
    alter_path = './data/duke/bounding_box_train/'
    save_file_path = './output/fig3/'

    gen_file_lv1 = save_file_path+'%d'%epoch

    if not os.path.exists(gen_file_lv1):
        os.makedirs(gen_file_lv1)

    for _t in query_id:
        gen_file_lv2 = gen_file_lv1 + '/%d'%(_t)
        if not os.path.exists(gen_file_lv2):
            os.makedirs(gen_file_lv2)
        img_path = data_path + path_mem[_t]
        alt_img_path = alter_path+ path_mem[_t]
        copy_path = gen_file_lv2+'/'+'Query_'+path_mem[_t]
        if os.path.isfile(img_path):
            os.system("cp %s %s"%(img_path,copy_path))
        else:
            os.system("cp %s %s"%(alt_img_path,copy_path))


    for _t in range(len(ss_id)):
        gen_file_lv2 = gen_file_lv1 + '/%d'%(query_id[_t])
        for _i,_j in enumerate(ss_id[_t]):
            if path_mem[_j]==None:
                continue
            else:
                img_path = data_path + path_mem[_j]
                alt_img_path = alter_path + path_mem[_j]
                copy_path = gen_file_lv2+'/'+'%d-epoch_%d_ss_'%(epoch,_i)+path_mem[_j]
                if os.path.isfile(img_path):
                    os.system("cp %s %s"%(img_path,copy_path))
                else:
                    os.system("cp %s %s"%(alt_img_path,copy_path))

    for _t in range(len(graph_id)):
        gen_file_lv2 = gen_file_lv1 + '/%d'%(query_id[_t])
        for _i,_j in enumerate(graph_id[_t]):
            if path_mem[_j]==None:
                continue
            else:
                img_path = data_path + path_mem[_j]
                alt_img_path = alter_path+ path_mem[_j]
                copy_path = gen_file_lv2+'/'+'%d-epoch_%d_graph_'%(epoch,_i)+path_mem[_j]
                if os.path.isfile(img_path):
                    os.system("cp %s %s"%(img_path,copy_path))
                else:
                    os.system("cp %s %s"%(alt_img_path,copy_path))

    for _t in range(len(positive_id)):
        gen_file_lv2 = gen_file_lv1 + '/%d'%(query_id[_t])
        for _i,_j in enumerate(positive_id[_t]):
            if path_mem[_j]==None:
                continue
            else:
                img_path = data_path + path_mem[_j]
                alt_img_path = alter_path + path_mem[_j]
                copy_path = gen_file_lv2+'/'+'%d-epoch_%d_positive_'%(epoch,_i)+path_mem[_j]
                if os.path.isfile(img_path):
                    os.system("cp %s %s"%(img_path,copy_path))
                else:
                    os.system("cp %s %s"%(alt_img_path,copy_path))
    return 0
