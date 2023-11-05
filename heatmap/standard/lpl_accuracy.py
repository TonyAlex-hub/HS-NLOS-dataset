'''lpl 19/07/02
精确度评价指标'''
'''拼接后的影像精确度评价指标'''
import sys
import cv2 as cv
from standard import divide_to_patches
sys.path.append("/home/test/lpl/vgg/standard")
sys.path.append("/home/test/lpl/vgg")
def acc_2D(prediction,label):
    shape = label.shape
    TP = 0;FP = 0
    FN = 0;TN = 0
    for i in  range (0,shape[0]):
        for j in range(0,shape[1]):
            if label[i][j]!=0 and prediction[i][j]!=0:
                TP+=1
            elif label[i][j]!=0 and prediction[i][j]==0:
                FN+=1
            elif label[i][j]==0 and prediction[i][j]!=0:
                FP+=1
            elif label[i][j]==0 and prediction[i][j]==0:
                TN+=1
    if TP + FP!=0:
        precision = float(TP) / float(TP + FP)
    else:
        precision=0
    if TP + FN!=0:
        recall = float(TP) / float(TP + FN)
    else:
        recall=0
    return precision, recall
'''拼接后的影像精确度_relx评价指标'''
def acc_2D_relx(prediction,label,N):
    shape = label.shape
    TP = 0;FP = 0;TP_relax=0
    FN = 0;TN = 0
    for i in  range (0,shape[0]):
        for j in range(0,shape[1]):
            if IsRight_N(label, i, j, 255, N) and prediction[i][j] != 0:
                TP_relax = TP_relax + 1
            # if label[i][j]!=0 and prediction[i][j]!=0:
            #     TP+=1
            elif label[i][j]!=0 and prediction[i][j]==0:
                FN+=1
            elif label[i][j]==0 and prediction[i][j]!=0:
                FP+=1
            elif label[i][j]==0 and prediction[i][j]==0:
                TN+=1
    if TP + FP!=0:
        precision = float(TP_relax) / float(TP_relax + FP)
    else:
        precision=0
    if TP + FN!=0:
        recall = float(TP_relax) / float(TP_relax + FN)
    else:
        recall=0
    return precision, recall
'''判断预测像素的临近范围内是否有相应对的标签。邻域大小为该像素的上、小、左、右个N像素'''
def IsRight_N(label,pos_row,pos_column,key_pixel,N):
    label_shape=label.shape
    start_row=0 if pos_row-N<0 else pos_row -N
    end_row=label_shape[0]-1 if pos_row+N>=label_shape[0] else pos_row+N
    start_column= 0 if pos_column-N<0 else pos_column-N
    end_column=label_shape[1]-1 if pos_column+N>=label_shape[1] else pos_column+N
    # '''corss shape'''
    # for i in range(start_row,end_row+1):
    #     if label[i][pos_column]==key_pixel:
    #         return True
    # for j in range(start_column,end_column+1):
    #     if label[pos_row][j]==key_pixel:
    #         return True
    #田字形缓冲区
    for i in range(start_row,end_row+1):
        for j in  range(start_column,end_column+1):
            if label[i][j]==key_pixel:
                return True
    return False
'''计算单类别预测目标的精确度，比如，道路和建筑物，可以同时计算两个目标的精确度。标签形式为3D-矩阵，
矩阵中的每个元素是一个三维的向量(v1,v2，v3)。v1代表该像素属于背景的的概率，v2代表属于预测目标1的概率，
v3代表属于预测目标2的概率。
已改动'''
def acc_3D(pre,label,key_pixel1=255,key_pixel2=76):
    shape=label.shape
    TP_1=0;FP_1=0;FN_1=0;TN_1=0
    TP_2 = 0;FP_2 = 0;FN_2 = 0;TN_2 = 0
    for i in range (0,shape[0]):
        for j in range (0,shape[1]):
            # 目标1的精度
            if label[i][j]==key_pixel1 and pre[i][j]==key_pixel1:
                TP_1 = TP_1 + 1
            elif label[i][j] ==key_pixel1 and pre[i][j] != key_pixel1:
                FN_1 = FN_1 + 1
            elif label[i][j] != key_pixel1 and pre[i][j] == key_pixel1:
                FP_1 = FP_1 + 1
            elif label[i][j] != key_pixel1 and pre[i][j] != key_pixel1:
                TN_1 = TN_1 + 1
            # 目标1的精度
            if label[i][j]==key_pixel2 and pre[i][j]==key_pixel2:
                TP_2 = TP_2 + 1
            elif label[i][j] ==key_pixel2 and pre[i][j] != key_pixel2:
                FN_2 = FN_2 + 1
            elif label[i][j] != key_pixel2 and pre[i][j] == key_pixel2:
                FP_2 = FP_2 + 1
            elif label[i][j] != key_pixel2 and pre[i][j] != key_pixel2:
                TN_2 = TN_2 + 1
    #目标1的精度
    if TP_1+FP_1 !=0:
        precision1=float(TP_1)/float(TP_1+FP_1)
    else:
        precision1=0
    if TP_1+FN_1 !=0:
        recall1=float(TP_1)/float(TP_1+FN_1)
    else:
        recall1=0
    # 目标2的精度
    if TP_2+FP_2 !=0:
        precision2=float(TP_2)/float(TP_2+FP_2)
    else:
        precision2=0
    if TP_2+FN_2 !=0:
        recall2=float(TP_2)/float(TP_2+FN_2)
    else:
        recall2=0
    return precision1,recall1,precision2,recall2
def acc_3D_relx(pre,label,key_pixel1=255,key_pixel2=76,N=3):
    shape=label.shape
    TP_1=0;FP_1=0;FN_1=0;TN_1=0;TP_1_relx=0
    TP_2 = 0;FP_2 = 0;FN_2 = 0;TN_2 = 0;TP_2_relx=0
    for i in range (0,shape[0]):
        for j in range (0,shape[1]):
            # 目标1的精度
            if IsRight_N(label, i, j, key_pixel1, N) and pre[i][j] ==key_pixel1:
                TP_1_relx = TP_1_relx + 1
            if label[i][j]==key_pixel1 and pre[i][j]==key_pixel1:
                TP_1 = TP_1 + 1
            elif label[i][j] ==key_pixel1 and pre[i][j] != key_pixel1:
                FN_1 = FN_1 + 1
            elif label[i][j] != key_pixel1 and pre[i][j] == key_pixel1:
                FP_1 = FP_1 + 1
            elif label[i][j] != key_pixel1 and pre[i][j] != key_pixel1:
                TN_1 = TN_1 + 1
            # 目标2的精度
            if IsRight_N(label, i, j, key_pixel2, N) and pre[i][j] ==key_pixel2:
                TP_2_relx = TP_2_relx + 1
            if label[i][j]==key_pixel2 and pre[i][j]==key_pixel2:
                TP_2 = TP_2 + 1
            elif label[i][j] ==key_pixel2 and pre[i][j] != key_pixel2:
                FN_2 = FN_2 + 1
            elif label[i][j] != key_pixel2 and pre[i][j] == key_pixel2:
                FP_2 = FP_2 + 1
            elif label[i][j] != key_pixel2 and pre[i][j] != key_pixel2:
                TN_2 = TN_2 + 1
    #目标1的精度
    if TP_1+FP_1 !=0:
        pre1_relx=float(TP_1_relx)/float(TP_1+FP_1)
    else:
        pre1_relx=0
    if TP_1+FN_1 !=0:
        rec1_relx=float(TP_1_relx)/float(TP_1+FN_1)
    else:
        rec1_relx=0
    # 目标2的精度
    if TP_2+FP_2 !=0:
        pre2_relx=float(TP_2_relx)/float(TP_2+FP_2)
    else:
        pre2_relx=0
    if TP_2+FN_2 !=0:
        rec2_relx=float(TP_2_relx)/float(TP_2+FN_2)
    else:
        rec2_relx=0
    return pre1_relx,rec1_relx,pre2_relx,rec2_relx
#F1 score
def acc_f1(pre,rec):
    if pre+rec!=0:
        f1=2*pre*rec/(pre+rec)
    else:
        f1=0
    return f1
if __name__ == '__main__':
    map_path='/home/lpl/lpl/distributed/merge_roads/images/11128870_15_label.tif'
    pre_path = '/home/lpl/lpl/distributed/merge_roads/images/11128870_15_wuda.tif'
    im_path='/public/lpl/datasets/source_datasets/mass_roads/test/sat_t1/11128870_15.tiff'
    sat_im = cv.imread(im_path, cv.IMREAD_COLOR)
    map = cv.imread(map_path, cv.IMREAD_GRAYSCALE)
    pre=cv.imread(pre_path, cv.IMREAD_GRAYSCALE)
    map_patches=map;pre_patches=pre
    P, R = acc_2D(pre, map);f1=acc_f1(P,R)
    print ('p=', P, 'r=', R, 'f1=', f1)
    # size=512
    # sat_patches, map_patches = \
    #     divide_to_patches.divide_to_patches(size, size, size, sat_im, map)
    # sat_patches, pre_patches = \
    #     divide_to_patches.divide_to_patches(size, size, size, sat_im, pre)
    # sum_P=0;sum_R=0;sum_f1=0;count=0
    # for i in range(0,len(pre_patches)):
    #     pre_label=pre_patches[i];label=map_patches[i]
    #     P,R=acc_2D(pre_label,label)
    #     sum_P=sum_P+P;sum_R=sum_R+R
    #     f1=acc_f1(P,R)
    #     sum_f1=sum_f1+f1
    #     print(P,R,f1)
    #     count=count+1
    #     if P== 0 and R==0:
    #         sum_P=sum_P+1;sum_R=sum_R+1
    #         f1=acc_f1(1,1);sum_f1=sum_f1+f1
    # print ('mean_p=',sum_P/count,'mean_r=',sum_R/count,'mean_f1=',sum_f1/count)
    print ('done')