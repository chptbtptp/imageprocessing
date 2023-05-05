python q2l_train.py --dataname 'heart' -a 'Q2L-SwinL-384' --img_size 448 -b 16 --epochs 100 --lr 0.00000005 --num_labels 2 #--resume checkpoint/Q2L-SwinL-384/heart/12/model_best.pth.tar
# python q2l_train.py --dataname 'iuxray' -a 'Q2L-SwinL-384' --img_size 384 -b 16 --epochs 100 --lr 0.000000005 --inference --num_labels 43 --resume checkpoint/Q2L-SwinL-384/iuxray/43/model_best.pth.tar
# 
# python q2l_train.py --dataname 'iuxray' -a 'Q2L-SwinL-384' --img_size 384 -b 32 --epochs 100 --lr 0.000005 --num_labels 43
