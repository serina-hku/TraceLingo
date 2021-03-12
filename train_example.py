import argparse
import collections
import time
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.nn.init as INIT
import torch.optim as optim
from torch.utils.data import DataLoader
from gene_dataset import Datagenerator
import torch
# from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import networkx as nx
import spacy
nlp = spacy.load('en')
import dgl
from wwl import *
from dgl.data.tree import SST, SSTBatch
from operator import itemgetter
from tree_lstm_feature import TreeLSTM
from convtree_lstm_feature import ConvLSTM
import heapq
import networkx
from convtree_lstm_feature import ConvLSTM
# torch.set_printoptions(threshold='nan')
import sklearn as sk
# from minepy import MINE

import math
def euclidean(v1,v2):
#如果两数据集数目不同，计算两者之间都对应有的数
  sq=np.square(v1 - v2)
  su=np.sum(sq)
  d= np.sqrt(su)
  return math.sqrt(d)


def getdistances(data):
    data=np.array(data)
    KNN = np.zeros([data.shape[0], data.shape[0]])
    for i in range(data.shape[0]):
        for j in  range(data.shape[0]):
          vec1 = data[i]
          vec2 = data[j]
          KNN[i][j]=euclidean(vec1,vec2)
    return KNN
def mape(y_true, y_pred):
    return th.mean(th.abs((y_pred- y_true) / y_true))
def batcher(device):
    def batcher_dev(batch):
        batch_trees = dgl.batch(batch)
        return SSTBatch(graph=batch_trees,
                        mask=batch_trees.ndata['mask'].to(device),
                        wordid=batch_trees.ndata['x'].to(device),
                        label=batch_trees.ndata['y'].to(device))
    return batcher_dev
def cos(x,y):
   return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
def main(args):
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed(args.seed)

    best_epoch = -1
    cuda = args.gpu >= 0
    device = th.device('cuda:{}'.format(args.gpu)) if cuda else th.device('cpu')
    if cuda:
        th.cuda.set_device(args.gpu)
    trainset = SST()
    graphset,train_graphset,node_attrs,G,A,G0,g_wwl,rootid=Datagenerator()
    model =  TreeLSTM(trainset.num_vocabs,
                     args.x_size,
                     args.h_size,
                     trainset.num_classes,
                     args.dropout,
                     # cell_type='childsum' if args.child_sum else 'nary',
                      cell_type='childsum',
                     pretrained_emb = trainset.pretrained_emb).to(device)
    params_ex_emb =[x for x in list(model.parameters()) if x.requires_grad and x.size(0)!=trainset.num_vocabs]
    params_emb = list(model.embedding.parameters())

    for p in params_ex_emb:
        if p.dim() > 1:
            INIT.xavier_uniform_(p)
    optimizer = optim.Adam([
        {'params':params_ex_emb, 'lr':args.lr, 'weight_decay':args.weight_decay},
        {'params':params_emb, 'lr':0.1*args.lr}])
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    dur = []
    #Reorganize the read dataframe into a list
    label_duration = []
    feature_name = []
    feature_name_word = []
    Roleinstance_name = []
    ActivityStart=[]
    NodeID=[]
    RootActivityId=[]
    ParentActivityId=[]
    ActivityId=[]
    labelclass=[]
    Tid=[]
    for k, v in node_attrs.items():
        count = 0
        vec = []
        for k1, v1 in v.items():
            # print("")
            if len(v) == 2:
                if count == 0:
                    label_duration.append(v1)
                if count == 1:
                    doc = nlp(v1)
                    vec = doc.vector

                    feature_name.append(vec.tolist())
                    feature_name_word.append(v1)
                    vec = vec[0:25].tolist()

                if count == 2:
                    ActivityStart.append(v1)
                if count == 3:
                    NodeID.append(v1)
                if count == 4:
                    RootActivityId.append(v1)
                if count == 5:
                    ParentActivityId.append(v1)
                if count == 6:
                    ActivityId.append(v1)
                if count == 7:
                    Tid.append(v1)
                count = count + 1
            else:
                if count == 1:
                    label_duration.append(v1)
                if count == 2:
                    # print("2 v1", v1)
                    doc = nlp(v1)
                    vec1 = doc.vector
                    vec=vec1[0:20].tolist()
                    feature_name_word.append(v1)
                if count == 3:
                    # print("3 v1",v1)
                    doc = nlp(v1)
                    vec1 = doc.vector
                    vec.extend(vec1[0:5].tolist())
                    # ActivityStart.append(v1)
                if count == 4:
                    labelclass.append(int(v1))
                if count == 6:
                    ##cluster

                    doc = nlp(v1)
                    vec1 = doc.vector
                    Roleinstance_name.append(v1)
                    vec.extend(vec1[0:5].tolist())
                if count == 7:
                    ##cluster
                    doc = nlp(v1)
                    ActivityId.append(v1)
                if count == 8:
                    labelclass.append(int(v1))
                count = count + 1
        feature_name.append(vec)
    feature_name_np = np.array(feature_name)
    kernel_matrix, node_representations = wwl(g_wwl, node_features=feature_name_np, num_iterations=1)


    feature_name_np2 = np.column_stack((node_representations[0][0:feature_name_np.shape[0]], feature_name_np,))
    feature_name_np_tensor = th.tensor(feature_name_np2, dtype=th.float32)
    g = graphset[0]
    n = g.number_of_nodes()
    feature_name_np_tensor1 = feature_name_np_tensor
    label_duration_tensor = th.tensor(label_duration, dtype=th.float32)
    labelclass = th.tensor(labelclass, dtype=th.float32)



    """
    train part
    """

    label_duration_tensor1 = label_duration_tensor.type(th.FloatTensor)
    label_duration_tensor1 = label_duration_tensor1.reshape(label_duration_tensor1.shape[0], 1)

    feature_name_np_tensor_aggragte = np.zeros([feature_name_np.shape[0], 32])
    feature_name_np_tensor_aggragte_2np = np.zeros([feature_name_np.shape[0], 50])

    for i in range(feature_name_np.shape[1]-2):

       path_all=networkx.shortest_path(G0,source=(i+1))
       pathlist=list(path_all.values())[-1]
       for k in range(len(pathlist)):

           feature_name_np_tensor_aggragte[i]=feature_name_np_tensor1[pathlist[k]]+feature_name_np_tensor_aggragte[i]

       feature_name_np_tensor_aggragte_2np[i][0:32] = feature_name_np_tensor1[i]
       feature_name_np_tensor_aggragte_2np[i][32:50] = (feature_name_np_tensor_aggragte[i][0:18])
    feature_name_np_tensor_aggragte_2 = torch.from_numpy(feature_name_np_tensor_aggragte_2np).type(torch.FloatTensor)
    import pickle
    picklefile1 = open("feature_name_np_tensor_aggragte_2np.pkl", "wb")
    pickle.dump(feature_name_np_tensor_aggragte_2np, picklefile1)
    picklefile1.close()
    ####################################################################

    labelclass_session=labelclass[rootid]

    # for epoch in range(1000):
    #     t_epoch = time.time()
    #     model.train()
    #
    #     t0 = time.time() # tik
    #
    #     h = th.zeros((feature_name_np_tensor1.shape[0], feature_name_np_tensor1.shape[1]))
    #     c = th.zeros((feature_name_np_tensor1.shape[0], feature_name_np_tensor1.shape[1]))
    #     # logits ,classlogits= model(g,G, h, c,feature_name_np_tensor1)
    #     logits, classlogits = model(g, G, h, c, feature_name_np_tensor_aggragte_2,rootid,epoch)
    #     logp=logits.type(th.FloatTensor)
    #
    #
    #     labelclass=  labelclass_session.type(th.LongTensor)
    #     # logp=logp.reshape(k,1)
    #     labelclass = labelclass.reshape(len(rootid))
    #
    #     loss = F.mse_loss(logp, labelclass, size_average=False)
    #
    #     logp_class=F.log_softmax(classlogits, dim=1)
    #
    #     logp_class=logp_class.type(th.FloatTensor)
    #
    #     logp_class = logp_class.reshape([ len(rootid), 2])
    #
    #     loss1 = F.nll_loss(logp_class, labelclass)
    #
    #     labelclass =np.array(labelclass)
    #     labelclass=torch.from_numpy(labelclass).type(torch.LongTensor)
    #
    #     optimizer.zero_grad()
    #     loss1.backward()
    #     optimizer.step()
    #     dur.append(time.time() - t0) # tok
    #     pred = logp_class.data.max(1, keepdim=True)[1]
    #     acc = pred.eq(labelclass.data.view_as(pred)).cpu().sum().item() / float(labelclass.size()[0])
    #
    #     print("Epoch {:05d} | Step {:05d} | Loss {:.4f} | Acc {:.4f} | Root Acc {:.4f} | Time(s) {:.4f}",
    #                 epoch, loss1.item(),acc)
    #     file_handle1 = open(
    #         '1029_loss_sumVMOnCreate611_nodenumtrain_bin1.txt',
    #         mode='a')
    #     print(str(epoch), file=file_handle1)
    #     print(str(loss.item()), file=file_handle1)
    #     file_handle1.close()
    #
    # th.save(model.state_dict(), 'train.pkl'.format(args.seed))
    ###############################################################################################
    """
        test part
        """
    model.load_state_dict(th.load('train.pkl'.format(args.seed)))
    accs = []
    model.eval()
    # label_duration_tensor_test = label_duration_tensor.type(th.FloatTensor)
    label_duration_tensor_test = labelclass.type(th.FloatTensor)
    feature_name_np_tensor_test = feature_name_np_tensor
    feature_name_word_test = feature_name_word
    for step in range(500):
        g = graphset[0]
        n = g.number_of_nodes()
        with th.no_grad():
            h = th.zeros((n, args.h_size)).to(device)
            c = th.zeros((n, args.h_size)).to(device)

            logits, classlogits = model(g, G, h, c, feature_name_np_tensor_aggragte_2,rootid,epoch)

            # logp_class=classlogits
            logp_class = F.log_softmax(classlogits, dim=1)

            file_handle3 = open('logp_class.txt', mode='a')
            logp_class.numpy()
            import pickle
            picklefile=open("logp_class_abnormal_normal.pkl","wb")
            pickle.dump(logp_class,picklefile)
            picklefile.close()

            print("logp_class", logp_class.numpy().tolist(), file=file_handle3)
            file_handle3.close()
            logp_class = logp_class.type(th.FloatTensor)

            logp = logits.type(th.FloatTensor)
            # pred = logp_class.data.max(1, keepdim=True)[1]

    import pandas as pd

    logpnp=np.array(logp)

    test_acc = 91
    label_duration_tensor_test = th.tensor(label_duration_tensor_test, dtype=th.int)

    label_duration_tensor_test = label_duration_tensor_test.reshape(len(rootid), 1)
    """
        caculate mape
        """

    loss_test = mape(logp, label_duration_tensor_test)


    logp = logp.reshape([1, len(rootid)])

    label_duration_tensor_test = label_duration_tensor_test.reshape([1, len(rootid)])
    # label_duration_tensor_test = label_duration_tensor_test.reshape([1, 200])
    print("label_duration_tensor_test", label_duration_tensor_test.shape)
    print("logp", logp.shape)

    # logp1.dtype='float32'
    # print("logp", logp1.dtype)

    label_duration_tensor_test1=np.array(label_duration_tensor_test,dtype=np.int32)
    # label_duration_tensor_test.dtype='float32'
    print("label_duration_tensor_test", label_duration_tensor_test.dtype)
    label_duration_tensor_test1=label_duration_tensor_test1.tolist()[0]


    print("label_duration_tensor_test1", len(label_duration_tensor_test1))
    print("label_duration_tensor_test1",label_duration_tensor_test1)



    distribution=torch.argmax(logp_class, dim=1)
    print("distribution", distribution)


    # logp1= distribution.reshape([4, 261])
    logp1 = np.array(distribution, dtype=np.int32)
    selector = SelectKBest(chi2, k=2)
    input=[]

    for i in range(len(feature_name_np_tensor_aggragte_2.numpy().tolist())):

        input.append(list(map(abs, feature_name_np_tensor_aggragte_2.numpy().tolist()[i])))

    X=feature_name_np_tensor_aggragte_2np
    # print("X_new.scores_", selector.transform(X))
    logp1 = logp1.tolist()


    listlog = distribution.numpy().tolist()
    label_duration_tensor_test1_np = np.array(label_duration_tensor_test1)
    Abnormlist_np= np.where((distribution ==2) | ( distribution ==1) ,)
    # K = cos(logp_class, logp_class.t())
    K = getdistances(logp_class)
    for i in range(Abnormlist_np[0].shape[0]):
        causeroot=[]
        similarity=[]
        if i !=0:

         path = networkx.shortest_path(G0, source=Abnormlist_np[0][i])
         print("path",path)
         list(path.values())
         list_path = list(path.values())
         print("list_path", list_path)
         rootcausedep = []
         for iii in range(len(list_path)):
            for jjj in range(len(list_path[iii])):
                if list_path[iii][jjj] not in rootcausedep and (list_path[iii][jjj]!=Abnormlist_np[0][i]):
                    rootcausedep.append(list_path[iii][jjj])
                    # similarity.append(K[Abnormlist_np[0][i]][list_path[iii][jjj]])
         print("rootcausedep", rootcausedep)
        # similarity

         for j in range(len(rootcausedep)):
            KJ=0
            for jk in range(len(rootcausedep)):
                if jk is not j:
                    KJ=K[rootcausedep[j]][rootcausedep[jk]]+KJ
            KJ=KJ+K[rootcausedep[j]][Abnormlist_np[0][i]]
            if KJ is  not 0:
              similarity.append(KJ)


         print("similarity",similarity)
         if len(similarity) >0 :
          max_index = similarity.index(max(similarity, key=abs))
          print("rootcausedep",rootcausedep, rootcausedep[max_index])


    print("test 0", sum(distribution==0))
    print("test 1", sum(distribution==1))
    print("test 2", sum(distribution==2))
    print("test 3", sum(distribution==3))
    print("label 0", label_duration_tensor_test1.count(0))
    print("label 1", label_duration_tensor_test1.count(1))
    print("label 2", label_duration_tensor_test1.count(2))
    print("label 3", label_duration_tensor_test1.count(3))
    # logp1
    print("logp1",len(logp1))
    print("label_duration_tensor_test1", len(label_duration_tensor_test1))
    f1score=sk.metrics.f1_score(logp1,label_duration_tensor_test1, average='micro')
    print("f1score",f1score)
    print("Epoch {:05d} | Test Acc {:.4f} | MAPE Loss {:.4f},f1score",
          best_epoch, test_acc, loss_test, f1score)
    # loss_test = mape(logp, label_duration_tensor_test[0:522])

    abs_duration=abs(label_duration_tensor_test - logp)
    # abs_duration = abs(label_duration_tensor_test[0:522] - logp)
    abs_duration=abs_duration
    id = th.where(abs_duration>0.05)
    id1 = th.where(abs_duration > 0.1)
    id11 = th.where(abs_duration >=1)
    id4 = th.where(abs_duration > 0.4)
    id44=np.array( id[0])
    id44list=id44.tolist()
    feature_name_wordkk=[]
    ActivityStartkk=[]
    ActivityIdkk = []
    label_durationkk=[]
    logpkk=[]
    abs_duration = (abs_duration).numpy()
    idk = heapq.nlargest(3000, range(len(abs_duration)), abs_duration.__getitem__)
    idklist = idk
    id44list = idklist
    logpk=[]
    print("len(idklist)",len(idklist))
    print("len(feature_name_word_test)",len(feature_name_word_test))
    for i in range(len(id44list)):
        print("i",i)
        feature_name_wordkk.append(feature_name_word_test[id44list[i]])

        label_durationkk.append(label_duration[id44list[i]])
        logpkk.append(abs_duration[id44list[i]])
        logpk.append(logp[id44list[i]])
    print("id0.05",id)
    print("id0.05", len(id[0]))
    print("id0.1", id1)
    print("id0.1", len(id1[0]))
    print("id0.01", id11)
    print("id0.01", len(id11[0]))
    print("id0.01", len(id11[0])/100)
    print("AnomalyID>0.01", len(id44list))

    """
        save result txt
        """
    file_handle2 = open('1029sum_fristVMOnCreate611_nodenum_bin1.txt', mode='a')
    from collections import Counter
    import operator
    # 进行统计
    a = dict(Counter(feature_name_wordkk))
    # 给得出的word进行排序
    b = sorted(a.items(), key=operator.itemgetter(1), reverse=True)

    for i in range(len(id44list)):
      print("index", str(i), file=file_handle2)
      print("indexcsv", str(id44list[i]), file=file_handle2)
      print("activity name",str(feature_name_wordkk[i]), file=file_handle2)
      # print("ActivityId",str(ActivityIdkk[i]), file=file_handle2)
      print("label duration",str(label_durationkk[i]), file=file_handle2)
      print("abs_duration",logpkk[i], file=file_handle2)
      print("predict duration", logpk[i], file=file_handle2)
    file_handle2.close()
    file_handle3 = open('0127sumaccVMOnCreate_nodenum_bin1.txt', mode='a')
    print("ActivityId", str(b), file=file_handle3)
    file_handle3.close()
    print('------------------------------------------------------------------------------------')
    print("Epoch {:05d} | Test Acc {:.4f} | MAPE Loss {:.4f},f1score",
        best_epoch, test_acc,loss_test,f1score)
    file_handle4 = open(
        '0127mean_mapeVMOnCreate611_nodenum_bin1.txt',
        mode='a')
    print("mape", file=file_handle4)
    print(str(loss_test), file=file_handle4)
    file_handle4.close()
    file_handle1 = open(
        '0127_loss_sumVMOnCreate611_nodenumtest.txt',
        mode='a')
    # print(str(epoch), file=file_handle1)
    print(str(test_acc), file=file_handle1)
    # print(str(loss.item()), file=file_handle1)
    file_handle1.close()
    # print(str(), file=file_handle1)
    print("node_representations", node_representations)
    print("rootid",rootid)
    label_session=[]
    for i in range(len(rootid)):
        label_session.append(label_duration_tensor_test1[i])

    print("sessionlabel", label_session)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--child-sum', action='store_true')
    parser.add_argument('--x-size', type=int, default=35)
    parser.add_argument('--h-size', type=int, default=35)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log-every', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    args = parser.parse_args()
    # print(args)
    main(args)
