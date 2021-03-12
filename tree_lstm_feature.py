import time
import itertools
import networkx as nx
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
# import dgl
import math

class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(50, 3*50, bias=False)
        self.U_iou = nn.Linear(50, 3*50, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1,3*50))
        self.U_f = nn.Linear(50, 50)
        self.i_u = nn.Linear(12, 10)


    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        # print("nodes.mailbox['h'].size(0)",nodes.mailbox['h'].size(0))
        h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
        f = th.sigmoid(self.U_f(h_cat)).view(*nodes.mailbox['h'].size())
        c = th.sum(f * nodes.mailbox['c'], 1)

        return {'iou': self.U_iou(h_cat), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou


        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data['c']
        h = o * th.tanh(c)
        return {'h' : h, 'c' : c}

class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(50, 3 *50, bias=False)
        self.U_iou = nn.Linear(50, 3 *50, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * 50))

        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges):

        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_tild = th.sum(nodes.mailbox['h'], 1)
        f = th.sigmoid(self.U_f(nodes.mailbox['h']))
        c = th.sum(f * nodes.mailbox['c'], 1)
        # print("c",c)
        return {'iou': self.U_iou(h_tild), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou

        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)

        c = i * u + nodes.data['c']

        h = o * th.tanh(c)
        # print("nodes.data['h']", nodes.data['h'])

        return {'h': h, 'c': c}

def euclidean(v1,v2):
#如果两数据集数目不同，计算两者之间都对应有的数

  sq=np.square(v1 - v2)
  su=np.sum(sq)
  d= np.sqrt(su)
  # print("d",d)
  return math.sqrt(d)


def getdistances(data):
    distancelist = []
    KNN = np.zeros([data.shape[0], data.shape[0]])
    data = np.array(data)
    for i in range(len(data)):
        for j in range(len(data)):
            vec1 = data[i]
            vec2 = data[j]
            KNN[i][j] = euclidean(vec1, vec2)

    # distancelist.sort()
    return KNN
class TreeLSTM(nn.Module):
    def __init__(self,
                 num_vocabs,
                 x_size,
                 h_size,
                 num_classes,
                 dropout,
                 cell_type='nary',
                 pretrained_emb=None):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size

        self.L1 = nn.Linear(50, 17)
        self.L1_2 = nn.Linear(17, 1)
        self.L1_3 = nn.Linear(17,10)
        self.L1_4= nn.Linear(10, 2)

        self.embedding = nn.Embedding(num_vocabs, x_size)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(pretrained_emb)
            self.embedding.weight.requires_grad = True
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(10,1)
        cell = ChildSumTreeLSTMCell
        self.cell = cell(x_size, h_size)
        self.atte = nn.Parameter(th.ones(50, 50))  ##train
        self.conv = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2)
    def forward(self, g, G, h, c,emd,rootid,epoch):
        """Compute tree-lstm prediction given a batch.
        Parameters
        ----------
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.
        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        # print("g",g)
        # g=G.graph
        # print("g",g)
        """
               tree-lstm 训练每个参数化的节点（输入为一个矩阵）
               1.使用全部作为输入，可以提高训练时间
               2.g代表所有session的结构，通过idx区分
               
               """
        ########################################treelstm#########
        # g.register_message_func(self.cell.message_func)
        # g.register_reduce_func(self.cell.reduce_func)
        # g.register_apply_node_func(self.cell.apply_node_func)
        # g.ndata['iou'] = self.cell.W_iou(emd)
        # g.ndata['h'] = h
        # g.ndata['c'] = c
        # dgl.prop_nodes_topo( g)
        # h = g.ndata.pop('h')
        #####################################################
        # print("h",h)
        # print("h",h)
########################################
        # emd=h.reshape([1,3105,34])
        # emd = emd.permute(0, 2, 1)
        # emd=self.conv(emd)
        # emd = emd.permute(0, 2, 1)
        """
                    通过id对所有session的进行区分
                      
                      """

############################################### id区分###########################
        session_emd=th.ones([len(rootid),emd.shape[1]])
        # for i in range(len(rootid)):
        #     if i==0:
        #         session_emd[i] = th.sum(emd[0:rootid[i], :], dim=0)/(rootid[i]-0)
        #     else:
        #         session_emd[i]=th.sum(emd[rootid[i-1]:rootid[i],:],dim=0)/(rootid[i]-rootid[i-1])
        # h = session_emd
        ##################################
        """
                   使用attention对计算每个session中的id得分
                   1.topk pool为选取k个重要的节点默认k=50
                   nx1 nxd dxd  nxd
                             """


        att_seesion = th.mm(emd,self.atte)
        self.att=th.sum(att_seesion,dim=1)
        # self.att=att_seesion
        k=5
        h = att_seesion
        for i in range(len(rootid)):
            if i == 0:
              topSpan=th.topk(self.att[0:rootid[i], :], k, dim=0)
              topSpan_list=topSpan[1].reshape([k]).numpy().tolist()

            else:

              topSpan = th.topk(self.att[rootid[i-1]:rootid[i], :], k, dim=0)
              topSpan_list = topSpan[1].reshape([k]).numpy().tolist()
              # print("topSpan[:,0]",topSpan_list)
            # session_emd[i]=th.sum(att_seesion[topSpan_list], dim=0)
            session_emd[i] = th.sum(emd[topSpan_list], dim=0)
            # print("session_emd",session_emd)
            # print("topSpan_list",topSpan_list)
            # print("session_emd", session_emd.shape)
        # session_emd=session_emd.reshape([1,session_emd.shape[0]])


              # th.sum(att_seesion[0:th.max(self.att[0:rootid[i], :], dim=0),:])
            # else:
            #     th.sum(att_seesion[0:th.max(self.att[0:rootid[i], :], dim=0), :])
        ################train 的时候保留#############################

        if epoch==999:
            import pickle
            file = open('1201norm_train_emd.pkl', 'rb')
            adj = pickle.load(file)
            adk=th.cat([session_emd, adj], dim= 0)
            adk=adk.detach().numpy()
            K = getdistances(adk)
            print("K",K)
            picklefile1 = open("K_18764_2.pkl", "wb")
            pickle.dump(K, picklefile1)
            picklefile1.close()
            picklefile2= open("attention_train.pkl", "wb")
            pickle.dump(self.att, picklefile2)
            picklefile2.close()

        #     import pickle
        #     picklefile1 = open("1201norm_train_emd.pkl", "wb")
        #     pickle.dump(session_emd, picklefile1)
        #     picklefile1.close()
        ################train 的时候保留#############################

        h=session_emd
        # h=emd+h
        h1=self.L1(h)
        h = self.L1_2(h1)
        logits = h
        h2 = self.L1_3(h1)
        h2=self.L1_4(h2)

        return logits,h2
