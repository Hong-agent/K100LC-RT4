#!/usr/bin/env python3
"""把运行时导出的 FA 输入（RT_DUMP_ATTN=层号）反解成 f32，和 dump 出来的 q/k/v 对照，
逐元素检查 Q/K/V 的量化与打包是否符合内核约定。"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt4_io import read_dump
H,KV,D,BG,QG = 24,4,256,64,128
KD, NQG = D//8, D//QG
recs = read_dump(sys.argv[1]); il=int(sys.argv[2]); T=int(sys.argv[3])
qq = np.fromfile('build/attn_qq.bin',np.uint32); qs=np.fromfile('build/attn_qs.bin',np.float32)
K  = np.fromfile('build/attn_k.bin',np.uint32);  Ks=np.fromfile('build/attn_ks.bin',np.float32)
CAP = K.size//(KV*KD)     # KV cache 的分配容量（行距）
print('kv 容量', CAP)
V  = np.fromfile('build/attn_v.bin',np.uint32);  Vs=np.fromfile('build/attn_vs.bin',np.float32)
out= np.fromfile('build/attn_out.bin',np.float32)
TP = qq.size//(H*KD); nk=CAP
print('TP=%d nk=%d'%(TP,nk))
def unpack(v):  # int4 解包：按字节，低半字节=偶数下标
    b = v.astype(np.uint32).view(np.uint8)          # 4 字节/dword
    lo = (b & 0xF).astype(np.int16); hi = ((b >> 4) & 0xF).astype(np.int16)
    w = np.empty(b.shape[:-1] + (b.shape[-1] * 2,), np.float32)
    w[..., 0::2] = np.where(lo >= 8, lo - 16, lo)
    w[..., 1::2] = np.where(hi >= 8, hi - 16, hi)
    return w
# 1) Q：内核把 Q 量化成 [head][row][g] + 尺度 [head][g][row]
qref = recs[1000+il*100+2].reshape(T,H,D).astype(np.float64)   # 含 1/sqrt(256)
qdec = np.zeros((H,TP,D))
for h in range(H):
    codes = unpack(qq.reshape(H,TP,KD)[h])                    # [TP][D]
    sc = qs.reshape(H,NQG,TP)[h]                              # [g][row]
    qdec[h] = codes * np.repeat(sc.T, QG, axis=1)             # 行尺度展开
print('Q[0,0,:4] 导出解码=%s  期望=%s'%(np.round(qdec[0,0,:4],5), np.round(qref[0,0,:4],5)))
for h in range(4):
    d=np.abs(qdec[h,:T]-qref[:,h,:]).mean()/np.abs(qref[:,h,:]).mean()
    print('  Q head %d rel=%.3e'%(h,d))
# 2) K
kref = recs[1000+il*100+6].reshape(T,KV,D).astype(np.float64)
kdec = np.zeros((KV,nk,D))
for h in range(KV):
    codes = unpack(K.reshape(KV,CAP,KD)[h][:nk])
    sc = Ks.reshape(KV,NQG,nk)[h]
    kdec[h] = codes*np.repeat(sc.T,QG,axis=1)
print('K[0,0,:4] 导出解码=%s  期望=%s'%(np.round(kdec[0,0,:4],5), np.round(kref[0,0,:4],5)))
for h in range(KV):
    print('  K head %d rel=%.3e'%(h,np.abs(kdec[h,:T]-kref[:,h,:]).mean()/np.abs(kref[:,h,:]).mean()))
# 3) V（沿 key 转置打包）
vref = recs[1000+il*100+5].reshape(T,KV,D).astype(np.float64)
vdec = np.zeros((KV,nk,D))
for h in range(KV):
    codes = np.zeros((nk//8,8,D),np.float32)
    for k in range(nk):
        w = V.reshape(KV,CAP//8,D)[h,k//8]
        c = (w>>(4*(k%8)))&0xF
        c = np.where(c>=8,c-16,c)
        codes[k//8,k%8] = c
    vdec[h]=codes.reshape(-1,D)*Vs.reshape(KV,-1,D)[h,0]
print('V[0,0,:4] 导出解码=%s  期望=%s'%(np.round(vdec[0,0,:4],5),np.round(vref[0,0,:4],5)))
for h in range(KV):
    print('  V head %d rel=%.3e'%(h,np.abs(vdec[h,:T]-vref[:,h,:]).mean()/np.abs(vref[:,h,:]).mean()))
