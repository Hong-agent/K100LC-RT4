import subprocess, sys, os, time, re
sys.path.insert(0,'/rt/tools'); import tok as T
ROOT='/rt'
M=os.environ.get('RT_RT4', ROOT+'/models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b.rt4'); J=M+'.json'
p=subprocess.Popen([ROOT+'/build/rt','--engine','--model',M,'--json',J],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
while True:                            # 引擎起不来时要报错，别死等
    _line=p.stdout.readline()
    if not _line: raise SystemExit('引擎启动失败：检查权重是否存在（%s）'%M)
    if _line.startswith('READY'): break
base="The history of computing began with mechanical calculators and continued through the industrial revolution. "
for n in (64, 512, 2048):
    txt=(base*(n//20+1))
    ids=T.encode(txt)[:n]
    p.stdin.write('PREFILL '+','.join(map(str,ids))+'\n'); p.stdin.flush()
    line=p.stdout.readline().strip()
    print('prefill %5d tok: %s'%(len(ids), line[:80]))
for temp in (0.0, 0.7):
    p.stdin.write('PREFILL '+','.join(map(str,T.encode(base)))+'\n'); p.stdin.flush(); p.stdout.readline()
    t0=time.time(); p.stdin.write('GEN 64 %f 1 0 1234 248044,248046\n'%temp); p.stdin.flush()
    cnt=0
    while True:
        l=p.stdout.readline()
        if l.startswith('TOK'): cnt+=1
        elif l.startswith('END'): print('decode temp=%.1f: %s  墙钟 %.2fs -> %.1f tok/s'%(temp,l.strip(),time.time()-t0,cnt/(time.time()-t0))); break
p.stdin.write('QUIT\n')
