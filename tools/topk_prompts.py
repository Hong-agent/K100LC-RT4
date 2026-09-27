import subprocess, sys, os, re
sys.path.insert(0,'/rt/tools'); import tok as T
ROOT='/rt'
M=os.environ.get('RT_RT4',ROOT+'/models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b.rt4')
ps=["The capital of France is","1+1=","2+2=","The quick brown fox jumps over the lazy",
    "Water is made of hydrogen and","My name is","I live in a small town. The weather today is"]
eng=subprocess.Popen([ROOT+'/build/rt','--engine','--model',M,'--json',M+'.json'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
while True:                            # 引擎起不来时要报错，别死等
    _line=eng.stdout.readline()
    if not _line: raise SystemExit('引擎启动失败：检查权重是否存在（%s）'%M)
    if _line.startswith('READY'): break
for p in ps:
    ids=T.encode(p)
    eng.stdin.write('PREFILL '+','.join(map(str,ids))+'\n'); eng.stdin.flush()
    line=eng.stdout.readline().strip()
    m=re.findall(r'(\d+):(-?[\d.]+)',line)
    tops=[int(a) for a,b in m][:5]
    print('%-45s | n=%d | %s'%(p,len(ids),' / '.join(repr(T.decode([t])) for t in tops)))
eng.stdin.write('QUIT\n')
