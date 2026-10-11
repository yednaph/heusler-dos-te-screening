"""Colour-vision check for figure palettes (Python stand-in for the dataviz validator).
Simulates protan/deutan/tritan (Machado et al. 2009, severity 1.0) and reports OKLab dE x100
for every pair; targets: CVD dE >= 8, normal-vision dE >= 15."""
import sys, itertools, numpy as np
M = {'protan': [[0.152286,1.052583,-0.204868],[0.114503,0.786281,0.099216],[-0.003882,-0.048116,1.051998]],
     'deutan': [[0.367322,0.860646,-0.227968],[0.280085,0.672501,0.047413],[-0.011820,0.042940,0.968881]],
     'tritan': [[1.255528,-0.076749,-0.178779],[-0.078411,0.930809,0.147602],[0.004733,0.691367,0.303900]]}
def lin(c): c = np.asarray(c); return np.where(c <= 0.04045, c/12.92, ((c+0.055)/1.055)**2.4)
def oklab(rgb_lin):
    l,m,s = (np.array([[0.4122214708,0.5363325363,0.0514459929],[0.2119034982,0.6806995451,0.1073969566],[0.0883024619,0.2817188376,0.6299787005]]) @ rgb_lin)
    l,m,s = np.cbrt([l,m,s])
    return np.array([[0.2104542553,0.7936177850,-0.0040720468],[1.9779984951,-2.4285922050,0.4505937099],[0.0259040371,0.7827717662,-0.8086757660]]) @ np.array([l,m,s])
def hex2(h): return np.array([int(h[i:i+2],16)/255 for i in (1,3,5)])
cols = sys.argv[1].split(',')
for a,b in itertools.combinations(cols,2):
    la, lb = lin(hex2(a)), lin(hex2(b))
    out = {'normal': 100*np.linalg.norm(oklab(la)-oklab(lb))}
    for k,m in M.items(): out[k] = 100*np.linalg.norm(oklab(np.clip(np.array(m)@la,0,1))-oklab(np.clip(np.array(m)@lb,0,1)))
    worst = min(out[k] for k in M)
    flag = 'OK' if (worst>=8 and out['normal']>=15) else ('WARN' if worst>=6 and out['normal']>=15 else 'FAIL')
    print(f"{a} {b}  normal {out['normal']:5.1f}  " + '  '.join(f"{k} {out[k]:5.1f}" for k in M) + f"  -> {flag}")
