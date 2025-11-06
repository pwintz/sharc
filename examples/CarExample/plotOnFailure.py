import re, json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

log = Path(sys.argv[1])
txt = log.read_text()

# Extract k, t
k_pat = re.compile(r'^k = (\d+), t = ([0-9.]+)', re.M)
kt = [(int(m.group(1)), float(m.group(2))) for m in k_pat.finditer(txt)]
K = [k for k,_ in kt]
T = [t for _,t in kt]

# Extract x lines like: Received x=[0, 3, 0, 0]
x_pat = re.compile(r'Received x=\[([^\]]+)\]')
X = []
for m in x_pat.finditer(txt):
    vals = [float(v.strip().replace(',', '')) for v in m.group(1).split()]
    X.append(vals)

# Extract u lines like: Send "u" to Python: [0, 0]
u_pat = re.compile(r'Send "u" to Python:\s*\[([^\]]+)\]')
U = []
for m in u_pat.finditer(txt):
    vals = [float(v.strip()) for v in m.group(1).split(',')]
    U.append(vals)

# Align lengths (truncate to min)
n = min(len(T), len(X), len(U))
T = np.array(T[:n]); X = np.array(X[:n]); U = np.array(U[:n])

print(f"Parsed {n} steps from log")

fig, axs = plt.subplots(4,1, figsize=(10,12), sharex=True)
axs[0].plot(T, X[:,0]); axs[0].set_ylabel('x0')
axs[1].plot(T, X[:,1]); axs[1].set_ylabel('x1')
axs[2].plot(T, X[:,2] if X.shape[1]>2 else np.nan); axs[2].set_ylabel('x2')
axs[3].plot(T, U[:,0], label='u0')
if U.shape[1]>1: axs[3].plot(T, U[:,1], '--', label='u1')
axs[3].set_ylabel('u'); axs[3].set_xlabel('time [s]'); axs[3].legend()
plt.tight_layout()
plt.show()
