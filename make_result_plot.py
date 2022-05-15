import matplotlib.pyplot as plt
import numpy as np

results = '''Inv & 0.87 & Geo & 0.82 & Inv & 0.67 & Rep & 0.66 & Inv & 0.65 & Inv & 0.61 & Rep & 0.58 & Rep & 0.53 & Geo & 0.53 & Geo & 0.51 \\
Inv & -33.13 & Inv & -37.20 & Inv & -40.41 & Inv & -40.98 & Inv & -41.18 & Inv & -41.91 & Inv & -42.31 & Inv & -42.66 & Rep & -42.85 & Inv & -43.22 \\
Bin & 40.13 & Bin & 29.33 & Bin & 29.09 & Inv & 12.23 & Inv & 8.46 & Rep & 5.75 & Inv & 4.21 & Rep & 3.95 & Rep & 3.14 & Rep & 2.40 \\
Inv & -18.5 & Inv & -19.1 & Inv & -19.6 & Inv & -20.3 & Geo & -20.9 & Rep & -20.9 & Rep & -21.0 & Rep & -21.0 & Rep & -21.0 & Rep & -21.0 \\
Geo & 6.3 & Idt & 5.8 & Idt & 5.7 & Idt & 5.5 & Idt & 5.4 & Idt & 5.3 & Rep & 4.9 & Idt & 4.6 & Inv & 4.1 & Rep & 3.6 \\
'''

lines = results.split('\\\n')[:-1]
tokens = [line.split(" & ") for line in lines]
print(tokens)
types = [[v for i,v in enumerate(linetoks) if i % 2 == 0] for linetoks in tokens]
vals = [[v for i,v in enumerate(linetoks) if i % 2 == 1] for linetoks in tokens]

colormap = {
    "Inv": "red",
    "Bin": "green",
    "Geo": "blue",
    "Rep": "yellow",
    "Idt": "black",
}
shapes = {
    "Inv": "o",
    "Bin": "^",
    "Geo": "s",
    "Rep": "v",
    "Idt": "D",
}
label_name = {
    "Inv": "Inversion",
    "Bin": "Binary",
    "Geo": "Geometric",
    "Rep": "Inversion with Replacement",
    "Idt": "Identity",
}
env_names = [
    "Cooperative Pong",
    "KAZ",
    "Prospector",
    "Pong",
    "Entombed Cooperative",
]
label_handles = {}
# plt.subplot(211)
fig, axes = plt.subplots(5, 1)

for i, (typevs, valv) in enumerate(zip(types, vals)):
    ax = axes[i]
    valv = np.array(valv,dtype="float64")
    minv = valv.min()
    maxv = valv.max()
    avgd_v = (valv - minv) / (maxv - minv) 
    for typev, v in zip(typevs, valv):
        handle, = ax.plot([v], [env_names[i]], '|', color=colormap[typev], markersize=15, label=label_name[typev])
        label_handles[typev] = handle
    ax.invert_xaxis()
plt.figlegend(handles=list(label_handles.values()), bbox_to_anchor=(1,1), loc="upper left")
plt.xlabel("Avg. reward")
# plt.figlegend([line, rand_line],['Trained Agent vs Random Agent', 'Random Agent vs Random Agent'], fontsize='x-large', loc='lower center', ncol=1, labelspacing=.2, columnspacing=.25, borderpad=.25, bbox_to_anchor=(0.68,0.06))

fig.tight_layout()

# plt.show()
fig.savefig("arg.png",bbox_inches='tight')
print(types)
print(vals)
