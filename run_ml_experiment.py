import numpy as np, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib import rcParams
rcParams.update({'font.size':9.5,'figure.dpi':150})
np.random.seed(0)
gl=lambda k,r:(1-r)*k**2+4*r
gu=lambda k,r:(1-r)*(9-k**2)+4*r

FW,FH=15.0,4.9
fig=plt.figure(figsize=(FW,FH))
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,FW); ax.set_ylim(0,FH); ax.axis('off')
f2y=lambda y: y/FH          # data -> figure fraction
f2x=lambda x: x/FW

# vertical grid
BOT,TOP = 0.42, 4.35
TITLE_Y  = 4.20             # va=top
CONT_LO, CONT_HI = 1.80, 3.40   # inset / content zone
CAP_Y    = 1.30             # va=top

COL=['#e8f1f8','#eaf5ec','#fdf1e3','#f6ecf5']
EDG=['#2166ac','#1b7837','#b35806','#8c5aa8']
TIT=['Step 1\nNoisy observations','Step 2\nSurrogate training',
     'Step 3\n$q$-Prabhakar operator','Step 4\nInclusion diagnostic']
X0=[0.35,4.05,7.75,11.45]; W=3.15
BOX=dict(boxstyle='round,pad=0.30,rounding_size=0.12',lw=1.3)

for i in range(4):
    ax.add_patch(FancyBboxPatch((X0[i],BOT),W,TOP-BOT,facecolor=COL[i],edgecolor=EDG[i],**BOX))
    ax.text(X0[i]+W/2,TITLE_Y,TIT[i],ha='center',va='top',fontsize=10,
            fontweight='bold',color=EDG[i],linespacing=1.35)
for i in range(3):
    ax.add_patch(FancyArrowPatch((X0[i]+W+0.06,(BOT+TOP)/2),(X0[i+1]-0.06,(BOT+TOP)/2),
                 arrowstyle='-|>',mutation_scale=17,lw=1.6,color='0.4'))

IW=(W-0.62)          # inset width in data units
def inset(i):
    return fig.add_axes([f2x(X0[i]+0.31), f2y(CONT_LO), f2x(IW), f2y(CONT_HI-CONT_LO)])

# ---------- Step 1 ----------
a=inset(0); kk=np.linspace(0,1,80)
for r,c in [(0.2,'#2166ac'),(0.7,'#8fb9d8')]:
    a.plot(kk,gl(kk,r),color=c,lw=1.5); a.plot(kk,gu(kk,r),color=c,lw=1.5)
    ks=np.random.uniform(0,1,42)
    a.plot(ks,gl(ks,r)+np.random.normal(0,.10,42),'o',ms=2.0,color=c,alpha=.55)
    a.plot(ks,gu(ks,r)+np.random.normal(0,.10,42),'o',ms=2.0,color=c,alpha=.55)
a.set_xlabel(r'$\kappa$',fontsize=8,labelpad=1); a.set_ylabel('endpoints',fontsize=8,labelpad=1)
a.tick_params(labelsize=6.5); a.grid(alpha=.2,ls=':')
ax.text(X0[0]+W/2,CAP_Y,r'$N=3000$ samples,  $\sigma=0.10$'+'\n800 held out for testing',
        ha='center',va='top',fontsize=8.6,linespacing=1.4)

# ---------- Step 2 : network ----------
cx=X0[1]+W/2
ys_io=[2.95,2.20]; ys_h=[3.20,2.70,2.20,1.70]
layers=[(cx-1.15,ys_io,[r'$\kappa$',r'$\varrho$']),
        (cx-0.38,ys_h,None),(cx+0.38,ys_h,None),
        (cx+1.15,ys_io,[r'$\hat f^{\,\mathrm{lo}}$',r'$\hat f^{\,\mathrm{up}}$'])]
pos=[[(x,y) for y in ys] for x,ys,_ in layers]
for A,B in zip(pos[:-1],pos[1:]):
    for (x1,y1) in A:
        for (x2,y2) in B: ax.plot([x1,x2],[y1,y2],color='0.75',lw=.5,zorder=1)
for li,(x,ys,labs) in enumerate(layers):
    fc='#ffffff' if li in (0,3) else '#1b7837'
    for j,y in enumerate(ys):
        ax.add_patch(Circle((x,y),0.115,facecolor=fc,edgecolor='#1b7837',lw=1.1,zorder=3))
        if labs: ax.text(x,y,labs[j],ha='center',va='center',fontsize=8,zorder=4)
ax.text(cx-1.15,3.46,'input',ha='center',fontsize=7.6,color='0.35')
ax.text(cx,3.46,'hidden (80, 80)',ha='center',fontsize=7.6,color='0.35')
ax.text(cx+1.15,3.46,'output',ha='center',fontsize=7.6,color='0.35')
ax.text(cx,CAP_Y,'flexible network   vs.\nstructure-aware model (degree 2, ridge)',
        ha='center',va='top',fontsize=8.6,linespacing=1.4)

# ---------- Step 3 ----------
cx=X0[2]+W/2
ax.text(cx,3.30,r'apply $_{\xi_1}E^{\eta,\mu}_{\gamma,\lambda;q}$ to the'+'\nlearned endpoints',
        ha='center',va='top',fontsize=8.8,linespacing=1.4)
ax.text(cx,2.42,r'$\hat{L}\ \supseteq\ \hat{M}\ \supseteq\ \hat{R}$',
        ha='center',va='center',fontsize=13,color='#b35806')
bars=[(0.00,2.60,'#1b7837',7,.55),(0.22,2.38,'#e08214',4,.9),(0.44,2.16,'#2166ac',1.8,1)]
for lo,hi,c,lw,al in bars:
    ax.plot([cx-1.30+lo*0.46,cx-1.30+hi*0.46],[1.80,1.80],color=c,lw=lw,alpha=al,
            solid_capstyle='butt')
ax.text(cx,CAP_Y,'three terms of Theorem 3.2\n'+r'evaluated at each level $\varrho$',
        ha='center',va='top',fontsize=8.6,linespacing=1.4)

# ---------- Step 4 ----------
d=inset(3); rs=np.linspace(0,1,11)
mlp=np.array([0.0601,0.0612,0.0653,0.0694,0.0339,-0.0032,-0.0131,-0.0122,-0.0086,-0.0595,-0.1528])
stw=np.full(11,0.0795)
d.plot(rs,mlp,color='#c0392b',lw=1.5,marker='x',ms=3.4)
d.plot(rs,stw,color='#1b7837',lw=1.5,marker='o',ms=3.0)
d.fill_between(rs,np.minimum(mlp,0),0,color='#c0392b',alpha=.15)
d.axhline(0,color='0.3',lw=1)
d.set_xlabel(r'$\varrho$',fontsize=8,labelpad=1); d.set_ylabel('margin',fontsize=8,labelpad=1)
d.tick_params(labelsize=6.5); d.grid(alpha=.2,ls=':')
ax.text(X0[3]+W/2,CAP_Y,'smallest of the four margins\nnegative  '+r'$\Rightarrow$'+'  theorem violated',
        ha='center',va='top',fontsize=8.6,linespacing=1.4)

plt.savefig('fig_ml_pipeline.png',bbox_inches='tight',facecolor='white')
plt.close(); print("saved")
