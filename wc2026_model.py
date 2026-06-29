"""WC2026 model v2: Dixon-Coles tau correction + major-tournament intercept."""
import csv, os, urllib.request, datetime as dt
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson
DATA_URL="https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
CACHE="results.csv"; LOCAL="data/results_2026.csv"
HALF_LIFE=540.0; FRIENDLY_W=0.6; RIDGE=1.0; MAXG=10
MAJORS={'FIFA World Cup','UEFA Euro','Copa América','African Cup of Nations',
 'AFC Asian Cup','Gold Cup','FIFA Confederations Cup','OFC Nations Cup'}
def load_matches(download=False):
    if download or not os.path.exists(CACHE): urllib.request.urlretrieve(DATA_URL,CACHE)
    rows=[]; today=dt.date.today()
    cutoff=today-dt.timedelta(days=int(5*365.25)); xi=np.log(2)/HALF_LIFE
    for src in [CACHE]+([LOCAL] if os.path.exists(LOCAL) else []):
        for r in csv.DictReader(open(src)):
            if r['home_score'] in ('NA',''): continue
            d=dt.date.fromisoformat(r['date'])
            if d<cutoff or d>today: continue
            w=np.exp(-xi*(today-d).days)
            if 'Friendly' in r['tournament']: w*=FRIENDLY_W
            rows.append((r['home_team'],r['away_team'],int(r['home_score']),
                int(r['away_score']),r['neutral']=='TRUE',w,r['tournament'] in MAJORS))
    return rows
def fit(rows):
    teams=sorted({t for r in rows for t in (r[0],r[1])}); tix={t:i for i,t in enumerate(teams)}; N=len(teams)
    H=np.array([tix[r[0]] for r in rows]); A=np.array([tix[r[1]] for r in rows])
    GH=np.array([r[2] for r in rows],float); GA=np.array([r[3] for r in rows],float)
    HOME=np.array([not r[4] for r in rows],float); W=np.array([r[5] for r in rows])
    MAJ=np.array([r[6] for r in rows],float); S=W.sum()
    def ng(p):
        atk,dfn,mu,hadv,gam=p[:N],p[N:2*N],p[2*N],p[2*N+1],p[2*N+2]
        lh=np.exp(mu+atk[H]-dfn[A]+hadv*HOME+gam*MAJ); la=np.exp(mu+atk[A]-dfn[H]+gam*MAJ)
        ll=(W*(GH*np.log(lh)-lh+GA*np.log(la)-la)).sum()
        f=-ll/S*1000+RIDGE*(atk@atk+dfn@dfn)/N+100*(atk.mean()**2+dfn.mean()**2)
        rh=W*(GH-lh); ra=W*(GA-la); ga=np.zeros(N); gd=np.zeros(N)
        np.add.at(ga,H,rh); np.add.at(ga,A,ra); np.add.at(gd,A,-rh); np.add.at(gd,H,-ra)
        g=np.zeros(2*N+3)
        g[:N]=-ga/S*1000+2*RIDGE*atk/N+200*atk.mean()/N
        g[N:2*N]=-gd/S*1000+2*RIDGE*dfn/N+200*dfn.mean()/N
        g[2*N]=-(rh.sum()+ra.sum())/S*1000; g[2*N+1]=-(rh*HOME).sum()/S*1000
        g[2*N+2]=-((rh+ra)*MAJ).sum()/S*1000
        return f,g
    p0=np.zeros(2*N+3); p0[2*N]=np.log(1.3); p0[2*N+1]=0.25
    res=minimize(ng,p0,jac=True,method='L-BFGS-B',options={'maxiter':50000,'ftol':1e-12,'gtol':1e-8})
    p=res.x; atk,dfn,mu,hadv,gam=p[:N],p[N:2*N],p[2*N],p[2*N+1],p[2*N+2]
    lh=np.exp(mu+atk[H]-dfn[A]+hadv*HOME+gam*MAJ); la=np.exp(mu+atk[A]-dfn[H]+gam*MAJ)
    x=GH.astype(int); y=GA.astype(int)
    def tll(rho):
        t=np.ones(len(rows)); m00=(x==0)&(y==0);m01=(x==0)&(y==1);m10=(x==1)&(y==0);m11=(x==1)&(y==1)
        t[m00]=1-lh[m00]*la[m00]*rho;t[m01]=1+lh[m01]*rho;t[m10]=1+la[m10]*rho;t[m11]=1-rho
        return -(W*np.log(np.clip(t,1e-6,None))).sum()
    rho=minimize_scalar(tll,bounds=(-0.2,0.2),method='bounded').x
    return dict(atk=atk,dfn=dfn,mu=mu,hadv=hadv,gam=gam,rho=rho,tix=tix,conv=res.success)
def lambdas(P,t1,t2,major=True):
    HOSTS={'Mexico','United States','Canada'}; i,j=P['tix'][t1],P['tix'][t2]
    lh=np.exp(P['mu']+P['atk'][i]-P['dfn'][j]+(P['hadv'] if t1 in HOSTS else 0)+P['gam']*major)
    la=np.exp(P['mu']+P['atk'][j]-P['dfn'][i]+(P['hadv'] if t2 in HOSTS else 0)+P['gam']*major)
    return lh,la
def predict_match(P,h,a,major=True,neutral=True):
    lh,la=lambdas(P,h,a,major)
    M=np.outer(poisson.pmf(range(MAXG),lh),poisson.pmf(range(MAXG),la)); rho=P['rho']
    M[0,0]*=1-lh*la*rho; M[0,1]*=1+lh*rho; M[1,0]*=1+la*rho; M[1,1]*=1-rho
    M=np.clip(M,0,None); M/=M.sum()
    pH=np.tril(M,-1).sum(); pD=np.trace(M); pA=np.triu(M,1).sum()
    top=sorted(((i,j,M[i,j]) for i in range(MAXG) for j in range(MAXG)),key=lambda z:-z[2])[:3]
    return lh,la,pH,pD,pA,M,top
