# -*- coding: utf-8 -*-
"""
Python prototype implementation of TSI, the vectorized MC algorithm
described in 'Guaranteed inference for probabilistic programs:
a parallelisable, small-step operational approach'([1]), Section 6.

In this file we consider a simplified translation T that can be applied 
for statements S of P_0 that rule out either 'while' or 'if-then-else'.

TSI returns guaranteed estimates for the expectation of random 
variables defined by probabilistic programs, guarantees are 
given in the form of confidence intervals.

The implementation is based on TensorFlow and autograph:
https://github.com/tensorflow/tensorflow/blob/master/
tensorflow/python/autograph/g3doc/reference/index.md.



The main functions are:
    
   1. def translate_seq(S,x,ind=''):
          Given a probabilistic program written in language P_0 described in [1]
          it returns its tanslation into TensorFlow.
             - S: model describing the probabilistic program we are
                 studying, written in language P_0.
             - x: list of variables involved in the program.
             
   2. def compute_statistics(res,xl, e, eps, maxe=1):
          It computes a posteriori statistics for the random variables defined 
          by the considered probabilistic program.
             - res: output samples of the considered probabilistic program.
             - xl: list of variables involved in the program.
             - e: random variable for which we are interested in obtaining 
                  guarantees.
             - eps: width of confidence interval for expectation.
                
        
The main workflow is the following:
    
    #1. define the model
    var('r y i')
    S=seq(draw(r,rhoU()),whl(abs(y)<1,seq(draw(y,rhoG(y,2*r)),setx(i,i+1)),i>=3))
    xlist=['r','y','i']
    
    #2. traslate the model in TF
    tr_S=translate_sc(S_s,xlist) 
    
    #3. create manually a working TF definition, starting from tr_S and using tfd.(...).sample() for vectorial sampling: 
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32)])
    def f(r,y,i,m):
        print("Tracing")
        r = tfd.Uniform(low=r).sample()#draw(r,rho_U())
        def body_f1(r,y,i,m):
            y = tfd.Normal(loc=y, scale=2*r).sample()#draw(y,rho_G(y, 2*r))
            i+=1
            return r,y,i,m
        def body1(r,y,i,m):
            res = tf.where(tf.less(tf.abs(y) ,1.0) & tf.greater(m,0),tf.concat(list(body_f1(r,y,i,m)),axis=0),tf.concat((r,y,i,m),axis=0))
            return  tuple([res[tf.newaxis,j] for j in range(4)]) # slicing tensor res 
        r,y,i,m=tf.while_loop(lambda *_: True, body1, (r,y,i,m), maximum_iterations=100)
        m=tf.where(tf.logical_or(tf.logical_not(tf.less(tf.abs(y) ,1.0)) , tf.equal(m,0.0)),  m * tf.cast(tf.greater_equal(i, 3.0), tf.float32), np.NaN)
        return r,y,i,m
    
    #4. define inputs for the function, and execute the model
    # Tracing
    N=1 
    rr = tf.zeros((1,N))
    yy = tf.zeros((1,N))
    ii = tf.zeros((1,N))
    m = tf.constant(1.0,shape=(1,N))
    res=f(rr,yy,ii,m)  
    # actual execution
    N=10**6
    rr = tf.zeros((1,N))
    yy = tf.zeros((1,N))
    ii = tf.zeros((1,N))
    m = tf.constant(1.0,shape=(1,N))
    res=f(rr,yy,ii,m)
    
    #5. compute posteriori statistics
    [r,y,i]   
    e=r
    eps=0.005
    maxe=1
    exp, lower_prob,conf=compute_statistics(res,xl, e, eps, maxe)
  

Examples and experiments are at the end of the script. 
"""

from  sympy import *
import numpy as np
import time
# Added these two lines to disable oneDNN opts, otherwise it will give me a warning
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

nil=Function('nil')
setx=Function('setx')
draw=Function('draw')
obs=Function('obs')
ite=Function('ite')
whl=Function('whl')

rho=Function('rho')    
rhoU=Function('rhoU')  
rhoG=Function('rhoG')  
B=Function('B')        
G=Function('G')        
g=Function('g') 
K=10
K_str=str(K)
skip=Function('skip')
seq=Function('seq')
c=0



def translate_seq(S,x,ind=''):
    '''
    Given a probabilistic program written in language P_0 described in [1],
    it returns its tanslation into TensorFlow.
    
       - S: model describing the probabilistic program we are
           studying, written in language P_0.
       - x: list of variables involved in the program.
       
    '''
    global c
    xargs=','.join(x)
    f=S.func
    args=S.args
    if f == skip:
        return ""
    elif f == setx:
        xi,g=args
        return f"{ind}{str(xi)}={str(g)}"
    elif f == draw:
        xi,rho=args
        return f"{ind}{str(xi)}=draw({str(rho)})"
    elif f== obs:    
        phi = args
        return f"{ind}m = m * tf.cast({str(phi)},tf.float32)"
    elif f==ite:
        phi, S1, S2 = args
        c=c+1
        c1=str(c)
        c=c+1
        c2=str(c)
        f1 = translate_seq(S1,x,ind+'    ')
        f2 = translate_seq(S2,x,ind+'    ')
        phi_str=str(phi)
        c=c+1
        return ind+f"def f{c1}({xargs},m):\n{f1}\n{ind}    return {xargs},m\n{ind}def f{c2}({xargs},m):\n{f2}\n{ind}    return {xargs},m\n{ind}mask = {phi_str}\n{ind}res=tf.where(mask, tf.concat(f{c1}({xargs},m),axis=0), tf.concat(f{c2}({xargs},m),axis=0))\n{ind}{xargs},m = tuple(res[tf.newaxis,j] for j in range({str(len(x)+1)})) # slicing tensor res"
    elif f==whl:
        phi, S1, psi = args
        phi_str=str(phi)
        psi_str=str(psi)
        c=c+1
        c1=str(c)
        S_tr = translate_seq(S1,x,ind+'    ')
        S_f=       f"def body_f{c1}({xargs},m):\n{S_tr}\n{ind}    return {xargs},m"
        def_body = f"def body{c1}({xargs},m):\n{ind}    res = tf.where(({phi_str}) & tf.greater(m,0.0),tf.concat(body_f{c1}({xargs},m),axis=0),tf.concat(({xargs},m),axis=0))\n{ind}    return tuple([res[tf.newaxis,j] for j in range({str(len(x)+1)})]) # slicing tensor res "
        post=      f"m=tf.where(tf.logical_or(tf.logical_not({phi_str}) , tf.equal(m,0.0)),  m * tf.cast({psi_str},tf.float32), np.NaN)"
        return     f"{ind}{S_f}\n{ind}{def_body}\n{ind}{xargs},m=tf.while_loop(lambda *_: True, body{c1}, ({xargs},m), maximum_iterations={K_str})\n{ind}{post}"
    elif f==seq:
        Slist=[translate_seq(Si,x,ind) for Si in args]           
        return "\n".join(Slist)            
    else:
        print("Syntax error")
        return None


def translate_sc(S,x):
    xargs=','.join(x)
    tr_S = translate_seq(S,x,ind='    ')
    arg_str=','.join(["tf.TensorSpec(shape=None, dtype=tf.float32)"]*(len(x)+1))
    header = f"@tf.function(input_signature=[{arg_str}])"
    return f"{header}\ndef f0({xargs},m):\n{tr_S}\n    return {xargs},m"

def compute_statistics_BA(res,xl, e, eps, maxe):
    m=res[4][0]
    N=m.shape[0]

    r=res[1][0]    
    term = (m==1.0)    
    fail = (m==0.0)      
    live = tf.logical_not(term|fail)
    lt2=    term|live
    na=live.numpy().sum()+term.numpy().sum()
    
    LB=r[lt2].numpy().sum()/na-eps   
    UB=r[lt2].numpy().sum()/na+eps     

    lower_prob=term.numpy().sum()/(term.numpy().sum()+live.numpy().sum())   
    delta = 2*np.exp(-2*N*eps**2/maxe**2)+np.exp(-2*na*eps**2/maxe**2)

    print('delta BA: %s' % delta)

    conf = 1-2*delta
    exp=[LB,UB]
    return exp, lower_prob, conf

def compute_statistics_MH(res,xl, e, eps, maxe):
    m=res[4][0]
    
    N=m.shape[0]
    r=res[3][0]    
    term = (m==1.0)    
    fail = (m==0.0)      
    live = tf.logical_not(term|fail)   
    lt2=    term|live
    na=live.numpy().sum()+term.numpy().sum()
    
    LB=r[lt2].numpy().sum()/na-eps     
    UB=r[lt2].numpy().sum()/na+eps     

    lower_prob=term.numpy().sum()/(term.numpy().sum()+live.numpy().sum())   
    
    delta = 2*np.exp(-2*N*eps**2/maxe**2)+np.exp(-2*na*eps**2/maxe**2)
    conf = 1-2*delta

    print('delta MH: %s' % delta)

    exp=[LB,UB]
    return exp, lower_prob, conf

# My compute statistics functions

def compute_statistics_brp(res, eps, maxe):
    m = res[3][0]

    N=m.shape[0]
    failed=res[1][0]
    failed_indicator = (failed==10.0)
    term = (m==1.0)
    fail = (m==0.0)
    live = tf.logical_not(term|fail)
    lt2=    term|live
    na=live.numpy().sum()+term.numpy().sum()
    LB=failed_indicator[lt2].numpy().sum()/na-eps     
    UB=failed_indicator[lt2].numpy().sum()/na+eps     

    lower_prob=term.numpy().sum()/(term.numpy().sum()+live.numpy().sum())   
    
    delta = 2*np.exp(-2*N*eps**2/maxe**2)+np.exp(-2*na*eps**2/maxe**2)
    conf = 1-2*delta

    print('delta brp: %s' % delta)

    exp=[LB,UB]
    return exp, lower_prob, conf

def compute_statistics_brp_finite_family(res, eps, maxe):
    m = res[4][0]

    N=m.shape[0]
    failed=res[2][0]
    failed_indicator = (failed==5.0)
    term = (m==1.0)
    fail = (m==0.0)
    live = tf.logical_not(term|fail)
    lt2=    term|live
    na=live.numpy().sum()+term.numpy().sum()

    LB=failed_indicator[lt2].numpy().sum()/na-eps     
    UB=failed_indicator[lt2].numpy().sum()/na+eps     

    lower_prob=term.numpy().sum()/(term.numpy().sum()+live.numpy().sum())   
    
    delta = 2*np.exp(-2*N*eps**2/maxe**2)+np.exp(-2*na*eps**2/maxe**2)
    conf = 1-2*delta

    print('delta brp fin: %s' % delta)

    exp=[LB,UB]
    return exp, lower_prob, conf

def compute_statistics_my_chain_5(res, eps, maxe):
    m = res[2][0]
    
    N=m.shape[0]
    coin=res[1][0]
    term = (m==1.0)
    fail = (m==0.0)      
    live = tf.logical_not(term|fail)
    lt2=    term|live
    na=live.numpy().sum()+term.numpy().sum()

    LB=coin[lt2].numpy().sum()/na-eps     
    UB=coin[lt2].numpy().sum()/na+eps     

    lower_prob=term.numpy().sum()/(term.numpy().sum()+live.numpy().sum())   
    
    delta = 2*np.exp(-2*N*eps**2/maxe**2)+np.exp(-2*na*eps**2/maxe**2)
    conf = 1-2*delta

    print('delta my chain: %s' % delta)

    exp=[LB,UB]
    return exp, lower_prob, conf

def compute_statistics_my_grid_small(res, eps, maxe):
    m = res[3][0]
    
    N=m.shape[0]
    a=res[0][0]
    b=res[1][0]
    a_is_zero_b_is_ten = (a==0) & (b==10)
    term = (m==1.0)
    fail = (m==0.0)      
    live = tf.logical_not(term|fail)
    lt2=    term|live
    na=live.numpy().sum()+term.numpy().sum()

    LB=a_is_zero_b_is_ten[lt2].numpy().sum()/na-eps     
    UB=a_is_zero_b_is_ten[lt2].numpy().sum()/na+eps     

    lower_prob=term.numpy().sum()/(term.numpy().sum()+live.numpy().sum())   
    
    delta = 2*np.exp(-2*N*eps**2/maxe**2)+np.exp(-2*na*eps**2/maxe**2)
    conf = 1-2*delta

    print('delta grid: %s' % delta)

    exp=[LB,UB]
    return exp, lower_prob, conf

def compute_statistics_normal(res, eps, maxe):
    m = res[1][0]
    
    N=m.shape[0]
    y=res[0][0]
    pos = y<=1.5
    
    term = (m==1.0)
    fail = (m==0.0)      
    live = tf.logical_not(term|fail)
    lt2=    term|live
    na=live.numpy().sum()+term.numpy().sum()

    LB=pos[lt2].numpy().sum()/na-eps     
    UB=pos[lt2].numpy().sum()/na+eps     

    lower_prob=term.numpy().sum()/(term.numpy().sum()+live.numpy().sum())   
    
    delta = 2*np.exp(-2*N*eps**2/maxe**2)+np.exp(-2*na*eps**2/maxe**2)
    conf = 1-2*delta

    print('delta normal: %s' % delta)

    exp=[LB,UB]
    return exp, lower_prob, conf

def compute_statistics_gambler(res, eps, maxe):
    m = res[4][0]
    
    N=m.shape[0]
    x=res[0][0]
    won = (x==4.0)
    
    term = (m==1.0)
    fail = (m==0.0)      
    live = tf.logical_not(term|fail)
    lt2=    term|live
    na=live.numpy().sum()+term.numpy().sum()

    LB=won[lt2].numpy().sum()/na-eps     
    UB=won[lt2].numpy().sum()/na+eps     

    lower_prob=term.numpy().sum()/(term.numpy().sum()+live.numpy().sum())   
    
    delta = 2*np.exp(-2*N*eps**2/maxe**2)+np.exp(-2*na*eps**2/maxe**2)
    conf = 1-2*delta

    print('delta gambler: %s' % delta)

    exp=[LB,UB]
    return exp, lower_prob, conf

def compute_statistics_geo0(res, eps, maxe):
    m = res[3][0]
    
    N=m.shape[0]
    z=res[0][0]
    
    term = (m==1.0)
    fail = (m==0.0)      
    live = tf.logical_not(term|fail)
    lt2=    term|live
    na=live.numpy().sum()+term.numpy().sum()

    LB=z[lt2].numpy().sum()/na-eps     
    UB=z[lt2].numpy().sum()/na+eps     

    lower_prob=term.numpy().sum()/(term.numpy().sum()+live.numpy().sum())   
    
    delta = 2*np.exp(-2*N*eps**2/maxe**2)+np.exp(-2*na*eps**2/maxe**2)
    conf = 1-2*delta

    print('delta geo0: %s' % delta)

    exp=[LB,UB]
    return exp, lower_prob, conf

def compute_statistics_geo0_1(res, eps, maxe, z_initial):
    m = res[3][0]
    
    N=m.shape[0]
    z=res[0][0]
    z_initial = tf.reshape(z_initial,shape=tf.shape(z))
    z-=z_initial
    
    term = (m==1.0)
    fail = (m==0.0)
    live = tf.logical_not(term|fail)
    lt2=    term|live
    na=live.numpy().sum()+term.numpy().sum()

    LB=z[lt2].numpy().sum()/na-eps     
    UB=z[lt2].numpy().sum()/na+eps     

    lower_prob=term.numpy().sum()/(term.numpy().sum()+live.numpy().sum())   
    
    delta = 2*np.exp(-2*N*eps**2/maxe**2)+np.exp(-2*na*eps**2/maxe**2)
    conf = 1-2*delta
    print('delta geo01: %s' % delta)

    exp=[LB,UB]
    return exp, lower_prob, conf

def compute_statistics_die_cond(res, eps, maxe):
    m = res[3][0]
    
    N=m.shape[0]
    d1=res[0][0]
    
    term = (m==1.0)
    fail = (m==0.0)      
    live = tf.logical_not(term|fail)
    lt2=    term|live
    na=live.numpy().sum()+term.numpy().sum()

    LB=d1[lt2].numpy().sum()/na-eps     
    UB=d1[lt2].numpy().sum()/na+eps     

    lower_prob=term.numpy().sum()/(term.numpy().sum()+live.numpy().sum())   
    
    delta = 2*np.exp(-2*N*eps**2/maxe**2)+np.exp(-2*na*eps**2/maxe**2)
    conf = 1-2*delta
    print('delta die cond: %s' % delta)

    exp=[LB,UB]
    return exp, lower_prob, conf

def compute_statistics_PrinSys(res, eps, maxe):
    m = res[2][0]
    
    N=m.shape[0]
    x=res[0][0]
    x_is_two = (x==2.0)
    
    term = (m==1.0)
    fail = (m==0.0)      
    live = tf.logical_not(term|fail)
    lt2=    term|live
    na=live.numpy().sum()+term.numpy().sum()

    LB=x_is_two[lt2].numpy().sum()/na-eps     
    UB=x_is_two[lt2].numpy().sum()/na+eps     

    lower_prob=term.numpy().sum()/(term.numpy().sum()+live.numpy().sum())   
    
    delta = 2*np.exp(-2*N*eps**2/maxe**2)+np.exp(-2*na*eps**2/maxe**2)
    conf = 1-2*delta
    print('delta prinsys: %s' % delta)

    exp=[LB,UB]
    return exp, lower_prob, conf

def compute_statistics_RevBin_1(res, eps, maxe, x_original, z_original):
    m = res[3][0]
    
    N=m.shape[0]
    z=res[1][0]
    
    term = (m==1.0)
    fail = (m==0.0)      
    live = tf.logical_not(term|fail)
    lt2=    term|live
    na=live.numpy().sum()+term.numpy().sum()

    LB=z[lt2].numpy().sum()/na-eps     
    UB=z[lt2].numpy().sum()/na+eps     

    lower_prob=term.numpy().sum()/(term.numpy().sum()+live.numpy().sum())   
    
    delta = 2*np.exp(-2*N*eps**2/maxe**2)+np.exp(-2*na*eps**2/maxe**2)
    conf = 1-2*delta
    print('delta revbin: %s' % delta)

    exp=[LB,UB]
    return exp, lower_prob, conf

def compute_statistics_sprdwalk(res, eps, maxe):
    m = res[3][0]
    
    N=m.shape[0]
    
    term = (m==1.0)
    fail = (m==0.0)
    
    term_or_fail_proportion = (term.numpy().sum() + fail.numpy().sum()) / N

    return term_or_fail_proportion

def compute_expected_value_approx(var, mask):
    return var[mask==1].numpy().sum()/(mask==1).numpy().sum() 


#-------------------------------------------------------------------------------------------
#-------------------------------------  MY EXPERIMENTS  ------------------------------------
#-------------------------------------------------------------------------------------------

#-------------------------------------  TACAS FOLDER  ------------------------------------

#------------------------------- My Example 1 : BRP 10, 0.5 ------------------------------

print('\n---------------------------- brp 10, 0.5 ----------------------------')

var('sent failed x')

S_s = whl((failed<10) & (sent<10),seq(draw(x,B(0.5)),ite(x==1,seq(setx(failed,0),setx(sent,sent+1)),setx(failed,failed+1))),true)
xlist=['sent','failed','x']
c=0
# tr_S=translate_sc(S_s,xlist)
# print(tr_S)

@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32)])
def f0(sent,failed,coin,m):
    B_probs = tf.zeros(shape=tf.shape(coin))
    B_probs += 0.5
    packets = 10
    def body_f1(sent,failed,coin,m):
        coin = tfd.Bernoulli(probs=B_probs).sample()
        coin = tf.cast(coin,dtype=tf.float32)
        def f2(sent,failed,coin,m):
            sent+=1.0
            failed-=failed
            return sent,failed,coin,m
        def f3(sent,failed,coin,m):
            failed+=1.0
            return sent,failed,coin,m
        # mask = tf.logical_and(tf.less(failed,10.0),tf.less(sent,packets)) # (failed<10) & (sent<packets)
        mask = tf.equal(coin,1.0)
        res=tf.where(mask, tf.concat(list(f2(sent,failed,coin,m)),axis=0), tf.concat(list(f3(sent,failed,coin,m)),axis=0)) # adjustment made originally no list
        sent,failed,coin,m = tuple(res[tf.newaxis,j] for j in range(4)) # slicing tensor res
        return sent,failed,coin,m
    def body1(sent,failed,coin,m):
        res = tf.where((tf.logical_and(tf.less(failed,10.0),tf.less(sent,packets))) & tf.greater(m,0.0),tf.concat(body_f1(sent,failed,coin,m),axis=0),tf.concat((sent,failed,coin,m),axis=0))  # adjustment made originally no list
        return tuple([res[tf.newaxis,j] for j in range(4)]) # slicing tensor res
    sent,failed,coin,m=tf.while_loop(lambda *_: True, body1, (sent,failed,coin,m), maximum_iterations=100)
    m=tf.where(tf.logical_or(tf.logical_not(tf.logical_and(tf.less(failed,10.0),tf.less(sent,packets))) , tf.equal(m,0.0)),  m * tf.cast(True,tf.float32), np.NaN)
    return sent,failed,coin,m

var('sent failed coin')

N=1 # Warm up 
sent = tf.zeros((1,N))
failed = tf.zeros((1,N))
coin = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(sent,failed,coin,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1  elem %s seconds -------        " % final_time)


N=10**6
sent = tf.zeros((1,N))
failed = tf.zeros((1,N))
coin = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(sent,failed,coin,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1M elems  %s seconds -------        " % final_time)

eps=0.005
maxe=1
exp, lower_prob,conf=compute_statistics_brp(res, eps, maxe)

print("exp %s" % exp)
# central_value = (exp[0] + exp[1]) / 2;
# print("central value %s" % central_value)

#------------------------------- Example 1 : BRP ------------------------------

print('\n---------------------------- brp ----------------------------')

var('sent failed x')

S_s = whl((failed<10) & (sent<8_000_000_000),seq(draw(x,B(0.99)),ite(x==1,seq(setx(failed,0),setx(sent,sent+1)),setx(failed,failed+1))),true)
xlist=['sent','failed','x']
c=0
# tr_S=translate_sc(S_s,xlist)
# print(tr_S)

@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32)])
def f0(sent,failed,coin,m):
    B_probs = tf.zeros(shape=tf.shape(coin))
    B_probs += 0.99
    packets = 8_000_000_000
    def body_f1(sent,failed,coin,m):
        coin = tfd.Bernoulli(probs=B_probs).sample()
        coin = tf.cast(coin,dtype=tf.float32)
        def f2(sent,failed,coin,m):
            sent+=1.0
            failed-=failed
            return sent,failed,coin,m
        def f3(sent,failed,coin,m):
            failed+=1.0
            return sent,failed,coin,m
        # mask = tf.logical_and(tf.less(failed,10.0),tf.less(sent,packets)) # (failed<10) & (sent<packets)
        mask = tf.equal(coin,1.0)
        res=tf.where(mask, tf.concat(list(f2(sent,failed,coin,m)),axis=0), tf.concat(list(f3(sent,failed,coin,m)),axis=0)) # adjustment made originally no list
        sent,failed,coin,m = tuple(res[tf.newaxis,j] for j in range(4)) # slicing tensor res
        return sent,failed,coin,m
    def body1(sent,failed,coin,m):
        res = tf.where((tf.logical_and(tf.less(failed,10.0),tf.less(sent,packets))) & tf.greater(m,0.0),tf.concat(body_f1(sent,failed,coin,m),axis=0),tf.concat((sent,failed,coin,m),axis=0))  # adjustment made originally no list
        return tuple([res[tf.newaxis,j] for j in range(4)]) # slicing tensor res
    sent,failed,coin,m=tf.while_loop(lambda *_: True, body1, (sent,failed,coin,m), maximum_iterations=10)
    m=tf.where(tf.logical_or(tf.logical_not(tf.logical_and(tf.less(failed,10.0),tf.less(sent,packets))) , tf.equal(m,0.0)),  m * tf.cast(True,tf.float32), np.NaN)
    return sent,failed,coin,m

var('sent failed coin')

N=1 # Warm up 
sent = tf.zeros((1,N))
failed = tf.zeros((1,N))
coin = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(sent,failed,coin,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1  elem %s seconds -------        " % final_time)


N=10**6
sent = tf.zeros((1,N))
failed = tf.zeros((1,N))
coin = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(sent,failed,coin,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1M elems  %s seconds -------        " % final_time)

eps=0.005
maxe=1
exp, lower_prob,conf=compute_statistics_brp(res, eps, maxe)

print("exp %s" % exp)
# central_value = (exp[0] + exp[1]) / 2;
# print("central value %s" % central_value)

# With 100 as max iterations

@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32)])
def f0(sent,failed,coin,m):
    B_probs = tf.zeros(shape=tf.shape(coin))
    B_probs += 0.99
    packets = 8_000_000_000
    def body_f1(sent,failed,coin,m):
        coin = tfd.Bernoulli(probs=B_probs).sample()
        coin = tf.cast(coin,dtype=tf.float32)
        def f2(sent,failed,coin,m):
            sent+=1.0
            failed-=failed
            return sent,failed,coin,m
        def f3(sent,failed,coin,m):
            failed+=1.0
            return sent,failed,coin,m
        # mask = tf.logical_and(tf.less(failed,10.0),tf.less(sent,packets)) # (failed<10) & (sent<packets)
        mask = tf.equal(coin,1.0)
        res=tf.where(mask, tf.concat(list(f2(sent,failed,coin,m)),axis=0), tf.concat(list(f3(sent,failed,coin,m)),axis=0)) # adjustment made originally no list
        sent,failed,coin,m = tuple(res[tf.newaxis,j] for j in range(4)) # slicing tensor res
        return sent,failed,coin,m
    def body1(sent,failed,coin,m):
        res = tf.where((tf.logical_and(tf.less(failed,10.0),tf.less(sent,packets))) & tf.greater(m,0.0),tf.concat(body_f1(sent,failed,coin,m),axis=0),tf.concat((sent,failed,coin,m),axis=0))  # adjustment made originally no list
        return tuple([res[tf.newaxis,j] for j in range(4)]) # slicing tensor res
    sent,failed,coin,m=tf.while_loop(lambda *_: True, body1, (sent,failed,coin,m), maximum_iterations=100)
    m=tf.where(tf.logical_or(tf.logical_not(tf.logical_and(tf.less(failed,10.0),tf.less(sent,packets))) , tf.equal(m,0.0)),  m * tf.cast(True,tf.float32), np.NaN)
    return sent,failed,coin,m

var('sent failed coin')

N=1 # Warm up 
sent = tf.zeros((1,N))
failed = tf.zeros((1,N))
coin = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(sent,failed,coin,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1  elem %s seconds -------        " % final_time)


N=10**6
sent = tf.zeros((1,N))
failed = tf.zeros((1,N))
coin = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(sent,failed,coin,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1M elems  %s seconds -------        " % final_time)

eps=0.005
maxe=1
exp, lower_prob,conf=compute_statistics_brp(res, eps, maxe)

print("exp %s" % exp)
# central_value = (exp[0] + exp[1]) / 2;
# print("central value %s" % central_value)

#---------------------------------- Example: brp_finite_family ----------------------------

print("---------------- brp_finite_family ----------------")

var('sent maxsent failed coin')

S_s = whl(((failed < 5) & (sent < maxsent) & (maxsent < 8000000)),seq(draw(coin,B(0.99)),ite(coin==1,seq(setx(failed,0),setx(sent,sent+1)),setx(failed,failed+1))),true)
xlist=['sent', 'maxsent', 'failed', 'coin']
# tr_S=translate_sc(S_s,xlist)
# print(tr_S)

@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32)])
def f0(sent,maxsent,failed,coin,m):
    B_probs = tf.zeros(shape=tf.shape(coin))
    B_probs += 0.99
    def body_f1(sent,maxsent,failed,coin,m):
        coin = tfd.Bernoulli(probs=B_probs).sample()
        coin = tf.cast(coin,dtype=tf.float32)
        def f2(sent,maxsent,failed,coin,m):
            failed-=failed
            sent+=1
            return sent,maxsent,failed,coin,m
        def f3(sent,maxsent,failed,coin,m):
            failed+=1
            return sent,maxsent,failed,coin,m
        mask = coin==1.0
        res=tf.where(mask, tf.concat(f2(sent,maxsent,failed,coin,m),axis=0), tf.concat(f3(sent,maxsent,failed,coin,m),axis=0))
        sent,maxsent,failed,coin,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
        return sent,maxsent,failed,coin,m
    def body1(sent,maxsent,failed,coin,m):
        res = tf.where(((failed < 5) & (maxsent < 8000000) & (sent < maxsent)) & tf.greater(m,0.0),tf.concat(body_f1(sent,maxsent,failed,coin,m),axis=0),tf.concat((sent,maxsent,failed,coin,m),axis=0))
        return tuple([res[tf.newaxis,j] for j in range(5)]) # slicing tensor res
    sent,maxsent,failed,coin,m=tf.while_loop(lambda *_: True, body1, (sent,maxsent,failed,coin,m), maximum_iterations=10)
    m=tf.where(tf.logical_or(tf.logical_not((failed < 5) & (maxsent < 8000000) & (sent < maxsent)) , tf.equal(m,0.0)),  m * tf.cast(True,tf.float32), np.NaN)
    return sent,maxsent,failed,coin,m

var('sent maxsent failed coin')


N=1 # Warm up 
sent = tf.zeros((1,N))
maxsent = tfd.Uniform(low=tf.zeros((1,N)), high=8000000.0).sample()
failed = tf.zeros((1,N))
coin = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(sent,maxsent,failed,coin,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1  elem %s seconds -------        " % final_time)

N=10**6
sent = tf.zeros((1,N))
maxsent = tfd.Uniform(low=tf.zeros((1,N)), high=8000000.0).sample()
failed = tf.zeros((1,N))
coin = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(sent,maxsent,failed,coin,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1M  elem %s seconds -------        " % final_time)

eps=0.005
maxe=1
exp, lower_prob,conf=compute_statistics_brp_finite_family(res, eps, maxe)

print("exp %s" % exp)
# central_value = (exp[0] + exp[1]) / 2;
# print("central value %s" % central_value)

#------------------------------- My Example 2.1 : My Chain 5, 0.2 ------------------------------

print('\n---------------------------- my chain 5, 0.2 ----------------------------')

var('x coin')

S_s = whl((coin==0) & (x < 5), seq(draw(coin,B(0.2)), ite((coin==0), setx(x,x+1), setx(coin,1))), true)
xlist=['x', 'coin']
c=0
# tr_S=translate_sc(S_s,xlist)
# print(tr_S)

@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32)])
def f0(x,coin,m):
    B_probs = tf.zeros(shape=tf.shape(coin))
    B_probs += 0.2
    def body_f1(x,coin,m):
        coin = tfd.Bernoulli(probs=B_probs).sample()
        coin = tf.cast(coin,dtype=tf.float32)
        def f2(x,coin,m):
            x+=1
            return x,coin,m
        def f3(x,coin,m):
            return x,coin,m
        mask = coin==0
        res=tf.where(mask, tf.concat(f2(x,coin,m),axis=0), tf.concat(f3(x,coin,m),axis=0))
        x,coin,m = tuple(res[tf.newaxis,j] for j in range(3)) # slicing tensor res
        return x,coin,m
    def body1(x,coin,m):
        res = tf.where((tf.logical_and(tf.equal(coin,0), tf.less(x,5))) & tf.greater(m,0.0),tf.concat(body_f1(x,coin,m),axis=0),tf.concat((x,coin,m),axis=0))
        return tuple([res[tf.newaxis,j] for j in range(3)]) # slicing tensor res
    x,coin,m=tf.while_loop(lambda *_: True, body1, (x,coin,m), maximum_iterations=10)
    m=tf.where(tf.logical_or(tf.logical_not(tf.logical_and(tf.equal(coin,0), tf.less(x,5))) , tf.equal(m,0.0)),  m * tf.cast(True,tf.float32), np.NaN)
    return x,coin,m

var('x coin')

N=1 # Warm up 
x = tf.zeros((1,N))
coin = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(x,coin,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1  elem %s seconds -------        " % final_time)

N=10**6
x = tf.zeros((1,N))
coin = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(x,coin,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1  elem %s seconds -------        " % final_time)

eps=0.005
maxe=1
exp, lower_prob,conf=compute_statistics_my_chain_5(res, eps, maxe)

print("exp %s" % exp)
# central_value = (exp[0] + exp[1]) / 2;
# print("central value %s" % central_value)


#------------------------------- Example 3 : Grid Small ------------------------------

print('\n---------------------------- grid small ----------------------------')

var('a b coin')

S_s = whl((a < 10) & (b < 10), seq(draw(coin,B(0.5)), ite((coin==0), setx(a,a+1), setx(b,b+1))), true)
xlist=['a', 'b', 'coin']
# tr_S=translate_sc(S_s,xlist)
# print(tr_S)

@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32)])
def f0(a,b,coin,m):
    B_probs = tf.zeros(shape=tf.shape(coin))
    B_probs += 0.5
    def body_f1(a,b,coin,m):
        coin = tfd.Bernoulli(probs=B_probs).sample()
        coin = tf.cast(coin,dtype=tf.float32)
        def f2(a,b,coin,m):
            a+=1
            return a,b,coin,m
        def f3(a,b,coin,m):
            b+=1
            return a,b,coin,m
        mask = tf.equal(coin,0.0)
        res=tf.where(mask, tf.concat(f2(a,b,coin,m),axis=0), tf.concat(f3(a,b,coin,m),axis=0))
        a,b,coin,m = tuple(res[tf.newaxis,j] for j in range(4)) # slicing tensor res
        return a,b,coin,m
    def body1(a,b,coin,m):
        res = tf.where(tf.logical_and(tf.less(a,10),tf.less(b,10)) & tf.greater(m,0.0),tf.concat(body_f1(a,b,coin,m),axis=0),tf.concat((a,b,coin,m),axis=0))
        return tuple([res[tf.newaxis,j] for j in range(4)]) # slicing tensor res
    a,b,coin,m=tf.while_loop(lambda *_: True, body1, (a,b,coin,m), maximum_iterations=20)
    m=tf.where(tf.logical_or(tf.logical_not(tf.logical_and(tf.less(a,10),tf.less(b,10))) , tf.equal(m,0.0)),  m * tf.cast(True,tf.float32), np.NaN)
    return a,b,coin,m

var('a b coin')

N=1 # Warm up 
a = tf.zeros((1,N))
b = tf.zeros((1,N))
coin = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(a,b,coin,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1  elem %s seconds -------        " % final_time)

N=10**6
a = tf.zeros((1,N))
b = tf.zeros((1,N))
coin = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(a,b,coin,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1M  elem %s seconds -------        " % final_time)

eps=0.005
maxe=1
exp, lower_prob,conf=compute_statistics_my_grid_small(res, eps, maxe)

print("exp %s" % exp)
# central_value = (exp[0] + exp[1]) / 2;
# print("central value %s" % central_value)

#-------------------------------------  TACAS23_EXIST FOLDER  ------------------------------------

#---------------------------------- Example: Gambler ----------------------------

print("---------------- Gambler ----------------")

var('x y z coin')

S_s = whl((0 < x) & (x < y), seq(draw(coin,B(0.5)),ite(coin==1,setx(x,x+1),setx(x,x-1)),setx(z,z+1)),true)
xlist=['x', 'y', 'z', 'coin']
# tr_S=translate_sc(S_s,xlist)
# print(tr_S)

@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32)])
def f0(x,y,z,coin,m):
    B_probs = tf.zeros(shape=tf.shape(coin))
    B_probs += 0.5
    def body_f1(x,y,z,coin,m):
        coin = tfd.Bernoulli(probs=B_probs).sample()
        coin = tf.cast(coin,dtype=tf.float32)
        def f2(x,y,z,coin,m):
            x+=1
            return x,y,z,coin,m
        def f3(x,y,z,coin,m):
            x-=1
            return x,y,z,coin,m
        mask = (coin==1.0)
        res=tf.where(mask, tf.concat(f2(x,y,z,coin,m),axis=0), tf.concat(f3(x,y,z,coin,m),axis=0))
        x,y,z,coin,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
        z+=1
        return x,y,z,coin,m
    def body1(x,y,z,coin,m):
        res = tf.where(((x > 0) & (x < y)) & tf.greater(m,0.0),tf.concat(body_f1(x,y,z,coin,m),axis=0),tf.concat((x,y,z,coin,m),axis=0))
        return tuple([res[tf.newaxis,j] for j in range(5)]) # slicing tensor res
    x,y,z,coin,m=tf.while_loop(lambda *_: True, body1, (x,y,z,coin,m), maximum_iterations=20)
    m=tf.where(tf.logical_or(tf.logical_not((x > 0) & (x < y)) , tf.equal(m,0.0)),  m * tf.cast(True,tf.float32), np.NaN)
    return x,y,z,coin,m

var('x y z coin')

N=1 # Warm up 
x = tf.constant(2.0,shape=(1,N))
y = tf.constant(4.0,shape=(1,N))
z = tf.zeros((1,N))
coin = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(x,y,z,coin,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1  elem %s seconds -------        " % final_time)

N=10**6
x = tf.constant(2.0,shape=(1,N))
y = tf.constant(4.0,shape=(1,N))
z = tf.zeros((1,N))
coin = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(x,y,z,coin,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1M  elem %s seconds -------        " % final_time)

eps=0.005
maxe=1
exp, lower_prob,conf=compute_statistics_gambler(res, eps, maxe)

print("exp %s" % exp)
# central_value = (exp[0] + exp[1]) / 2;
# print("central value %s" % central_value)

#---------------------------------- Example: geo0 ----------------------------

print("---------------- geo0 ----------------")

var('z flip coin')

S_s = whl((flip==0), seq(draw(coin,B(0.2)),ite(coin==1,setx(flip,1),setx(z,z+1))),true)
xlist=['z', 'flip', 'coin']
# tr_S=translate_sc(S_s,xlist)
# print(tr_S)

@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32)])
def f0(z,flip,coin,m):
    B_probs = tf.zeros(shape=tf.shape(coin))
    B_probs += 0.2
    def body_f1(z,flip,coin,m):
        coin = tfd.Bernoulli(probs=B_probs).sample()
        coin = tf.cast(coin,dtype=tf.float32)
        def f2(z,flip,coin,m):
            flip+=1
            return z,flip,coin,m
        def f3(z,flip,coin,m):
            z+=1
            return z,flip,coin,m
        mask = (coin==1.0)
        res=tf.where(mask, tf.concat(f2(z,flip,coin,m),axis=0), tf.concat(f3(z,flip,coin,m),axis=0))
        z,flip,coin,m = tuple(res[tf.newaxis,j] for j in range(4)) # slicing tensor res
        return z,flip,coin,m
    def body1(z,flip,coin,m):
        res = tf.where((tf.equal(flip,0.0)) & tf.greater(m,0.0),tf.concat(body_f1(z,flip,coin,m),axis=0),tf.concat((z,flip,coin,m),axis=0))
        return tuple([res[tf.newaxis,j] for j in range(4)]) # slicing tensor res
    z,flip,coin,m=tf.while_loop(lambda *_: True, body1, (z,flip,coin,m), maximum_iterations=20)
    m=tf.where(tf.logical_or(tf.logical_not(tf.equal(flip,0.0)) , tf.equal(m,0.0)),  m * tf.cast(True,tf.float32), np.NaN)
    return z,flip,coin,m

var('x y z coin')

N=1 # Warm up 
z = tf.constant(1.0,shape=(1,N))
flip = tf.zeros((1,N))
coin = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(z,flip,coin,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1  elem %s seconds -------        " % final_time)

N=10**6
z = tf.constant(1.0,shape=(1,N))
flip = tf.zeros((1,N))
coin = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(z,flip,coin,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1M  elem %s seconds -------        " % final_time)

eps=0.005
maxe=1
exp, lower_prob,conf=compute_statistics_geo0(res, eps, maxe)

print("exp %s" % exp)
# central_value = (exp[0] + exp[1]) / 2;
# print("central value %s" % central_value)

N=1 # Warm up 
z = tf.random.uniform(shape=(1,N),maxval=1_000_000)
flip = tf.zeros((1,N))
coin = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(z,flip,coin,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1  elem %s seconds -------        " % final_time)

N=10**6
z = tf.random.uniform(shape=(1,N),maxval=1_000_000)
flip = tf.zeros((1,N))
coin = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(z,flip,coin,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1M  elem %s seconds -------        " % final_time)

eps=0.005
maxe=1
exp, lower_prob,conf=compute_statistics_geo0_1(res, eps, maxe, z)

print("exp %s" % exp)
# central_value = (exp[0] + exp[1]) / 2;
# print("central value %s" % central_value)

#---------------------------------- Example: geo0_obs ----------------------------

print("---------------- geo0_obs ----------------")

var('z flip coin')

S_s = whl((flip==0), seq(draw(coin,B(0.2)),ite(coin==1,setx(flip,1),setx(z,z+1))),z>2)
xlist=['z', 'flip', 'coin']
# tr_S=translate_sc(S_s,xlist)
# print(tr_S)

@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32)])
def f0(z,flip,coin,m):
    B_probs = tf.zeros(shape=tf.shape(coin))
    B_probs += 0.2
    def body_f1(z,flip,coin,m):
        coin = tfd.Bernoulli(probs=B_probs).sample()
        coin = tf.cast(coin,dtype=tf.float32)
        def f2(z,flip,coin,m):
            flip+=1
            return z,flip,coin,m
        def f3(z,flip,coin,m):
            z+=1
            return z,flip,coin,m
        mask = (coin==1.0)
        res=tf.where(mask, tf.concat(f2(z,flip,coin,m),axis=0), tf.concat(f3(z,flip,coin,m),axis=0))
        z,flip,coin,m = tuple(res[tf.newaxis,j] for j in range(4)) # slicing tensor res
        return z,flip,coin,m
    def body1(z,flip,coin,m):
        res = tf.where(tf.equal(flip,0.0) & tf.greater(m,0.0),tf.concat(body_f1(z,flip,coin,m),axis=0),tf.concat((z,flip,coin,m),axis=0))
        return tuple([res[tf.newaxis,j] for j in range(4)]) # slicing tensor res
    z,flip,coin,m=tf.while_loop(lambda *_: True, body1, (z,flip,coin,m), maximum_iterations=20)
    m=tf.where(tf.logical_or(tf.logical_not(tf.equal(flip,0.0)) , tf.equal(m,0.0)),  m * tf.cast(tf.greater(z,2),tf.float32), np.NaN)
    return z,flip,coin,m

var('x y z coin')

N=1 # Warm up 
z = tf.zeros((1,N))
flip = tf.zeros((1,N))
coin = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(z,flip,coin,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1  elem %s seconds -------        " % final_time)

N=10**6
z = tf.zeros((1,N))
flip = tf.zeros((1,N))
coin = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(z,flip,coin,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1M  elem %s seconds -------        " % final_time)

eps=0.005
maxe=1
exp, lower_prob,conf=compute_statistics_geo0(res, eps, maxe)

print("exp %s" % exp)
# central_value = (exp[0] + exp[1]) / 2;
# print("central value %s" % central_value)

#---------------------------------- Example: die_conditioning ----------------------------

print("---------------- die_conditioning ----------------")

var('d1 d2 sum')

S_s = seq(draw(d1,rhoU(1,6)),draw(d2,rhoU(1,6)),setx(sum,d1+d2),obs(sum==10))
xlist=['d1', 'd2', 'sum']
# tr_S=translate_sc(S_s,xlist)
# print(tr_S)

@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32)])
def f0(d1,d2,sum,m):
    probs = tf.constant([1/6] * 6, dtype=tf.float32)
    outcomes = tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.float32)
    dice = tfd.FiniteDiscrete(outcomes, probs)
    d1=dice.sample(tf.shape(d1))
    d2=dice.sample(tf.shape(d2))
    sum=d1 + d2
    m = m * tf.cast((sum==10),tf.float32)
    return d1,d2,sum,m

var('d1 d2 sum')

N=1 # Warm up 
d1 = tf.zeros((1,N))
d2 = tf.zeros((1,N))
sum = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(d1,d2,sum,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1  elem %s seconds -------        " % final_time)

N=10**6
d1 = tf.zeros((1,N))
d2 = tf.zeros((1,N))
sum = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(d1,d2,sum,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1M  elem %s seconds -------        " % final_time)

eps=0.005
maxe=1
exp, lower_prob,conf=compute_statistics_die_cond(res, eps, maxe)

print("exp %s" % exp)
# central_value = (exp[0] + exp[1]) / 2;
# print("central value %s" % central_value)

#---------------------------------- Example: PrinSys ----------------------------

print("---------------- PrinSys ----------------")

var('x coin')

S_s = whl((x==0), seq(draw(coin,B(0.5)),ite(coin==1,setx(x,x-x),seq(draw(coin,B(0.5)),ite(coin==1,setx(x,1),setx(x,2))))),true)
xlist=['x', 'coin']
# tr_S=translate_sc(S_s,xlist)
# print(tr_S)

@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32)])
def f0(x,coin,m):
    B_probs = tf.zeros(shape=tf.shape(coin))
    B_probs += 0.5
    def body_f1(x,coin,m):
        coin = tfd.Bernoulli(probs=B_probs).sample()
        coin = tf.cast(coin,dtype=tf.float32)
        def f2(x,coin,m):
            x-=x
            return x,coin,m
        def f3(x,coin,m):
            coin = tfd.Bernoulli(probs=B_probs).sample()
            coin = tf.cast(coin,dtype=tf.float32)
            def f4(x,coin,m):
                x+=1
                return x,coin,m
            def f5(x,coin,m):
                x+=2
                return x,coin,m
            mask = (coin==1.0)
            res=tf.where(mask, tf.concat(f4(x,coin,m),axis=0), tf.concat(f5(x,coin,m),axis=0))
            x,coin,m = tuple(res[tf.newaxis,j] for j in range(3)) # slicing tensor res
            return x,coin,m
        mask = (coin==1.0)
        res=tf.where(mask, tf.concat(f2(x,coin,m),axis=0), tf.concat(f3(x,coin,m),axis=0))
        x,coin,m = tuple(res[tf.newaxis,j] for j in range(3)) # slicing tensor res
        return x,coin,m
    def body1(x,coin,m):
        res = tf.where((tf.equal(x,0.0)) & tf.greater(m,0.0),tf.concat(body_f1(x,coin,m),axis=0),tf.concat((x,coin,m),axis=0))
        return tuple([res[tf.newaxis,j] for j in range(3)]) # slicing tensor res
    x,coin,m=tf.while_loop(lambda *_: True, body1, (x,coin,m), maximum_iterations=10)
    m=tf.where(tf.logical_or(tf.logical_not(tf.equal(x,0.0)) , tf.equal(m,0.0)),  m * tf.cast(True,tf.float32), np.NaN)
    return x,coin,m

var('x coin')

N=1 # Warm up 
x = tf.constant(2.0,shape=(1,N))
coin = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(x,coin,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1  elem %s seconds -------        " % final_time)

N=10**6
x = tf.constant(2.0,shape=(1,N))
coin = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(x,coin,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1M  elem %s seconds -------        " % final_time)

eps=0.005
maxe=1
exp, lower_prob,conf=compute_statistics_PrinSys(res, eps, maxe)

print("exp %s" % exp)
# central_value = (exp[0] + exp[1]) / 2;
# print("central value %s" % central_value)

#---------------------------------- Example: RevBin ----------------------------

print("---------------- RevBin ----------------")

var('x z coin')

S_s = whl(0<x,seq(draw(coin,B(0.5)),ite(coin==1,setx(x,x-1),setx(x,x)),setx(z,z+1)),true)
xlist=['x', 'z', 'coin']
# tr_S=translate_sc(S_s,xlist)
# print(tr_S)

@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32)])
def f0(x,z,coin,m):
    B_probs = tf.zeros(shape=tf.shape(coin))
    B_probs += 0.5
    def body_f1(x,z,coin,m):
        coin = tfd.Bernoulli(probs=B_probs).sample()
        coin = tf.cast(coin,dtype=tf.float32)
        def f2(x,z,coin,m):
            x=x - 1
            return x,z,coin,m
        def f3(x,z,coin,m):
            x=x
            return x,z,coin,m
        mask = coin==1.0
        res=tf.where(mask, tf.concat(f2(x,z,coin,m),axis=0), tf.concat(f3(x,z,coin,m),axis=0))
        x,z,coin,m = tuple(res[tf.newaxis,j] for j in range(4)) # slicing tensor res
        z=z + 1
        return x,z,coin,m
    def body1(x,z,coin,m):
        res = tf.where((x > 0) & tf.greater(m,0.0),tf.concat(body_f1(x,z,coin,m),axis=0),tf.concat((x,z,coin,m),axis=0))
        return tuple([res[tf.newaxis,j] for j in range(4)]) # slicing tensor res
    x,z,coin,m=tf.while_loop(lambda *_: True, body1, (x,z,coin,m), maximum_iterations=20)
    m=tf.where(tf.logical_or(tf.logical_not(x > 0) , tf.equal(m,0.0)),  m * tf.cast(True,tf.float32), np.NaN)
    return x,z,coin,m

var('x z coin')

N=1 # Warm up 
x = tf.constant(5.0,shape=(1,N))
z = tf.constant(3.0,shape=(1,N))
coin = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(x,z,coin,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1  elem %s seconds -------        " % final_time)

N=10**6
x = tf.constant(5.0,shape=(1,N))
z = tf.constant(3.0,shape=(1,N))
coin = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(x,z,coin,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1M  elem %s seconds -------        " % final_time)

eps=0.005
maxe=1
exp, lower_prob,conf=compute_statistics_RevBin_1(res, eps, maxe, x, z)

print("exp %s" % exp)
# central_value = (exp[0] + exp[1]) / 2;
# print("central value %s" % central_value)

#-------------------------------------  TACAS_ABSYNTH FOLDER  ------------------------------------

#---------------------------------- Example: sprdwalk ----------------------------

print("---------------- sprdwalk ----------------")

var('x n coin')

S_s = whl(x<n,seq(draw(coin,B(0.5)),ite(coin==1,setx(x,x),setx(x,x+1))),true)
xlist=['x', 'n', 'coin']
# tr_S=translate_sc(S_s,xlist)
# print(tr_S)

@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32)])
def f0(x,n,coin,m):
    B_probs = tf.zeros(shape=tf.shape(coin))
    B_probs += 0.5
    def body_f1(x,n,coin,m):
        coin = tfd.Bernoulli(probs=B_probs).sample()
        coin = tf.cast(coin,dtype=tf.float32)
        def f2(x,n,coin,m):
            x=x
            return x,n,coin,m
        def f3(x,n,coin,m):
            x=x + 1
            return x,n,coin,m
        mask = coin==1.0
        res=tf.where(mask, tf.concat(f2(x,n,coin,m),axis=0), tf.concat(f3(x,n,coin,m),axis=0))
        x,n,coin,m = tuple(res[tf.newaxis,j] for j in range(4)) # slicing tensor res
        return x,n,coin,m
    def body1(x,n,coin,m):
        res = tf.where((x < n) & tf.greater(m,0.0),tf.concat(body_f1(x,n,coin,m),axis=0),tf.concat((x,n,coin,m),axis=0))
        return tuple([res[tf.newaxis,j] for j in range(4)]) # slicing tensor res
    x,n,coin,m=tf.while_loop(lambda *_: True, body1, (x,n,coin,m), maximum_iterations=10)
    m=tf.where(tf.logical_or(tf.logical_not(x < n) , tf.equal(m,0.0)),  m * tf.cast(True,tf.float32), np.NaN)
    return x,n,coin,m

var('x n coin')

N=1 # Warm up 
x = tf.zeros((1,N))
n = tf.random.uniform(shape=(1,N),maxval=5)
coin = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(x,n,coin,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1  elem %s seconds -------        " % final_time)

N=10**6
x = tf.constant(2.0,shape=(1,N))
n = tf.random.uniform(shape=(1,N),maxval=5)
coin = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(x,n,coin,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1M  elem %s seconds -------        " % final_time)

eps=0.005
maxe=1
prop=compute_statistics_sprdwalk(res, eps, maxe)

print("prop %s" % prop)

#-------------------------------------  MY EXAMPLE  ------------------------------------

#---------------------------------- Example: Normal ----------------------------

print("---------------- Normal ----------------")

var('y pos')

S_s = seq(draw(y,rhoU(0,1)),setx(y,y+rhoU(0,1)),setx(y,y+rhoU(0,1))) # ,ite(0<y-1.5,setx(pos,1),setx(pos,0))
xlist=['y']
# tr_S=translate_sc(S_s,xlist)
# print(tr_S)

@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32)])
def f0(y,m):
    probs = tf.constant([1/11] * 11, dtype=tf.float32)
    outcomes = tf.constant([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], dtype=tf.float32)
    d = tfd.FiniteDiscrete(outcomes, probs)
    y+=d.sample(tf.shape(y))
    y+=d.sample(tf.shape(y))
    y+=d.sample(tf.shape(y))
    return y,m

var('y')

N=1 # Warm up 
y = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(y,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1  elem %s seconds -------        " % final_time)

N=10**6
y = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(y,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1M  elem %s seconds -------        " % final_time)

eps=0.005
maxe=1
exp, lower_prob,conf=compute_statistics_normal(res, eps, maxe)

print("exp %s" % exp)
# central_value = (exp[0] + exp[1]) / 2;
# print("central value %s" % central_value)

# #-------------------------------------------------------------------------------------------
# #-------------------------------------  EXPERIMENTS  ------------------------------------
# #-------------------------------------------------------------------------------------------

#---------------------------------- Example 6: Monty Hall  ----------------------------

print("---------------- Monty Hall ----------------")

var('x car guest host guest2 win')

def ternary(x,pL=[1/3,1/3,1/3],vL=[0,1,2]):   
    pL=np.cumsum(pL) 
    S=seq( draw(x,rhoU()), ite(x<=pL[0],setx(x,vL[0]), ite(x<=pL[1],setx(x,vL[1]),setx(x,vL[2]))))
    return S

def bern(x,e=1/2,vL=[0,1]):  
    S=seq(draw(x,rhoU()),ite(x>=e,setx(x,vL[0]),setx(x,vL[1])))
    return S  
    
MH = seq(ternary(car),
                  ternary(guest),      
                  ite(Eq(car,0) & Eq(guest,1), setx(host,2) ,setx(host,host)),
                  ite(Eq(car,0) & Eq(guest,2), setx(host,1),setx(host,host)),                 
                  ite(Eq(car,1) & Eq(guest,0),setx(host,2),setx(host,host)),
                  ite(Eq(car,1) & Eq(guest,2),setx(host,0),setx(host,host)),
                  ite(Eq(car,2) & Eq(guest,0), setx(host,1),setx(host,host)),
                  ite(Eq(car,2) & Eq(guest,1), setx(host,0),setx(host,host)),
                  ite(Eq(car,0) & Eq(guest,0), seq(draw(x,rhoU()),ite(x>=0.5,setx(host,1),setx(host,2))),setx(host,host)),
                  ite(Eq(car,1) & Eq(guest,1), seq(draw(x,rhoU()),ite(x>=0.5,setx(host,0),setx(host,2))),setx(host,host)),
                  ite(Eq(car,2) & Eq(guest,2), seq(draw(x,rhoU()),ite(x>=0.5,setx(host,0),setx(host,1))),setx(host,host)),
                  ite(Eq(guest,1) & Eq(host,2),setx(win,0),setx(win,win)),
                  ite(Eq(guest,0) & Eq(host,2),setx(guest,1),setx(win,win)),
                  ite(Eq(guest,0) & Eq(host,1),setx(win,2),setx(win,win)),
                  ite(Eq(guest,2) & Eq(host,1),setx(win,0),setx(win,win)),                  
                  ite(Eq(guest,1) & Eq(host,0),setx(win,2),setx(win,win)),
                  ite(Eq(guest,2) & Eq(host,0),setx(win,1),setx(win,win)),                  
                  ite(Eq(win,car),
                       setx(win,1),
                       setx(win,0)))

xlist_MH='car, guest, host, win'.split(',')
c=0

@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32)])
def f0(r, car, guest, host, win,m):
    car=tfd.Uniform(low=r).sample()
    def f1(car, guest, host, win,m):
        car=r
        return car, guest, host, win,m
    def f2(car, guest, host, win,m):
        def f3(car, guest, host, win,m):
            car=r+1
            return car, guest, host, win,m
        def f4(car, guest, host, win,m):
            car=r+2
            return car, guest, host, win,m
        mask = car <= 0.666666666666667
        res=tf.where(mask, tf.concat(f3(car, guest, host, win,m),axis=0), tf.concat(f4(car, guest, host, win,m),axis=0))
        car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
        return car, guest, host, win,m
    mask = car <= 0.333333333333333
    res=tf.where(mask, tf.concat(f1(car, guest, host, win,m),axis=0), tf.concat(f2(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res

   
    guest=tfd.Uniform(low=r).sample()
    def f7(car, guest, host, win,m):
        guest=r
        return car, guest, host, win,m
    def f8(car, guest, host, win,m):
        def f9(car, guest, host, win,m):
            guest=r+1
            return car, guest, host, win,m
        def f10(car, guest, host, win,m):
            guest=r+2
            return car, guest, host, win,m
        mask = guest <= 0.666666666666667
        res=tf.where(mask, tf.concat(f9(car, guest, host, win,m),axis=0), tf.concat(f10(car, guest, host, win,m),axis=0))
        car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
        return car, guest, host, win,m
    mask = guest <= 0.333333333333333
    res=tf.where(mask, tf.concat(f7(car, guest, host, win,m),axis=0), tf.concat(f8(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
    
    def f13(car, guest, host, win,m):
        host=r+2
        return car, guest, host, win,m
    def f14(car, guest, host, win,m):
        host=host
        return car, guest, host, win,m
    
    mask1 = (car==r)
    mask2 = (guest==r+1)
    #mask = (car==r) and (guest==r+1)
    res=tf.where(mask1 & mask2, tf.concat(f13(car, guest, host, win,m),axis=0), tf.concat(f14(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
    
   
    def f16(car, guest, host, win,m):
        host=r+1
        return car, guest, host, win,m
    def f17(car, guest, host, win,m):
        host=host
        return car, guest, host, win,m
    mask1 = (car==r)
    mask2 = (guest==r+2)
    res=tf.where(mask1 & mask2, tf.concat(f16(car, guest, host, win,m),axis=0), tf.concat(f17(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
   
    def f19(car, guest, host, win,m):
        host=r+2
        return car, guest, host, win,m
    def f20(car, guest, host, win,m):
        host=host
        return car, guest, host, win,m
    mask1 = (car==r+1)
    mask2 = (guest==r)
    res=tf.where(mask1 & mask2, tf.concat(f19(car, guest, host, win,m),axis=0), tf.concat(f20(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
   
    def f22(car, guest, host, win,m):
        host=r
        return car, guest, host, win,m
    def f23(car, guest, host, win,m):
        host=host
        return car, guest, host, win,m
    mask1 = (car==r+1)
    mask2 = (guest==r+2)
    res=tf.where(mask1 & mask2, tf.concat(f22(car, guest, host, win,m),axis=0), tf.concat(f23(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
    
      
    def f25(car, guest, host, win,m):
        host=r+1
        return car, guest, host, win,m
    def f26(car, guest, host, win,m):
        host=host
        return car, guest, host, win,m    
    mask1 = (car==r+2)
    mask2 = (guest==r)
    res=tf.where(mask1 & mask2, tf.concat(f25(car, guest, host, win,m),axis=0), tf.concat(f26(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
    
    def f28(car, guest, host, win,m):
        host=r+0
        return car, guest, host, win,m
    def f29(car, guest, host, win,m):
        host=host
        return car, guest, host, win,m
    mask1 = (car==r+2)
    mask2 = (guest==r+1)
    res=tf.where(mask1 & mask2, tf.concat(f28(car, guest, host, win,m),axis=0), tf.concat(f29(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res

    
    def f31(car, guest, host, win,m):
        x=tfd.Uniform(low=r).sample()
        def f33(car, guest, host, win,m):
            host=r+1
            return car, guest, host, win,m
        def f34(car, guest, host, win,m):
            host=r+2
            return car, guest, host, win,m
        mask = x >= 0.5
        res=tf.where(mask, tf.concat(f33(car, guest, host, win,m),axis=0), tf.concat(f34(car, guest, host, win,m),axis=0))
        car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
        return car, guest, host, win,m
    def f32(car, guest, host, win,m):
        host=host
        return car, guest, host, win,m
    mask1 = (car==r)
    mask2 = (guest==r)
    res=tf.where(mask1 & mask2, tf.concat(f31(car, guest, host, win,m),axis=0), tf.concat(f32(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
    
    def f37(car, guest, host, win,m):
        x=tfd.Uniform(low=r).sample()
        def f39(car, guest, host, win,m):
            host=r
            return car, guest, host, win,m
        def f40(car, guest, host, win,m):
            host=r+2
            return car, guest, host, win,m
        mask = x >= 0.5
        res=tf.where(mask, tf.concat(f39(car, guest, host, win,m),axis=0), tf.concat(f40(car, guest, host, win,m),axis=0))
        car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
        return car, guest, host, win,m
    def f38(car, guest, host, win,m):
        host=host
        return car, guest, host, win,m
    mask1 = (car==r+1)
    mask2 = (guest==r+1)
    res=tf.where(mask1 & mask2, tf.concat(f37(car, guest, host, win,m),axis=0), tf.concat(f38(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
    
    def f43(car, guest, host, win,m):
        x=tfd.Uniform(low=r).sample()
        def f45(car, guest, host, win,m):
            host=r+0
            return car, guest, host, win,m
        def f46(car, guest, host, win,m):
            host=r+1
            return car, guest, host, win,m
        mask = x >= 0.5
        res=tf.where(mask, tf.concat(f45(car, guest, host, win,m),axis=0), tf.concat(f46(car, guest, host, win,m),axis=0))
        car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
        return car, guest, host, win,m
    def f44(car, guest, host, win,m):
        host=host
        return car, guest, host, win,m
    mask1 = (car==r+2)
    mask2 = (guest==r+2)
    res=tf.where(mask1 & mask2, tf.concat(f43(car, guest, host, win,m),axis=0), tf.concat(f44(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
    
    #-------------  
    def f49(car, guest, host, win,m):
        win=r
        return car, guest, host, win,m
    def f50(car, guest, host, win,m):
        win=win
        return car, guest, host, win,m
    mask1 = (guest==r+1)
    mask2 = (host==r+2)
    res=tf.where(mask1 & mask2, tf.concat(f49(car, guest, host, win,m),axis=0), tf.concat(f50(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
    def f52(car, guest, host, win,m):
        win=r+1
        return car, guest, host, win,m
    def f53(car, guest, host, win,m):
        win=win
        return car, guest, host, win,m
    mask1 = (guest==r)
    mask2 = (host==r+2)
    res=tf.where(mask1 & mask2, tf.concat(f52(car, guest, host, win,m),axis=0), tf.concat(f53(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
    def f55(car, guest, host, win,m):
        win=r+2
        return car, guest, host, win,m
    def f56(car, guest, host, win,m):
        win=win
        return car, guest, host, win,m
    mask1 = (guest==r)
    mask2 = (host==r+1)
    res=tf.where(mask1 & mask2, tf.concat(f55(car, guest, host, win,m),axis=0), tf.concat(f56(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
    def f58(car, guest, host, win,m):
        win=r
        return car, guest, host, win,m
    def f59(car, guest, host, win,m):
        win=win
        return car, guest, host, win,m
    mask1 = (guest==r+2)
    mask2 = (host==r+1)   
    res=tf.where(mask1 & mask2, tf.concat(f58(car, guest, host, win,m),axis=0), tf.concat(f59(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
    def f61(car, guest, host, win,m):
        win=r+2
        return car, guest, host, win,m
    def f62(car, guest, host, win,m):
        win=win
        return car, guest, host, win,m
    mask1 = (guest==r+1)
    mask2 = (host==r)
    res=tf.where(mask1 & mask2, tf.concat(f61(car, guest, host, win,m),axis=0), tf.concat(f62(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
    def f64(car, guest, host, win,m):
        win=r+1
        return car, guest, host, win,m
    def f65(car, guest, host, win,m):
        win=win
        return car, guest, host, win,m
    mask1 = (guest==r+2)
    mask2 = (host==r)
    res=tf.where(mask1 & mask2, tf.concat(f64(car, guest, host, win,m),axis=0), tf.concat(f65(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
    
    
    def f67(car, guest, host, win,m):
        win=r+1
        return car, guest, host, win,m
    def f68(car, guest, host, win,m):
        win=r
        return car, guest, host, win,m
    mask = win==car
    res=tf.where(mask, tf.concat(f67(car, guest, host, win,m),axis=0), tf.concat(f68(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res

    return car, guest, host, win,m



N=1
rr = tf.zeros((1,N))
bb=tf.zeros(shape=(1,N))
m=tf.fill(dims=[1,N],value=1.0)
start_time=time.time()
res=f0(rr,bb,bb,bb,bb,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1 elems  %s seconds ------      " % final_time)



N=10**6
rr = tf.zeros((1,N))
bb=tf.zeros(shape=(1,N))
m=tf.fill(dims=[1,N],value=1.0)
start_time=time.time()
res=f0(rr,bb,bb,bb,bb,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 10**6 elems  %s seconds -------        " % final_time)



var('car guest host guest2 win')
xl=[car,guest,host, win]   
e=win
eps=0.005
maxe=1
exp, lower_prob,conf=compute_statistics_MH(res,xl, e, eps, maxe)


print("exp %s" % exp)
# central_value = (exp[0] + exp[1]) / 2;
# print("central value %s" % central_value)


#--------------------------------------- Example 3: Burglar Alarm -----------------------------

print("\n---------------- Burglar alarm ----------------")

var('earthquake burglary phoneWorking maryWakes alarm called')
BA = seq(draw(earthquake,B(0.001)), 
         draw(burglary,B(0.01)) ,   
         setx(alarm , (earthquake>0) | (burglary>0)),
         ite(earthquake>0, draw(phoneWorking,B(0.6)), draw(phoneWorking,B(0.99))),
         ite(alarm & (earthquake>0), 
             draw(maryWakes,B(0.8)), 
             ite(alarm, 
                 draw(maryWakes,B(0.6)), 
                 draw(maryWakes,B(0.2))
                )
            ),
         setx(called , (maryWakes>0) & (phoneWorking>0)),
         obs(called)
)

xlist_BA='earthquake, burglary, phoneWorking, maryWakes'.split(',')
c=0
#tr_BA=translate_sc(BA,xlist_BA)
#print(tr_BA)



# working TF function definition:
@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32)]+[tf.TensorSpec(shape=None, dtype=tf.float32)]*7)
def f0(earthquake, burglary, phoneWorking, maryWakes,m,p1,p2,p3,p4,p5,p6,p7):
    print("Tracing")
    earthquake=tfd.Bernoulli(dtype=tf.float32, probs=p1).sample()
    burglary=tfd.Bernoulli(dtype=tf.float32,probs=p2).sample()
    alarm=(burglary > 0) | (earthquake > 0)
    def f1(earthquake, burglary, phoneWorking, maryWakes,m):
        phoneWorking=tfd.Bernoulli(dtype=tf.float32,probs=p3).sample()
        return earthquake, burglary, phoneWorking, maryWakes,m
    def f2(earthquake, burglary, phoneWorking, maryWakes,m):
        phoneWorking=tfd.Bernoulli(dtype=tf.float32,probs=p4).sample()
        return earthquake, burglary, phoneWorking, maryWakes,m
    mask = earthquake > 0
    res=tf.where(mask, tf.concat(f1(earthquake, burglary, phoneWorking, maryWakes,m),axis=0), tf.concat(f2(earthquake, burglary, phoneWorking, maryWakes,m),axis=0))
    earthquake, burglary, phoneWorking, maryWakes,m = tuple(res[tf.newaxis,j] for j in range(5))
    def f4(earthquake, burglary, phoneWorking, maryWakes,m):
        maryWakes=tfd.Bernoulli(dtype=tf.float32,probs=p5).sample()
        return earthquake, burglary, phoneWorking, maryWakes,m
    def f5(earthquake, burglary, phoneWorking, maryWakes,m):
        def f6(earthquake, burglary, phoneWorking, maryWakes,m):
            maryWakes=tfd.Bernoulli(dtype=tf.float32,probs=p6).sample()
            return earthquake, burglary, phoneWorking, maryWakes,m
        def f7(earthquake, burglary, phoneWorking, maryWakes,m):
            maryWakes=tfd.Bernoulli(dtype=tf.float32,probs=p7).sample()
            return earthquake, burglary, phoneWorking, maryWakes,m
        mask = alarm
        res=tf.where(mask, tf.concat(f6(earthquake, burglary, phoneWorking, maryWakes,m),axis=0), tf.concat(f7(earthquake, burglary, phoneWorking, maryWakes,m),axis=0))
        earthquake, burglary, phoneWorking, maryWakes,m = tuple(res[tf.newaxis,j] for j in range(5)) 
        return earthquake, burglary, phoneWorking, maryWakes,m
    mask = alarm & (earthquake > 0)
    res=tf.where(mask, tf.concat(f4(earthquake, burglary, phoneWorking, maryWakes,m),axis=0), tf.concat(f5(earthquake, burglary, phoneWorking, maryWakes,m),axis=0))
    earthquake, burglary, phoneWorking, maryWakes,m = tuple(res[tf.newaxis,j] for j in range(5))
    called=(maryWakes > 0) & (phoneWorking > 0)
    m = m * tf.cast((called),tf.float32)
    return earthquake, burglary, phoneWorking, maryWakes,m



# Warm up
N=1
bb=tf.zeros(shape=(1,N))
m=tf.fill(dims=[1,N],value=1.0)
shp=(1,N)
p1=tf.constant(.001 , shape=shp)
p2=tf.constant(.01 , shape=shp)
p3=tf.constant(.6 , shape=shp)
p4=tf.constant(.99 , shape=shp)
p5=tf.constant(.8 , shape=shp)
p6=tf.constant(.6 , shape=shp)
p7=tf.constant(.2 , shape=shp)
start_time=time.time()
res=f0(bb,bb,bb,bb,m,p1,p2,p3,p4,p5,p6,p7)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1 elems  %s seconds -------        " % final_time)


N=10**6
bb=tf.zeros(shape=(1,N))
m=tf.fill(dims=[1,N],value=1.0)
shp=(1,N)
p1=tf.constant(.001 , shape=shp)
p2=tf.constant(.01 , shape=shp)
p3=tf.constant(.6 , shape=shp)
p4=tf.constant(.99 , shape=shp)
p5=tf.constant(.8 , shape=shp)
p6=tf.constant(.6 , shape=shp)
p7=tf.constant(.2 , shape=shp)

start_time=time.time()
res=f0(bb,bb,bb,bb,m,p1,p2,p3,p4,p5,p6,p7)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 10**6 elems  %s seconds -------        " % final_time)

compute_expected_value_approx(res[1][0], res[4][0])

var('earthquake, burglary, phoneWorking, maryWakes')
xl=[earthquake, burglary, phoneWorking, maryWakes]   
e=burglary
eps=0.005
maxe=1
exp, lower_prob,conf=compute_statistics_BA(res,xl, e, eps, maxe)

print("exp %s" % exp)
# central_value = (exp[0] + exp[1]) / 2;
# print("central value %s" % central_value)