"""Matrix-only conditional search across numeric contexts; no field labels."""
import json, hashlib, time, sys
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from model import CoefficientDistribution


def train_worker(inputs,out):
    cfg=json.loads((inputs/'config.json').read_text())
    with np.load(inputs/'matrix.npz') as data:
        assert data.files==['train']
        x=data['train']
    torch.set_num_threads(1); torch.manual_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(cfg['seed'])
    contexts,rows,terms=x.shape
    model=CoefficientDistribution(terms,cfg['alphabet'],512,128,contexts)
    with torch.no_grad():
        # Generic small-integer prior, identical for every term and context.
        values=torch.tensor(model.alphabet,dtype=torch.float32)
        model.coefficient_head.bias.reshape(terms,-1).sub_(cfg.get('integer_prior',0.)*values.square())
        model.value_head.bias.sub_(torch.log(torch.tensor(model.positive,dtype=torch.float32)))
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    scaled=x/np.maximum(np.abs(x).max(axis=2,keepdims=True),1e-30)
    gram=torch.tensor(scaled.transpose(0,2,1)@scaled/rows,dtype=torch.float64)
    floor=np.maximum(np.abs(x).max(axis=(1,2))*1e-10,1e-30)
    cache={}; replay={}; history=[]; priorities=np.zeros(contexts); seen=set(); calls=0; first=None; visits=np.zeros(contexts,dtype=int)
    start=time.perf_counter(); first_proposal=None
    passed_contexts=set(); passed_archive={}; remaining=[]; focus_start=0; focus=None
    def evaluate(ctx,coefficients,proposal_type):
        nonlocal calls,first,cache,first_proposal
        numerator=np.empty((len(ctx),rows));scale=np.empty_like(numerator)
        for sector in np.unique(ctx):
            indices=np.flatnonzero(ctx==sector)
            numerator[indices]=(x[sector]@coefficients[indices].T).T
            scale[indices]=(np.abs(x[sector])@np.abs(coefficients[indices]).T).T
        active=scale>floor[ctx,None]
        rel=numerator/np.maximum(scale,floor[ctx,None])
        rms=np.sqrt((rel**2*active).sum(1)/np.maximum(active.sum(1),1))
        maximum=(np.abs(rel)*active).max(1); coverage=active.mean(1)
        support=np.count_nonzero(coefficients,axis=1)
        reward=1/(1+cfg['reward_scale']*rms)-cfg['complexity']*(support-1)
        reward[(support<2)|(coverage<.8)]=-1
        for j,(sector,c) in enumerate(zip(ctx,coefficients)):
            digest=hashlib.sha256(np.r_[sector,c].astype(np.int16).tobytes()).digest();seen.add(digest)
            passed=bool(maximum[j]<cfg['success_tol'] and coverage[j]>=.8 and 2<=support[j]<=cfg['max_terms'])
            record={'context':int(sector),'coefficients':c.tolist(),'reward':float(reward[j]),'rms':float(rms[j]),'max':float(maximum[j]),'coverage':float(coverage[j]),'passed':passed,'proposal_type':proposal_type}
            if first is None and passed and int(sector) in cfg['eligible_contexts']:
                first=calls+j+1
                first_proposal=proposal_type
            cache[digest]=record
            if passed:
                passed_contexts.add(int(sector))
                primitive=c//np.gcd.reduce(np.abs(c))
                if primitive[np.flatnonzero(primitive)[0]]<0: primitive=-primitive
                key=(int(sector),tuple(primitive.tolist()))
                passed_archive.setdefault(key,record)
            if support[j]>=2 and coverage[j]>=.8:
                priorities[sector]=max(priorities[sector],float(reward[j]))
        calls+=len(ctx)
        # Bounded cache; no target vectors supplied.
        if len(cache)>1024:
            cache=dict(sorted(cache.items(),key=lambda pair:(not pair[1]['passed'],-pair[1]['reward']))[:256])
        return reward
    completed=0
    with threadpool_limits(limits=1):
        for step in range(cfg['steps']):
            # Cover all contexts in each cycle, independent of the answers.
            warmup=int(np.ceil(contexts/cfg['batch_size']))
            if step<warmup:
                ctx=(np.arange(cfg['batch_size'])+step*cfg['batch_size'])%contexts
            else:
                if focus is None or step-focus_start>=cfg.get('focus_steps',2000) or focus in passed_contexts:
                    # Field-blind random context choice: no coefficient labels.
                    if not remaining:
                        remaining=[i for i in cfg['eligible_contexts'] if i not in passed_contexts]
                    if not remaining:
                        break
                    focus=int(rng.choice(remaining));remaining.remove(focus);focus_start=step
                ctx=np.full(cfg['batch_size'],focus)
            visits+=np.bincount(ctx,minlength=contexts)
            dist=model.distribution(len(ctx),contexts=torch.tensor(ctx))
            c,tokens=model.sample_distribution(dist)
            greedy,gtokens=model.sample_distribution(dist,greedy=True)
            r=evaluate(ctx,c.detach().numpy(),'categorical_sample'); gr=evaluate(ctx,greedy.detach().numpy(),'conditional_mode')
            alphabet=torch.tensor(model.alphabet,dtype=dist['probabilities'].dtype)
            means=dist['probabilities']@alphabet
            rounded=means.detach().round().clamp(min(model.alphabet),max(model.alphabet)).to(torch.long)
            rounded_ids=(rounded[:,:,None]==alphabet[None,None,:]).long().argmax(-1)
            rounded_tokens=torch.cat([dist['controls'],rounded_ids,dist['contexts'][:,None]],1)
            rr=evaluate(ctx,rounded.numpy(),'rounded_neural_mean')
            mean,var,scale,support=model.quadratic_moments(dist,gram)
            beta=cfg['variance_start']+(cfg['variance_end']-cfg['variance_start'])*min((step+1)/cfg['variance_ramp_steps'],1)
            guide_rows=torch.log1p((mean+beta*var).clamp_min(0)/scale)
            guide=guide_rows.mean()
            loss=guide+cfg.get('integer_penalty',0.)*min((step+1)/2000,1)*torch.sin(torch.pi*means).square().mean()+cfg.get('mean_l1_weight',0.)*means.abs().mean()+cfg['support_weight']*support.mean()+cfg.get('minimum_support_weight',1.)*torch.relu(2-support).square().mean()-cfg['control_entropy_weight']*dist['control_entropy'].mean()
            rewards=torch.tensor(np.maximum(np.maximum(r,gr),rr),dtype=torch.float32)-cfg.get('control_guide_weight',0.)*guide_rows.detach().float()
            advantage=(rewards-rewards.mean()).detach()/rewards.std().detach().clamp_min(.03)
            loss-=cfg['control_pg_weight']*(advantage*dist['control_logprob']).mean()
            for seqs,scores in ((tokens,r),(gtokens,gr),(rounded_tokens,rr)):
                for seq,score in zip(seqs.detach().tolist(),scores):
                    context=seq[-1]
                    local=replay.setdefault(context,{})
                    local[tuple(seq)]=float(score)
                    replay[context]=dict(sorted(local.items(),key=lambda pair:-pair[1])[:8])
            # Balanced self-replay: no context can monopolize the training signal.
            sequences=[]; memory_weights=[]
            active_contexts=np.unique(ctx)
            for context in active_contexts:
                local=list(replay[int(context)].items())[:2]
                values=np.array([r for _,r in local]);weights=np.exp((values-values.max())/.04);weights/=weights.sum()*len(active_contexts)
                sequences.extend(seq for seq,_ in local);memory_weights.extend(weights)
            mem=torch.tensor(sequences)
            loss-=cfg['replay_weight']*(torch.tensor(memory_weights)*model.log_probability(mem)).sum()/terms
            optimizer.zero_grad(set_to_none=True);loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not torch.isfinite(loss) or not torch.isfinite(grad): raise RuntimeError('Nonfinite training')
            optimizer.step()
            completed=step+1
            if step%cfg['log_every']==0 or (first is not None and calls-first<3*len(ctx)) or step+1==cfg['steps']:
                best=min(cache.values(),key=lambda d:(not d['passed'],-d['reward']))
                history.append({'step':step+1,'seconds':time.perf_counter()-start,'loss':float(loss.detach()),'relaxation_loss':float(guide.detach()),'best_rms':best['rms'],'best_max':best['max'],'evaluations':calls,'context':best['context'],'active_context':int(ctx[0])})
                print(f"boson step {step+1}/{cfg['steps']} best={best['max']:.3g} active_context={ctx[0]} best_context={best['context']} support={float(support.detach().mean()):.2f}",flush=True)
            if first is not None and visits.min()>0 and cfg.get('stop_on_success',False): break
    out.mkdir()
    initial_hash=hashlib.sha256(b''.join(initial[k].numpy().tobytes() for k in sorted(initial))).hexdigest()
    torch.save({'policy_kind':'conditional_coefficients','architecture':model.config,'state_dict':model.state_dict(),'initial_state_sha256':initial_hash,'seed':cfg['seed'],'completed_steps':completed},out/'policy.pt')
    result={'history':history,'completed_steps':completed,'budget':calls,'unique_candidates':len(seen),'first_success_call':first,'first_success_proposal':first_proposal,
            'seconds':time.perf_counter()-start,'context_visits':visits.tolist(),'parameter_count':sum(p.numel() for p in model.parameters()),
            'parameter_l2_change':sum((v-initial[k]).square().sum().item() for k,v in model.state_dict().items())**.5,
            'candidates':list(passed_archive.values())+sorted((c for c in cache.values() if not c['passed']),key=lambda d:-d['reward'])[:16],
            'numerically_passing_contexts':sorted(passed_contexts),
            'forbidden_modules_absent':set(n for n in sys.modules if n.split('.')[0] in {'model','train','main','physics','validation','plot_tool','utils','test_core'}).issubset({'model','train'}),
            'proposal_types':['categorical_sample','conditional_mode','rounded_neural_mean'],
            'worker_inputs':['config.json','matrix.npz:train'],'linear_solver_used':False,'replay_initially_empty':True}
    (out/'search.json').write_text(json.dumps(result,indent=2))
