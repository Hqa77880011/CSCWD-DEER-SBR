"""Pipeline unchanged batch-one GPU inference with exact CPU postprocessing.

Only scheduling changes. Every frame still uses the frozen FP32 decoder,
postprocessor and metadata, and results retain the original sample order.
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import sys
import time

_POSTPROCESS = None


def initialize_cpu_worker(repo):
    global _POSTPROCESS
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    sys.path.insert(0,repo)
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    from opencood.project.evaluation import postprocess_prediction_arrays
    _POSTPROCESS=postprocess_prediction_arrays


def postprocess_frame(arrays,sample):
    started=time.perf_counter()
    predictions,counts=_POSTPROCESS(arrays['boxes'],arrays['scores'],arrays['prediction_ids'],sample)
    return predictions,counts,time.perf_counter()-started


def export_loader_parallel(model,loader,dataset,device,*,profile_callback=None,workers=8,max_inflight=32):
    import torch
    from project_tools.export_predictions import decode_prediction_arrays
    from project_tools.run_experiments import shutdown_loader,REPO
    from opencood.project.training import select_model_inputs,to_device
    if workers<1 or max_inflight<workers:
        raise ValueError('parallel export needs bounded inflight work >= worker count')
    def stage(name,operation):
        if profile_callback is None:
            return operation()
        synchronize=name in ('h2d','forward','decode') and torch.device(device).type=='cuda'
        if synchronize:torch.cuda.synchronize(device)
        started=time.perf_counter();value=operation()
        if synchronize:torch.cuda.synchronize(device)
        profile_callback(dict(stage=name,seconds=time.perf_counter()-started))
        return value
    modes={name:module.training for name,module in model.named_modules()}
    records=[];pending=deque();iterator=None;pool=None
    def collect_one():
        record,future=pending.popleft()
        predictions,counts,seconds=future.result()
        record['predictions']=predictions;record['filter_counts']=counts
        records.append(record)
        if profile_callback:
            profile_callback(dict(stage='postprocess',seconds=seconds))
    model.eval()
    try:
        pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize_cpu_worker,initargs=(str(REPO),))
        iterator=iter(loader)
        with torch.inference_mode():
            while True:
                try:
                    batch=stage('loader_wait',lambda:next(iterator))
                except StopIteration:
                    break
                if any(module.training for module in model.modules()):
                    raise ValueError('parallel export requires all modules in eval mode')
                ego=batch['ego']
                inputs=stage('h2d',lambda:to_device(select_model_inputs(ego),device))
                output=stage('forward',lambda:model(inputs,return_aux=False))
                decoded=stage('decode',lambda:decode_prediction_arrays(output,dataset.anchor_box,ego['sample_ids']))
                for index,arrays in enumerate(decoded):
                    sample=dataset.manifest['samples'][arrays['sample_id']]
                    record=dict(sample_id=arrays['sample_id'],predictions=None)
                    record.update({key:sample[key] for key in ('split','sequence','timestamp','ego_id','agent_ids','gt_sha256')})
                    record.update(dataset='v2xr',evaluation_weather=dataset.weather,protocol_sha256=dataset.manifest['protocol_sha256'])
                    agents=ego['metadata'][index]
                    record.update(source_agent_ids=sample['agent_ids'],agent_ids=[a['agent_id'] for a in agents],
                                  evaluated_agent_ids=[a['agent_id'] for a in agents],
                                  agent_weather={a['agent_id']:a['weather'] for a in agents})
                    future=pool.submit(postprocess_frame,arrays,sample)
                    pending.append((record,future))
                    if len(pending)>=max_inflight:
                        collect_one()
                # Keep enough outstanding work for all CPU processes, while
                # promptly reporting already-finished frames between forwards.
                while len(pending)>workers and pending[0][1].done():
                    collect_one()
                del batch,inputs,output,decoded
            while pending:
                collect_one()
    finally:
        for _,future in pending:
            future.cancel()
        if pool is not None:
            pool.shutdown(wait=True,cancel_futures=True)
        for name,module in model.named_modules():
            module.training=modes[name]
        shutdown_loader({'loader':loader,'iterator':iterator})
    return records
