#!/usr/bin/env python3
"""Bounded train-only resource probes. Never selects models or launches train mode."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time
import yaml

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from opencood.project.config import load_config

def memory_snapshot():
    base=Path('/sys/fs/cgroup')
    stat=dict(line.split() for line in (base/'memory.stat').read_text().splitlines())
    return {k:int(stat.get(k,0))/1024**3 for k in ('anon','file','shmem','inactive_file')} | {
        'current_gib':int((base/'memory.current').read_text())/1024**3}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(REPO/'cscwd_configs/project_v4_W_AB.yaml'))
    p.add_argument('--output',default='/root/autodl-tmp/CSCWD/outputs/resource_probes')
    p.add_argument('--steps',type=int,default=8)
    p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--profiles',nargs='+',default=['fp32:4:8'],
                   help='precision:micro_batch:workers, sequential GPU probes')
    args=p.parse_args()
    if not 1<=args.steps<=20 or not 0<=args.warmup<=5:p.error('bounded probes allow 1–20 measured / 0–5 warmup updates')
    cfg=load_config(args.config)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    stamp=time.strftime('%Y%m%d_%H%M%S',time.gmtime())
    results=[]
    for profile in args.profiles:
        precision,micro,workers=profile.split(':');micro=int(micro);workers=int(workers)
        if precision != 'fp32' or micro not in (1,2,4,8) or not 0<=workers<=16:
            p.error('profiles must use stable FP32; rejected AMP candidates are not resource profiles')
        name=f'{stamp}_{precision}_m{micro}_w{workers}'
        c=deepcopy(cfg);c['experiment']['run_id']=name
        c['training'].update(precision=precision,micro_batch_size=micro,accumulation_steps=8//micro)
        c['resources'].update(num_workers=workers,cpu_threads=4)
        c['artifacts']['output_root']=str(root)
        config_path=root/(name+'.yaml');config_path.write_text(yaml.safe_dump(c,sort_keys=False))
        log_path=root/(name+'.log');telemetry=[]
        command=[sys.executable,str(REPO/'project_tools/run_experiments.py'),'--config',str(config_path),
                 '--mode','benchmark','--steps',str(args.steps),'--warmup',str(args.warmup)]
        start=time.monotonic()
        with log_path.open('w') as log:
            proc=subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
            while proc.poll() is None:
                raw=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],
                                   capture_output=True,text=True,check=False)
                row={'elapsed':time.monotonic()-start,**memory_snapshot()}
                step_log=root/name/'benchmark/steps.rank0.jsonl'
                row['optimizer_steps_logged']=len(step_log.read_text().splitlines()) if step_log.exists() else 0
                if raw.returncode==0:
                    util,mem,power=map(float,raw.stdout.strip().split(','))
                    row.update(gpu_utilization=util,gpu_memory_mib=mem,gpu_power_w=power)
                telemetry.append(row);time.sleep(.5)
        (root/(name+'.telemetry.json')).write_text(json.dumps(telemetry,indent=2))
        report_path=root/name/'benchmark/report.rank0.json'
        report=json.loads(report_path.read_text()) if report_path.exists() else {'status':'failed','log':str(log_path)}
        report.update(profile=profile,exit_code=proc.returncode,wall_seconds=time.monotonic()-start,
                      peak_cgroup_anon_gib=max((x['anon'] for x in telemetry),default=0),peak_cgroup_shmem_gib=max((x['shmem'] for x in telemetry),default=0))
        active=[x for x in telemetry if x.get('gpu_utilization',0)>0]
        if active:report.update(active_gpu_util_mean=sum(x['gpu_utilization'] for x in active)/len(active),
                                peak_nvidia_memory_mib=max(x['gpu_memory_mib'] for x in active))
        measured=[x for x in telemetry if args.warmup<=x['optimizer_steps_logged']<args.warmup+args.steps
                  and 'gpu_utilization' in x]
        if measured:report.update(measured_gpu_util_mean=sum(x['gpu_utilization'] for x in measured)/len(measured),
                                  measured_gpu_power_w_mean=sum(x['gpu_power_w'] for x in measured)/len(measured),
                                  measured_gpu_poll_count=len(measured))
        results.append(report)
        (root/(stamp+'_summary.json')).write_text(json.dumps(results,indent=2))
        print(json.dumps(report),flush=True)

if __name__=='__main__':main()
