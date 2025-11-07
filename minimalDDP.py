import os, sys, torch, torch.distributed as dist, traceback

os.environ.setdefault("NCCL_DEBUG","INFO")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING","1")
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG","DETAIL")

def fail(msg):
    print(msg, file=sys.stderr, flush=True)
    sys.exit(1)

for k in ["RANK","WORLD_SIZE","LOCAL_RANK"]:
    if k not in os.environ:
        fail(f"Missing {k}. Use: torchrun --standalone --nproc_per_node=2 minimalDDP.py")

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])

print(f"[rank {rank}] env RANK={rank} LOCAL_RANK={local_rank} WORLD_SIZE={world} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

dev_count = torch.cuda.device_count()
print(f"[rank {rank}] device_count={dev_count}", flush=True)
if local_rank >= dev_count:
    fail(f"[rank {rank}] local_rank {local_rank} >= device_count {dev_count}")

torch.cuda.set_device(local_rank)
print(f"[rank {rank}] set_device ok current_device={torch.cuda.current_device()}", flush=True)

# Simple all-reduce test BEFORE DDP
t = torch.full((1,), rank, device=f"cuda:{local_rank}")
dist.init_process_group("nccl")
dist.all_reduce(t)
print(f"[rank {rank}] pre-DDP all_reduce sum={t.item()}", flush=True)

model = torch.nn.Linear(10, 10).to(f"cuda:{local_rank}")
print(f"[rank {rank}] model param device {next(model.parameters()).device}", flush=True)

model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
print(f"[rank {rank}] DDP wrap done", flush=True)

x = torch.ones(1, device=f"cuda:{local_rank}") * rank
dist.all_reduce(x)
print(f"[rank {rank}] post-DDP all_reduce sum={x.item()}", flush=True)

dist.barrier()
print(f"[rank {rank}] finished", flush=True)