import os, sys, torch, torch.distributed as dist, traceback, datetime

os.environ.setdefault("NCCL_DEBUG","INFO")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING","1")
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG","DETAIL")
# Optional: make NCCL avoid problematic interconnects/NICs
# os.environ.setdefault("NCCL_IB_DISABLE","1")
# os.environ.setdefault("NCCL_SOCKET_IFNAME","eth0")  # set to your NIC

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
if dev_count == 0:
    print(f"[rank {rank}] No CUDA devices; using gloo on CPU", flush=True)

if dev_count and local_rank >= dev_count:
    fail(f"[rank {rank}] local_rank {local_rank} >= device_count {dev_count}")

# Choose backend
backend = "nccl" if (dev_count > 0 and sys.platform != "win32") else "gloo"

if dev_count:
    torch.cuda.set_device(local_rank)
    print(f"[rank {rank}] set_device ok current_device={torch.cuda.current_device()}", flush=True)

# Init process group with shorter timeout to avoid endless hang
print(f"[rank {rank}] initializing process group with backend={backend} ...", flush=True)
dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=120))
print(f"[rank {rank}] process group initialized", flush=True)

# Rendezvous sanity barrier
dist.barrier()
print(f"[rank {rank}] passed first barrier", flush=True)

# Simple all-reduce test BEFORE DDP
device = f"cuda:{local_rank}" if dev_count else "cpu"
t = torch.full((1,), rank, device=device)
dist.all_reduce(t)
print(f"[rank {rank}] pre-DDP all_reduce sum={t.item()}", flush=True)

model = torch.nn.Linear(10, 10).to(device)
print(f"[rank {rank}] model param device {next(model.parameters()).device}", flush=True)

if backend == "nccl":
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
else:
    # gloo/CPU or Windows
    model = torch.nn.parallel.DistributedDataParallel(model)

print(f"[rank {rank}] DDP wrap done", flush=True)

x = torch.ones(1, device=device) * rank
dist.all_reduce(x)
print(f"[rank {rank}] post-DDP all_reduce sum={x.item()}", flush=True)

dist.barrier()
print(f"[rank {rank}] finished", flush=True)

dist.destroy_process_group()