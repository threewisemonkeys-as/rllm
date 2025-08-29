# actor = DataParallelPPOActor(config=config.actor, actor_module=actor_module_fsdp, actor_optimizer=actor_optimizer)
# metrics = self.actor.update_policy(data=data)
# entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)
import torch
from transformers import Qwen3ForCausalLM
from verl.workers.fsdp_workers import create_device_mesh, get_fsdp_wrap_policy, get_sharding_strategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoConfig, AutoModelForTokenClassification
from torch.distributed.fsdp import MixedPrecision
import os
import torch.distributed as dist
from torch.distributed.fsdp import CPUOffload

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
# Initialize distributed training
# if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
#     init_process_group(backend="nccl")
#     torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    # Each process gets its own GPU
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return device

device = setup_distributed()
data = {
    "input_ids": torch.ones((4, 14596), dtype=torch.long).to(device),
    "attention_mask": torch.ones((4, 14596), dtype=torch.long).to(device),
    "position_ids": torch.ones((4, 14596), dtype=torch.long).to(device),
}

final_input_ids = torch.ones((4, 14596), dtype=torch.long).to(device)

def init_fn(x: torch.nn.Module):
    if torch.distributed.get_rank() != 0:
        x = x.to_empty(device=torch.cuda.current_device(), recurse=False)
        torch.cuda.empty_cache()
    return x

actor_model_config = AutoConfig.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True, attn_implementation="flash_attention_2")

actor_module = Qwen3ForCausalLM.from_pretrained(
                pretrained_model_name_or_path="Qwen/Qwen3-8B",
                torch_dtype="bfloat16",
                config=actor_model_config,
                trust_remote_code=True,
            ).to(device)

fsdp_config = {'strategy': 'fsdp', 'ppo_mini_batch_size': 1, 'ppo_num_mini_batches': 1, 'ppo_micro_batch_size': None, 
               'ppo_micro_batch_size_per_gpu': 1, 'use_dynamic_bsz': False, 'use_dynamic_mini_batch': False,
               'ppo_max_token_len_per_gpu': 32000, 'grad_clip': 1.0, 'clip_ratio': 0.2, 'clip_ratio_low': 0.2, 
               'clip_ratio_high': 0.28, 'clip_ratio_c': 3.0, 'loss_agg_mode': 'seq-mean-token-sum', 
               'entropy_coeff': 0.0, 'use_kl_loss': False, 'use_torch_compile': True, 
               'kl_loss_coef': 0.001, 'kl_loss_type': 'low_var_kl', 'ppo_epochs': 1, 
               'shuffle': False, 'ulysses_sequence_parallel_size': 2, 
               'checkpoint': {'contents': ['model', 'optimizer', 'extra']}, 
               'optim': {'lr': 1e-06, 'lr_warmup_steps': -1, 'lr_warmup_steps_ratio': 0.0, 
                         'min_lr_ratio': None, 'num_cycles': 0.5, 'warmup_style': 'constant', 
                         'total_training_steps': 125000, 'weight_decay': 0.01}, 
               'fsdp_config': {'wrap_policy': {'min_num_params': 0}, 'param_offload': True, 
                               'optimizer_offload': True, 'offload_policy': False, 
                               'reshard_after_forward': True, 'fsdp_size': -1}, 
               'grad_norm_threshold': 100000.0, 'use_remove_padding': True, 'use_fused_kernels': False}
world_size = torch.distributed.get_world_size()
device_mesh = create_device_mesh(world_size=world_size, fsdp_size=-1)
auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get('fsdp_config').get("wrap_policy", None), is_lora=False)
sharding_strategy = get_sharding_strategy(device_mesh)
mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=CPUOffload(offload_params=True),
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                # device_id=torch.cuda.current_device(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=device_mesh,
                forward_prefetch=False,
            )

output = actor_module_fsdp(
    input_ids=data["input_ids"],
    attention_mask=data["attention_mask"],
    # position_ids=data["position_ids"],
    use_cache=False,
)
print(output)