import math
import torch
import copy
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import OPTConfig
from modeling_rmt.long_mem_cross_attn import CrossAttentionMemory
from accelerate.logging import get_logger
from torch.profiler import profile, record_function, ProfilerActivity
import random
import evaluate
from huggingface_hub import PyTorchModelHubMixin

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class MemoryCell(torch.nn.Module):
    def __init__(self, base_model, num_mem_tokens, num_prepend):
        super().__init__()
        self.model = base_model
        self.n_prepend = num_prepend
        self.prepend_list = None
        if isinstance(self.model.config, OPTConfig):
            self.mem_map = MemoryMap(getattr(self.model.config, 'n_embd', self.model.config.word_embed_proj_dim))
            print("MemoryCell uses emb_dim:", self.mem_map.linear.in_features)
        else:
            self.mem_map = MemoryMap(getattr(self.model.config, 'n_embd', self.model.config.hidden_size))
            print("MemoryCell uses emb_dim:", self.mem_map.linear.in_features)
        self.create_memory(num_mem_tokens)

    def create_memory(self, num_mem_tokens):
        self.num_mem_tokens = num_mem_tokens
        embeddings = self.model.get_input_embeddings()
        if num_mem_tokens > 0:
            if isinstance(self.model.config, OPTConfig):
                memory_dim = getattr(self.model.config, 'n_embd', self.model.config.word_embed_proj_dim)
            else:
                memory_dim = getattr(self.model.config, 'n_embd', self.model.config.hidden_size)
            memory_weights = torch.randn((num_mem_tokens, memory_dim)) * embeddings.weight.data.std()
            self.register_parameter('memory', torch.nn.Parameter(memory_weights, requires_grad=True))
            # signifier_weights = torch.randn((1, memory_dim)) * embeddings.weight.data.std()
            # self.register_parameter('signifier', torch.nn.Parameter(signifier_weights, requires_grad=False))

        self.read_memory_position = range(num_mem_tokens)
        self.write_memory_position = range(-num_mem_tokens, 0)

    def set_memory(self, input_shape):
        memory = self.memory.repeat(input_shape[0], 1, 1)
        return memory

    def forward(self, input_ids, memory_state=None, prepend_state=None, browse=False, switch=False, **kwargs):
        input_ids = input_ids.cuda()
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                kwargs[k] = v.cuda()
        if memory_state is None and self.num_mem_tokens > 0:
            memory_state = self.set_memory(input_ids.shape)

        if switch:
            if self.mem_map.mode == 'forward':
                self.mem_map.set_mode('backward')
            else:
                self.mem_map.set_mode('forward')
            memory_state = self.mem_map(memory_state)

        seg_kwargs = self.process_input(input_ids, memory_state, prepend_state=prepend_state, **kwargs)

        out = self.model(**seg_kwargs)
        n_prepend = self.n_prepend//2 if browse else self.n_prepend
        out, new_memory_state = self.process_output(out, 0 if prepend_state is None else n_prepend, **kwargs)
        input_ids = input_ids.cpu()
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                kwargs[k] = v.cpu()
        return out, new_memory_state, self.prepend_list
    
    def generate(self, input_ids, memory_state, prepend_state, attention_mask, **generate_kwargs):
        if memory_state is None and self.num_mem_tokens > 0:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(input_ids, memory_state, prepend_state=prepend_state, generate=True, attention_mask=attention_mask)        
        out = self.model.generate(inputs_embeds=seg_kwargs['inputs_embeds'], attention_mask=seg_kwargs['attention_mask'], **generate_kwargs)
        return out

    def process_input(self, input_ids, memory_state, prepend_state=None, generate=False, **kwargs):
        seg_kwargs = dict(**kwargs)

        inputs_embeds = kwargs.get('inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        if prepend_state is not None:
            prepend_embeds = self.model.get_input_embeddings()(prepend_state)
            inputs_embeds = torch.cat([prepend_embeds, inputs_embeds], dim=1)
        if memory_state is not None:
            if generate:
                inputs_embeds = torch.cat([memory_state, inputs_embeds], dim=1)
            else:
                inputs_embeds = torch.cat([memory_state, inputs_embeds, memory_state], dim=1)
        
        self.prepend_list = input_ids[:,-self.n_prepend:] if self.n_prepend != 0 else None

        seg_kwargs['input_ids'] = None
        seg_kwargs['inputs_embeds'] = inputs_embeds
        if kwargs.get('attention_mask') is not None:
            seg_kwargs['attention_mask'] = self.pad_attention_mask(kwargs['attention_mask'], inputs_embeds.shape, 0 if prepend_state is None else self.n_prepend, generate)
        seg_kwargs['output_hidden_states'] = True
        return seg_kwargs
    
    def pad_attention_mask(self, attention_mask, shape, n_prepend, generate=False):
        if self.num_mem_tokens in {0, None}:
            mask = torch.ones(*shape[:2], dtype=torch.int64).to(attention_mask.device)
            mask[:, (n_prepend):] = attention_mask
            return mask
        else:
            mask = torch.ones(*shape[:2], dtype=torch.int64).to(attention_mask.device)
            if generate:
                mask[:, (self.num_mem_tokens+n_prepend):] = attention_mask
            else:
                mask[:, (self.num_mem_tokens+n_prepend):-self.num_mem_tokens] = attention_mask
            return mask
    
    def process_output(self, model_outputs, n_prepend, **kwargs):
        if self.num_mem_tokens not in {0, None}:
            out = CausalLMOutputWithCrossAttentions()
            memory_state = model_outputs.hidden_states[-1][:, -self.num_mem_tokens:]
            out['logits'] = model_outputs.logits[:, (self.num_mem_tokens+n_prepend):-self.num_mem_tokens]
            out['logits'] = out['logits'].cpu()

            if kwargs.get('output_hidden_states'):
                out['hidden_states'] = [lh[:, (self.num_mem_tokens+n_prepend):-self.num_mem_tokens] for lh in model_outputs.hidden_states]
            if kwargs.get('output_attentions'):
                out['attentions'] = model_outputs['attentions']
        else:
            out = CausalLMOutputWithCrossAttentions()
            memory_state = None
            out['logits'] = model_outputs.logits[:, (n_prepend):]
            
            if kwargs.get('output_hidden_states'):
                out['hidden_states'] = [lh[:, (n_prepend):] for lh in model_outputs.hidden_states]
            if kwargs.get('output_attentions'):
                out['attentions'] = model_outputs['attentions']
            
        return out, memory_state 

class SegmentIterator:
    def __init__(self, **kwargs):
        self.iter_content = kwargs
        self.pointer = 0
        self.empty = False
    
    def next(self, segment_length):
        segment = {}
        for k, tensor in self.iter_content.items():
            if tensor is not None:
                if self.pointer >= tensor.shape[1]:
                    self.empty = True
                    return None
                segment[k] = tensor[:, self.pointer:self.pointer+segment_length]
        
        self.pointer += segment_length
        return segment
    
    def is_empty(self):
        for k, tensor in self.iter_content.items():
            if tensor is not None:
                if self.pointer >= tensor.shape[1]:
                    self.empty = True
                    return True
        return False

class MemoryMap(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.mode = 'forward' # 'backward' do inverse mapping
        # try linear projection first
        self.linear = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        self.inv_linear = torch.nn.Linear(emb_dim, emb_dim, bias=False)

    def set_mode(self, mode):
        self.mode = mode
    
    def forward(self, inputs):
        if self.mode == 'forward':
            return self.linear(inputs)
        else:
            return self.inv_linear(inputs)

class RecurrentWrapper(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, memory_cell, emb=None, word_emb_dim=4096, hidden_dim=4096, ltm_context=100, **rmt_kwargs):
        super().__init__()
        self.memory_cell = memory_cell
        self.rmt_config = rmt_kwargs
        self.ltm_context = ltm_context
        self.logger = get_logger('')
        if emb is not None:
            memory_weights = torch.randn((1, word_emb_dim)) * emb.weight.data.std()
            self.register_parameter('mem', torch.nn.Parameter(memory_weights, requires_grad=True))
            self.cross_attn = CrossAttentionMemory(word_emb_dim, hidden_dim)
        else:
            self.cross_attn = None
        
        self.rouge = evaluate.load('rouge')
        self.f1 = evaluate.load("f1")
        print("RecurrentWrapper word_emb_dim:", word_emb_dim, "hidden_dim:", hidden_dim)
        self.memory_attn_manager = MemoryAttentionManager(embed_dim=word_emb_dim, num_heads=8)

    def forward(self, 
            input_ids, 
            labels=None, 
            labels_mask=None, 
            inputs_embeds=None, 
            attention_mask=None, 
            mask_size=None,  # Size of the attention mask used to compute the loss, it should be the length of the labels. If it's None, then self.mask_size is used. 
            output_attentions=None, 
            output_hidden_states=None, 
            sum_fraction=0.5,
            segment_size=1022, 
            extra_size=16, 
            mode='train', 
            prof=False,
            switch_at=-1,
            pos_mask=None,
            **kwargs
        ):

        mask_size = self.rmt_config.get('mask_size') if mask_size is None else mask_size

        memory_state = None
        prepend_state = None
        seg_iter = SegmentIterator(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        cell_outputs = []
        n_cell_out = self.rmt_config.get('n_cell_out')
        memory_seq = None

        total_hist = []

        seg_num = 0
        segment = None
        browse_count = 0
        total_ret_loss = 0

        while True:
            segment = seg_iter.next(segment_size)
            if segment is None:
                break

            browse = False
            if self.cross_attn is not None:
                s_mem = self.mem.repeat(segment['input_ids'].shape[0], 1, 1)
                seg = copy.deepcopy(segment)
                seg['input_ids'] = seg['input_ids'][:,:int(round(segment_size * sum_fraction))]
                seg['attention_mask'] = seg['attention_mask'][:,:int(round(segment_size * sum_fraction))]
                _, q_mem, _ = self.memory_cell(**seg, memory_state=s_mem)
                browse_thres = 0
                if mode == 'test':
                    browse_thres = 3
                pos_mask_n = None
                if pos_mask is not None:
                    pos_mask_n = pos_mask[..., :seg_num]
                if seg_iter.is_empty():
                    last_seg = True
                else:
                    last_seg = False
                memory_state, hist, browse, ret_loss = self.cross_attn(memory_seq, q_mem, mode, seg_num if seg_num < self.ltm_context else self.ltm_context, browse_thres, pos_mask_n, last_seg)
                if seg_iter.is_empty():
                    total_ret_loss += ret_loss
                if hist is not None:
                    total_hist.extend(hist)
            
            if (browse and mode == 'test') or (switch_at <= seg_num and switch_at >= 0):
                # proceed extra tokens
                browse_count += 1
                extra_seg = seg_iter.next(extra_size)
                if extra_seg is not None:
                    for k, tensor in extra_seg.items():
                        segment[k] = torch.cat([segment[k], tensor], dim=1)

            browse = browse or mode == 'browse' or (switch_at <= seg_num and switch_at >= 0)

            if prof:
                with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_stack=True) as prof_m:
                    with record_function("model_inference"):
                        cell_out, memory_state, prepend_state = self.memory_cell(**segment, memory_state=memory_state, prepend_state=prepend_state, browse=browse, switch=(switch_at==seg_num), output_hidden_states=True)
                
                with open('model_profile_dump.txt', 'w') as file:
                    file.write(prof_m.key_averages().table(sort_by="cuda_time_total"))
                
                prof_m.export_chrome_trace("model_trace.json")
                exit(0)
            else:
                cell_out, memory_state, prepend_state = self.memory_cell(**segment, memory_state=memory_state, prepend_state=prepend_state, browse=browse, switch=(switch_at==seg_num), output_hidden_states=True)

            cell_outputs.append(cell_out)
            if len(cell_outputs) > n_cell_out:
                cell_outputs.pop(0)
            
            if self.cross_attn is not None:
                device = memory_state.device  # Ensure device consistency
                if memory_seq is None:
                    memory_seq = memory_state.detach().to(device)
                else:
                    memory_seq = torch.cat([memory_seq, memory_state.detach().to(device)], dim=1)
                #print("memory_state shape:", memory_state.shape)
                #print("memory_seq shape before cat:", memory_seq.shape if memory_seq is not None else None)
                #print("memory_seq shape after cat:", memory_seq.shape)
                #print("Current memory_seq length:", memory_seq.shape[1], "ltm_context:", self.ltm_context)
                if memory_seq.shape[1] > self.ltm_context:
                    memory_seq = memory_seq.to(device)  # Ensure memory_seq is on the correct device
                    memory_seq, removed_idx = self.memory_attn_manager(memory_seq)
                    #print("Removed memory index (per batch):", removed_idx)

            if memory_state is not None:
                self.manage_gradients(memory_state, seg_num)

            seg_num+=1
        
        out = self.process_outputs(cell_outputs, labels=labels, 
                                   labels_mask=labels_mask,
                                   output_attentions=output_attentions, 
                                   output_hidden_states=output_hidden_states,
                                   mask_size=mask_size,
                                   ret_loss=total_ret_loss)
        return out, total_hist
    
    def generate(self, input_ids, attention_mask, segment_size, mem_seq=None, **generate_kwargs):
        memory_state = None
        prepend_state = None
        segmented = self.segment(segment_size, input_ids=input_ids, attention_mask=attention_mask)
        memory_seq = mem_seq
        enum_seg = None
        if memory_seq is not None:
            iter_len = (len(segmented)-1-memory_seq.shape[1])
            enum_seg = segmented[-(iter_len+1):-1]
            n_prepend = self.memory_cell.n_prepend
            prepend_state = segmented[-(iter_len+2)]['input_ids'][:,-n_prepend:].cuda()
        else:
            enum_seg = segmented[:-1]

        for seg_num, segment in enumerate(enum_seg):
            for k, v in segment.items():
                segment[k] = v.cuda()
            
            if self.cross_attn is not None:
                s_mem = self.mem.repeat(segment['input_ids'].shape[0], 1, 1)
                seg = copy.deepcopy(segment)
                seg['input_ids'] = seg['input_ids'][:,:(segment_size//2)]
                seg['attention_mask'] = seg['attention_mask'][:,:(segment_size//2)]
                _, q_mem, _ = self.memory_cell(**seg, memory_state=s_mem)
                memory_state, _, _, _ = self.cross_attn(memory_seq, q_mem, 'generate', seg_num if seg_num < self.ltm_context else self.ltm_context, generate=True)
            
            with torch.no_grad():
                _, memory_state, prepend_state = self.memory_cell(**segment, memory_state=memory_state, prepend_state=prepend_state, output_hidden_states=True)
            
            if self.cross_attn is not None:
                if memory_seq is None:
                    memory_seq = memory_state.cpu()
                else:
                    memory_seq = torch.cat([memory_seq, memory_state.cpu()], dim=1)
                    if memory_seq.shape[1] > self.ltm_context:
                        #memory_seq = memory_seq[:,-self.ltm_context:,:] FIFO
                        memory_seq, removed_idx = self.memory_attn_manager(memory_seq) #Our New Memory Attention Manager
                        #print("Removed memory index (per batch):", removed_idx)

            for k, v in segment.items():
                segment[k] = v.cpu()
        
        final_segment = segmented[-1]
        for k, v in final_segment.items():
            final_segment[k] = v.cuda()

        seg_num = len(segmented) - 1
        if self.cross_attn is not None:
            s_mem = self.mem.repeat(final_segment['input_ids'].shape[0], 1, 1)
            seg = copy.deepcopy(final_segment)
            seg['input_ids'] = seg['input_ids'][:,:(segment_size//2)]
            seg['attention_mask'] = seg['attention_mask'][:,:(segment_size//2)]
            _, q_mem, _ = self.memory_cell(**seg, memory_state=s_mem)
            memory_state, _, _, _ = self.cross_attn(memory_seq, q_mem, 'generate', seg_num if seg_num < self.ltm_context else self.ltm_context, generate=True)
        
        out = self.memory_cell.generate(**final_segment, memory_state=memory_state, prepend_state=prepend_state, **generate_kwargs)

        return out

    def segment(self, segment_size, **kwargs):
        segments = []
        for k, tensor in kwargs.items():
            if tensor is not None:
                k_segments = self.split_tensor(tensor, segment_size)
                for s, k_seg in enumerate(k_segments):
                    if s < len(segments):
                        segments[s][k] = k_seg
                    else:
                        segments.append({k: k_seg})

        return segments
    
    def split_tensor(self, tensor, segment_size):
        align = self.rmt_config.get('segment_alignment')
        if align in {'left', None}:
            split_inds = list(range(0, tensor.shape[1], segment_size)) + [tensor.shape[1]]
            segments = [tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])]
        elif align in {'right', None}:
            split_inds = (list(range(tensor.shape[1], 0, -segment_size)) + [0])[::-1]
            segments = [tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])]
        elif align == 'center':
            n_seg = math.ceil(tensor.shape[1] / segment_size)
            segments = torch.chunk(tensor, n_seg, dim=1)
        else:
            raise NotImplementedError
        return segments

    def process_outputs(self, cell_outputs, **kwargs):
        out = CausalLMOutputWithCrossAttentions()
        full_logits = torch.cat([o.logits for o in cell_outputs], dim=1)
        full_hidden_states = tuple([torch.cat(layer_hs, dim=1) for layer_hs in zip(*[o.hidden_states for o in cell_outputs])])
        
        mask_size = kwargs.get('mask_size')
        ret_loss = kwargs.get('ret_loss')

        labels = kwargs.get('labels')
        if labels is not None:
            if labels.shape[1] <= mask_size:
                mask_size = labels.shape[1]-1
            shift_labels = labels[..., -mask_size:].contiguous()
            shift_logits = full_logits[..., -(mask_size+1):-1, :].contiguous()
            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            
            loss_fct = CrossEntropyLoss()
            gen_loss = loss_fct(flat_logits.cuda(), flat_labels.cuda())
            out['loss'] = gen_loss + 0.05 * ret_loss
            out['ppl'] = torch.exp(gen_loss)
            out['ret_loss'] = ret_loss

            # Compute the F1 score
            predictions = torch.argmax(torch.Tensor(flat_logits), dim=-1)
            flat_labels_cpu, predictions_cpu = flat_labels.detach().cpu(), predictions.detach().cpu()
            precision, recall, f1, _ = precision_recall_fscore_support(flat_labels_cpu, predictions_cpu, average='weighted')
            accuracy = accuracy_score(flat_labels_cpu, predictions_cpu)
            out['f1'] = {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}
        else:
            out['loss'] = 0
            out['ppl'] = 0
            out['ret_loss'] = 0

        out['logits'] = full_logits
        segment_keys = ['loss', 'logits']
        if kwargs.get('output_attentions'):
            segment_keys.append('attentions')
        if kwargs.get('output_hidden_states'):
            segment_keys.append('hidden_states')
            out['hidden_states'] = full_hidden_states

        for seg_num, o in enumerate(cell_outputs):
            for key, value in o.items():
                if any(sk in key for sk in segment_keys):
                    out[f'{key}_{seg_num}'] = value

        return out 
        
    def manage_gradients(self, memory_state, seg_num):
        k2, max_n_segments = self.rmt_config.get('k2'), self.rmt_config.get('max_n_segments')
        if seg_num == 0 \
            or k2 in {-1, None} \
            or seg_num + k2 > max_n_segments:
                return True
        
        memory_state = memory_state.detach()
        return False

import torch.nn as nn
import torch.nn.functional as F

class MemoryAttentionManager(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, memory_seq):
        # memory_seq: (batch, seq_len, embed_dim)
        attn_output, attn_weights = self.self_attn(memory_seq, memory_seq, memory_seq)
        # attn_weights: (batch, num_heads, seq_len, seq_len)
        avg_attn = attn_weights.mean(dim=1).mean(dim=1)  # (batch, seq_len)
        batch_size, seq_len = avg_attn.shape if avg_attn.dim() == 2 else (1, avg_attn.shape[0])

        # Always get remove_idx as a 1D tensor of length batch_size
        if avg_attn.dim() == 1:
            remove_idx = avg_attn.argmin(dim=0, keepdim=True)
        else:
            remove_idx = avg_attn.argmin(dim=1)

        # Ensure remove_idx is 1D and matches batch size
        remove_idx = remove_idx.view(-1)
        
        #print(f"[MemoryAttentionManager] avg_attn: {avg_attn.detach().cpu().numpy()}")
        #print(f"[MemoryAttentionManager] remove_idx: {remove_idx.detach().cpu().numpy()}")

        new_memory_seq = []
        for b in range(memory_seq.size(0)):
            mask = torch.ones(memory_seq.size(1), dtype=torch.bool, device=memory_seq.device)
            idx = remove_idx[b] if remove_idx.numel() > 1 else remove_idx[0]
            # Clamp idx to valid range
            idx = min(idx.item(), memory_seq.size(1) - 1)
            mask[idx] = False
            filtered = memory_seq[b][mask]
            # Pad if needed
            if filtered.size(0) < memory_seq.size(1):
                filtered = F.pad(filtered, (0,0,0,memory_seq.size(1)-filtered.size(0)))
            new_memory_seq.append(filtered)
        new_memory_seq = torch.stack(new_memory_seq)
        return new_memory_seq, remove_idx