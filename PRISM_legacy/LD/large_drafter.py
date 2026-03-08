import copy
import json
import torch
from torch import nn
from typing import Optional, List
from transformers import AutoTokenizer

from model.cnets_hass import Model
from model.cnets import Model as InferModel
from model.cnets_hass import LlamaDecoderLayer
from model.cnets import LlamaDecoderLayer as InferLlamaDecoderLayer
from model.configs import EConfig
from model.ea_model import EaModel

from LD.moes import DCMoe


class LlamaDecoderLayerMoE(LlamaDecoderLayer):
    def __init__(self, config, index, moe_config=None):
        super().__init__(config=config, index=index)

        if moe_config:
            # redefine self.mlp
            self.mlp = DCMoe(**moe_config)


class InferLlamaDecoderLayerMoE(InferLlamaDecoderLayer):
    def __init__(self, config, index, moe_config=None):
        super().__init__(config=config, index=index)

        if moe_config:
            # redefine self.mlp
            self.mlp = DCMoe(**moe_config)


class LargeDrafter(Model):
    def __init__(
            self,
            config,
            load_emb=False,
            path=None,
            bias=True,
            total_tokens=63,
            depth=5,
            top_k=8,
            threshold=1.0,
            hass_path: str = None
    ):
        eagle_config = EConfig(**config["eagle_config"])
        super().__init__(
            config=eagle_config,
            load_emb=load_emb,
            path=path,
            bias=bias,
            total_tokens=total_tokens,
            depth=depth,
            top_k=top_k,
            threshold=threshold,
        )

        self.num_steps = config["num_steps"]  # the logical number of step models
        self.num_step_models = config["num_step_models"]  # the physical number of step models
        self.step_mapping = config["step_mapping"]  # the map of steps to step models

        self.stepModels = nn.ModuleList()
        moe_config = config.get("moe_config", None)
        for _ in range(self.num_step_models):
            stepModel = nn.ModuleList(
                [
                    LlamaDecoderLayerMoE(eagle_config, index, moe_config)
                    for index in range(eagle_config.num_hidden_layers)
                ]
            )
            self.stepModels.append(stepModel)

        self.stepFCs = nn.ModuleList()
        for _ in range(self.num_step_models):
            self.stepFCs.append(
                nn.Linear(
                    2 * eagle_config.hidden_size,
                    eagle_config.hidden_size,
                    bias=bias,
                )
            )

        self.use_adapter = config.get("use_adapter", False)
        if self.use_adapter:
            self.stepAdapters = nn.ModuleList()
            for _ in range(self.num_step_models):
                self.stepAdapters.append(
                    nn.Linear(
                        eagle_config.hidden_size,
                        eagle_config.hidden_size,
                        bias=bias,
                    )
                )

        if hass_path:
            self.load_state_dict(torch.load(hass_path, map_location='cpu', weights_only=True), strict=False)
            for step in range(1, self.num_step_models):
                self.stepModels[step] = copy.deepcopy(self.stepModels[0])
                self.stepFCs[step] = copy.deepcopy(self.stepFCs[0])

        # replace
        self.current_step = 0
        del self.layers
        self.layers = self.stepModels[self.step_mapping[str(self.current_step)]]
        del self.fc
        self.fc = self.stepFCs[self.step_mapping[str(self.current_step)]]
        if self.use_adapter:
            self.stepAdapter = self.stepAdapters[self.step_mapping[str(self.current_step)]]

    def forward(
            self,
            hidden_states,
            input_ids,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            std=None,
            q_hidden_states=None,
    ):
        results = super().forward(
            hidden_states,
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            std=std,
            q_hidden_states=q_hidden_states,
        )

        if use_cache:
            output_hidden_states = results[0]
        else:
            output_hidden_states = results

        if self.use_adapter:
            sample_hidden_states = self.stepAdapter(output_hidden_states)
        else:
            sample_hidden_states = None

        self.switch_step()

        if use_cache:
            return results[0], results[1], sample_hidden_states

        return results, sample_hidden_states

    def switch_step(self):
        if self.current_step < self.num_steps - 1:
            self.current_step += 1
            self.layers = self.stepModels[self.step_mapping[str(self.current_step)]]
            self.fc = self.stepFCs[self.step_mapping[str(self.current_step)]]
            if self.use_adapter:
                self.stepAdapter = self.stepAdapters[self.step_mapping[str(self.current_step)]]

    def reset_step(self):
        self.current_step = 0
        self.layers = self.stepModels[self.step_mapping[str(self.current_step)]]
        self.fc = self.stepFCs[self.step_mapping[str(self.current_step)]]
        if self.use_adapter:
            self.stepAdapter = self.stepAdapters[self.step_mapping[str(self.current_step)]]


class InferLargeDrafter(InferModel):
    def __init__(
            self,
            config,
            load_emb=False,
            path=None,
            bias=True,
            total_tokens=63,
            depth=5,
            top_k=8,
            threshold=1.0,
    ):
        eagle_config = EConfig(**config["eagle_config"])
        super().__init__(
            config=eagle_config,
            load_emb=load_emb,
            path=path,
            bias=bias,
            total_tokens=total_tokens,
            depth=depth,
            top_k=top_k,
            threshold=threshold,
        )

        self.num_steps = config["num_steps"]  # the logical number of step models
        self.num_step_models = config["num_step_models"]  # the physical number of step models
        self.step_mapping = config["step_mapping"]  # the map of steps to step models

        self.stepModels = nn.ModuleList()
        moe_config = config.get("moe_config", None)
        for _ in range(self.num_step_models):
            stepModel = nn.ModuleList(
                [
                    InferLlamaDecoderLayerMoE(eagle_config, index, moe_config)
                    for index in range(eagle_config.num_hidden_layers)
                ]
            )
            self.stepModels.append(stepModel)

        self.stepFCs = nn.ModuleList()
        for _ in range(self.num_step_models):
            self.stepFCs.append(
                nn.Linear(
                    2 * eagle_config.hidden_size,
                    eagle_config.hidden_size,
                    bias=bias,
                )
            )

        self.use_adapter = config.get("use_adapter", False)
        if self.use_adapter:
            self.stepAdapters = nn.ModuleList()
            for _ in range(self.num_step_models):
                self.stepAdapters.append(
                    nn.Linear(
                        eagle_config.hidden_size,
                        eagle_config.hidden_size,
                        bias=bias,
                    )
                )

        # replace
        self.current_step = 0
        self.layers = nn.ModuleList([InferLlamaDecoderLayerMoE(eagle_config, index, moe_config)
                                     for index in range(eagle_config.num_hidden_layers)])
        self.fc = nn.Linear(2 * eagle_config.hidden_size, eagle_config.hidden_size, bias=bias)
        if self.use_adapter:
            self.stepAdapter = nn.Linear(eagle_config.hidden_size, eagle_config.hidden_size, bias=bias)

    def forward(
            self,
            hidden_states,
            input_ids,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            std=None,
    ):
        results = super().forward(
            hidden_states,
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            std=std,
        )

        if use_cache:
            output_hidden_states = results[0]
        else:
            output_hidden_states = results

        if self.use_adapter:
            sample_hidden_states = self.stepAdapter(output_hidden_states)
        else:
            sample_hidden_states = None

        self.switch_step()

        if use_cache:
            return results[0], results[1], sample_hidden_states

        return results, sample_hidden_states

    def switch_step(self):
        if self.current_step < self.num_steps - 1:
            self.current_step += 1
            self.layers = self.stepModels[self.step_mapping[str(self.current_step)]]
            self.fc = self.stepFCs[self.step_mapping[str(self.current_step)]]
            if self.use_adapter:
                self.stepAdapter = self.stepAdapters[self.step_mapping[str(self.current_step)]]

    def reset_step(self):
        self.current_step = 0
        self.layers = self.stepModels[self.step_mapping[str(self.current_step)]]
        self.fc = self.stepFCs[self.step_mapping[str(self.current_step)]]
        if self.use_adapter:
            self.stepAdapter = self.stepAdapters[self.step_mapping[str(self.current_step)]]

    @torch.no_grad()
    def topK_genrate(self, hidden_states, input_ids, head, logits_processor):
        self.reset_step()

        input_ids = input_ids.to(hidden_states.device)
        total_tokens = self.total_tokens
        depth = self.depth
        top_k = self.top_k

        sample_token = input_ids[:, -1]

        scores_list = []
        parents_list = []
        ss_token = []

        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)

        len_posi = input_ids.shape[1]
        self.reset()

        # with Timer("draft many"):
        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            out_hidden, past_key_values, sample_hidden = self(
                hidden_states,
                input_ids=input_ids[:, kv_len:],
                past_key_values=self.stable_kv,
                use_cache=True,
            )
        else:
            out_hidden, past_key_values, sample_hidden = self(
                hidden_states, input_ids=input_ids, use_cache=True
            )
        self.stable_kv = past_key_values
        last_hidden = out_hidden[:, -1]

        if not self.use_adapter:
            last_headout = head(last_hidden)
        else:
            sample_hidden = sample_hidden[:, -1, :]
            last_headout = head(sample_hidden)

        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        ss_token.append(topk_index)
        input_ids = topk_index
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)

        # 4
        for i in range(depth):
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids
            # with Timer("draft one"):
            out_hidden, past_key_values, sample_hidden = self(
                input_hidden,
                input_ids=input_ids,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=True,
            )
            len_posi += 1

            # with Timer("sort1"):
            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = topk_cs_index + bias
            parents_list.append(parents)

            if not self.use_adapter:
                last_headout = head(out_hidden[0])
            else:
                last_headout = head(sample_hidden[0])

            last_p = self.logsoftmax(last_headout)

            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values

            cu_scores = topk_p + scores[:, None]

            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            out_ids = topk_cs_index // top_k
            input_hidden = out_hidden[:, out_ids]
            # with Timer("2index"):
            #     in_ids = topk_cs_index % top_k
            #     input_ids = topk_index[out_ids, in_ids][None]
            # with Timer("1index"):
            input_ids = topk_index.view(-1)[topk_cs_index][None]
            # print(input_ids.equal(input_ids0))

            ss_token.append(topk_index)
            scores_list.append(cu_scores)
            tree_mask = torch.cat(
                (tree_mask[:, :, out_ids], self.tree_mask_init), dim=3
            )

            # if self.threshold < 0 and cu_scores.max() < self.threshold:
            #     break

        # del parents_list,scores_list,ss_token
        # return draft_tokens, mask_index,tree_mask,tree_position_ids

        # with Timer("post"):

        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(
            top_scores_index, draft_parents - 1, right=False
        )
        # mask_index[(top_scores_index[mask_index]!=draft_parents - 1)]=-1
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()
        # with Timer("mask"):
        tree_mask = torch.eye(total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for i in range(total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

        # with Timer("mask1"):
        #     tree_mask0 = [[False for _ in range(total_tokens + 1)] for _ in range(total_tokens + 1)]
        #     tree_mask0[0][0] = True
        #     for i in range(total_tokens):
        #         #tree_mask0[i + 1][0]=True
        #         tree_mask0[i + 1][i + 1] = True
        #         p=mask_index_list[i]
        #         tree_mask0[i + 1][p] = True
        #         while p:
        #             p=mask_index_list[p-1]
        #             tree_mask0[i + 1][p] = True
        #     tree_mask0 = torch.tensor(tree_mask0, dtype=torch.bool)
        #
        # print(tree_mask0.equal(tree_mask))
        tree_position_ids = torch.sum(tree_mask, dim=1) - 1

        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]

        del parents_list, scores_list, ss_token, ss_token_list, draft_parents

        # with Timer("retrieve"):

        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = total_tokens - noleaf_num

        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()

        for i in range(total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        if logits_processor is not None:
            maxitem = total_tokens + 5

            def custom_sort(lst):
                # sort_keys=[len(list)]
                sort_keys = []
                for i in range(len(lst)):
                    sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
                return sort_keys

            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        del (
            mask_index,
            mask_index_list,
            noleaf_index,
            noleaf_num,
            leaf_num,
            max_depth,
            rid,
        )
        tree_position_ids = tree_position_ids.to(hidden_states.device)

        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids


class LDModel(EaModel):
    def __init__(
            self,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
    ):

        nn.Module.__init__(self)
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name_or_path, use_fast=False
        )

        with open(ea_model_path, "r") as f:
            config = json.loads(f.read())
        try:
            bias = config["bias"]
        except:
            bias = True

        self.ea_layer = InferLargeDrafter(
            config,
            bias=bias,
            total_tokens=total_token,
            depth=depth,
            top_k=top_k,
            threshold=threshold,
        )

        low_memory = False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device != base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device
        else:
            self.ea_layer.diff_device = False

        self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()
