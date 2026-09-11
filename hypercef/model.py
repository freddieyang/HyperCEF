"""HyperCEF model for reaction-feasibility prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv

from .reaction import reaction_metabolites

try:
    from torch_scatter import scatter_mean

    SCATTER_AVAILABLE = True
except ImportError:  # pragma: no cover - optional acceleration dependency
    scatter_mean = None
    SCATTER_AVAILABLE = False


MODEL_NAME = "HyperCEF"
MODEL_VERSION = "1.0.0"
FINAL_MOLECULAR_FUSION_ALPHA = 0.5
OUTPUT_KEYS = (
    "final_logit",
    "chemistry_logit",
    "host_logit",
    "topology_logit",
    "memory_logit",
    "topology_weight",
    "stats",
    "invalid_probability",
    "semantic_correction",
    "isolation_gate",
)


class HyperCEF(nn.Module):
    """Joint chassis-independent and chassis-based reaction predictor.

    The molecular branch represents an unordered set of reaction participants.
    The chassis branch encodes a fold-specific reaction hypergraph and calibrates
    its score against native and chassis-absent reaction references.  A bounded,
    stop-gradient molecular residual supplies one-way information to the chassis
    branch while preserving the meaning of the molecular prediction.
    """

    model_name = MODEL_NAME
    model_version = MODEL_VERSION
    compact_formula = "s_fuse = s_emb + 0.5 * tanh(stopgrad(z_ind))"
    output_keys = OUTPUT_KEYS

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout=0.05,
        num_layers=2,
        heads=2,
        max_boundary_prototypes=512,
        boundary_topk=8,
        use_identity_skip=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.num_layers = num_layers
        self.heads = heads
        if int(num_layers) != 2:
            raise ValueError("HyperCEF uses exactly two hypergraph-convolution layers")
        self.max_boundary_prototypes = int(max_boundary_prototypes)
        self.boundary_topk = int(boundary_topk)
        self.use_identity_skip = bool(use_identity_skip)

        self.chem_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        with torch.no_grad():
            flat_dim = self.chem_cnn(torch.zeros(1, 1, in_channels)).view(1, -1).size(1)
        self.feature_encoder = nn.Sequential(
            nn.Linear(flat_dim, hidden_channels),
            nn.ReLU(),
        )

        self.topo_reaction_norm = nn.LayerNorm(hidden_channels)
        self.chem_reaction_norm = nn.LayerNorm(hidden_channels)
        gate_in = hidden_channels * 2 + 3
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(gate_in),
            nn.Linear(gate_in, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Sigmoid(),
        )

        classifier_in = hidden_channels * 3 + 3
        self.query_proj = nn.Linear(hidden_channels, hidden_channels)
        self.memory_key = nn.Linear(hidden_channels, hidden_channels)
        self.memory_value = nn.Linear(hidden_channels, hidden_channels)
        self.memory_classifier = nn.Sequential(
            nn.LayerNorm(classifier_in),
            nn.Linear(classifier_in, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Linear(hidden_channels // 2, out_channels),
        )

        # Active HGNN encoder used by the host-memory branch.
        self.hgnn_conv1 = HypergraphConv(in_channels, hidden_channels, use_attention=False)
        self.hgnn_conv2 = HypergraphConv(hidden_channels, hidden_channels, use_attention=False)
        self.top_k_memory = 64
        self.singleton_penalty = 1.75
        self.low_order_penalty = 0.75

        chunks = min(32, in_channels)
        while chunks > 1 and in_channels % chunks != 0:
            chunks -= 1
        self.raw_chunks = chunks
        self.raw_chunk_dim = in_channels // chunks
        self.raw_emb = nn.Linear(self.raw_chunk_dim, hidden_channels)
        raw_heads = min(4, max(1, heads * 2))
        while hidden_channels % raw_heads != 0 and raw_heads > 1:
            raw_heads -= 1
        raw_layer = nn.TransformerEncoderLayer(
            hidden_channels,
            raw_heads,
            hidden_channels * 2,
            dropout,
            batch_first=True,
            activation="gelu",
        )
        self.raw_transformer = nn.TransformerEncoder(raw_layer, num_layers=2)
        self.raw_head = nn.Linear(hidden_channels, out_channels)

        self.protected_hgnn_conv1 = HypergraphConv(in_channels, hidden_channels, use_attention=False)
        self.protected_hgnn_conv2 = HypergraphConv(hidden_channels, hidden_channels, use_attention=False)
        self.protected_hgnn_classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels),
        )
        self.protected_identity_norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        self.protected_identity_skip = nn.Linear(in_channels, hidden_channels, bias=False)
        nn.init.normal_(self.protected_identity_skip.weight, std=in_channels**-0.5)
        for param in self.protected_identity_skip.parameters():
            param.requires_grad = False
        self.register_buffer("protected_identity_scale", torch.tensor(0.05))
        self.register_buffer(
            "fixed_molecular_fusion_alpha",
            torch.tensor(float(FINAL_MOLECULAR_FUSION_ALPHA)),
        )

        self.insertion_support_gate = nn.Sequential(
            nn.LayerNorm(7),
            nn.Linear(7, hidden_channels // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 4, out_channels),
            nn.Sigmoid(),
        )
        self.set_metabolite_proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        set_heads = min(max(1, heads), 4)
        while hidden_channels % set_heads != 0 and set_heads > 1:
            set_heads -= 1
        self.set_query = nn.Parameter(torch.zeros(1, 1, hidden_channels))
        nn.init.normal_(self.set_query, std=0.02)
        self.set_attention = nn.MultiheadAttention(
            hidden_channels,
            num_heads=set_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.set_head = nn.Sequential(
            nn.LayerNorm(hidden_channels + 2),
            nn.Linear(hidden_channels + 2, hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )
        self.chemset_scale = nn.Parameter(torch.tensor(0.05))

        self.rare_floor_raw = nn.Parameter(torch.tensor(1.0))
        self.core_cap_raw = nn.Parameter(torch.tensor(1.25))
        self.node_logit_mix = nn.Parameter(torch.tensor(-0.25))

        self.native_boundary_sets = []
        self.hard_boundary_sets = []
        self.counterfactual_rescue_scale = nn.Parameter(torch.tensor(0.25))
        self.counterfactual_veto_scale = nn.Parameter(torch.tensor(-8.0))
        self.counterfactual_temperature_raw = nn.Parameter(torch.tensor(1.50))
        self.counterfactual_tail_loss_weight = 0.20
        self.counterfactual_core_loss_weight = 0.00

    @staticmethod
    def _canonical_reaction(reaction):
        return tuple(reaction_metabolites(reaction))

    def _select_reaction_sets(self, reactions):
        unique = sorted({self._canonical_reaction(reaction) for reaction in reactions})
        unique = [reaction for reaction in unique if reaction]
        if len(unique) <= self.max_boundary_prototypes:
            return unique
        step = len(unique) / float(self.max_boundary_prototypes)
        return [unique[min(int(i * step), len(unique) - 1)] for i in range(self.max_boundary_prototypes)]

    def fit_topology_boundary_prototypes(self, x, edge_index, native_reactions, hard_negative_reactions):
        del x, edge_index
        self.native_boundary_sets = self._select_reaction_sets(native_reactions)
        self.hard_boundary_sets = self._select_reaction_sets(hard_negative_reactions)

    def encode(self, x, edge_index):
        h_topo = F.relu(self.hgnn_conv1(x, edge_index))
        h_topo = self.hgnn_conv2(h_topo, edge_index)
        h_chem = self.chem_cnn(x.unsqueeze(1)).view(x.size(0), -1)
        h_chem = self.feature_encoder(h_chem)
        return h_topo, h_chem

    def _manual_mean(self, feats, batch_idx, num_groups):
        out = torch.zeros(num_groups, feats.size(1), device=feats.device, dtype=feats.dtype)
        counts = torch.zeros(num_groups, 1, device=feats.device, dtype=feats.dtype)
        out.index_add_(0, batch_idx, feats)
        counts.index_add_(
            0,
            batch_idx,
            torch.ones(batch_idx.numel(), 1, device=feats.device, dtype=feats.dtype),
        )
        return out / counts.clamp(min=1.0)

    def _query_membership(self, reaction_list, num_nodes, device):
        met_indices = []
        batch_indices = []
        for rxn_idx, reaction in enumerate(reaction_list):
            for met_idx in reaction_metabolites(reaction):
                met_idx = int(met_idx)
                if 0 <= met_idx < num_nodes:
                    met_indices.append(met_idx)
                    batch_indices.append(rxn_idx)
        if not met_indices:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty
        return (
            torch.tensor(met_indices, dtype=torch.long, device=device),
            torch.tensor(batch_indices, dtype=torch.long, device=device),
        )

    def _pool_reactions(self, node_embeddings, reaction_list):
        met_idx, batch_idx = self._query_membership(
            reaction_list,
            node_embeddings.size(0),
            node_embeddings.device,
        )
        if met_idx.numel() == 0:
            return torch.zeros(
                len(reaction_list),
                node_embeddings.size(1),
                device=node_embeddings.device,
                dtype=node_embeddings.dtype,
            )
        feats = node_embeddings[met_idx]
        if SCATTER_AVAILABLE:
            return scatter_mean(feats, batch_idx, dim=0, dim_size=len(reaction_list))
        return self._manual_mean(feats, batch_idx, len(reaction_list))

    def _pool_hyperedges(self, node_embeddings, edge_index):
        if edge_index is None or edge_index.numel() == 0:
            return torch.zeros(1, node_embeddings.size(1), device=node_embeddings.device, dtype=node_embeddings.dtype)
        node_idx, hyperedge_idx = edge_index[0], edge_index[1]
        num_hyperedges = int(hyperedge_idx.max().item()) + 1
        feats = node_embeddings[node_idx]
        if SCATTER_AVAILABLE:
            return scatter_mean(feats, hyperedge_idx, dim=0, dim_size=num_hyperedges)
        return self._manual_mean(feats, hyperedge_idx, num_hyperedges)

    def _node_degrees(self, edge_index, num_nodes):
        if edge_index is None or edge_index.numel() == 0:
            device = edge_index.device if edge_index is not None else None
            return torch.zeros(num_nodes, device=device)
        return torch.bincount(edge_index[0], minlength=num_nodes).float()

    def _reaction_stats(self, reaction_list, node_degrees, device, dtype):
        if node_degrees is None:
            return torch.zeros(len(reaction_list), 3, device=device, dtype=dtype)
        met_idx, batch_idx = self._query_membership(reaction_list, node_degrees.numel(), device)
        if met_idx.numel() == 0:
            return torch.zeros(len(reaction_list), 3, device=device, dtype=dtype)
        degrees = node_degrees[met_idx].to(device=device, dtype=dtype)
        observed = (degrees > 0).to(dtype).unsqueeze(-1)
        log_degree = torch.log1p(degrees).unsqueeze(-1)
        ones = torch.ones_like(log_degree)
        coverage = self._manual_mean(observed, batch_idx, len(reaction_list))
        mean_log_degree = self._manual_mean(log_degree, batch_idx, len(reaction_list))
        reaction_size = self._manual_mean(ones, batch_idx, len(reaction_list))
        return torch.cat([coverage, mean_log_degree, torch.log1p(reaction_size) / 4.0], dim=-1)

    def _degree_censoring_stats(self, reaction_list, node_degrees, device, dtype):
        met_idx, batch_idx = self._query_membership(reaction_list, node_degrees.numel(), device)
        if met_idx.numel() == 0:
            return torch.zeros(len(reaction_list), 4, device=device, dtype=dtype)
        degrees = node_degrees[met_idx].to(device=device, dtype=dtype)
        zero_degree = (degrees <= 0).to(dtype).unsqueeze(-1)
        rare_degree = (degrees <= 1).to(dtype).unsqueeze(-1)
        core_degree = torch.sigmoid(torch.log1p(degrees).unsqueeze(-1) - 3.0)
        inv_degree = 1.0 / torch.sqrt(torch.log1p(degrees).unsqueeze(-1) + 1.0)
        return torch.cat(
            [
                self._manual_mean(zero_degree, batch_idx, len(reaction_list)),
                self._manual_mean(rare_degree, batch_idx, len(reaction_list)),
                self._manual_mean(core_degree, batch_idx, len(reaction_list)),
                self._manual_mean(inv_degree, batch_idx, len(reaction_list)),
            ],
            dim=-1,
        )

    def _hyperedge_sizes(self, edge_index, device, dtype):
        if edge_index is None or edge_index.numel() == 0:
            return torch.ones(1, device=device, dtype=dtype)
        hyperedge_idx = edge_index[1]
        num_hyperedges = int(hyperedge_idx.max().item()) + 1 if hyperedge_idx.numel() else 1
        return torch.bincount(hyperedge_idx, minlength=num_hyperedges).to(device=device, dtype=dtype).clamp_min(1.0)

    def _size_bias(self, sizes):
        log_size = torch.log1p(sizes)
        centered_size = log_size - log_size.mean()
        singleton_fraction = (sizes <= 1).to(sizes.dtype).mean()
        low_order_fraction = (sizes <= 2).to(sizes.dtype).mean()
        singleton_penalty = (sizes <= 1).to(sizes.dtype) * (
            self.singleton_penalty + 2.0 * singleton_fraction
        )
        low_order_penalty = ((sizes > 1) & (sizes <= 2)).to(sizes.dtype) * (
            self.low_order_penalty + 0.5 * low_order_fraction
        )
        return centered_size - singleton_penalty - low_order_penalty

    def _attend_memory(self, query, memory, edge_index):
        q = self.query_proj(query)
        k = self.memory_key(memory)
        v = self.memory_value(memory)
        logits = torch.matmul(q, k.t()) / (q.size(-1) ** 0.5)
        sizes = self._hyperedge_sizes(edge_index, logits.device, logits.dtype)
        if sizes.numel() < logits.size(1):
            sizes = F.pad(sizes, (0, logits.size(1) - sizes.numel()), value=1.0)
        logits = logits + self._size_bias(sizes[: logits.size(1)]).unsqueeze(0)

        if self.top_k_memory is None or logits.size(1) <= self.top_k_memory:
            attn = torch.softmax(logits, dim=-1)
            return torch.matmul(attn, v), attn

        k_eff = min(self.top_k_memory, logits.size(1))
        top_values, top_indices = torch.topk(logits, k=k_eff, dim=-1)
        top_weights = torch.softmax(top_values, dim=-1)
        top_memory = v[top_indices]
        context = (top_weights.unsqueeze(-1) * top_memory).sum(dim=1)
        attn = logits.new_zeros(logits.shape)
        attn.scatter_(1, top_indices, top_weights)
        return context, attn

    def _memory_features(self, h_topo, h_chem, reaction_list, node_degrees, edge_index):
        rxn_topo_raw = self._pool_reactions(h_topo, reaction_list)
        rxn_topo = self.topo_reaction_norm(rxn_topo_raw)
        rxn_chem = self.chem_reaction_norm(self._pool_reactions(h_chem, reaction_list))
        stats = self._reaction_stats(reaction_list, node_degrees, rxn_topo.device, rxn_topo.dtype)
        gate = self.fusion_gate(torch.cat([rxn_topo, rxn_chem, stats], dim=-1))
        query = gate * rxn_topo + (1.0 - gate) * rxn_chem
        if edge_index is None:
            memory = rxn_topo.new_zeros(1, rxn_topo.size(1))
        else:
            memory = self.topo_reaction_norm(self._pool_hyperedges(h_topo, edge_index)) + self.chem_reaction_norm(
                self._pool_hyperedges(h_chem, edge_index)
            )
        context, attn = self._attend_memory(query, memory, edge_index)
        max_attn = attn.max(dim=-1, keepdim=True).values
        entropy = -(attn.clamp_min(1e-12) * attn.clamp_min(1e-12).log()).sum(dim=-1, keepdim=True)
        entropy = entropy / torch.log(torch.tensor(float(max(attn.size(1), 2)), device=attn.device, dtype=attn.dtype))
        rxn_embedding = torch.cat([query, context, torch.abs(query - context), stats], dim=-1)
        return rxn_topo_raw, rxn_embedding, stats, attn, max_attn, entropy, gate, context, query

    def _raw_reaction_vectors(self, x, reaction_list):
        vectors = []
        num_nodes = x.size(0)
        for reaction in reaction_list:
            valid = [
                int(met_idx)
                for met_idx in reaction_metabolites(reaction)
                if 0 <= int(met_idx) < num_nodes
            ]
            if not valid:
                vectors.append(torch.zeros(x.size(1), device=x.device, dtype=x.dtype))
                continue
            idx = torch.tensor(valid, dtype=torch.long, device=x.device)
            vectors.append(x[idx].sum(dim=0))
        if not vectors:
            return x.new_zeros((0, x.size(1)))
        return torch.stack(vectors, dim=0)

    def _raw_descriptor_chemistry_logit(self, x, reaction_list):
        raw = self._raw_reaction_vectors(x, reaction_list)
        seq = raw.view(raw.size(0), self.raw_chunks, self.raw_chunk_dim)
        encoded = self.raw_transformer(self.raw_emb(seq)).mean(dim=1)
        return self.raw_head(encoded)

    def _raw_chemistry_logit(self, x, reaction_list):
        descriptor_logit = self._raw_descriptor_chemistry_logit(x, reaction_list)
        set_logit = self._set_semantic_logit(x, reaction_list)
        return descriptor_logit + self.chemset_scale * torch.tanh(set_logit)

    def _protected_identity_gate(self, edge_index, num_nodes, device, dtype):
        degrees = self._node_degrees(edge_index, num_nodes).to(device=device, dtype=dtype).view(-1, 1)
        return (1.0 / (1.0 + degrees)).clamp(0.0, 1.0)

    def _protected_hgnn_embeddings(self, x, edge_index):
        h_anchor = F.relu(self.protected_hgnn_conv1(x, edge_index))
        h_anchor = self.protected_hgnn_conv2(h_anchor, edge_index)
        if not self.use_identity_skip:
            return h_anchor
        identity_gate = self._protected_identity_gate(edge_index, x.size(0), x.device, h_anchor.dtype)
        identity_skip = self.protected_identity_skip(self.protected_identity_norm(x))
        identity_scale = self.protected_identity_scale.to(device=x.device, dtype=h_anchor.dtype)
        return h_anchor + identity_scale * identity_gate * identity_skip

    def _pool_protected_reactions(self, h_anchor, reaction_list, node_degrees):
        met_idx, batch_idx = self._query_membership(reaction_list, h_anchor.size(0), h_anchor.device)
        if met_idx.numel() == 0:
            return h_anchor.new_zeros(len(reaction_list), h_anchor.size(1))
        weight = (node_degrees.to(device=h_anchor.device)[met_idx] > 0.0).to(dtype=h_anchor.dtype).view(-1, 1)
        observed_sum = h_anchor.new_zeros(len(reaction_list), h_anchor.size(1))
        observed_sum.index_add_(0, batch_idx, h_anchor[met_idx] * weight)
        observed_count = h_anchor.new_zeros(len(reaction_list), 1)
        observed_count.index_add_(0, batch_idx, weight)
        observed_mean = observed_sum / observed_count.clamp(min=1.0)
        fallback_mean = self._pool_reactions(h_anchor, reaction_list)
        return torch.where(observed_count > 0.0, observed_mean, fallback_mean)

    def _native_prototype_features(self, query_anchor, h_anchor, edge_index):
        if edge_index is None or edge_index.numel() == 0:
            return query_anchor.new_zeros(query_anchor.size(0), 3)
        native_memory = self._pool_hyperedges(h_anchor, edge_index)
        query_norm = F.normalize(query_anchor, dim=-1)
        memory_norm = F.normalize(native_memory, dim=-1)
        sims = torch.matmul(query_norm, memory_norm.t())
        k_eff = min(8, sims.size(1))
        top_sims = torch.topk(sims, k=k_eff, dim=-1).values
        max_sim = top_sims[:, 0:1]
        topk_mean = top_sims.mean(dim=-1, keepdim=True)
        weights = torch.softmax(top_sims, dim=-1)
        entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(dim=-1, keepdim=True)
        entropy = entropy / torch.log(
            torch.tensor(float(max(k_eff, 2)), device=query_anchor.device, dtype=query_anchor.dtype)
        )
        return torch.cat([max_sim, topk_mean, entropy], dim=-1)

    def _prototype_vectors(self, h_anchor, reaction_sets):
        if not reaction_sets:
            return h_anchor.new_zeros((0, h_anchor.size(1)))
        return self._pool_reactions(h_anchor, reaction_sets)

    def _topk_context_and_summary(self, query_vectors, prototype_vectors):
        if prototype_vectors.numel() == 0:
            zeros = query_vectors.new_zeros((query_vectors.size(0), 3))
            return query_vectors, zeros
        query_norm = F.normalize(query_vectors, dim=-1)
        prototype_norm = F.normalize(prototype_vectors, dim=-1)
        sims = torch.matmul(query_norm, prototype_norm.t())
        k_eff = min(max(1, self.boundary_topk), sims.size(1))
        top_values, top_indices = torch.topk(sims, k=k_eff, dim=-1)
        weights = torch.softmax(top_values, dim=-1)
        top_vectors = prototype_vectors[top_indices]
        context = (weights.unsqueeze(-1) * top_vectors).sum(dim=1)
        max_sim = top_values[:, 0:1]
        mean_sim = top_values.mean(dim=-1, keepdim=True)
        entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(dim=-1, keepdim=True)
        entropy = entropy / torch.log(
            torch.tensor(float(max(k_eff, 2)), device=query_vectors.device, dtype=query_vectors.dtype)
        )
        return context, torch.cat([max_sim, mean_sim, entropy], dim=-1)

    def _counterfactual_boundary_context(self, query_anchor, h_anchor):
        native_vectors = self._prototype_vectors(h_anchor, self.native_boundary_sets)
        hard_vectors = self._prototype_vectors(h_anchor, self.hard_boundary_sets)
        native_context, native_summary = self._topk_context_and_summary(query_anchor, native_vectors)
        _, hard_summary = self._topk_context_and_summary(query_anchor, hard_vectors)
        margin = native_summary[:, 0:1] - hard_summary[:, 0:1]
        return native_context, native_summary, hard_summary, margin

    def _counterfactual_completion_correction(
        self,
        base_logit,
        counterfactual_logit,
        stats,
        degree_stats,
        margin,
    ):
        coverage = stats[:, 0:1].clamp(0.0, 1.0)
        zero_fraction = degree_stats[:, 0:1].clamp(0.0, 1.0)
        rare_fraction = degree_stats[:, 1:2].clamp(0.0, 1.0)
        core_fraction = degree_stats[:, 2:3].clamp(0.0, 1.0)

        temperature = F.softplus(self.counterfactual_temperature_raw).to(
            device=base_logit.device,
            dtype=base_logit.dtype,
        )
        native_certificate = torch.sigmoid(temperature * margin)
        hard_certificate = torch.sigmoid(-temperature * margin)
        rare_gate = torch.sigmoid(8.0 * (zero_fraction + 0.5 * rare_fraction - 0.15))
        rare_gate = rare_gate * torch.sigmoid(2.0 * (0.75 - base_logit)) * native_certificate
        core_gate = coverage * core_fraction * torch.sigmoid(base_logit) * hard_certificate

        rescue_scale = F.softplus(self.counterfactual_rescue_scale).to(
            device=base_logit.device,
            dtype=base_logit.dtype,
        )
        veto_scale = F.softplus(self.counterfactual_veto_scale).to(
            device=base_logit.device,
            dtype=base_logit.dtype,
        )
        rare_rescue = rescue_scale * rare_gate * F.relu(counterfactual_logit - base_logit)
        core_veto = veto_scale * core_gate * F.relu(base_logit - counterfactual_logit - margin)
        correction = rare_rescue - core_veto
        diagnostics = {
            "rare_gate": rare_gate,
            "core_gate": core_gate,
            "rare_rescue": rare_rescue,
            "core_veto": core_veto,
            "native_certificate": native_certificate,
            "hard_certificate": hard_certificate,
        }
        return correction, diagnostics

    def _censored_node_logits(self, x, edge_index):
        h_anchor = self._protected_hgnn_embeddings(x, edge_index)
        node_degrees = self._node_degrees(edge_index, x.size(0)).to(x.device)
        node_logit = self.protected_hgnn_classifier(h_anchor)
        degree_column = node_degrees.to(device=x.device, dtype=node_logit.dtype).view(-1, 1)
        rare_floor = -F.softplus(self.rare_floor_raw).to(device=x.device, dtype=node_logit.dtype)
        core_cap = F.softplus(self.core_cap_raw).to(device=x.device, dtype=node_logit.dtype)
        rare_censored = torch.where(degree_column <= 0, torch.maximum(node_logit, rare_floor), node_logit)
        core_weight = torch.sigmoid(torch.log1p(degree_column) - 3.0)
        capped = torch.minimum(rare_censored, core_cap)
        censored_logit = (1.0 - core_weight) * rare_censored + core_weight * capped
        return censored_logit, h_anchor, node_degrees

    def _pool_node_logits(self, node_logits, reaction_list):
        return self._pool_reactions(node_logits, reaction_list)

    def _molecular_to_chassis_residual(self, molecular_logit):
        """Return the fixed, bounded molecular residual for chassis scoring."""

        scale = self.fixed_molecular_fusion_alpha.to(
            device=molecular_logit.device,
            dtype=molecular_logit.dtype,
        )
        return scale * torch.tanh(molecular_logit.detach())

    def _base_protected_topology_logit(
        self,
        x,
        edge_index,
        reaction_list,
        molecular_logit=None,
    ):
        censored_node_logit, h_anchor, node_degrees = self._censored_node_logits(x, edge_index)
        base_anchor = self._pool_reactions(h_anchor, reaction_list)
        base_logit = self.protected_hgnn_classifier(base_anchor)
        if molecular_logit is not None:
            if molecular_logit.shape != base_logit.shape:
                raise ValueError(
                    "molecular_logit must match the chassis embedding logit shape: "
                    f"{tuple(molecular_logit.shape)} != {tuple(base_logit.shape)}"
                )
            base_logit = base_logit + self._molecular_to_chassis_residual(molecular_logit)
        pooled_logit = self._pool_node_logits(censored_node_logit, reaction_list)
        prototype_features = self._native_prototype_features(base_anchor, h_anchor, edge_index)
        stats = self._reaction_stats(reaction_list, node_degrees, x.device, x.dtype)
        degree_stats = self._degree_censoring_stats(reaction_list, node_degrees, x.device, x.dtype)

        max_sim = prototype_features[:, 0:1]
        zero_fraction = degree_stats[:, 0:1].clamp(0.0, 1.0)
        rare_fraction = degree_stats[:, 1:2].clamp(0.0, 1.0)
        core_fraction = degree_stats[:, 2:3].clamp(0.0, 1.0)
        coverage = stats[:, 0:1].clamp(0.0, 1.0)

        prototype_support = F.relu(max_sim)
        native_like = torch.sigmoid(4.0 * max_sim)
        sparse_support = (zero_fraction + rare_fraction).clamp(0.0, 1.0)
        native_gate = native_like * sparse_support
        shortcut_gate = coverage * core_fraction * (1.0 - native_like)

        mix = torch.sigmoid(self.node_logit_mix)
        censored_logit = (1.0 - mix) * base_logit + mix * pooled_logit
        return censored_logit + 0.50 * native_gate * prototype_support - 0.50 * shortcut_gate

    def _protected_topology_logit(
        self,
        x,
        edge_index,
        reaction_list,
        molecular_logit=None,
    ):
        base_logit = self._base_protected_topology_logit(
            x,
            edge_index,
            reaction_list,
            molecular_logit=molecular_logit,
        )
        if not self.native_boundary_sets or not self.hard_boundary_sets:
            return base_logit

        _, h_anchor, node_degrees = self._censored_node_logits(x, edge_index)
        query_anchor = self._pool_reactions(h_anchor, reaction_list)
        native_context, _, _, margin = self._counterfactual_boundary_context(query_anchor, h_anchor)
        counterfactual_logit = self.protected_hgnn_classifier(native_context)
        stats = self._reaction_stats(reaction_list, node_degrees, x.device, x.dtype)
        degree_stats = self._degree_censoring_stats(reaction_list, node_degrees, x.device, x.dtype)
        correction, _ = self._counterfactual_completion_correction(
            base_logit,
            counterfactual_logit,
            stats,
            degree_stats,
            margin,
        )
        return base_logit + correction

    def extra_semantic_loss(self, x, edge_index, reaction_list, batch_groups, final_logit):
        del x, edge_index, reaction_list, batch_groups
        return final_logit.new_zeros(())

    def _boundary_prior(self, stats):
        coverage = stats[:, 0:1].clamp(min=0.03, max=0.97)
        mean_log_degree = stats[:, 1:2]
        return torch.logit(coverage) + 0.10 * (mean_log_degree - 1.0)

    def _boundary_violation_penalty(self, stats):
        boundary_prior = self._boundary_prior(stats)
        violation_gate = torch.sigmoid(-boundary_prior)
        return violation_gate * F.softplus(-boundary_prior)

    def _memory_certificate(self, coverage, max_attn, entropy, memory_logit, topology_logit):
        agreement = torch.exp(-torch.abs(memory_logit - topology_logit).clamp(max=6.0))
        return torch.sigmoid(
            1.50 * coverage
            + 1.25 * agreement
            + 0.75 * max_attn
            + 1.00 * (1.0 - entropy)
            - 2.25
        )

    def _set_semantic_logit(self, x, reaction_list):
        batch_size = len(reaction_list)
        if batch_size == 0:
            return x.new_zeros((0, self.out_channels))

        valid_reactions = [
            [
                int(met_idx)
                for met_idx in reaction_metabolites(reaction)
                if 0 <= int(met_idx) < x.size(0)
            ]
            for reaction in reaction_list
        ]
        max_len = max((len(reaction) for reaction in valid_reactions), default=0)
        if max_len == 0:
            return x.new_zeros((batch_size, self.out_channels))

        token_tensor = x.new_zeros((batch_size, max_len, x.size(1)))
        padding_mask = torch.ones((batch_size, max_len), device=x.device, dtype=torch.bool)
        lengths = []
        for row_idx, reaction in enumerate(valid_reactions):
            lengths.append(len(reaction))
            if not reaction:
                continue
            idx = torch.tensor(reaction, dtype=torch.long, device=x.device)
            token_tensor[row_idx, : len(reaction)] = x[idx]
            padding_mask[row_idx, : len(reaction)] = False

        tokens = self.set_metabolite_proj(token_tensor)
        query = self.set_query.expand(batch_size, -1, -1)
        pooled, _ = self.set_attention(query, tokens, tokens, key_padding_mask=padding_mask)
        pooled = pooled.squeeze(1)
        length_tensor = torch.tensor(lengths, device=x.device, dtype=x.dtype).view(batch_size, 1)
        length_features = torch.cat(
            [
                torch.log1p(length_tensor) / 3.0,
                (length_tensor / 12.0).clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        return self.set_head(torch.cat([pooled, length_features], dim=-1))

    def forward_outputs(self, x, mp_edge_index, query_reactions):
        """Return the logits and intermediate readouts used by HyperCEF."""

        h_topo, h_chem = self.encode(x, mp_edge_index)
        node_degrees = self._node_degrees(mp_edge_index, x.size(0)).to(x.device)
        _, rxn_embedding, stats, _, _, _, _, _, _ = self._memory_features(
            h_topo,
            h_chem,
            query_reactions,
            node_degrees,
            mp_edge_index,
        )
        chemistry_logit = self._raw_chemistry_logit(x, query_reactions)
        protected_topology_logit = self._protected_topology_logit(
            x,
            mp_edge_index,
            query_reactions,
            molecular_logit=chemistry_logit,
        )
        memory_logit = self.memory_classifier(rxn_embedding)
        invalid_probability = torch.sigmoid(-(chemistry_logit + 1.0))
        final_logit = protected_topology_logit
        molecular_residual = self._molecular_to_chassis_residual(chemistry_logit)
        zeros = torch.zeros_like(final_logit)
        return {
            "final_logit": final_logit,
            "chemistry_logit": chemistry_logit,
            "host_logit": final_logit,
            "topology_logit": protected_topology_logit,
            "memory_logit": memory_logit,
            "topology_weight": torch.ones_like(final_logit),
            "stats": stats,
            "invalid_probability": invalid_probability,
            "semantic_correction": molecular_residual,
            "isolation_gate": zeros,
            "molecular_residual": molecular_residual,
            "molecular_fusion_scale": self.fixed_molecular_fusion_alpha.to(
                device=final_logit.device,
                dtype=final_logit.dtype,
            ),
        }

    def _output_parts(self, x, mp_edge_index, query_reactions):
        outputs = self.forward_outputs(x, mp_edge_index, query_reactions)
        return tuple(outputs[name] for name in self.output_keys)

    def prediction_logits(self, x, mp_edge_index, query_reactions):
        final_logit, chemistry_logit, host_logit, *_ = self._output_parts(x, mp_edge_index, query_reactions)
        return final_logit, chemistry_logit, host_logit

    def forward(self, x, mp_edge_index, query_reactions, return_embedding=False):
        parts = self._output_parts(x, mp_edge_index, query_reactions)
        if return_embedding:
            return torch.cat([part for part in parts if part.dim() == 2], dim=-1)
        return parts[0]

    def score_decomposition(self, x, mp_edge_index, query_reactions):
        """Decompose the final chassis logit into its model components.

        The score is decomposed into the topology-only base, the bounded
        molecular contribution after mixing, and the subsequent calibration.
        """

        flow = self.forward_outputs(x, mp_edge_index, query_reactions)
        base_topology_logit = self._base_protected_topology_logit(
            x,
            mp_edge_index,
            query_reactions,
        )
        fused_base_logit = self._base_protected_topology_logit(
            x,
            mp_edge_index,
            query_reactions,
            molecular_logit=flow["chemistry_logit"],
        )
        mixed_molecular_contribution = fused_base_logit - base_topology_logit
        calibration_delta = flow["topology_logit"] - fused_base_logit
        delta_certified = torch.zeros_like(flow["final_logit"])
        chemistry_veto = torch.zeros_like(flow["final_logit"])
        reconstructed_logit = (
            base_topology_logit
            + mixed_molecular_contribution
            + calibration_delta
        )

        return {
            "final_logit": flow["final_logit"],
            "reconstructed_logit": reconstructed_logit,
            "base_topology_logit": base_topology_logit,
            "molecular_residual": flow["molecular_residual"],
            "mixed_molecular_contribution": mixed_molecular_contribution,
            "molecular_fusion_scale": flow["molecular_fusion_scale"],
            "calibration_delta": calibration_delta,
            "delta_cchc": calibration_delta,
            "delta_certified": delta_certified,
            "chemistry_veto": chemistry_veto,
            "chemistry_logit": flow["chemistry_logit"],
            "host_logit": flow["host_logit"],
            "topology_logit": flow["topology_logit"],
            "memory_logit": flow["memory_logit"],
            "topology_weight": flow["topology_weight"],
            "coverage": flow["stats"][:, 0:1],
            "mean_log_degree": flow["stats"][:, 1:2],
            "reaction_size": flow["stats"][:, 2:3],
            "invalid_probability": flow["invalid_probability"],
            "semantic_correction": flow["semantic_correction"],
            "isolation_gate": flow["isolation_gate"],
        }

    def explain_reactions(self, x, mp_edge_index, query_reactions):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            explanation = self.score_decomposition(x, mp_edge_index, query_reactions)
            explanation["score"] = torch.sigmoid(explanation["final_logit"])
            explanation["logit"] = explanation["final_logit"]
        if was_training:
            self.train()
        return explanation


__all__ = [
    "MODEL_NAME",
    "MODEL_VERSION",
    "HyperCEF",
]
