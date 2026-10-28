"""Large GRU policy for a conditional distribution over all primitive coefficients.

The network chooses a nonzero anchor index AND its positive integer value.
This removes the all-zero degeneracy without selecting any physical term by
hand. Given these two sampled controls, the other coefficients (including
zero, meaning absent) have neural categorical distributions. Their conditional
quadratic residual has an exactly computable expectation from the train Gram
matrix; no coefficient fitting or linear-system solution is performed.
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class CoefficientDistribution(nn.Module):
    def __init__(self, terms, alphabet, hidden=512, embedding=128, contexts=1):
        super().__init__()
        self.terms = int(terms)
        self.contexts = contexts
        self.alphabet = tuple(sorted(set(alphabet) | {0}))
        self.positive = tuple(c for c in self.alphabet if c > 0)
        if not self.positive or self.terms < 2:
            raise ValueError("Expected several terms and positive coefficient choices")
        self.vocabulary = len(self.alphabet)
        self.config = {"terms": terms, "alphabet": list(self.alphabet), "hidden": hidden, "embedding": embedding, "contexts": contexts}
        self.token_embedding = nn.Embedding(terms+self.vocabulary+1, embedding)
        self.position_embedding = nn.Embedding(3, embedding)
        self.cell = nn.GRUCell(2*embedding, hidden)
        self.initial_state = nn.Parameter(torch.zeros(hidden))
        self.anchor_head = nn.Linear(hidden, terms)
        self.value_head = nn.Linear(hidden, len(self.positive))
        self.coefficient_head = nn.Linear(hidden, terms*self.vocabulary)
        self.context_embedding = nn.Embedding(contexts, hidden) if contexts > 1 else None

    def _step(self, tokens, position, hidden):
        pos = torch.full_like(tokens, position)
        inputs = torch.cat([self.token_embedding(tokens), self.position_embedding(pos)], dim=-1)
        return self.cell(inputs, hidden)

    def distribution(self, count, controls=None, greedy_controls=False, contexts=None):
        device = self.initial_state.device
        hidden = self.initial_state.expand(count, -1)
        if self.contexts > 1:
            contexts = torch.randint(self.contexts,(count,),device=device) if contexts is None else contexts
            hidden = hidden+self.context_embedding(contexts)
        bos = torch.full((count,), self.terms+self.vocabulary, dtype=torch.long, device=device)
        hidden = self._step(bos, 0, hidden)
        anchors = Categorical(logits=self.anchor_head(hidden))
        index = controls[:, 0] if controls is not None else (anchors.logits.argmax(-1) if greedy_controls else anchors.sample())
        hidden = self._step(index, 1, hidden)
        values = Categorical(logits=self.value_head(hidden))
        magnitude_id = controls[:, 1] if controls is not None else (values.logits.argmax(-1) if greedy_controls else values.sample())
        positive = torch.as_tensor(self.positive, dtype=torch.long, device=device)
        magnitude = positive[magnitude_id]
        alphabet = torch.as_tensor(self.alphabet, dtype=torch.long, device=device)
        value_tokens = (magnitude[:, None] == alphabet[None]).to(torch.int64).argmax(-1)
        hidden = self._step(self.terms+value_tokens, 2, hidden)
        logits = self.coefficient_head(hidden).reshape(count, self.terms, self.vocabulary)
        probabilities = logits.softmax(-1)
        mask = torch.nn.functional.one_hot(index, self.terms).bool()
        fixed = torch.nn.functional.one_hot(value_tokens, self.vocabulary).to(probabilities.dtype)
        probabilities = torch.where(mask[:, :, None], fixed[:, None, :], probabilities)
        return {"contexts":contexts,"probabilities": probabilities, "logits": logits, "anchor_mask": mask,
                "controls": torch.stack([index, magnitude_id], 1), "anchor": index, "magnitude": magnitude,
                "control_logprob": anchors.log_prob(index)+values.log_prob(magnitude_id),
                "control_entropy": anchors.entropy()+values.entropy()}

    def sample_distribution(self, dist, greedy=False):
        probabilities = dist["probabilities"]
        ids = probabilities.argmax(-1) if greedy else Categorical(probs=probabilities).sample()
        alphabet = torch.as_tensor(self.alphabet, device=ids.device)
        coefficients = alphabet[ids]
        tokens = torch.cat([dist["controls"], ids], 1)
        if self.contexts > 1:
            tokens = torch.cat([tokens,dist["contexts"][:,None]],1)
        return coefficients, tokens

    def sample(self, count, greedy=False):
        dist = self.distribution(count, greedy_controls=greedy)
        coefficients, tokens = self.sample_distribution(dist, greedy)
        return coefficients, tokens

    def log_probability(self, tokens):
        dist = self.distribution(len(tokens), controls=tokens[:, :2], contexts=tokens[:, -1] if self.contexts>1 else None)
        chosen = dist["logits"].log_softmax(-1).gather(-1, tokens[:, 2:2+self.terms, None]).squeeze(-1)
        return dist["control_logprob"]+chosen.masked_fill(dist["anchor_mask"], 0).sum(-1)

    def decode(self, tokens):
        return np.asarray(self.alphabet)[np.asarray(tokens)[:, 2:2+self.terms]]

    def quadratic_moments(self, dist, gram):
        # Focused batches share one Gram matrix; avoid 64 copies of it.
        if gram.ndim==3 and torch.all(dist['contexts']==dist['contexts'][0]):
            gram=gram[dist['contexts'][0]]
        p = dist["probabilities"].to(gram.dtype)
        values = torch.as_tensor(self.alphabet, dtype=gram.dtype, device=gram.device)
        mean = p @ values
        variance = (p @ values.square()-mean.square()).clamp_min(0)
        if gram.ndim == 3:
            selected = gram[dist['contexts']]
            diagonal = selected.diagonal(dim1=-2,dim2=-1)
            mean_energy = (torch.bmm(mean[:,None,:],selected).squeeze(1)*mean).sum(-1)
            variance_energy = (variance*diagonal).sum(-1)
            normalization = diagonal.gather(1,dist['anchor'][:,None]).squeeze(1)*dist['magnitude'].to(gram.dtype).square()
        else:
            mean_energy = ((mean @ gram)*mean).sum(-1)
            variance_energy = variance @ gram.diag()
            normalization = gram.diag()[dist["anchor"]]*dist["magnitude"].to(gram.dtype).square()
        support = (1-p[:, :, self.alphabet.index(0)]).sum(-1)
        return mean_energy, variance_energy, normalization.clamp_min(1e-20), support


def load_policy(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("policy_kind") != "conditional_coefficients":
        raise ValueError("Expected a current six-point coefficient model")
    model = CoefficientDistribution(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
