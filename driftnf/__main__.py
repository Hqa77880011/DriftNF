import argparse
import copy
import json
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
from scipy.special import expit
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from torch import nn
from torch.nn import functional as F


def descriptors(x):
    median = x.median(-1).values
    slope = (x[:, :, -1] - x[:, :, 0]) / (x.shape[-1] - 1)
    mad = (x - median[..., None]).abs().median(-1).values * 1.4826
    return torch.stack([median, slope, mad], -1).flatten(1)


class Block(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.dilation = dilation

    def forward(self, x):
        return x + F.dropout(F.relu(self.conv(F.pad(x, (2 * self.dilation, 0)))), .1, self.training)


class Detector(nn.Module):
    def __init__(self, dimensions, c):
        super().__init__()
        self.c = c
        self.encoder = nn.Sequential(nn.Conv1d(dimensions, 64, 1), *[Block(64, d) for d in [1, 2, 4]])
        self.centers = nn.Parameter(torch.zeros(c['max_rules'], 3 * dimensions))
        self.log_width = nn.Parameter(torch.zeros_like(self.centers))
        self.gates = nn.Parameter(torch.zeros_like(self.centers))
        self.predictor = nn.Parameter(torch.randn(c['max_rules'], 3 * dimensions, 65) * .01)
        self.register_buffer('residual_variance', torch.ones_like(self.centers))
        self.register_buffer('active', torch.arange(c['max_rules']) < c['initial_rules'])
        self.register_buffer('utility', torch.zeros(c['max_rules']))

    def antecedents(self):
        continuous = self.gates.sigmoid()
        if self.c['variant'] == 'dense':
            return torch.ones_like(continuous)
        values, indices = continuous.topk(min(6, continuous.shape[1]), dim=1)
        selected = torch.zeros_like(continuous).scatter(1, indices, (values > .5).float())
        empty = selected.sum(1) == 0
        selected[empty, continuous[empty].argmax(1)] = 1
        return selected - continuous.detach() + continuous if self.training else selected

    def score(self, q, h):
        width = F.softplus(self.log_width).clamp_min(.1)
        gate = self.antecedents()
        logphi = (-.5 * ((q[:, None] - self.centers) / width).square() * gate).sum(-1)
        logphi = logphi.masked_fill(~self.active[None], -torch.inf)
        weights = logphi.softmax(1)
        augmented = F.pad(h, (0, 1), value=1)
        predicted = torch.einsum('rkh,nh->nrk', self.predictor, augmented)
        error = (q[:, None] - predicted).square() / self.residual_variance.clamp_min(1e-5)
        score = (weights * error.sum(-1)).sum(1) - self.c['novelty_weight'] * logphi.max(1).values.clamp_min(-16.118)
        contribution = (weights[..., None] * gate * ((q[:, None] - self.centers).abs() / width +
                       (q[:, None] - predicted).abs() / self.residual_variance.clamp_min(1e-5).sqrt())).sum(1)
        sensors = contribution.reshape(len(q), -1, 3).sum(-1)
        sensors = sensors / sensors.sum(1, keepdim=True).clamp_min(1e-8)
        return score, weights, sensors, (weights[..., None] * predicted).sum(1)


def causal_fill(x, start):
    x = x.copy()
    previous = start.copy()
    for i in range(len(x)):
        good = np.isfinite(x[i])
        previous[good] = x[i, good]
        x[i] = previous
    return x


def windows(x, c):
    return torch.tensor(x, dtype=torch.float32).unfold(0, c['window'], 1)


def prepare(c, out, source):
    raw = np.load(source, allow_pickle=False)
    train = raw['train'].astype(float)
    boundary = int(.8 * len(train))
    median = np.nanmedian(train[:boundary], axis=0)
    scale = 1.4826 * np.nanmedian(abs(train[:boundary] - median), axis=0)
    keep = np.isfinite(scale) & (scale > 1e-8)
    filled_train = causal_fill(train, median)
    normalized = (filled_train - median) / np.maximum(scale, 1e-8)
    test = (causal_fill(raw['test'].astype(float), filled_train[-1]) - median) / np.maximum(scale, 1e-8)
    np.savez_compressed(out / 'data.npz', fit=normalized[:boundary, keep], val=normalized[boundary:, keep], test=test[:, keep])
    np.savez(out / 'preprocessing.npz', median=median, scale=scale, retained_channels=np.flatnonzero(keep))


def train(c, out):
    torch.manual_seed(c['seed'])
    data = np.load(out / 'data.npz')
    x = windows(data['fit'], c)
    q = descriptors(x)[1:]
    model = Detector(x.shape[1], c)
    clusters = KMeans(c['initial_rules'], n_init=10, random_state=c['seed']).fit(q.numpy())
    with torch.no_grad():
        model.centers[:c['initial_rules']] = torch.tensor(clusters.cluster_centers_)
        for r in range(c['initial_rules']):
            spread = q[clusters.labels_ == r].std(0).nan_to_num(.1).clamp_min(.1)
            model.log_width[r] = torch.log(torch.expm1(spread))
            model.gates[r] = torch.randn(q.shape[1]) * .1
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'])
    for _ in range(c['epochs']):
        model.train()
        for ids in torch.randperm(len(q)).split(c['batch_size']):
            h = model.encoder(x[ids])[:, :, -1]
            score, weight, contribution, predicted = model.score(q[ids], h)
            loss = score.mean() + c['sparsity_weight'] * model.gates.sigmoid().mean()
            centers = model.centers[model.active]
            overlap = torch.exp(-torch.cdist(centers, centers).square())
            loss += c['overlap_weight'] * (overlap.sum() - len(centers)) / max(1, len(centers) ** 2)
            if c['variant'] != 'no-counterfactual':
                interventions = []
                for sensor in range(x.shape[1]):
                    repaired = q[ids].clone()
                    repaired[:, 3 * sensor:3 * sensor + 3] = predicted[:, 3 * sensor:3 * sensor + 3]
                    interventions.append(F.relu(score - model.score(repaired, h)[0]))
                intervention = torch.stack(interventions, 1)
                intervention /= intervention.sum(1, keepdim=True).clamp_min(1e-8)
                loss += c['counterfactual_weight'] * (contribution - intervention).abs().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        h = torch.cat([model.encoder(batch)[:, :, -1] for batch in x[:-1].split(c['batch_size'])])
        augmented = F.pad(h, (0, 1), value=1)
        prediction = torch.einsum('rkh,nh->nrk', model.predictor, augmented)
        error = (q[:, None] - prediction).square()
        weight = model.score(q, h)[1]
        model.residual_variance.copy_((weight[..., None] * error).sum(0) / weight.sum(0)[:, None].clamp_min(1e-5))
        vx = windows(data['val'], c)
        vq = descriptors(vx)[1:]
        vh = torch.cat([model.encoder(batch)[:, :, -1] for batch in vx[:-1].split(c['batch_size'])])
        scores = model.score(vq, vh)[0].numpy()
        if len(vq) < 2 * c['buffer']:
            raise ValueError('Validation stream must contain at least two full drift buffers after windowing')
        qdrift, rdrift = [], []
        for end in range(2 * c['buffer'], len(vq) + 1, c['buffer']):
            old, recent = vq[end - 2 * c['buffer']:end - c['buffer']], vq[end - c['buffer']:end]
            qdrift.append(gaussian_shift(old.numpy(), recent.numpy()))
            p = model.score(old, vh[end - 2 * c['buffer']:end - c['buffer']])[1].mean(0).numpy()
            r = model.score(recent, vh[end - c['buffer']:end])[1].mean(0).numpy()
            rdrift.append(float(jensenshannon(p, r) ** 2))
    np.save(out / 'training_descriptors.npy', q.numpy())
    torch.save(model.state_dict(), out / 'model.pt')
    np.savez(out / 'validation.npz', q=vq.numpy(), h=vh.numpy())
    calibration = dict(alarm=float(np.quantile(scores, .995)), temperature=max(float(scores.std()), 1e-5),
                       descriptor=max(float(np.quantile(qdrift, .995)), 1e-5), occupancy=max(float(np.quantile(rdrift, .995)), 1e-5),
                       audit_range=max(float(np.quantile(scores, .995)) * c['audit_scale'], 1e-3))
    (out / 'calibration.json').write_text(json.dumps(calibration))


def gaussian_shift(old, recent):
    m0, m1 = old.mean(0), recent.mean(0)
    v0, v1 = old.var(0).clip(1e-6), recent.var(0).clip(1e-6)
    return float(.25 * np.sum(v0 / v1 + v1 / v0 - 2 + (m0 - m1) ** 2 * (1 / v0 + 1 / v1)))


def bound(values, cal, c, upper=False):
    z = np.clip(values, -cal['audit_range'], cal['audit_range'])
    if len(z) < 2:
        return np.inf if upper else -np.inf
    radius = np.sqrt(2 * z.var(ddof=1) * np.log(6 / c['alpha']) / len(z)) + 6 * cal['audit_range'] * np.log(6 / c['alpha']) / len(z)
    if c['variant'] == 'raw-means':
        radius = 0
    return float(z.mean() + radius if upper else z.mean() - radius)


def run(c, out):
    data = np.load(out / 'data.npz')
    validation = np.load(out / 'validation.npz')
    cal = json.loads((out / 'calibration.json').read_text(encoding='utf-8'))
    model = Detector(data['test'].shape[1], c)
    model.load_state_dict(torch.load(out / 'model.pt', weights_only=True, map_location='cpu'))
    model.eval()
    x = windows(data['test'], c)
    ref = torch.tensor(validation['q'][-c['buffer']:])
    anchor_q = torch.tensor(validation['q'][:c['anchor_capacity']])
    anchor_h = torch.tensor(validation['h'][:c['anchor_capacity']])
    buffer_q, buffer_h = deque(maxlen=c['buffer']), deque(maxlen=c['buffer'])
    ledger = deque(maxlen=c['regimes'])
    with torch.no_grad():
        initial_rule = int(model.score(ref, torch.zeros(len(ref), 64))[1].mean(0).argmax())
    ledger.append(dict(mean=ref.mean(0).numpy(), rule=initial_rule))
    persistent = 0
    rows, audits = [], []
    isolation = IsolationForest(random_state=c['seed']).fit(np.load(out / 'training_descriptors.npy'))
    isolation_threshold = np.quantile(-isolation.score_samples(validation['q']), .995)
    channel_ids = np.load(out / 'preprocessing.npz')['retained_channels']
    with torch.no_grad():
        h_previous = model.encoder(x[:1])[:, :, -1]
        for t in range(1, len(x)):
            start = time.perf_counter()
            q = descriptors(x[t:t + 1])
            score, weights, contributions, predicted = model.score(q, h_previous)
            ranked = contributions[0].topk(min(3, contributions.shape[1])).indices
            repaired = q.clone()
            repaired[:, ranked[0] * 3:ranked[0] * 3 + 3] = predicted[:, ranked[0] * 3:ranked[0] * 3 + 3]
            fidelity = float(F.relu(score - model.score(repaired, h_previous)[0]) / score.abs().clamp_min(1e-8))
            perturbed = descriptors(x[t:t + 1] + torch.randn_like(x[t:t + 1]) * .01)
            other = model.score(perturbed, h_previous)[2][0].topk(len(ranked)).indices
            union = len(set(ranked.tolist()) | set(other.tolist()))
            stability = len(set(ranked.tolist()) & set(other.tolist())) / max(1, union)
            buffer_q.append(q[0])
            buffer_h.append(h_previous[0])
            committed = False
            if len(buffer_q) == c['buffer'] and c['variant'] != 'frozen':
                recent, recent_h = torch.stack(list(buffer_q)), torch.stack(list(buffer_h))
                dq = gaussian_shift(ref.numpy(), recent.numpy()) / cal['descriptor']
                occ0 = model.score(ref, recent_h)[1].mean(0).numpy()
                occ1 = model.score(recent, recent_h)[1].mean(0).numpy()
                dr = float(jensenshannon(occ0, occ1) ** 2) / cal['occupancy']
                agreement = 2 * dq * dr / max(dq + dr, 1e-8)
                if c['variant'] == 'descriptor-only':
                    agreement = dq
                elif c['variant'] == 'occupancy-only':
                    agreement = dr
                persistent = persistent + 1 if agreement > 1 else 0
                if persistent >= c['persistence']:
                    before = model.score(recent, recent_h)[0].numpy()
                    risk = expit((before - cal['alarm']) / cal['temperature'])
                    clean = risk <= c['sample_risk']
                    veto = 'contamination'
                    record = dict(position=t + c['window'] - 1, descriptor=dq, occupancy=dr,
                                  contamination=float(risk.mean()), committed=False, veto=veto, slack=-1.,
                                  fit_lcb=None, old_ucb=None, fault_lcb=None, before=before.tolist(), after=before.tolist())
                    if risk.mean() <= c['maximum_contamination'] and clean.sum() >= 2:
                        candidate = copy.deepcopy(model)
                        tentative_ledger = deque(ledger, maxlen=c['regimes'])
                        signature = recent.mean(0).numpy()
                        match = None
                        if c['variant'] != 'no-recurrence':
                            for entry in ledger:
                                if np.linalg.norm(signature - entry['mean']) / np.sqrt(len(signature)) < c['recurrence_distance']:
                                    match = entry['rule']
                                    break
                        new_rule = False
                        slot = int(model.score(recent, recent_h)[1].mean(0).argmax())
                        if match is not None:
                            slot = match
                        elif float(model.score(recent, recent_h)[1].max(1).values.mean()) < c['match_weight']:
                            new_rule = True
                            free = torch.nonzero(~candidate.active).flatten()
                            slot = int(free[0]) if len(free) else int(candidate.utility.argmin())
                            tentative_ledger = deque([e for e in ledger if e['rule'] != slot], maxlen=c['regimes'])
                        u = torch.tensor(1 - risk, dtype=recent.dtype)
                        u /= u.sum()
                        center = (u[:, None] * recent).sum(0)
                        candidate.centers[slot].lerp_(center, 1. if new_rule else c['online_rate'])
                        width = (u[:, None] * (recent - center).square()).sum(0).sqrt().clamp_min(.1)
                        candidate.log_width[slot].lerp_(torch.log(torch.expm1(width)), 1. if new_rule else c['online_rate'])
                        design = F.pad(recent_h, (0, 1), value=1)
                        weighted = design * u.sqrt()[:, None]
                        beta = torch.linalg.solve(weighted.T @ weighted + c['ridge'] * torch.eye(65), weighted.T @ (recent * u.sqrt()[:, None]))
                        candidate.predictor[slot].lerp_(beta.T, 1. if new_rule else c['online_rate'])
                        if new_rule:
                            relevant = ((recent - model.centers[slot]).abs().mean(0) / width).topk(min(6, recent.shape[1])).indices
                            candidate.gates[slot].fill_(-8)
                            candidate.gates[slot, relevant] = 8
                        residual = (recent - design @ candidate.predictor[slot].T).square()
                        candidate.residual_variance[slot] = (u[:, None] * residual).sum(0).clamp_min(1e-5)
                        candidate.active[slot] = True
                        others = torch.nonzero(candidate.active & (torch.arange(c['max_rules']) != slot)).flatten()
                        if len(others):
                            distance = ((candidate.centers[others] - candidate.centers[slot]) / F.softplus(candidate.log_width[others]).clamp_min(.1)).square().mean(1).sqrt()
                            closest = int(others[distance.argmin()])
                            if float(distance.min()) < c['merge_distance']:
                                candidate.centers[slot].lerp_(candidate.centers[closest], .5)
                                candidate.predictor[slot].lerp_(candidate.predictor[closest], .5)
                                candidate.residual_variance[slot].lerp_(candidate.residual_variance[closest], .5)
                                candidate.active[closest] = False
                                tentative_ledger = deque([dict(e, rule=slot if e['rule'] == closest else e['rule']) for e in tentative_ledger], maxlen=c['regimes'])
                        after = candidate.score(recent, recent_h)[0].numpy()
                        old_damage = F.relu(candidate.score(anchor_q, anchor_h)[0] - model.score(anchor_q, anchor_h)[0]).numpy()
                        pseudo = recent[clean].clone()
                        sensor = int(contributions[0].argmax())
                        pseudo[:, sensor * 3] += c['pseudo_shift'] * torch.where(pseudo[:, sensor * 3] >= center[sensor * 3], 1, -1)
                        fault_margin = (candidate.score(pseudo, recent_h[clean])[0] - candidate.score(recent[clean], recent_h[clean])[0]).numpy()
                        fit_lcb = bound(before[clean] - after[clean], cal, c)
                        old_ucb = bound(old_damage, cal, c, upper=True)
                        fault_lcb = bound(fault_margin, cal, c)
                        checks = [fit_lcb >= c['fit_margin'], old_ucb <= c['anchor_damage'], fault_lcb >= c['fault_margin']]
                        if c['variant'] == 'no-anchor':
                            checks[1] = True
                        if c['variant'] == 'no-fault':
                            checks[2] = True
                        committed = all(checks) or c['variant'] == 'direct'
                        veto = 'none' if committed else ['fit', 'anchor', 'pseudo-fault'][checks.index(False)]
                        slack = min((c['maximum_contamination'] - risk.mean()) / c['maximum_contamination'],
                                    (fit_lcb - c['fit_margin']) / c['fit_margin'],
                                    (c['anchor_damage'] - old_ucb) / c['anchor_damage'],
                                    (fault_lcb - c['fault_margin']) / c['fault_margin'])
                        record.update(committed=committed, veto=veto, slack=float(slack), fit_lcb=fit_lcb,
                                      old_ucb=old_ucb, fault_lcb=fault_lcb, after=(after if committed else before).tolist(),
                                      candidate_after=after.tolist(), recalled=match is not None)
                        if committed:
                            model = candidate
                            ledger = tentative_ledger
                            ref = recent.clone()
                            ledger.append(dict(mean=signature, rule=slot))
                    audits.append(record)
                    persistent = 0
            model.utility.mul_(.99).add_(weights[0], alpha=.01)
            latency_ms = 1000 * (time.perf_counter() - start)
            baseline_start = time.perf_counter()
            isolation_value = float(-isolation.score_samples(q.numpy())[0])
            isolation_latency_ms = 1000 * (time.perf_counter() - baseline_start)
            rows.append(dict(position=t + c['window'] - 1, score=float(score[0]), alarm=bool(score[0] > cal['alarm']),
                             isolation_score=isolation_value, isolation_latency_ms=isolation_latency_ms,
                             isolation_alarm=bool(isolation_value > isolation_threshold), committed=committed,
                             rules=int(model.active.sum()), regimes=len(ledger), deletion_fidelity=fidelity,
                             explanation_stability=stability, top_sensors=json.dumps(channel_ids[ranked.numpy()].tolist()),
                             peak_rss_mib=psutil.Process().memory_info().rss / 2 ** 20,
                             parameters=sum(p.numel() for p in model.parameters()),
                             latency_ms=latency_ms))
            h_previous = model.encoder(x[t:t + 1])[:, :, -1]
    pd.DataFrame(rows).to_csv(out / 'predictions.csv', index=False)
    (out / 'audit.json').write_text(json.dumps(audits))


def evaluate(c, out, labels):
    from affiliation.generics import convert_vector_to_events
    from affiliation.metrics import pr_from_events
    from timeeval.metrics import RangePrVUS
    d = pd.read_csv(out / 'predictions.csv')
    truth = np.load(labels, allow_pickle=False)
    y = truth['label'][d.position]
    def affiliation_f1(prediction):
        if not np.any(y):
            return np.nan
        if not np.any(prediction):
            return 0.
        events = pr_from_events(convert_vector_to_events(np.asarray(prediction).astype(int).tolist()), convert_vector_to_events(y.astype(int).tolist()), (0, len(y)))
        precision, recall = events['precision'], events['recall']
        return 2 * precision * recall / max(precision + recall, 1e-12)

    result = dict(method=c['variant'], affiliation_f1=affiliation_f1(d.alarm.to_numpy()),
                  vus_pr=float(RangePrVUS(max_buffer_size=c['vus_buffer'])(y.astype(int), d.score.to_numpy())) if len(np.unique(y)) == 2 else np.nan,
                  far=float(d.alarm[y == 0].mean()), deletion_fidelity=d.deletion_fidelity.mean(),
                  explanation_stability=d.explanation_stability.mean(), maximum_rules=int(d.rules.max()),
                  maximum_regimes=int(d.regimes.max()), p95_latency_ms=float(d.latency_ms.quantile(.95)), peak_rss_mib=d.peak_rss_mib.max(), parameters=int(d.parameters.iloc[0]))
    if 'affected_sensor' in truth.files:
        scores = []
        for position, top in zip(d.position, d.top_sensors):
            affected = set(np.flatnonzero(truth['affected_sensor'][position]))
            if affected:
                scores.append(len(affected & set(json.loads(top))) / len(affected))
        result['top3_recall'] = float(np.mean(scores)) if scores else np.nan
    audits = json.loads((out / 'audit.json').read_text(encoding='utf-8'))
    if 'change_points' in truth.files:
        boundaries = np.asarray(truth['change_points'])
        proposals = [a['position'] for a in audits]
        unmatched = set(range(len(boundaries)))
        hits = 0
        for proposal in proposals:
            candidates = [i for i in unmatched if abs(boundaries[i] - proposal) <= c['buffer']]
            if candidates:
                unmatched.remove(min(candidates, key=lambda i: abs(boundaries[i] - proposal)))
                hits += 1
        result['drift_precision'] = hits / len(proposals) if proposals else 0.
        result['drift_recall'] = hits / len(boundaries) if len(boundaries) else np.nan
    if 'regime_return' in truth.files:
        delays = []
        for returned in truth['regime_return']:
            after_return = np.flatnonzero(d.position >= returned)
            recovered = None
            for j in after_return:
                if j + 1 < c['buffer']:
                    continue
                ids = np.arange(j + 1 - c['buffer'], j + 1)
                normal = y[ids] == 0
                if normal.any() and d.alarm.iloc[ids].to_numpy()[normal].mean() <= .005:
                    recovered = int(d.position.iloc[j] - returned)
                    break
            delays.append(recovered if recovered is not None else int(d.position.max() - returned))
        result['recurrence_recovery_samples'] = float(np.mean(delays)) if delays else np.nan
    if audits and 'fault_mask' in truth.files:
        good, confidence, retention = [], [], []
        for a in audits:
            start = a['position'] - len(a['before']) + 1
            mask = truth['fault_mask'][start:a['position'] + 1].astype(bool)
            harmful = bool(mask.any())
            good.append(a['committed'] != harmful)
            confidence.append(float(expit(abs(a['slack']))))
            if mask.any():
                threshold = json.loads((out / 'calibration.json').read_text(encoding='utf-8'))['alarm']
                before = np.array(a['before'])[mask] - threshold
                after = np.array(a['after'])[mask] - threshold
                valid = before > 1e-8
                retention.extend((after[valid] / before[valid]).tolist())
        result['correct_action_rate'] = float(np.mean(good))
        result['anomaly_retention_ratio'] = float(np.mean(retention)) if retention else np.nan
        groups = np.array_split(np.argsort(confidence), min(10, len(confidence)))
        result['certificate_ece'] = sum(len(g) / len(good) * abs(np.mean(np.array(good)[g]) - np.mean(np.array(confidence)[g])) for g in groups)
        for veto in ['contamination', 'fit', 'anchor', 'pseudo-fault']:
            selected = [i for i, a in enumerate(audits) if a['veto'] == veto]
            result[veto + '_veto_precision'] = float(np.mean(np.array(good)[selected])) if selected else np.nan
    baseline = dict(method='IsolationForest', affiliation_f1=affiliation_f1(d.isolation_alarm.to_numpy()),
                    vus_pr=float(RangePrVUS(max_buffer_size=c['vus_buffer'])(y.astype(int), d.isolation_score.to_numpy())) if len(np.unique(y)) == 2 else np.nan,
                    far=float(d.isolation_alarm[y == 0].mean()), p95_latency_ms=float(d.isolation_latency_ms.quantile(.95)))
    pd.DataFrame([result, baseline]).to_csv(out / 'metrics.csv', index=False)


def plot(out):
    d = pd.read_csv(out / 'predictions.csv')
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(d.position, d.score)
    ax.scatter(d[d.committed].position, d[d.committed].score, s=12)
    ax.set(xlabel='Timestamp index', ylabel='Causal anomaly score')
    fig.tight_layout()
    fig.savefig(out / 'monitor.png', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'train', 'run', 'evaluate', 'plot'])
    p.add_argument('--input')
    p.add_argument('--config', default='config.json')
    p.add_argument('--output', default='runs/default')
    p.add_argument('--variant')
    args = p.parse_args()
    c = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.variant:
        c['variant'] = args.variant
    if c['variant'] not in ['full', 'frozen', 'direct', 'raw-means', 'no-anchor', 'no-fault', 'no-recurrence', 'descriptor-only', 'occupancy-only', 'no-counterfactual', 'dense']:
        p.error('Unknown variant')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    settings = out / 'settings.json'
    if args.variant is None and settings.exists():
        c['variant'] = json.loads(settings.read_text(encoding='utf-8')).get('variant', c.get('variant', 'full'))
    settings.write_text(json.dumps(c, indent=2))
    if args.command in ['prepare', 'evaluate']:
        if not args.input:
            p.error('--input required')
        globals()[args.command](c, out, args.input)
    elif args.command == 'plot':
        plot(out)
    else:
        globals()[args.command](c, out)


if __name__ == '__main__':
    main()
