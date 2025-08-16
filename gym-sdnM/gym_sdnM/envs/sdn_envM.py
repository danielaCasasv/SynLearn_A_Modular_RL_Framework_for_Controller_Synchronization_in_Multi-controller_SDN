import networkx as nx
import numpy as np
import heapq
import gymnasium as gym
import gym_sdnM.topologies as tp
import logging
from gymnasium.spaces import Box, Discrete

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuração do ambiente/topologia ---
WEIGHTS_CHANGE_TIME = 2
SYNC_TIME = 2
arrival_rates = [32, 64, 18, 51, 34, 56, 94, 93, 94, 90, 97, 91, 92, 98,
                 53, 80, 80, 14, 22, 34, 15, 38, 19, 73, 57, 68, 70, 79, 70,
                 44, 36, 34, 77]
max_tslots = 30
n_domains = 5

# Ações: índice 
ACTION_MAP = [
    [0, 0, 0, 0],  # 0 = no sync
    [1, 0, 0, 0],  # 1 = sync com B
    [0, 1, 0, 0],  # 2 = sync com C
    [0, 0, 1, 0],  # 3 = sync com D
    [0, 0, 0, 1],  # 4 = sync com E
]

_EVENT_PRIORITY = {'weights_change': 0, 'sync': 1}

# === parámetros de reward/costo ===
SYNC_COST = 0.05   # en [0,1]
ETA = 2
KAPPA = 1.2        # peso de mejora incremental Δ
BETA = 0.6         # peso del costo de sync
GAMMA = 0.1        # ancla contra diffs grandes

# Umbrales para shaping
TAU_MISFIRE = 0.01   # mejora mínima para considerar que valió la pena
TAU_TREND   = 0.01   # umbral de 'está empeorando'
DELAY_W     = 0.6    # penalización por demorar
MISFIRE_W   = 1.2    # penalización por sync sin mejora
 
# ---------- Tráfego/ Pesos dinâmicos ----------
def dynamic_arrival_rates_with_noise(t, base_arrival_rates, amplitude=30, period=60, noise_std=3):
    base_lambda = np.array(base_arrival_rates) + amplitude * np.sin(2 * np.pi * t / period)
    noise = np.random.normal(0, noise_std, size=len(base_arrival_rates))
    return np.maximum(1, base_lambda + noise)

def get_new_weights(t, base_arrival_rates):
    lam_t = dynamic_arrival_rates_with_noise(t, base_arrival_rates)
    return np.random.poisson(lam=lam_t).astype(int).tolist()

def update_weights(controllers, real_net, new_weights=None):
    if new_weights is None:
        new_weights = get_new_weights(0, arrival_rates)
    real_links = real_net.topology.graph['links']
    for i, link in enumerate(real_links):
        link['weight'] = new_weights[i]
    for ctrl in controllers:
        for i, link_ctrl in enumerate(ctrl.network.topology.graph['links']):
            if link_ctrl.get('domain') in (ctrl.domain, ctrl.interdomain):
                link_ctrl['weight'] = new_weights[i]

# ---------- Sincronização ----------
def sync(controller_X, controller_Y):
    links_X = controller_X.network.topology.graph['links']
    links_Y = controller_Y.network.topology.graph['links']
    for i, lY in enumerate(links_Y):
        if lY.get('domain') in (controller_Y.domain, controller_Y.interdomain):
            links_X[i]['weight'] = lY['weight']
    pos = {'B': 0, 'C': 1, 'D': 2, 'E': 3}
    controller_X.desync_list[pos[controller_Y.domain]] = 0

# ---------- Métrica / Recompensa ----------
def get_reward(controller, real_net):
    path_costs, diffs = [], []

    ctrl_G, real_G = nx.Graph(), nx.Graph()

    for link in controller.network.topology.graph['links']:
        u, v = link['source'], link['target']
        ctrl_G.add_edge(u, v, weight=link['weight'])

    for link in real_net.topology.graph['links']:
        u, v = link['source'], link['target']
        real_G.add_edge(u, v, weight=link['weight'])
        
    path_costs, diffs = [], []

    for src in range(4):
        # single_source_dijkstra devuelve distancias y predecesores
        _, paths_ctrl = nx.single_source_dijkstra(ctrl_G, src, weight='weight')
        for dst in range(4, 23):
            if dst not in paths_ctrl:
                continue
            path = paths_ctrl[dst]
            cost_ctrl = sum(ctrl_G[u][v]['weight'] for u, v in zip(path, path[1:]))
            cost_real = sum(real_G[u][v]['weight'] for u, v in zip(path, path[1:]))
            path_costs.append(cost_real)
            diffs.append(abs(cost_real - cost_ctrl))
    if not diffs:
        return 0.0, 0.0
    return float(np.mean(diffs)), float(np.mean(path_costs))

# ---------- Estruturas stma de eventos ----------
class Network:
    def __init__(self, topology):
        self.topology = topology

class Controller:
    def __init__(self, domain, interdomain=''):
        self.domain = domain
        self.interdomain = interdomain
        self.network = Network(tp.get_topo('multidomain_topo'))
        self.desync_list = [0] * (n_domains - 1)

class Evento:
    def __init__(self, tipo, inicio, extra, function):
        self.tipo = tipo
        self.inicio = inicio
        self.extra = extra
        self.function = function
    def __repr__(self):
        return f'Evento({self.tipo}@{self.inicio})'

# ---------- Ambiente baseado em gymnasium ----------
class SdnEnvM(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, alpha=0.5, shaping_coeff=0.1, sync_cost=0.02, envelope_decay=0.995):
        super().__init__()
        self.network = Network(tp.get_topo('multidomain_topo'))
        self.controllers = [
            Controller(d, i) for d, i in zip(
                ['A', 'B', 'C', 'D', 'E'],
                ['AB', 'BC', 'CD', 'DE', '']
            )
        ]
        self.horario = 0
        self.run_till = max_tslots
        self.event_heap = []
        self._init_sim()

        # ---- Nuevos estados/contadores ----
        self.last_sync_step = 0
        self.steps = 0
        self.trend_smooth = 0.0
        self.trend_beta = 0.5  # EMA del trend

        self.alpha = alpha
        self.shaping_coeff = shaping_coeff

        G = nx.node_link_graph(self.network.topology.graph)
        self.max_hops = nx.diameter(G.to_undirected())
        self.apc_max = (np.max(arrival_rates) + 30) * self.max_hops
        # Envelopes dinâmicos
        self.envelope_decay = envelope_decay
        
        # Normalizações dinâmicas 
        self.diff_max_dyn = 1.0
        self.apc_max_dyn = 1.0

        self._last_apc = None
        self._last_diff = None
        self._last_mean_diff = None

        self.sync_cost = sync_cost  # custo unitário por sync (se ação != 0)

        self.action_space = Discrete(len(ACTION_MAP))
        dims = 2 * (n_domains - 1)
        self.observation_space = Box(low=0.0, high=1.0, shape=(dims+2,), dtype=np.float32)

    def _add_event(self, evt):
        prio = _EVENT_PRIORITY[evt.tipo]
        heapq.heappush(self.event_heap, (evt.inicio, prio, evt))

    def _init_sim(self):
        self._add_event(Evento('weights_change', WEIGHTS_CHANGE_TIME, None, func_update_weights))
        self._add_event(Evento('sync', SYNC_TIME, None, func_synchronize))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.horario = 0
        self.network = Network(tp.get_topo('multidomain_topo'))
        self.controllers = [
            Controller(d, i) for d, i in zip(
                ['A', 'B', 'C', 'D', 'E'],
                ['AB', 'BC', 'CD', 'DE', '']
            )
        ]
        self.event_heap = []
        self._init_sim()

        # reinicia envelopes
        self.diff_max_dyn = 1.0
        self.apc_max_dyn = 1.0
        self.steps = 0
        self.last_sync_step = 0
        self.trend_smooth = 0.0
        self._last_mean_diff = None

        return self._get_observation(), {}

    def _step_until_sync(self, action):
        while self.event_heap:
            t, _, evt = heapq.heappop(self.event_heap)
            self.horario = t
            if evt.tipo == 'weights_change':
                # Usa o callback padrão (mantém desync_list e usa pesos dinâmicos)
                evt.function(self, evt)
                self._add_event(Evento('weights_change', t + WEIGHTS_CHANGE_TIME, None, func_update_weights))
            else:
                evt.function(self, evt, action)
                self._add_event(Evento('sync', t + SYNC_TIME, None, func_synchronize))
                return

    def _get_observation(self):
        desync = np.array(self.controllers[0].desync_list, dtype=np.float32) / max_tslots
        diffs_por_dom = []
        real_links = self.network.topology.graph['links']
        # ctrl_links = self.controllers[0].network.topology.graph['links']
        for i in range(n_domains - 1):
            diffs = []
            ctrl_links_dom = self.controllers[i+1].network.topology.graph['links']
            for idx, l_real in enumerate(real_links):
                if ctrl_links_dom[idx].get('domain') in (self.controllers[i+1].domain, self.controllers[i+1].interdomain):
                    diffs.append(abs(l_real['weight'] - ctrl_links_dom[idx]['weight']))
            diffs_por_dom.append(np.mean(diffs))

        # normaliza por envelope dinâmico da dif (usa fallback pra evitar div/0)
        denom = max(1.0, self.diff_max_dyn)
        diffs_por_dom = np.array(diffs_por_dom, dtype=np.float32) / denom
        # trend normalizado (EMA) y urgencia temporal desde último sync
        trend_norm = np.array([self.trend_smooth], dtype=np.float32)
        since_last_sync = (self.steps - self.last_sync_step) / max_tslots
        since_last_sync = np.array([np.clip(since_last_sync, 0.0, 1.0)], dtype=np.float32)

        return np.concatenate([desync, diffs_por_dom, trend_norm, since_last_sync], axis=0)

    def step(self, action):
        self._step_until_sync(action)
        self.steps += 1
        mean_diff, apc = get_reward(self.controllers[0], self.network)
        
        self.diff_max_dyn = max(mean_diff, self.diff_max_dyn * self.envelope_decay)
        self.apc_max_dyn  = max(apc,       self.apc_max_dyn  * self.envelope_decay)

        mean_diff_norm = mean_diff / max(1.0, self.diff_max_dyn)

        if self._last_mean_diff is None:
            delta = 0.0
            raw_trend = 0.0

        else:
            delta = (self._last_mean_diff - mean_diff) / max(1.0, self.diff_max_dyn)
            raw_trend = (mean_diff - self._last_mean_diff) / max(1.0, self.diff_max_dyn)  # >0 si empeora

        self._last_mean_diff = mean_diff

        self.trend_smooth = (1 - self.trend_beta) * self.trend_smooth + self.trend_beta * raw_trend

        sync_flag = 1 if action != 0 else 0

        improve = max(delta, 0.0)
        worsen  = max(-delta, 0.0)
        reward = (
            ETA * improve
            - KAPPA * worsen
            - BETA * self.sync_cost * sync_flag
            - GAMMA * mean_diff_norm
        )

        obs_tmp = self._get_observation()
        urgency_time = float(obs_tmp[-1])
        urgency_desync = float(np.mean(obs_tmp[:(n_domains-1)]))
        urgency = float(np.clip(0.5*urgency_time + 0.5*urgency_desync, 0.0, 1.0))

        misfire = 1 if (sync_flag == 1 and improve <= TAU_MISFIRE) else 0
        delay   = 1 if (sync_flag == 0 and self.trend_smooth > TAU_TREND and urgency > 0.5) else 0

        if misfire:
            reward -= MISFIRE_W * urgency
        if delay:
            reward -= DELAY_W * (urgency - 0.5) * 2.0

        state = obs_tmp
        done = (self.horario >= self.run_till)
        info = {
            'APC': apc,
            'mean_diff': mean_diff,
            'mean_diff_norm': mean_diff_norm,
            'delta_diff': delta,
            'trend': self.trend_smooth,
            'sync_cost': self.sync_cost * sync_flag,
            'reward': reward,
            'urgency': urgency,
            'misfire': misfire,
            'delay': delay
        }
        return state, reward, done, False, info

# ---------- Callbacks de eventos ----------
def func_update_weights(sim, evt):
    new_w = get_new_weights(sim.horario, arrival_rates)
    update_weights(sim.controllers, sim.network, new_weights=new_w)
    for ctrl in sim.controllers:
        ctrl.desync_list = [d + WEIGHTS_CHANGE_TIME for d in ctrl.desync_list]

def func_synchronize(sim, evt, action):
    action_vec = ACTION_MAP[action]
    if 1 in action_vec:
        idx = action_vec.index(1)
        sync(sim.controllers[0], sim.controllers[idx+1])
        sim.last_sync_step = sim.steps
