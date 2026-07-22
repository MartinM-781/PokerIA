//! Résolution de sous-jeu en temps réel (« search ») : CFR range-contre-range
//! sur l'arbre d'enchères restant de la street courante.
//!
//! Feuilles :
//! - fold : le pot (contributions réelles) va au survivant ;
//! - all-in payé : équité exacte main-contre-main sur runouts échantillonnés ;
//! - fin de street (mises égalisées) : partage du pot par la même équité —
//!   approximation « on checke jusqu'à l'abattage » des streets suivantes.
//!
//! Les ranges des deux joueurs sont fournies par l'appelant (traquées via le
//! blueprint). Renvoie la stratégie moyenne du héros pour SA main réelle.

use crate::eval::evaluate7;
use crate::game::{ALL_IN, BB, CHECK_CALL, FOLD, RAISE_CAP, RAISE_HALF, RAISE_POT, RAISE_THIRD};
use crate::rng::Rng;

const N_ACT: usize = 6;

/// Matrice d'équité main-vs-main sur un board partiel : equity[i][j] = P(main i
/// bat main j), estimée sur `n_runouts` tirages de fin de board communs.
fn equity_matrix(
    board: &[u8],
    hands: &[(u8, u8)],
    n_runouts: usize,
    rng: &mut Rng,
) -> Vec<Vec<f32>> {
    let n = hands.len();
    let mut wins = vec![vec![0f32; n]; n];
    let need = 5 - board.len();
    let mut blocked = [false; 52];
    for &c in board {
        blocked[c as usize] = true;
    }
    let base_deck: Vec<u8> = (0..52u8).filter(|&c| !blocked[c as usize]).collect();

    let mut seven = [0u8; 7];
    seven[2..2 + board.len()].copy_from_slice(board);

    for _ in 0..n_runouts {
        let mut deck = base_deck.clone();
        rng.partial_shuffle(&mut deck, need);
        let runout = &deck[..need];
        // score de chaque main pour ce runout (usize::MAX si conflit board/runout)
        let mut scores: Vec<u32> = Vec::with_capacity(n);
        for &(a, b) in hands {
            if runout.contains(&a) || runout.contains(&b) {
                scores.push(u32::MAX); // main impossible sur ce runout
                continue;
            }
            seven[0] = a;
            seven[1] = b;
            seven[2 + board.len()..].copy_from_slice(runout);
            scores.push(evaluate7(&seven));
        }
        for i in 0..n {
            if scores[i] == u32::MAX {
                continue;
            }
            for j in 0..n {
                if i == j || scores[j] == u32::MAX {
                    continue;
                }
                if scores[i] > scores[j] {
                    wins[i][j] += 1.0;
                } else if scores[i] == scores[j] {
                    wins[i][j] += 0.5;
                }
            }
        }
    }
    for i in 0..n {
        for j in 0..n {
            wins[i][j] /= n_runouts as f32;
        }
    }
    wins
}

/// Un nœud de l'arbre d'enchères restant de la street.
struct Node {
    player: usize,           // 0 = héros, 1 = adversaire
    legal: Vec<u8>,
    children: Vec<usize>,    // indices des enfants, alignés sur legal
    terminal: Option<Leaf>,
}

#[derive(Clone, Copy)]
enum Leaf {
    Fold { winner: usize, pot_winner: i32 },
    Showdown { pot_half: i32 }, // chacun a misé pot_half au total
}

/// Construit l'arbre d'enchères restant (street courante uniquement).
struct TreeBuilder {
    nodes: Vec<Node>,
}

#[allow(clippy::too_many_arguments)]
impl TreeBuilder {
    fn build(
        &mut self,
        player: usize,
        invested: [i32; 2], // contributions TOTALES à la main (streets passées incluses)
        bets: [i32; 2],     // mises de la street en cours
        stacks: [i32; 2],
        last_raise: i32,
        raises: u8,
        acted: [bool; 2],
    ) -> usize {
        let o = 1 - player;
        let to_call = bets[o] - bets[player];
        // légales — copie de game.rs (simplifiée : mêmes règles)
        let mut legal = Vec::with_capacity(6);
        if to_call > 0 {
            legal.push(FOLD);
        }
        legal.push(CHECK_CALL);
        if stacks[player] > to_call && stacks[o] > 0 {
            if raises < RAISE_CAP {
                if raises == 0 {
                    legal.push(RAISE_THIRD);
                }
                legal.push(RAISE_HALF);
                legal.push(RAISE_POT);
            }
            legal.push(ALL_IN);
        }

        let idx = self.nodes.len();
        self.nodes.push(Node { player, legal: legal.clone(), children: Vec::new(), terminal: None });

        let mut children = Vec::with_capacity(legal.len());
        for &a in &legal {
            let child = self.apply(a, player, invested, bets, stacks, last_raise, raises, acted);
            children.push(child);
        }
        self.nodes[idx].children = children;
        idx
    }

    #[allow(clippy::too_many_arguments)]
    fn apply(
        &mut self,
        action: u8,
        player: usize,
        mut invested: [i32; 2],
        mut bets: [i32; 2],
        mut stacks: [i32; 2],
        mut last_raise: i32,
        mut raises: u8,
        mut acted: [bool; 2],
    ) -> usize {
        let o = 1 - player;
        let to_call = bets[o] - bets[player];

        if action == FOLD {
            let idx = self.nodes.len();
            // Valeur nette du fold : le foldeur perd sa mise égalisée, le
            // survivant la gagne (sa propre mise non suivie lui est rendue).
            let matched = invested[player].min(invested[o]);
            self.nodes.push(Node {
                player: o,
                legal: Vec::new(),
                children: Vec::new(),
                terminal: Some(Leaf::Fold { winner: o, pot_winner: matched }),
            });
            return idx;
        }

        if action == CHECK_CALL {
            let pay = to_call.min(stacks[player]);
            stacks[player] -= pay;
            invested[player] += pay;
            bets[player] += pay;
            acted[player] = true;
            let low = if bets[0] <= bets[1] { 0 } else { 1 };
            let settled = bets[0] == bets[1] || stacks[low] == 0;
            if acted[0] && acted[1] && settled {
                let idx = self.nodes.len();
                let matched = invested[0].min(invested[1]);
                self.nodes.push(Node {
                    player: o,
                    legal: Vec::new(),
                    children: Vec::new(),
                    terminal: Some(Leaf::Showdown { pot_half: matched }),
                });
                return idx;
            }
            return self.build(o, invested, bets, stacks, last_raise, raises, acted);
        }

        // relances
        let pot_now: i32 = invested[0] + invested[1];
        let pot_after_call = pot_now + to_call;
        let mut raise_by = match action {
            RAISE_THIRD => pot_after_call / 3,
            RAISE_HALF => pot_after_call / 2,
            RAISE_POT => pot_after_call,
            _ => stacks[player],
        };
        raise_by = raise_by.max(last_raise).max(BB);
        let add = (to_call + raise_by).min(stacks[player]);
        stacks[player] -= add;
        invested[player] += add;
        bets[player] += add;
        let actual = bets[player] - bets[o];
        if actual > 0 {
            last_raise = actual.max(BB);
            raises += 1;
        }
        acted = [false, false];
        acted[player] = true;
        self.build(o, invested, bets, stacks, last_raise, raises, acted)
    }
}

/// Résout la street courante par CFR range-vs-range et renvoie, pour chaque
/// main possible du héros, sa stratégie moyenne au nœud racine (n_hands × 6).
#[allow(clippy::too_many_arguments)]
pub fn solve_street(
    board: &[u8],
    hero_hands: &[(u8, u8)],
    hero_weights: &[f64],
    opp_hands: &[(u8, u8)],
    opp_weights: &[f64],
    invested: [i32; 2],  // héros = 0, adversaire = 1 (contributions totales)
    bets: [i32; 2],
    stacks: [i32; 2],
    last_raise: i32,
    raises: u8,
    acted: [bool; 2],
    hero_to_act: bool,
    iterations: usize,
    n_runouts: usize,
    seed: u64,
) -> Vec<[f64; N_ACT]> {
    let mut rng = Rng::new(seed);

    // 1. Matrice d'équité héros × adversaire
    let nh = hero_hands.len();
    let no = opp_hands.len();
    let mut all: Vec<(u8, u8)> = Vec::with_capacity(nh + no);
    all.extend_from_slice(hero_hands);
    all.extend_from_slice(opp_hands);
    let eq_all = equity_matrix(board, &all, n_runouts, &mut rng);
    // eq[i][j] = P(héros i bat adversaire j)
    let eq: Vec<Vec<f32>> = (0..nh)
        .map(|i| (0..no).map(|j| eq_all[i][nh + j]).collect())
        .collect();

    // 2. Arbre d'enchères de la street
    let first = if hero_to_act { 0usize } else { 1usize };
    let mut tb = TreeBuilder { nodes: Vec::new() };
    let root = tb.build(first, invested, bets, stacks, last_raise, raises, acted);
    let nodes = tb.nodes;

    // 3. CFR : regrets/stratégies par (nœud, main) du joueur au trait
    let mut regrets: Vec<Vec<[f64; N_ACT]>> = nodes
        .iter()
        .map(|nd| vec![[0f64; N_ACT]; if nd.player == 0 { nh } else { no }])
        .collect();
    let mut strat_sum = regrets.clone();

    // blocage des combinaisons impossibles (cartes partagées)
    let conflict: Vec<Vec<bool>> = hero_hands
        .iter()
        .map(|&(a, b)| {
            opp_hands
                .iter()
                .map(|&(c, d)| a == c || a == d || b == c || b == d)
                .collect()
        })
        .collect();

    // Traversée vectorisée sur les mains : à chaque nœud, on propage les
    // valeurs attendues main-par-main du joueur au trait contre la range
    // adverse courante. Implémentation « vanilla CFR » sur petit arbre.
    //
    // reach[h] : probabilité que le joueur concerné joue jusqu'ici avec la main h.
    fn cfr(
        node_idx: usize,
        nodes: &[Node],
        regrets: &mut [Vec<[f64; N_ACT]>],
        strat_sum: &mut [Vec<[f64; N_ACT]>],
        hero_reach: &[f64],
        opp_reach: &[f64],
        eq: &[Vec<f32>],
        conflict: &[Vec<bool>],
        hero_w: &[f64],
        opp_w: &[f64],
    ) -> Vec<f64> {
        let nd = &nodes[node_idx];
        let nh = hero_reach.len();
        let no = opp_reach.len();

        if let Some(leaf) = &nd.terminal {
            // valeur pour CHAQUE main du héros contre la range adverse atteinte
            let mut vals = vec![0f64; nh];
            for i in 0..nh {
                let mut num = 0f64;
                let mut den = 0f64;
                for j in 0..no {
                    if conflict[i][j] {
                        continue;
                    }
                    let w = opp_reach[j] * opp_w[j];
                    if w <= 0.0 {
                        continue;
                    }
                    let v = match *leaf {
                        Leaf::Fold { winner, pot_winner } => {
                            if winner == 0 { pot_winner as f64 } else { -(pot_winner as f64) }
                        }
                        Leaf::Showdown { pot_half } => {
                            (2.0 * eq[i][j] as f64 - 1.0) * pot_half as f64
                        }
                    };
                    num += w * v;
                    den += w;
                }
                vals[i] = if den > 0.0 { num / den } else { 0.0 };
            }
            let _ = hero_w;
            return vals;
        }

        let na = nd.legal.len();
        if nd.player == 0 {
            // stratégie courante par regret matching, main par main
            let mut sigma = vec![[0f64; N_ACT]; nh];
            for i in 0..nh {
                let mut tot = 0.0;
                for &a in &nd.legal {
                    let r = regrets[node_idx][i][a as usize].max(0.0);
                    sigma[i][a as usize] = r;
                    tot += r;
                }
                if tot > 1e-12 {
                    for &a in &nd.legal {
                        sigma[i][a as usize] /= tot;
                    }
                } else {
                    for &a in &nd.legal {
                        sigma[i][a as usize] = 1.0 / na as f64;
                    }
                }
            }
            // valeurs des enfants
            let mut child_vals: Vec<Vec<f64>> = Vec::with_capacity(na);
            for (k, &a) in nd.legal.iter().enumerate() {
                let mut hr = vec![0f64; nh];
                for i in 0..nh {
                    hr[i] = hero_reach[i] * sigma[i][a as usize];
                }
                child_vals.push(cfr(
                    nd.children[k], nodes, regrets, strat_sum, &hr, opp_reach,
                    eq, conflict, hero_w, opp_w,
                ));
            }
            let mut vals = vec![0f64; nh];
            for i in 0..nh {
                let mut v = 0.0;
                for (k, &a) in nd.legal.iter().enumerate() {
                    v += sigma[i][a as usize] * child_vals[k][i];
                }
                vals[i] = v;
                for (k, &a) in nd.legal.iter().enumerate() {
                    regrets[node_idx][i][a as usize] =
                        (regrets[node_idx][i][a as usize] + child_vals[k][i] - v).max(0.0);
                    strat_sum[node_idx][i][a as usize] += hero_reach[i] * sigma[i][a as usize];
                }
            }
            vals
        } else {
            // adversaire : même mécanique, mais les valeurs remontées sont
            // celles du héros ; l'adversaire minimise (il maximise son gain =
            // minimise le nôtre) via SES regrets sur -valeur.
            let mut sigma = vec![[0f64; N_ACT]; no];
            for j in 0..no {
                let mut tot = 0.0;
                for &a in &nd.legal {
                    let r = regrets[node_idx][j][a as usize].max(0.0);
                    sigma[j][a as usize] = r;
                    tot += r;
                }
                if tot > 1e-12 {
                    for &a in &nd.legal {
                        sigma[j][a as usize] /= tot;
                    }
                } else {
                    for &a in &nd.legal {
                        sigma[j][a as usize] = 1.0 / na as f64;
                    }
                }
            }
            let mut child_vals: Vec<Vec<f64>> = Vec::with_capacity(na);
            for (k, &a) in nd.legal.iter().enumerate() {
                let mut or_ = vec![0f64; no];
                for j in 0..no {
                    or_[j] = opp_reach[j] * sigma[j][a as usize];
                }
                child_vals.push(cfr(
                    nd.children[k], nodes, regrets, strat_sum, hero_reach, &or_,
                    eq, conflict, hero_w, opp_w,
                ));
            }
            // valeur héros = somme pondérée par la stratégie adverse « moyenne
            // de range » ; les regrets adverses utilisent la valeur RETOURNÉE
            // (approximation : valeur moyenne côté héros, suffisante pour un
            // sous-jeu d'une street à petit arbre)
            let mut vals = vec![0f64; nh];
            // poids moyen de chaque action adverse (pondéré par sa range)
            let mut awt = [0f64; N_ACT];
            let mut wtot = 0.0;
            for j in 0..no {
                let w = opp_reach[j] * opp_w[j];
                wtot += w;
                for &a in &nd.legal {
                    awt[a as usize] += w * sigma[j][a as usize];
                }
            }
            if wtot > 0.0 {
                for &a in &nd.legal {
                    awt[a as usize] /= wtot;
                }
            }
            for i in 0..nh {
                let mut v = 0.0;
                for (k, &a) in nd.legal.iter().enumerate() {
                    v += awt[a as usize] * child_vals[k][i];
                }
                vals[i] = v;
            }
            // mise à jour des regrets adverses : valeur adverse = -valeur héros
            // moyennée sur la range héros atteinte
            for j in 0..no {
                let mut per_action = [0f64; N_ACT];
                let mut base = 0.0;
                for (k, &a) in nd.legal.iter().enumerate() {
                    let mut num = 0.0;
                    let mut den = 0.0;
                    for i in 0..nh {
                        if conflict[i][j] {
                            continue;
                        }
                        let w = hero_reach[i] * hero_w[i];
                        num += w * (-child_vals[k][i]);
                        den += w;
                    }
                    let av = if den > 0.0 { num / den } else { 0.0 };
                    per_action[a as usize] = av;
                    base += sigma[j][a as usize] * av;
                }
                for &a in &nd.legal {
                    regrets[node_idx][j][a as usize] =
                        (regrets[node_idx][j][a as usize] + per_action[a as usize] - base).max(0.0);
                    strat_sum[node_idx][j][a as usize] += opp_reach[j] * sigma[j][a as usize];
                }
            }
            vals
        }
    }

    let hero_reach = vec![1f64; nh];
    let opp_reach = vec![1f64; no];
    for _ in 0..iterations {
        cfr(
            root, &nodes, &mut regrets, &mut strat_sum, &hero_reach, &opp_reach,
            &eq, &conflict, hero_weights, opp_weights,
        );
    }

    // Stratégie moyenne du héros à la racine (dans notre usage, la racine est
    // toujours le héros au trait).
    let root_node = &nodes[root];
    let src = root;
    let mut out = vec![[0f64; N_ACT]; nh];
    for i in 0..nh {
        let mut tot = 0.0;
        for a in 0..N_ACT {
            tot += strat_sum[src][i][a];
        }
        if tot > 1e-12 {
            for a in 0..N_ACT {
                out[i][a] = strat_sum[src][i][a] / tot;
            }
        } else {
            let na = root_node.legal.len().max(1);
            for &a in &root_node.legal {
                out[i][a as usize] = 1.0 / na as f64;
            }
        }
    }
    out
}
