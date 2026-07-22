//! Moteur heads-up no-limit — port exact, jeton pour jeton, de poker_ai/game.py.

use crate::eval::evaluate7;

pub const FOLD: u8 = 0;
pub const CHECK_CALL: u8 = 1;
pub const RAISE_HALF: u8 = 2;
pub const RAISE_POT: u8 = 3;
pub const ALL_IN: u8 = 4;
pub const RAISE_QUARTER: u8 = 5;
pub const RAISE_THIRD: u8 = 6;

pub const PREFLOP: u8 = 0;
pub const RIVER: u8 = 3;

pub const SB: i32 = 1;
pub const BB: i32 = 2;
pub const START_STACK: i32 = 200;

const BOARD_LEN: [usize; 4] = [0, 3, 4, 5];

#[derive(Clone)]
pub struct Hand {
    pub hole: [[u8; 2]; 2],
    pub full_board: [u8; 5],
    pub button: usize,
    pub stacks: [i32; 2],
    pub invested: [i32; 2],
    pub bets: [i32; 2],
    pub street: u8,
    pub acted: [bool; 2],
    pub last_raise: i32,
    pub raises_this_street: u8,
    pub raise_counts: [u8; 2],
    pub street_raise_counts: [u8; 2],
    pub last_aggressor: i8, // -1 = personne
    pub terminal: bool,
    pub showdown: bool,
    pub winner: i8, // -1 = partage, -2 = pas encore fini
    pub payoffs: [i32; 2],
    pub history: Vec<(u8, u8, u8)>, // (street, joueur, action effective)
    pub to_act: usize,
}

impl Hand {
    /// `deck` : les 9 premières cartes servent (2+2 fermées, 5 board).
    pub fn new(deck: &[u8], button: usize) -> Self {
        let mut h = Hand {
            hole: [[deck[0], deck[1]], [deck[2], deck[3]]],
            full_board: [deck[4], deck[5], deck[6], deck[7], deck[8]],
            button,
            stacks: [START_STACK, START_STACK],
            invested: [0, 0],
            bets: [0, 0],
            street: PREFLOP,
            acted: [false, false],
            last_raise: BB,
            raises_this_street: 0,
            raise_counts: [0, 0],
            street_raise_counts: [0, 0],
            last_aggressor: -1,
            terminal: false,
            showdown: false,
            winner: -2,
            payoffs: [0, 0],
            history: Vec::with_capacity(16),
            to_act: button,
        };
        h.commit(button, SB);
        h.commit(1 - button, BB);
        h
    }

    #[inline]
    pub fn pot(&self) -> i32 {
        self.invested[0] + self.invested[1]
    }

    pub fn board(&self) -> &[u8] {
        &self.full_board[..BOARD_LEN[self.street as usize]]
    }

    fn commit(&mut self, p: usize, amount: i32) {
        let a = amount.min(self.stacks[p]);
        self.stacks[p] -= a;
        self.invested[p] += a;
        self.bets[p] += a;
    }

    pub fn legal_actions(&self) -> Vec<u8> {
        let p = self.to_act;
        let o = 1 - p;
        let mut legal = Vec::with_capacity(5);
        if self.bets[o] > self.bets[p] {
            legal.push(FOLD);
        }
        legal.push(CHECK_CALL);
        if self.stacks[p] > self.bets[o] - self.bets[p] && self.stacks[o] > 0 {
            legal.push(RAISE_QUARTER);
            legal.push(RAISE_THIRD);
            legal.push(RAISE_HALF);
            legal.push(RAISE_POT);
            legal.push(ALL_IN);
        }
        legal
    }

    pub fn step(&mut self, action: u8) {
        debug_assert!(!self.terminal);
        let p = self.to_act;
        let o = 1 - p;
        let to_call = self.bets[o] - self.bets[p];

        if action == FOLD && to_call > 0 {
            self.history.push((self.street, p as u8, FOLD));
            self.finish_fold(o);
            return;
        }

        // Une action de relance = RAISE_HALF/POT/ALL_IN (2..4) ou RAISE_QUARTER/
        // THIRD (5..6). CHECK_CALL (1) et FOLD (0) n'en sont pas.
        let is_raise = action == RAISE_HALF || action == RAISE_POT || action == ALL_IN
            || action == RAISE_QUARTER || action == RAISE_THIRD;
        let can_raise = self.stacks[p] > to_call && self.stacks[o] > 0;
        let mut action = action;
        if is_raise && !can_raise {
            action = CHECK_CALL; // garde-fou : action illégale rétrogradée
        }

        if action == FOLD || action == CHECK_CALL {
            self.history.push((self.street, p as u8, CHECK_CALL));
            self.commit(p, to_call);
            self.acted[p] = true;
        } else {
            self.history.push((self.street, p as u8, action));
            let pot_after_call = self.pot() + to_call;
            let mut raise_by = match action {
                RAISE_QUARTER => pot_after_call / 4,
                RAISE_THIRD => pot_after_call / 3,
                RAISE_HALF => pot_after_call / 2,
                RAISE_POT => pot_after_call,
                _ => self.stacks[p],
            };
            raise_by = raise_by.max(self.last_raise).max(BB);
            self.commit(p, (to_call + raise_by).min(self.stacks[p]));
            let actual_raise = self.bets[p] - self.bets[o];
            if actual_raise > 0 {
                self.last_raise = actual_raise.max(BB);
                self.raises_this_street += 1;
                self.raise_counts[p] += 1;
                self.street_raise_counts[p] += 1;
                self.last_aggressor = p as i8;
            }
            self.acted = [false, false];
            self.acted[p] = true;
        }

        let low = if self.bets[0] <= self.bets[1] { 0 } else { 1 };
        let settled = self.bets[0] == self.bets[1] || self.stacks[low] == 0;
        if self.acted[0] && self.acted[1] && settled {
            self.next_street();
        } else {
            self.to_act = o;
        }
    }

    fn next_street(&mut self) {
        if self.street == RIVER || self.stacks[0] == 0 || self.stacks[1] == 0 {
            self.street = RIVER; // tapis : on déroule le board jusqu'au bout
            self.do_showdown();
            return;
        }
        self.street += 1;
        self.bets = [0, 0];
        self.acted = [false, false];
        self.last_raise = 0;
        self.raises_this_street = 0;
        self.street_raise_counts = [0, 0];
        self.to_act = 1 - self.button;
    }

    fn do_showdown(&mut self) {
        // Rembourse une éventuelle mise non suivie (all-in pour moins)
        let diff = self.invested[0] - self.invested[1];
        if diff > 0 {
            self.stacks[0] += diff;
            self.invested[0] -= diff;
        } else if diff < 0 {
            self.stacks[1] -= diff;
            self.invested[1] += diff;
        }
        let mut c0 = [0u8; 7];
        let mut c1 = [0u8; 7];
        c0[..2].copy_from_slice(&self.hole[0]);
        c0[2..].copy_from_slice(&self.full_board);
        c1[..2].copy_from_slice(&self.hole[1]);
        c1[2..].copy_from_slice(&self.full_board);
        let s0 = evaluate7(&c0);
        let s1 = evaluate7(&c1);
        let pot = self.pot();
        if s0 > s1 {
            self.winner = 0;
            self.stacks[0] += pot;
        } else if s1 > s0 {
            self.winner = 1;
            self.stacks[1] += pot;
        } else {
            self.winner = -1;
            self.stacks[0] += pot / 2;
            self.stacks[1] += pot - pot / 2;
        }
        self.showdown = true;
        self.finish();
    }

    fn finish_fold(&mut self, winner: usize) {
        // Rembourse la mise non suivie du gagnant avant de lui donner le pot
        let diff = self.invested[winner] - self.invested[1 - winner];
        if diff > 0 {
            self.stacks[winner] += diff;
            self.invested[winner] -= diff;
        }
        self.winner = winner as i8;
        self.stacks[winner] += self.pot();
        self.finish();
    }

    fn finish(&mut self) {
        self.terminal = true;
        self.payoffs = [self.stacks[0] - START_STACK, self.stacks[1] - START_STACK];
    }
}
