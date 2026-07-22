//! Xoshiro256++ semé par SplitMix64 — rapide, qualité largement suffisante
//! pour du Monte-Carlo (la parité avec NumPy est statistique, pas bit à bit).

pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let mut next_sm = || {
            sm = sm.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        Rng { s: [next_sm(), next_sm(), next_sm(), next_sm()] }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Entier uniforme dans [0, n) — méthode de Lemire.
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    /// Flottant uniforme dans [0, 1).
    #[inline]
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Mélange de Fisher-Yates partiel : les `k` premiers éléments sont un
    /// tirage uniforme sans remise.
    pub fn partial_shuffle(&mut self, arr: &mut [u8], k: usize) {
        let n = arr.len();
        for i in 0..k.min(n.saturating_sub(1)) {
            let j = i + self.below(n - i);
            arr.swap(i, j);
        }
    }
}
