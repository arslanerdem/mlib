//! A small, fast, seedable random number generator (SplitMix64).
//!
//! The library owns its generator so that a seed gives the same model on
//! every machine and every release — an external RNG crate may change its
//! streams between versions.

#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// A generator for the given seed. Different `stream` values give
    /// independent sequences for the same seed (one per tree, per fold, ...),
    /// so parallel work is reproducible regardless of thread scheduling.
    pub fn new(seed: u64, stream: u64) -> Rng {
        let mut rng = Rng { state: seed };
        let mixed_seed = rng.next_u64();
        rng.state = mixed_seed ^ stream.wrapping_mul(0xD6E8_FEB8_6659_FD93);
        rng.next_u64();
        rng
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform integer in `0..n`. `n` must be greater than zero.
    pub fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    /// Shuffle a slice in place (Fisher–Yates).
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            items.swap(i, self.below(i + 1));
        }
    }
}
