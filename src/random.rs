//! A small, fast, seedable random number generator (SplitMix64).
//!
//! The library owns its generator so that a seed gives the same model on
//! every machine and every release — an external RNG crate may change its
//! streams between versions.

/// The generator. For your own algorithms that need randomness which
/// must be reproducible.
///
/// ```
/// use mlib::random::Rng;
///
/// let mut rng = Rng::new(42, 0);
/// let mut again = Rng::new(42, 0);
/// assert_eq!(rng.next_u64(), again.next_u64()); // same seed, same numbers
///
/// assert!(rng.below(10) < 10);
/// assert!(rng.chance(1.0) && !rng.chance(0.0));
///
/// let mut rows = [0, 1, 2, 3, 4];
/// rng.shuffle(&mut rows);
/// rows.sort();
/// assert_eq!(rows, [0, 1, 2, 3, 4]); // shuffling loses nothing
/// ```
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

    /// The next random 64-bit number.
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

    /// `true` with the given probability (0 never, 1 always).
    pub fn chance(&mut self, probability: f64) -> bool {
        // The top 53 bits make a uniform number in [0, 1).
        ((self.next_u64() >> 11) as f64) < probability * (1u64 << 53) as f64
    }

    /// Shuffle a slice in place (Fisher–Yates).
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            items.swap(i, self.below(i + 1));
        }
    }
}
