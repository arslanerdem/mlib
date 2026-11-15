//! Finding the lowest point of a smooth function (L-BFGS).
//!
//! Standing at a point, the gradient says which way is downhill, but not
//! how far to go or how the slope will bend. L-BFGS remembers the last
//! few steps it took and how the gradient changed over each of them; from
//! that it estimates the bend and proposes a much better step than
//! "straight downhill". It never builds the full matrix of second
//! derivatives, so it stays cheap with many parameters.
//!
//! Each proposed step is then shortened until the function really has
//! gone down by a fair amount.

use std::collections::VecDeque;

/// How many past steps are remembered.
const MEMORY: usize = 10;

/// A step must lower the function by at least this share of what the
/// slope promised.
const ENOUGH_DECREASE: f64 = 1e-4;

/// Where a search ended.
pub(crate) struct Minimum {
    pub point: Vec<f64>,
    pub n_iterations: usize,
    /// Whether the slope fell below the tolerance, or steps stopped
    /// making any difference. `false` means the iterations ran out first.
    pub converged: bool,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// Minimise a function, starting from `start`.
///
/// `value_and_gradient(point)` returns the function's value at a point
/// and its gradient there. The search stops when no gradient entry is
/// larger than `tolerance` in size, or after `max_iterations` steps.
pub(crate) fn minimise(
    start: Vec<f64>,
    tolerance: f64,
    max_iterations: usize,
    value_and_gradient: impl Fn(&[f64]) -> (f64, Vec<f64>),
) -> Minimum {
    let mut point = start;
    let (mut value, mut gradient) = value_and_gradient(&point);
    // Past steps: (the step taken, the change in gradient over it), oldest first.
    let mut history: VecDeque<(Vec<f64>, Vec<f64>)> = VecDeque::new();

    for iteration in 0..max_iterations {
        let steepest = gradient.iter().fold(0.0, |largest: f64, entry| largest.max(entry.abs()));
        if steepest <= tolerance {
            return Minimum { point, n_iterations: iteration, converged: true };
        }

        let direction = downhill_direction(&gradient, &history);
        let slope = dot(&gradient, &direction);
        // With no history the direction is the raw gradient, whose length
        // says nothing about distance: start with a step of length one.
        let mut step = if history.is_empty() { 1.0 / dot(&gradient, &gradient).sqrt().max(1.0) } else { 1.0 };

        // Shorten the step until the function has gone down enough.
        let moved = loop {
            let candidate: Vec<f64> = point.iter().zip(&direction).map(|(x, d)| x + step * d).collect();
            let (new_value, new_gradient) = value_and_gradient(&candidate);
            if new_value <= value + ENOUGH_DECREASE * step * slope {
                break Some((candidate, new_value, new_gradient));
            }
            step /= 2.0;
            if step < 1e-12 {
                break None;
            }
        };
        // No step, however short, lowers the function: this is as low as
        // the arithmetic can tell.
        let Some((new_point, new_value, new_gradient)) = moved else {
            return Minimum { point, n_iterations: iteration, converged: true };
        };

        let taken: Vec<f64> = new_point.iter().zip(&point).map(|(new, old)| new - old).collect();
        let change: Vec<f64> = new_gradient.iter().zip(&gradient).map(|(new, old)| new - old).collect();
        // A step is worth remembering only if the slope along it increased,
        // as it does in a valley.
        if dot(&taken, &change) > 1e-12 * dot(&change, &change) {
            if history.len() == MEMORY {
                history.pop_front();
            }
            history.push_back((taken, change));
        }
        (point, value, gradient) = (new_point, new_value, new_gradient);
    }
    Minimum { point, n_iterations: max_iterations, converged: false }
}

/// The step L-BFGS proposes: downhill, corrected for the bend of the
/// function as learned from the remembered steps (the "two-loop
/// recursion"). With no history it is simply the negative gradient.
fn downhill_direction(gradient: &[f64], history: &VecDeque<(Vec<f64>, Vec<f64>)>) -> Vec<f64> {
    let mut direction: Vec<f64> = gradient.iter().map(|entry| -entry).collect();
    let add = |to: &mut [f64], amount: f64, of: &[f64]| to.iter_mut().zip(of).for_each(|(to, of)| *to += amount * of);

    // Newest step first.
    let mut amounts = Vec::new();
    for (taken, change) in history.iter().rev() {
        let amount = dot(taken, &direction) / dot(taken, change);
        add(&mut direction, -amount, change);
        amounts.push(amount);
    }
    // The newest step also gives the overall scale of the bend.
    if let Some((taken, change)) = history.back() {
        let scale = dot(taken, change) / dot(change, change);
        direction.iter_mut().for_each(|entry| *entry *= scale);
    }
    // Oldest step first.
    for ((taken, change), amount) in history.iter().zip(amounts.iter().rev()) {
        let correction = dot(change, &direction) / dot(taken, change);
        add(&mut direction, amount - correction, taken);
    }
    direction
}
