//! This module holds the various optimizers used to update parameters in a computation graph.
//! Currently only one is implemented.
use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use std::collections::HashMap;
use std::{f32, fmt};
use Idx;

#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize)]
struct OptimizerInstance {
    /// Accumulates average gradient. Used in Momentum and Adam
    momentum: Option<ArrayD<f32>>,
    /// Accumulates gradient squared, used in RMSProp and Adam
    magnitude: Option<ArrayD<f32>>,
    // param_magnitude for Adadelta
}

/// A Good Blog comparing different optimizers. The `Optimizer` builds `OptimizerInstance`s which
/// hold specific runtime information about the parameter being optimized.
/// http://ruder.io/optimizing-gradient-descent/index.html#momentum
// IDEA/TODO
// * Replace optimizer trait with this!
// * Builder?
// * Components
//     * Adadelta auto LR with RMS(param) / RMS(grad)
//     * Adamax {L1, L2, Max}-Norm for magnitude component
//     * don't do Nesterov accleration b/c needs graph ownership for lookahead?
#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize)]
pub struct Optimizer {
    pub learning_rate: f32,
    pub beta_momentum: f32,
    pub beta_magnitude: f32,
    pub epsilon: f32,
    // QUESTION why keep this instance info inside the optimizer intead of the parameter node?
    // * Need to make parameter node its own type for easier accessing
    // * Tiny memory impact in forward only "production" graph.
    data: HashMap<Idx, OptimizerInstance>,
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::sgd_default()
    }
}
impl fmt::Display for Optimizer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Customize so only `x` and `y` are denoted.
        write!(f,
            "Optimizer {{ learning_rate {:?}, beta_momentum: {:?}, beta_magnitude: {:?}, epsilon: {:?}}}",
            self.learning_rate,
            self.beta_momentum,
            self.beta_magnitude,
            self.epsilon,
        )?;
        Ok(())
    }
}

#[allow(dead_code)]
impl Optimizer {
    pub fn new(learning_rate: f32, beta_momentum: f32, beta_magnitude: f32, epsilon: f32) -> Self {
        let data = HashMap::new();
        Optimizer {
            learning_rate,
            beta_momentum,
            beta_magnitude,
            epsilon,
            data,
        }
    }
    /// Vanilla stochastic gradient descent with no added fluff.
    pub fn sgd_default() -> Self {
        Self::new(1e-3, 0.0, 0.0, 1e-8)
    }
    /// SGD with a momentum component. Add the geometric average of past gradients to the parameter
    /// instead of the gradient itself. This averaging dampens the stochasticity of the stochastic
    /// gradient descent.
    pub fn momentum_default() -> Self {
        Self::new(1e-3, 0.9, 0.0, 1e-8)
    }
    /// SGD with a magnitude component. Rescale gradients by dividing by the geometric mean of
    /// previous gradients squared. Parameters with frequent large gradients will see those
    /// gradients shrink while parameters with sparse gradients will have their gradients grow.
    pub fn rmsprop_default() -> Self {
        Self::new(1e-2, 0.0, 0.9, 1e-8)
    }
    /// Adam (Adaptive Moment Estimation) Combines the momentum component from `momentum` and the
    /// magnitude component from `rmsprop`.
    pub fn adam_default() -> Self {
        Self::new(1e-2, 0.9, 0.999, 1e-8)
    }
    pub fn register(&mut self, i: Idx, shape: &[usize]) {
        let momentum = if self.beta_momentum > f32::EPSILON {
            Some(ArrayD::zeros(shape))
        } else {
            None
        };
        let magnitude = if self.beta_magnitude > f32::EPSILON {
            Some(ArrayD::zeros(shape))
        } else {
            None
        };
        let instance = OptimizerInstance {
            momentum,
            magnitude,
        };
        self.data.insert(i, instance);
    }
    /// Apply gradient
    pub fn apply_gradient(
        &mut self,
        i: &Idx,
        mut param: ArrayViewMutD<f32>,
        grad: ArrayViewD<f32>,
    ) {
        let optimizer_instance = self
            .data
            .get_mut(i)
            .expect("Attempted to apply gradient to unregistered parameter");

        let mut delta = if let Some(ref mut mom) = optimizer_instance.momentum {
            let beta1 = self.beta_momentum;
            mom.zip_mut_with(&grad, |m, g| *m = (1.0 - beta1) * *m + beta1 * g);
            mom.to_owned() / (1.0 - self.beta_momentum)
        } else {
            grad.to_owned()
        };

        if let Some(ref mut mag) = optimizer_instance.magnitude {
            let beta2 = self.beta_magnitude;
            mag.zip_mut_with(&grad, |m, g| *m = (1.0 - beta2) * *m + beta2 * g);
            let e = self.epsilon;
            delta.zip_mut_with(mag, |d, m| *d /= m.sqrt() + e);
        }

        let lr = self.learning_rate;
        param.zip_mut_with(&delta, |p, d| *p += d * lr);
    }
}
