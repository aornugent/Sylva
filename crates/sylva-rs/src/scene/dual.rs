// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Second-order forward derivatives in two variables (3dtrees research 130 slice-collapsed): the ray likelihood is
//! written once over [`Real`], read as `f64` for its value and as [`Dual`] for its value, slope and curvature in an
//! element's stem front shift and reflectance level together, in one pass over its cells. `f64`'s operations are the
//! plain ones, in the same order, so the value is the same number either way.
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub};

/// The numbers the ray likelihood is written over: `f64`, or [`Dual`].
pub trait Real:
    Copy + Sum + Product + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self>
    + Add<f64, Output = Self> + Sub<f64, Output = Self> + Mul<f64, Output = Self> + Div<f64, Output = Self> + AddAssign + MulAssign
{
    /// A constant.
    fn c(x: f64) -> Self;
    /// The value.
    fn v(self) -> f64;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn log10(self) -> Self;
    /// `10^self`.
    fn exp10(self) -> Self;
    fn sqrt(self) -> Self;
    fn sq(self) -> Self;
    fn erfc(self) -> Self;
    fn hypot(self, o: Self) -> Self;
    /// The larger of it and `c` (its own derivatives where it is the larger or equal).
    fn max(self, c: f64) -> Self;
    fn clamp(self, lo: f64, hi: f64) -> Self;
}

impl Real for f64 {
    fn c(x: f64) -> Self {
        x
    }
    fn v(self) -> f64 {
        self
    }
    fn exp(self) -> Self {
        f64::exp(self)
    }
    fn ln(self) -> Self {
        f64::ln(self)
    }
    fn log10(self) -> Self {
        f64::log10(self)
    }
    fn exp10(self) -> Self {
        10f64.powf(self)
    }
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    fn sq(self) -> Self {
        self.powi(2)
    }
    fn erfc(self) -> Self {
        libm::erfc(self)
    }
    fn hypot(self, o: Self) -> Self {
        f64::hypot(self, o)
    }
    fn max(self, c: f64) -> Self {
        f64::max(self, c)
    }
    fn clamp(self, lo: f64, hi: f64) -> Self {
        f64::clamp(self, lo, hi)
    }
}

/// A value with its slope `g` and curvature `h` (xx, xy, yy) in two variables.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    pub v: f64,
    pub g: [f64; 2],
    pub h: [f64; 3],
}

impl Dual {
    /// Variable `i` (0 or 1) at `v`.
    pub fn var(v: f64, i: usize) -> Dual {
        let mut g = [0.0; 2];
        g[i] = 1.0;
        Dual { v, g, h: [0.0; 3] }
    }

    /// `f(self)` given `f`'s value `f0` and its first and second derivatives at `self.v`. A constant stays one, and so
    /// does a value past the floating point's range (an exponential that overflows: the likelihood reads it as a
    /// saturated chance, 0 or 1, whose derivatives are 0).
    fn chain(self, f0: f64, f1: f64, f2: f64) -> Dual {
        if (self.g == [0.0; 2] && self.h == [0.0; 3]) || !(f0.is_finite() && f1.is_finite() && f2.is_finite()) {
            return Dual::c(f0);
        }
        let [a, b] = self.g;
        Dual { v: f0, g: [f1 * a, f1 * b], h: [f1 * self.h[0] + f2 * a * a, f1 * self.h[1] + f2 * a * b, f1 * self.h[2] + f2 * b * b] }
    }
}

impl Real for Dual {
    fn c(x: f64) -> Self {
        Dual { v: x, g: [0.0; 2], h: [0.0; 3] }
    }
    fn v(self) -> f64 {
        self.v
    }
    fn exp(self) -> Self {
        let e = self.v.exp();
        self.chain(e, e, e)
    }
    fn ln(self) -> Self {
        self.chain(self.v.ln(), 1.0 / self.v, -1.0 / (self.v * self.v))
    }
    fn log10(self) -> Self {
        let k = std::f64::consts::LN_10;
        self.chain(self.v.log10(), 1.0 / (k * self.v), -1.0 / (k * self.v * self.v))
    }
    fn exp10(self) -> Self {
        let (p, k) = (10f64.powf(self.v), std::f64::consts::LN_10);
        self.chain(p, p * k, p * k * k)
    }
    /// At 0, where the root has no derivative, its derivatives are taken as 0 (a spread that vanishes).
    fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        if s == 0.0 {
            return Dual::c(s);
        }
        self.chain(s, 0.5 / s, -0.25 / (s * self.v))
    }
    fn sq(self) -> Self {
        self.chain(self.v.powi(2), 2.0 * self.v, 2.0)
    }
    fn erfc(self) -> Self {
        let d = -std::f64::consts::FRAC_2_SQRT_PI * (-self.v * self.v).exp();
        self.chain(libm::erfc(self.v), d, -2.0 * self.v * d)
    }
    fn hypot(self, o: Self) -> Self {
        Dual { v: self.v.hypot(o.v), ..(self.sq() + o.sq()).sqrt() }
    }
    fn max(self, c: f64) -> Self {
        if self.v >= c { self } else { Dual::c(c) }
    }
    fn clamp(self, lo: f64, hi: f64) -> Self {
        if self.v < lo {
            Dual::c(lo)
        } else if self.v > hi {
            Dual::c(hi)
        } else {
            self
        }
    }
}

impl Add for Dual {
    type Output = Dual;
    fn add(self, o: Dual) -> Dual {
        Dual { v: self.v + o.v, g: [self.g[0] + o.g[0], self.g[1] + o.g[1]], h: [self.h[0] + o.h[0], self.h[1] + o.h[1], self.h[2] + o.h[2]] }
    }
}

impl Neg for Dual {
    type Output = Dual;
    fn neg(self) -> Dual {
        Dual { v: -self.v, g: self.g.map(|x| -x), h: self.h.map(|x| -x) }
    }
}

impl Sub for Dual {
    type Output = Dual;
    fn sub(self, o: Dual) -> Dual {
        Dual { v: self.v - o.v, ..self + -o }
    }
}

impl Mul for Dual {
    type Output = Dual;
    fn mul(self, o: Dual) -> Dual {
        let ([a, b], [c, d]) = (self.g, o.g);
        Dual {
            v: self.v * o.v,
            g: [a * o.v + c * self.v, b * o.v + d * self.v],
            h: [self.h[0] * o.v + o.h[0] * self.v + 2.0 * a * c, self.h[1] * o.v + o.h[1] * self.v + a * d + b * c,
                self.h[2] * o.v + o.h[2] * self.v + 2.0 * b * d],
        }
    }
}

impl Div for Dual {
    type Output = Dual;
    fn div(self, o: Dual) -> Dual {
        let r = o.chain(1.0 / o.v, -1.0 / (o.v * o.v), 2.0 / (o.v * o.v * o.v));
        Dual { v: self.v / o.v, ..self * r }
    }
}

impl Add<f64> for Dual {
    type Output = Dual;
    fn add(self, c: f64) -> Dual {
        Dual { v: self.v + c, ..self }
    }
}

impl Sub<f64> for Dual {
    type Output = Dual;
    fn sub(self, c: f64) -> Dual {
        Dual { v: self.v - c, ..self }
    }
}

impl Mul<f64> for Dual {
    type Output = Dual;
    fn mul(self, c: f64) -> Dual {
        Dual { v: self.v * c, g: self.g.map(|x| x * c), h: self.h.map(|x| x * c) }
    }
}

impl Div<f64> for Dual {
    type Output = Dual;
    fn div(self, c: f64) -> Dual {
        Dual { v: self.v / c, g: self.g.map(|x| x / c), h: self.h.map(|x| x / c) }
    }
}

impl AddAssign for Dual {
    fn add_assign(&mut self, o: Dual) {
        *self = *self + o;
    }
}

impl MulAssign for Dual {
    fn mul_assign(&mut self, o: Dual) {
        *self = *self * o;
    }
}

impl Sum for Dual {
    fn sum<I: Iterator<Item = Dual>>(it: I) -> Dual {
        it.fold(Dual::c(-0.0), |a, b| a + b)
    }
}

impl Product for Dual {
    fn product<I: Iterator<Item = Dual>>(it: I) -> Dual {
        it.fold(Dual::c(1.0), |a, b| a * b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every operation's slope and curvature against central differences, on one function of two variables.
    #[test]
    fn derivatives_match_finite_differences() {
        fn f<T: Real>(x: T, y: T) -> T {
            let a = (x * y + 0.3).exp() / (y.sq() + 1.0).sqrt() + (x - y * 0.5).erfc() * x.hypot(y);
            a.ln() + (x * 0.2).exp10().log10() * (y + 2.0).max(0.0) + -(x / 3.0).clamp(-5.0, 5.0)
        }
        let (x, y) = (0.4, -0.7);
        let d = f(Dual::var(x, 0), Dual::var(y, 1));
        assert_eq!(d.v, f(x, y));
        let h = 1e-4;
        let fd = |dx: f64, dy: f64| f(x + dx, y + dy);
        let g = [(fd(h, 0.0) - fd(-h, 0.0)) / (2.0 * h), (fd(0.0, h) - fd(0.0, -h)) / (2.0 * h)];
        let hh = [(fd(h, 0.0) - 2.0 * fd(0.0, 0.0) + fd(-h, 0.0)) / (h * h), (fd(h, h) - fd(h, -h) - fd(-h, h) + fd(-h, -h)) / (4.0 * h * h),
                  (fd(0.0, h) - 2.0 * fd(0.0, 0.0) + fd(0.0, -h)) / (h * h)];
        for i in 0..2 {
            assert!((d.g[i] - g[i]).abs() < 1e-7, "{:?} vs {g:?}", d.g);
        }
        for i in 0..3 {
            assert!((d.h[i] - hh[i]).abs() < 1e-5, "{:?} vs {hh:?}", d.h);
        }
        // a chance that saturates (an exponential past the floating point's range) has no slope, not an infinite one
        let p = Dual::c(1.0) / ((Dual::var(800.0, 0) * 2.0).exp() + 1.0);
        assert!(p.v == 0.0 && p.g.iter().chain(&p.h).all(|x| x.is_finite()), "{p:?}");
        assert_eq!(Dual::var(0.0, 1).sq().sqrt().g, [0.0; 2]);
    }
}
