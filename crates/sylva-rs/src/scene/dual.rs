// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Forward derivatives in any number of variables (3dtrees research 130 slice-collapsed, slice-derivatives): the ray
//! likelihood is written once over [`Real`], read as `f64` for its value, as [`Dual`] for its value and slope (first
//! order: the geometry's Fisher information needs no more), and as [`Dual2`] for its value, slope and curvature
//! (second order: the own nodes' Newton). `f64`'s operations are the plain ones, in the same order, so the value is the
//! same number every way.
use nalgebra::{SMatrix, SVector};
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub};

/// The numbers the ray likelihood is written over: `f64`, [`Dual`] or [`Dual2`].
pub trait Real:
    Copy + Sum + Product + From<f64> + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self>
    + Add<f64, Output = Self> + Sub<f64, Output = Self> + Mul<f64, Output = Self> + Div<f64, Output = Self> + AddAssign + MulAssign
{
    /// A constant.
    fn c(x: f64) -> Self;
    /// The value.
    fn v(self) -> f64;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn ln_1p(self) -> Self;
    fn log10(self) -> Self;
    /// `10^self`.
    fn exp10(self) -> Self;
    fn powf(self, p: f64) -> Self;
    fn sqrt(self) -> Self;
    fn sq(self) -> Self;
    fn asin(self) -> Self;
    fn erfc(self) -> Self;
    fn hypot(self, o: Self) -> Self;
    /// The larger of it and `c` (its own derivatives where it is the larger or equal).
    fn max(self, c: f64) -> Self;
    /// The smaller of it and `c` (its own derivatives where it is the smaller or equal).
    fn min(self, c: f64) -> Self;
    fn clamp(self, lo: f64, hi: f64) -> Self;
    /// The larger of two (the first where they are equal).
    fn larger(self, o: Self) -> Self {
        if self.v() >= o.v() { self } else { o }
    }
    /// The smaller of two (the first where they are equal).
    fn smaller(self, o: Self) -> Self {
        if self.v() <= o.v() { self } else { o }
    }
}

/// A number `T` that carries another, `P`, in its arithmetic: the ray likelihood's own nodes' numbers over its
/// geometry's (`f64` geometry under [`Dual2`] own nodes in their Newton; [`Dual`] both together for the information).
pub trait Over<P>: Real + From<P> + Add<P, Output = Self> + Sub<P, Output = Self> + Mul<P, Output = Self> + Div<P, Output = Self> {}

impl<P, T: Real + From<P> + Add<P, Output = T> + Sub<P, Output = T> + Mul<P, Output = T> + Div<P, Output = T>> Over<P> for T {}

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
    fn ln_1p(self) -> Self {
        f64::ln_1p(self)
    }
    fn log10(self) -> Self {
        f64::log10(self)
    }
    fn exp10(self) -> Self {
        10f64.powf(self)
    }
    fn powf(self, p: f64) -> Self {
        f64::powf(self, p)
    }
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    fn sq(self) -> Self {
        self.powi(2)
    }
    fn asin(self) -> Self {
        f64::asin(self)
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
    fn min(self, c: f64) -> Self {
        f64::min(self, c)
    }
    fn clamp(self, lo: f64, hi: f64) -> Self {
        f64::clamp(self, lo, hi)
    }
    fn larger(self, o: Self) -> Self {
        f64::max(self, o)
    }
    fn smaller(self, o: Self) -> Self {
        f64::min(self, o)
    }
}

/// A value with its slope `g` in `N` variables: first-order forward derivatives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual<const N: usize> {
    pub v: f64,
    pub g: SVector<f64, N>,
}

/// A value with its slope `g` and curvature `h` (symmetric) in `N` variables: second-order forward derivatives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual2<const N: usize> {
    pub v: f64,
    pub g: SVector<f64, N>,
    pub h: SMatrix<f64, N, N>,
}

impl<const N: usize> Dual<N> {
    /// Variable `i` at `v`.
    pub fn var(v: f64, i: usize) -> Self {
        let mut g = SVector::zeros();
        g[i] = 1.0;
        Dual { v, g }
    }

    /// `f(self)` given `f`'s value `f0` and its first and second derivatives at `self.v` (the second unused here). A
    /// constant stays one, and so does a value past the floating point's range (an exponential that overflows: the
    /// likelihood reads it as a saturated chance, 0 or 1, whose derivatives are 0).
    fn chain(self, f0: f64, f1: f64, _f2: f64) -> Self {
        if self.g.iter().all(|&x| x == 0.0) || !(f0.is_finite() && f1.is_finite()) {
            return Dual::c(f0);
        }
        Dual { v: f0, g: self.g * f1 }
    }

    fn constant(x: f64) -> Self {
        Dual { v: x, g: SVector::zeros() }
    }

    /// `f` applied to the value and every derivative (a linear map: a scaling or a sign).
    fn map(self, f: impl Fn(f64) -> f64) -> Self {
        Dual { v: f(self.v), g: self.g.map(f) }
    }
}

impl<const N: usize> Dual2<N> {
    /// Variable `i` at `v`.
    pub fn var(v: f64, i: usize) -> Self {
        let mut g = SVector::zeros();
        g[i] = 1.0;
        Dual2 { v, g, h: SMatrix::zeros() }
    }

    /// As [`Dual`]'s, with the curvature `f1 h + f2 g gᵀ`. Each pair (i, j) is computed once, i ≤ j, so `h` stays
    /// exactly symmetric.
    fn chain(self, f0: f64, f1: f64, f2: f64) -> Self {
        if (self.g.iter().all(|&x| x == 0.0) && self.h.iter().all(|&x| x == 0.0)) || !(f0.is_finite() && f1.is_finite() && f2.is_finite()) {
            return Dual2::c(f0);
        }
        let (g, h) = (self.g, self.h);
        Dual2 { v: f0, g: g * f1, h: symmetric(|i, j| f1 * h[(i, j)] + f2 * g[i] * g[j]) }
    }

    fn constant(x: f64) -> Self {
        Dual2 { v: x, g: SVector::zeros(), h: SMatrix::zeros() }
    }

    fn map(self, f: impl Fn(f64) -> f64) -> Self {
        Dual2 { v: f(self.v), g: self.g.map(&f), h: self.h.map(f) }
    }
}

/// A symmetric matrix from its upper triangle, `f(i, j)` for i ≤ j, each pair computed once.
fn symmetric<const N: usize>(f: impl Fn(usize, usize) -> f64) -> SMatrix<f64, N, N> {
    let mut h = SMatrix::zeros();
    for j in 0..N {
        for i in 0..=j {
            let x = f(i, j);
            h[(i, j)] = x;
            h[(j, i)] = x;
        }
    }
    h
}

impl<const N: usize> Add for Dual<N> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Dual { v: self.v + o.v, g: self.g + o.g }
    }
}

impl<const N: usize> Add for Dual2<N> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Dual2 { v: self.v + o.v, g: self.g + o.g, h: self.h + o.h }
    }
}

impl<const N: usize> Mul for Dual<N> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Dual { v: self.v * o.v, g: SVector::from_fn(|i, _| self.g[i] * o.v + o.g[i] * self.v) }
    }
}

impl<const N: usize> Mul for Dual2<N> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        let (a, c) = (self.g, o.g);
        Dual2 {
            v: self.v * o.v,
            g: SVector::from_fn(|i, _| a[i] * o.v + c[i] * self.v),
            h: symmetric(|i, j| {
                let base = self.h[(i, j)] * o.v + o.h[(i, j)] * self.v;
                if i == j { base + 2.0 * a[i] * c[i] } else { base + a[i] * c[j] + a[j] * c[i] }
            }),
        }
    }
}

/// The operations [`Dual`] and [`Dual2`] share, from their `constant`, `chain`, `map`, sum and product.
macro_rules! real {
    ($d:ident) => {
        impl<const N: usize> From<f64> for $d<N> {
            fn from(x: f64) -> Self {
                <$d<N> as Real>::c(x)
            }
        }

        impl<const N: usize> Real for $d<N> {
            fn c(x: f64) -> Self {
                $d::constant(x)
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
            fn ln_1p(self) -> Self {
                let u = 1.0 + self.v;
                self.chain(self.v.ln_1p(), 1.0 / u, -1.0 / (u * u))
            }
            fn log10(self) -> Self {
                let k = std::f64::consts::LN_10;
                self.chain(self.v.log10(), 1.0 / (k * self.v), -1.0 / (k * self.v * self.v))
            }
            fn exp10(self) -> Self {
                let (p, k) = (10f64.powf(self.v), std::f64::consts::LN_10);
                self.chain(p, p * k, p * k * k)
            }
            fn powf(self, p: f64) -> Self {
                self.chain(self.v.powf(p), p * self.v.powf(p - 1.0), p * (p - 1.0) * self.v.powf(p - 2.0))
            }
            /// At 0, where the root has no derivative, its derivatives are taken as 0 (a spread that vanishes).
            fn sqrt(self) -> Self {
                let s = self.v.sqrt();
                if s == 0.0 {
                    return Self::c(s);
                }
                self.chain(s, 0.5 / s, -0.25 / (s * self.v))
            }
            fn sq(self) -> Self {
                self.chain(self.v.powi(2), 2.0 * self.v, 2.0)
            }
            /// At ±1 its slope is infinite: it reads as a constant there, as an overflow does.
            fn asin(self) -> Self {
                let u = 1.0 - self.v * self.v;
                self.chain(self.v.asin(), 1.0 / u.sqrt(), self.v / (u * u.sqrt()))
            }
            fn erfc(self) -> Self {
                let d = -std::f64::consts::FRAC_2_SQRT_PI * (-self.v * self.v).exp();
                self.chain(libm::erfc(self.v), d, -2.0 * self.v * d)
            }
            fn hypot(self, o: Self) -> Self {
                $d { v: self.v.hypot(o.v), ..(self.sq() + o.sq()).sqrt() }
            }
            fn max(self, c: f64) -> Self {
                if self.v >= c { self } else { Self::c(c) }
            }
            fn min(self, c: f64) -> Self {
                if self.v <= c { self } else { Self::c(c) }
            }
            fn clamp(self, lo: f64, hi: f64) -> Self {
                if self.v < lo {
                    Self::c(lo)
                } else if self.v > hi {
                    Self::c(hi)
                } else {
                    self
                }
            }
        }

        impl<const N: usize> Neg for $d<N> {
            type Output = Self;
            fn neg(self) -> Self {
                self.map(|x| -x)
            }
        }

        impl<const N: usize> Sub for $d<N> {
            type Output = Self;
            fn sub(self, o: Self) -> Self {
                $d { v: self.v - o.v, ..self + -o }
            }
        }

        impl<const N: usize> Div for $d<N> {
            type Output = Self;
            fn div(self, o: Self) -> Self {
                let r = o.chain(1.0 / o.v, -1.0 / (o.v * o.v), 2.0 / (o.v * o.v * o.v));
                $d { v: self.v / o.v, ..self * r }
            }
        }

        impl<const N: usize> Add<f64> for $d<N> {
            type Output = Self;
            fn add(self, c: f64) -> Self {
                $d { v: self.v + c, ..self }
            }
        }

        impl<const N: usize> Sub<f64> for $d<N> {
            type Output = Self;
            fn sub(self, c: f64) -> Self {
                $d { v: self.v - c, ..self }
            }
        }

        impl<const N: usize> Mul<f64> for $d<N> {
            type Output = Self;
            fn mul(self, c: f64) -> Self {
                self.map(|x| x * c)
            }
        }

        impl<const N: usize> Div<f64> for $d<N> {
            type Output = Self;
            fn div(self, c: f64) -> Self {
                self.map(|x| x / c)
            }
        }

        impl<const N: usize> AddAssign for $d<N> {
            fn add_assign(&mut self, o: Self) {
                *self = *self + o;
            }
        }

        impl<const N: usize> MulAssign for $d<N> {
            fn mul_assign(&mut self, o: Self) {
                *self = *self * o;
            }
        }

        impl<const N: usize> Sum for $d<N> {
            fn sum<I: Iterator<Item = Self>>(it: I) -> Self {
                it.fold(Self::c(-0.0), |a, b| a + b)
            }
        }

        impl<const N: usize> Product for $d<N> {
            fn product<I: Iterator<Item = Self>>(it: I) -> Self {
                it.fold(Self::c(1.0), |a, b| a * b)
            }
        }
    };
}

real!(Dual);
real!(Dual2);

#[cfg(test)]
mod tests {
    use super::*;

    fn f<T: Real>(x: T, y: T) -> T {
        let a = (x * y + 0.3).exp() / (y.sq() + 1.0).sqrt() + (x - y * 0.5).erfc() * x.hypot(y);
        a.ln() + (x * 0.2).exp10().log10() * (y + 2.0).max(0.0) + -(x / 3.0).clamp(-5.0, 5.0)
    }

    /// Every operation's slope and curvature against central differences, on one function of two variables, first
    /// and second order; the slope the same number either way.
    #[test]
    fn derivatives_match_finite_differences() {
        fn g<T: Real>(x: T, y: T) -> T {
            f(x, y) + (x * 0.5).asin() * (y.sq() * 0.1).ln_1p() + (x + 2.0).powf(1.7) / (y - 3.0).min(-2.5).larger(x - 9.0)
        }
        let (x, y) = (0.4, -0.7);
        let (d, d1) = (g(Dual2::<2>::var(x, 0), Dual2::var(y, 1)), g(Dual::<2>::var(x, 0), Dual::var(y, 1)));
        assert_eq!(d.v, g(x, y));
        assert_eq!((d1.v, d1.g), (d.v, d.g));
        let h = 1e-4;
        let fd = |dx: f64, dy: f64| g(x + dx, y + dy);
        let gr = [(fd(h, 0.0) - fd(-h, 0.0)) / (2.0 * h), (fd(0.0, h) - fd(0.0, -h)) / (2.0 * h)];
        let hh = [[(fd(h, 0.0) - 2.0 * fd(0.0, 0.0) + fd(-h, 0.0)) / (h * h), (fd(h, h) - fd(h, -h) - fd(-h, h) + fd(-h, -h)) / (4.0 * h * h)],
                  [0.0, (fd(0.0, h) - 2.0 * fd(0.0, 0.0) + fd(0.0, -h)) / (h * h)]];
        for i in 0..2 {
            assert!((d.g[i] - gr[i]).abs() < 1e-7, "{:?} vs {gr:?}", d.g);
            for j in i..2 {
                assert!((d.h[(i, j)] - hh[i][j]).abs() < 1e-5, "{:?} vs {hh:?}", d.h);
                assert_eq!(d.h[(i, j)], d.h[(j, i)]);
            }
        }
        // a chance that saturates (an exponential past the floating point's range) has no slope, not an infinite one
        let p = Dual2::<2>::c(1.0) / ((Dual2::var(800.0, 0) * 2.0).exp() + 1.0);
        assert!(p.v == 0.0 && p.g.iter().chain(p.h.iter()).all(|x| x.is_finite()), "{p:?}");
        assert_eq!(Dual2::<2>::var(0.0, 1).sq().sqrt().g, SVector::<f64, 2>::zeros());
        assert_eq!(Dual::<3>::var(1.0, 2).asin().g, SVector::<f64, 3>::zeros());
    }

    /// The generic second-order dual in two variables gives the slice-collapsed two-variable `Dual`'s numbers bit for
    /// bit (recorded from it at `221614e`'s successor `2b48822`), and in more variables the same numbers in its first two.
    #[test]
    fn the_generic_dual_matches_the_two_variable_one() {
        let old_v = f64::from_bits(4585502850188554452);
        let old_g = [13830299918018280290u64, 4605093704866443130].map(f64::from_bits);
        let old_h = [13809383381688355467u64, 4609778021506538047, 13822797546983780065].map(f64::from_bits);
        let d = f(Dual2::<2>::var(0.4, 0), Dual2::var(-0.7, 1));
        assert_eq!((d.v, [d.g[0], d.g[1]], [d.h[(0, 0)], d.h[(0, 1)], d.h[(1, 1)]]), (old_v, old_g, old_h));
        let d5 = f(Dual2::<5>::var(0.4, 0), Dual2::var(-0.7, 1));
        assert_eq!((d5.v, [d5.g[0], d5.g[1]], [d5.h[(0, 0)], d5.h[(0, 1)], d5.h[(1, 1)]]), (old_v, old_g, old_h));
        assert!(d5.g.rows(2, 3).iter().chain(d5.h.view((2, 0), (3, 5)).iter()).all(|&x| x == 0.0));
    }
}
