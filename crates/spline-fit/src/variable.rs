use nalgebra::{RealField, Scalar};
use num_traits::ToPrimitive;

pub trait Variable: Scalar + RealField + Copy + ToPrimitive {}
impl<T: Scalar + RealField + Copy + ToPrimitive> Variable for T {}
