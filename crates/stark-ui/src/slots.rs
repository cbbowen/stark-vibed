//! The shape of the quick-brush rack: hold a number, paint with that brush (§18.1.8).
//!
//! Two constants so far, and they are here because the **preset table** names one of
//! them: an app preset declares the digit it ships on ([`PresetEntry::slot`]), so the
//! rack's arithmetic and the list of tools cannot live on opposite sides of a crate
//! boundary. The rack itself — what a slot holds, how a held number swaps the live
//! brush and gives it back — is still a frontend's, and joins this module at its own
//! stage (§11.2).
//!
//! [`PresetEntry::slot`]: crate::presets::PresetEntry::slot

/// How many quick brushes there are — one per digit.
pub const COUNT: usize = 10;

/// The slot the pen's other end holds, and the one the `0` key holds: the digit past
/// the nine, which is where a tenth of anything goes on a keyboard.
pub const ERASER: usize = 0;

/// The slot a key **code** stands for, or `None` for every other key.
///
/// The physical key rather than the character it produces (`code`, not `key`),
/// which is what a hold binding wants: on a French layout the digit row types
/// `&é"'` unshifted, and a rack reachable only through Shift would be no rack at
/// all. The numeric keypad is the same ten slots — it is a digit row that happens
/// to be square.
pub fn of_code(code: &str) -> Option<usize> {
    let digit = code
        .strip_prefix("Digit")
        .or_else(|| code.strip_prefix("Numpad"))?;
    match digit.as_bytes() {
        [d @ b'0'..=b'9'] => Some((d - b'0') as usize),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The digit row and the keypad are the same ten slots, read by position.
    #[test]
    fn both_digit_rows_reach_the_same_slots() {
        for d in 0..10 {
            assert_eq!(of_code(&format!("Digit{d}")), Some(d));
            assert_eq!(of_code(&format!("Numpad{d}")), Some(d));
        }
        assert!(of_code("KeyA").is_none());
        assert!(of_code("Digit").is_none());
        assert!(of_code("DigitX").is_none());
    }

    /// Every slot a key can reach is a slot the rack has.
    #[test]
    fn no_key_names_a_slot_off_the_rack() {
        for d in 0..10 {
            assert!(of_code(&format!("Digit{d}")).is_some_and(|s| s < COUNT));
        }
    }
}
