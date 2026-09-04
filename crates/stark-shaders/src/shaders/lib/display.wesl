// The display transfer (§6.5): how a frame's target encodes light, decided by the
// surface it is presented on. One selector for the media pass's encode and the
// resolve's decode, so the two cannot disagree; a uniform branch rather than a
// pipeline variant, so one pipeline per target format serves every surface of it.

import package::lib::color::{linear_to_srgb, srgb_to_linear};

// The sRGB OETF over `[0,1]` — every 8-bit target, and every export.
const TRANSFER_SRGB: u32 = 0u;
// The sRGB OETF continued past 1 and mirrored through 0 — an fp16 canvas under the
// web's "extended" tone mapping; 1.0 is SDR white.
const TRANSFER_EXTENDED_SRGB: u32 = 1u;
// No encoding: linear light, 1.0 = SDR white — scRGB, the native HDR surface.
const TRANSFER_LINEAR: u32 = 2u;

// Odd (`f(−x) = −f(x)`), so a channel driven negative follows the mirrored curve
// rather than the linear toe `linear_to_srgb` puts every negative input on.
fn encode_display(transfer: u32, lin: vec3<f32>) -> vec3<f32> {
    if transfer == TRANSFER_LINEAR {
        return lin;
    }
    if transfer == TRANSFER_EXTENDED_SRGB {
        return sign(lin) * linear_to_srgb(abs(lin));
    }
    return linear_to_srgb(lin);
}

// The exact inverse of `encode_display` for the same selector.
fn decode_display(transfer: u32, enc: vec3<f32>) -> vec3<f32> {
    if transfer == TRANSFER_LINEAR {
        return enc;
    }
    if transfer == TRANSFER_EXTENDED_SRGB {
        return sign(enc) * srgb_to_linear(abs(enc));
    }
    return srgb_to_linear(enc);
}
