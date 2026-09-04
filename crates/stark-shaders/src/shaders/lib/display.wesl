// The display transfer (§6.5): how a frame's target encodes light, decided by the
// surface it is presented on. One selector for the media pass's encode and the
// resolve's decode, so the two cannot disagree; a uniform branch rather than a
// pipeline variant, so one pipeline per target format serves every surface of it.

import package::lib::color::{linear_to_srgb, srgb_to_linear};

// The sRGB OETF over `[0,1]` — every 8-bit sRGB target, and every export.
const TRANSFER_SRGB: u32 = 0u;
// The sRGB OETF continued past 1 and mirrored through 0 — an fp16 canvas under the
// web's "extended" tone mapping; 1.0 is SDR white.
const TRANSFER_EXTENDED_SRGB: u32 = 1u;
// No encoding: linear light, 1.0 = SDR white — scRGB, the native HDR surface. sRGB
// primaries, unbounded, so a wide-gamut color rides in a negative channel.
const TRANSFER_LINEAR: u32 = 2u;
// Display P3 primaries under the sRGB OETF over `[0,1]` — an 8-bit `display-p3`
// canvas: wide gamut, no headroom.
const TRANSFER_DISPLAY_P3: u32 = 3u;
// Display P3 primaries under the extended OETF — an fp16 `display-p3` canvas under
// extended tone mapping.
const TRANSFER_EXTENDED_DISPLAY_P3: u32 = 4u;

// Linear sRGB ↔ linear Display P3, both D65 — mirrors `color.rs`.
fn linear_srgb_to_p3(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c, vec3<f32>(0.8224621, 0.1775380, 0.0)),
        dot(c, vec3<f32>(0.0331941, 0.9668058, 0.0)),
        dot(c, vec3<f32>(0.0170827, 0.0723974, 0.9105199)),
    );
}

fn linear_p3_to_srgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c, vec3<f32>(1.2249401, -0.2249404, 0.0)),
        dot(c, vec3<f32>(-0.0420569, 1.0420571, 0.0)),
        dot(c, vec3<f32>(-0.0196376, -0.0786361, 1.0982735)),
    );
}

// The sRGB OETF made odd (`f(−x) = −f(x)`), so a channel driven negative follows the
// mirrored curve rather than the linear toe `linear_to_srgb` puts every negative
// input on — which is what an *extended* encoding is for.
fn encode_extended(lin: vec3<f32>) -> vec3<f32> {
    return sign(lin) * linear_to_srgb(abs(lin));
}

fn decode_extended(enc: vec3<f32>) -> vec3<f32> {
    return sign(enc) * srgb_to_linear(abs(enc));
}

fn encode_display(transfer: u32, lin: vec3<f32>) -> vec3<f32> {
    if transfer == TRANSFER_LINEAR {
        return lin;
    }
    if transfer == TRANSFER_EXTENDED_SRGB {
        return encode_extended(lin);
    }
    if transfer == TRANSFER_DISPLAY_P3 {
        return linear_to_srgb(linear_srgb_to_p3(lin));
    }
    if transfer == TRANSFER_EXTENDED_DISPLAY_P3 {
        return encode_extended(linear_srgb_to_p3(lin));
    }
    return linear_to_srgb(lin);
}

// The exact inverse of `encode_display` for the same selector.
fn decode_display(transfer: u32, enc: vec3<f32>) -> vec3<f32> {
    if transfer == TRANSFER_LINEAR {
        return enc;
    }
    if transfer == TRANSFER_EXTENDED_SRGB {
        return decode_extended(enc);
    }
    if transfer == TRANSFER_DISPLAY_P3 {
        return linear_p3_to_srgb(srgb_to_linear(enc));
    }
    if transfer == TRANSFER_EXTENDED_DISPLAY_P3 {
        return linear_p3_to_srgb(decode_extended(enc));
    }
    return srgb_to_linear(enc);
}
