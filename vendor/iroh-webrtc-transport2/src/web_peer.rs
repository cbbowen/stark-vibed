//! Browser twin of `str0m_peer.rs`: the browser's own `RTCPeerConnection` +
//! data channel, negotiated over the same [`Signaling`] envelopes and attached
//! to the same [`WebRtcTunnel`] bridge.
//!
//! NOT in upstream — upstream's browser demo keeps the data channel in
//! JavaScript and never bridges it into an iroh custom transport. This module
//! is what makes the transport usable from wasm. Event-handling patterns
//! (open/close/error oneshots, `bufferedamountlow` backpressure, ownership of
//! the peer connection by the outbound pump) follow the browser-verified code
//! in `vendor/iroh-webrtc-transport` (the SuddenlyHazel facade).
//!
//! Signaling stays the two-message offer/answer protocol shared with the
//! native side: ICE candidates ride inside the SDP ("vanilla ICE"), which the
//! browser produces by waiting for ICE gathering to complete before sending
//! its local description. str0m on the native side does the same with its
//! host candidate, so browser↔native negotiation interoperates.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::{Context as _, anyhow, bail};
use iroh_base::CustomAddr;
use js_sys::{Array, Object, Reflect, Uint8Array};
use tokio::sync::oneshot;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::{JsFuture, spawn_local};
use web_sys::{
    Event, MessageEvent, RtcConfiguration, RtcDataChannel, RtcDataChannelEvent,
    RtcDataChannelState, RtcDataChannelType, RtcIceGatheringState, RtcPeerConnection, RtcSdpType,
    RtcSessionDescriptionInit,
};

use crate::bridge::{AttachOptions, InboundPacket, WebRtcTunnel};
use crate::jsep_envelope::SignalEnvelope;
use crate::jsep_signaling::Signaling;
use crate::transport::WebRtcTransport;

/// Pause outbound sends when the channel buffers more than this.
const BUFFERED_HIGH_WATER: u32 = 1 << 20; // 1 MiB
/// Resume once the buffer drains below this (`bufferedamountlow`).
const BUFFERED_LOW_WATER: u32 = 256 << 10; // 256 KiB

/// ICE configuration for the browser peer connection.
#[derive(Debug, Clone)]
pub struct WebPeerConfig {
    /// STUN server URLs (e.g. `stun:stun.l.google.com:19302`). Empty means
    /// host candidates only — fine on one LAN, useless across NATs.
    pub stun_urls: Vec<String>,
}

impl Default for WebPeerConfig {
    fn default() -> Self {
        Self {
            stun_urls: vec!["stun:stun.l.google.com:19302".into()],
        }
    }
}

/// A negotiated browser WebRTC data channel (ready for I/O), analogous to the
/// native `Str0mPeer`.
pub struct WebRtcPeer {
    pc: RtcPeerConnection,
    dc: RtcDataChannel,
}

fn js_err(context: &'static str) -> impl FnOnce(JsValue) -> anyhow::Error {
    move |e| {
        let msg = e
            .as_string()
            .unwrap_or_else(|| format!("{:?}", e));
        anyhow!("{context}: {msg}")
    }
}

fn new_peer_connection(config: &WebPeerConfig) -> anyhow::Result<RtcPeerConnection> {
    let rtc_config = RtcConfiguration::new();
    if !config.stun_urls.is_empty() {
        let servers = Array::new();
        for url in &config.stun_urls {
            let server = Object::new();
            Reflect::set(&server, &JsValue::from_str("urls"), &JsValue::from_str(url))
                .map_err(js_err("set ICE server url"))?;
            servers.push(&server);
        }
        rtc_config.set_ice_servers(&servers);
    }
    RtcPeerConnection::new_with_configuration(&rtc_config)
        .map_err(js_err("RTCPeerConnection::new"))
}

/// Reads the `sdp` field of a JS session-description value.
fn sdp_of(value: &JsValue) -> anyhow::Result<String> {
    let sdp = Reflect::get(value, &JsValue::from_str("sdp"))
        .map_err(js_err("read sdp"))?
        .as_string()
        .context("session description sdp is not a string")?;
    if sdp.is_empty() {
        bail!("session description did not include SDP");
    }
    Ok(sdp)
}

/// Waits until ICE gathering completes, so the local description carries every
/// candidate (vanilla ICE — matches the two-message signaling protocol).
async fn wait_ice_gathering_complete(pc: &RtcPeerConnection) {
    if pc.ice_gathering_state() == RtcIceGatheringState::Complete {
        return;
    }
    let (tx, rx) = oneshot::channel::<()>();
    let tx = Rc::new(RefCell::new(Some(tx)));
    let handler = {
        let pc = pc.clone();
        let tx = tx.clone();
        Closure::wrap(Box::new(move |_event: Event| {
            if pc.ice_gathering_state() == RtcIceGatheringState::Complete
                && let Some(tx) = tx.borrow_mut().take()
            {
                let _ = tx.send(());
            }
        }) as Box<dyn FnMut(_)>)
    };
    let installed = pc
        .add_event_listener_with_callback(
            "icegatheringstatechange",
            handler.as_ref().unchecked_ref(),
        )
        .is_ok();
    // Close the install race: the state may have flipped before the listener
    // landed (or the listener may have failed to install at all).
    if !installed || pc.ice_gathering_state() == RtcIceGatheringState::Complete {
        if let Some(tx) = tx.borrow_mut().take() {
            let _ = tx.send(());
        }
    }
    let _ = rx.await;
    let _ = pc.remove_event_listener_with_callback(
        "icegatheringstatechange",
        handler.as_ref().unchecked_ref(),
    );
}

/// Waits until the data channel reports open (or fails).
async fn wait_channel_open(dc: &RtcDataChannel) -> anyhow::Result<()> {
    match dc.ready_state() {
        RtcDataChannelState::Open => return Ok(()),
        RtcDataChannelState::Closing | RtcDataChannelState::Closed => {
            bail!("data channel closed before opening")
        }
        _ => {}
    }

    #[derive(Clone, Copy)]
    enum Outcome {
        Open,
        Closed,
        Failed,
    }
    let (tx, rx) = oneshot::channel::<Outcome>();
    let tx = Rc::new(RefCell::new(Some(tx)));
    let make = |outcome: Outcome, tx: Rc<RefCell<Option<oneshot::Sender<Outcome>>>>| {
        Closure::wrap(Box::new(move |_event: Event| {
            if let Some(tx) = tx.borrow_mut().take() {
                let _ = tx.send(outcome);
            }
        }) as Box<dyn FnMut(_)>)
    };
    let on_open = make(Outcome::Open, tx.clone());
    let on_close = make(Outcome::Closed, tx.clone());
    let on_error = make(Outcome::Failed, tx.clone());
    let _ = dc.add_event_listener_with_callback("open", on_open.as_ref().unchecked_ref());
    let _ = dc.add_event_listener_with_callback("close", on_close.as_ref().unchecked_ref());
    let _ = dc.add_event_listener_with_callback("error", on_error.as_ref().unchecked_ref());
    // Close the install race.
    match dc.ready_state() {
        RtcDataChannelState::Open => {
            if let Some(tx) = tx.borrow_mut().take() {
                let _ = tx.send(Outcome::Open);
            }
        }
        RtcDataChannelState::Closing | RtcDataChannelState::Closed => {
            if let Some(tx) = tx.borrow_mut().take() {
                let _ = tx.send(Outcome::Closed);
            }
        }
        _ => {}
    }

    let outcome = rx.await.unwrap_or(Outcome::Failed);
    let _ = dc.remove_event_listener_with_callback("open", on_open.as_ref().unchecked_ref());
    let _ = dc.remove_event_listener_with_callback("close", on_close.as_ref().unchecked_ref());
    let _ = dc.remove_event_listener_with_callback("error", on_error.as_ref().unchecked_ref());
    match outcome {
        Outcome::Open => Ok(()),
        Outcome::Closed => bail!("data channel closed before opening"),
        Outcome::Failed => bail!("data channel failed before opening"),
    }
}

pub async fn negotiate_dc_as_offerer<S: Signaling + ?Sized>(
    sig: &mut S,
    dc_label: &str,
    config: &WebPeerConfig,
) -> anyhow::Result<WebRtcPeer> {
    let pc = new_peer_connection(config)?;
    let dc = pc.create_data_channel(dc_label);
    dc.set_binary_type(RtcDataChannelType::Arraybuffer);

    let offer = JsFuture::from(pc.create_offer())
        .await
        .map_err(js_err("create_offer"))?;
    let description = RtcSessionDescriptionInit::new(RtcSdpType::Offer);
    description.set_sdp(&sdp_of(&offer)?);
    JsFuture::from(pc.set_local_description(&description))
        .await
        .map_err(js_err("set_local_description (offer)"))?;
    wait_ice_gathering_complete(&pc).await;
    let sdp = pc
        .local_description()
        .context("no local description after ICE gathering")?
        .sdp();
    sig.send_envelope(&SignalEnvelope::Offer { sdp }).await?;

    let answer = match sig.recv_envelope().await? {
        SignalEnvelope::Answer { sdp } => sdp,
        other => bail!("expected answer, got {:?}", other),
    };
    let description = RtcSessionDescriptionInit::new(RtcSdpType::Answer);
    description.set_sdp(&answer);
    JsFuture::from(pc.set_remote_description(&description))
        .await
        .map_err(js_err("set_remote_description (answer)"))?;

    wait_channel_open(&dc).await?;
    Ok(WebRtcPeer { pc, dc })
}

pub async fn negotiate_dc_as_answerer<S: Signaling + ?Sized>(
    sig: &mut S,
    config: &WebPeerConfig,
) -> anyhow::Result<WebRtcPeer> {
    let pc = new_peer_connection(config)?;

    // Install before applying the offer: the channel event fires as a result
    // of `set_remote_description` seeing the offer's data-channel m-line.
    let (dc_tx, dc_rx) = oneshot::channel::<RtcDataChannel>();
    let dc_tx = Rc::new(RefCell::new(Some(dc_tx)));
    let on_channel = Closure::wrap(Box::new(move |event: RtcDataChannelEvent| {
        if let Some(tx) = dc_tx.borrow_mut().take() {
            let _ = tx.send(event.channel());
        }
    }) as Box<dyn FnMut(_)>);
    pc.set_ondatachannel(Some(on_channel.as_ref().unchecked_ref()));
    // Fires at most once, then idles for the (single-channel) peer's lifetime.
    on_channel.forget();

    let offer = match sig.recv_envelope().await? {
        SignalEnvelope::Offer { sdp } => sdp,
        other => bail!("expected offer, got {:?}", other),
    };
    let description = RtcSessionDescriptionInit::new(RtcSdpType::Offer);
    description.set_sdp(&offer);
    JsFuture::from(pc.set_remote_description(&description))
        .await
        .map_err(js_err("set_remote_description (offer)"))?;

    let answer = JsFuture::from(pc.create_answer())
        .await
        .map_err(js_err("create_answer"))?;
    let description = RtcSessionDescriptionInit::new(RtcSdpType::Answer);
    description.set_sdp(&sdp_of(&answer)?);
    JsFuture::from(pc.set_local_description(&description))
        .await
        .map_err(js_err("set_local_description (answer)"))?;
    wait_ice_gathering_complete(&pc).await;
    let sdp = pc
        .local_description()
        .context("no local description after ICE gathering")?
        .sdp();
    sig.send_envelope(&SignalEnvelope::Answer { sdp }).await?;

    let dc = dc_rx.await.context("no data channel from remote offer")?;
    dc.set_binary_type(RtcDataChannelType::Arraybuffer);
    wait_channel_open(&dc).await?;
    Ok(WebRtcPeer { pc, dc })
}

/// Waits for the buffered amount to drain below the low-water mark. Returns
/// `false` if the channel closed or failed instead.
async fn wait_send_capacity(dc: &RtcDataChannel) -> bool {
    if dc.buffered_amount() <= BUFFERED_LOW_WATER {
        return true;
    }
    if matches!(
        dc.ready_state(),
        RtcDataChannelState::Closing | RtcDataChannelState::Closed
    ) {
        return false;
    }
    let (tx, rx) = oneshot::channel::<bool>();
    let tx = Rc::new(RefCell::new(Some(tx)));
    let make = |ok: bool, tx: Rc<RefCell<Option<oneshot::Sender<bool>>>>| {
        Closure::wrap(Box::new(move |_event: Event| {
            if let Some(tx) = tx.borrow_mut().take() {
                let _ = tx.send(ok);
            }
        }) as Box<dyn FnMut(_)>)
    };
    let on_low = make(true, tx.clone());
    let on_close = make(false, tx.clone());
    let on_error = make(false, tx.clone());
    let _ = dc.add_event_listener_with_callback(
        "bufferedamountlow",
        on_low.as_ref().unchecked_ref(),
    );
    let _ = dc.add_event_listener_with_callback("close", on_close.as_ref().unchecked_ref());
    let _ = dc.add_event_listener_with_callback("error", on_error.as_ref().unchecked_ref());
    // Close the install race.
    if dc.buffered_amount() <= BUFFERED_LOW_WATER
        && let Some(tx) = tx.borrow_mut().take()
    {
        let _ = tx.send(true);
    }
    let ok = rx.await.unwrap_or(false);
    let _ = dc.remove_event_listener_with_callback(
        "bufferedamountlow",
        on_low.as_ref().unchecked_ref(),
    );
    let _ = dc.remove_event_listener_with_callback("close", on_close.as_ref().unchecked_ref());
    let _ = dc.remove_event_listener_with_callback("error", on_error.as_ref().unchecked_ref());
    ok
}

impl WebRtcTransport {
    /// Wire a negotiated browser data channel into this transport so iroh can
    /// send/receive QUIC datagrams on it. wasm counterpart of the native
    /// `attach_data_channel(Str0mPeer, ..)`.
    pub fn attach_data_channel(
        &self,
        peer: WebRtcPeer,
        remote_custom_addr: CustomAddr,
        opts: AttachOptions,
    ) -> anyhow::Result<()> {
        self.tunnel.attach_web_peer(peer, remote_custom_addr, opts)
    }
}

impl WebRtcTunnel {
    pub(crate) fn attach_web_peer(
        self: &Arc<Self>,
        peer: WebRtcPeer,
        remote_custom: CustomAddr,
        opts: AttachOptions,
    ) -> anyhow::Result<()> {
        let (route_id, mut out_rx) = self
            .add_route(remote_custom.clone())
            .map_err(|e| anyhow!("{e}"))?;
        let in_tx = self.inbound_sender();
        let wake = Arc::clone(self);
        let cleanup_tunnel = Arc::clone(self);
        let cleanup_remote = remote_custom.clone();

        let WebRtcPeer { pc, dc } = peer;
        dc.set_buffered_amount_low_threshold(BUFFERED_LOW_WATER);

        // Inbound: data channel message -> tunnel inbound queue -> iroh poll_recv.
        let on_message = {
            let tap = opts.tap_inbound_to.clone();
            let mirror = opts.mirror_sctp_echo;
            let dc = dc.clone();
            Closure::wrap(Box::new(move |event: MessageEvent| {
                let data = event.data();
                let Some(buffer) = data.dyn_ref::<js_sys::ArrayBuffer>() else {
                    tracing::debug!("dropping non-binary RTCDataChannel message");
                    return;
                };
                let bytes = Uint8Array::new(buffer).to_vec();
                if let Some(tap) = &tap {
                    let _ = tap.send(bytes.clone());
                }
                if mirror {
                    let _ = dc.send_with_u8_array(&bytes);
                }
                let packet = InboundPacket {
                    source_custom: remote_custom.clone(),
                    payload: bytes,
                };
                // Bounded queue: if iroh is not draining fast enough, drop —
                // these are QUIC datagrams, loss is handled above us.
                if in_tx.try_send(packet).is_ok() {
                    wake.wake_recv_pollers();
                } else {
                    tracing::debug!("tunnel inbound queue full; dropping datagram");
                }
            }) as Box<dyn FnMut(_)>)
        };
        dc.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
        // One handler per channel, alive for the page's lifetime rather than
        // threading teardown plumbing through. After the route is removed a
        // late message still enqueues an inbound packet; iroh ignores
        // datagrams from paths it no longer uses.
        on_message.forget();

        // Outbound pump: iroh poll_send -> route queue -> data channel. Owns
        // `pc` so the peer connection is not GC'd while active. The pump is an
        // inner async block so its `return`s fall through to the route
        // cleanup (route removal makes `is_valid_send_addr` false again, so
        // iroh stops using the path).
        spawn_local(async move {
            let _pc = pc;
            let pump = async {
                loop {
                    let Some(bytes) = out_rx.recv().await else {
                        tracing::debug!("route replaced or transport dropped; stopping WebRTC pump");
                        return;
                    };
                    if bytes.is_empty() {
                        continue;
                    }
                    if matches!(
                        dc.ready_state(),
                        RtcDataChannelState::Closing | RtcDataChannelState::Closed
                    ) {
                        tracing::debug!("data channel closed; stopping WebRTC pump");
                        return;
                    }
                    if dc
                        .buffered_amount()
                        .saturating_add(bytes.len() as u32)
                        > BUFFERED_HIGH_WATER
                        && !wait_send_capacity(&dc).await
                    {
                        tracing::debug!("data channel closed while draining; stopping WebRTC pump");
                        return;
                    }
                    if let Err(error) = dc.send_with_u8_array(&bytes) {
                        tracing::debug!(?error, "data channel send failed; stopping WebRTC pump");
                        return;
                    }
                }
            };
            pump.await;
            cleanup_tunnel.remove_route(&cleanup_remote, route_id);
        });

        Ok(())
    }
}
