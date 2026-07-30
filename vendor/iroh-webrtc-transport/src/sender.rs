use std::io;
use std::sync::Arc;
use std::task::{Context, Poll};

use iroh::endpoint::transports::{CustomSender, Transmit};
use iroh_base::CustomAddr;

use crate::bridge::WebRtcTunnel;

#[derive(Debug)]
pub(crate) struct WebRtcSender {
    tunnel: Arc<WebRtcTunnel>,
}

impl WebRtcSender {
    pub(crate) fn new(tunnel: Arc<WebRtcTunnel>) -> Self {
        Self { tunnel }
    }

    fn split_transmit<'a>(transmit: &'a Transmit<'a>) -> impl Iterator<Item = Vec<u8>> + 'a {
        let segment_size = transmit
            .segment_size
            .unwrap_or(transmit.contents.len())
            .max(1);
        transmit
            .contents
            .chunks(segment_size)
            .map(|c| c.to_vec())
    }
}

impl CustomSender for WebRtcSender {
    fn is_valid_send_addr(&self, addr: &CustomAddr) -> bool {
        self.tunnel.has_route(addr)
    }

    fn poll_send(
        &self,
        _cx: &mut Context,
        dst: &CustomAddr,
        _src: Option<&CustomAddr>,
        transmit: &Transmit<'_>,
    ) -> Poll<io::Result<()>> {
        let Some(out_tx) = self.tunnel.route_sender(dst) else {
            return Poll::Ready(Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "no WebRTC data channel attached for this CustomAddr",
            )));
        };

        for chunk in Self::split_transmit(transmit) {
            if out_tx.send(chunk).is_err() {
                return Poll::Ready(Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "WebRTC outbound queue closed",
                )));
            }
        }
        Poll::Ready(Ok(()))
    }
}
