//! A shared session as a **link** (§12.4): the address a peer opens, and the
//! ticket read back out of one.
//!
//! Nothing about the network is here — a ticket is an opaque string to this
//! module, and what it decodes to is `stark-net`'s. What is here is the two
//! string operations both frontends need and neither can own alone.
//!
//! # Why a frontend needs the web client's address
//!
//! The web app builds its invitation out of `location`: the page *is* the client,
//! so the link is this page's address with the ticket in the fragment. A native
//! window has no address, and a link naming nothing is a link nobody can open —
//! so it hands out [`WEB_APP`], the one client that lives somewhere.
//!
//! # Why the fragment
//!
//! A ticket rides after the `#` because that is the part of a URL a browser keeps
//! to itself: it is never sent to the server, so pasting a session link into a
//! chat window does not hand the drawing's key to the host of the page. The
//! alphabet is base64url for the same reason (`stark_net::SessionTicket`), which
//! is what makes [`ticket_in`] a `split` rather than a parse — no `#` can occur
//! inside a ticket, so the first one is the whole of the boundary.

/// Where the web client lives — what a native invitation names.
///
/// A constant rather than a setting, because it is a fact about this project
/// rather than a preference: there is one hosted build, and a link to anywhere
/// else would not open the app. A private deployment changes it here.
pub const WEB_APP: &str = "https://cbbowen.github.io/stark-vibed/";

/// The invitation to hand out: the hosted client, with `ticket` in its fragment.
pub fn invite_link(ticket: &str) -> String {
    format!("{WEB_APP}#{ticket}")
}

/// The ticket in a pasted link — everything after the first `#`.
///
/// **A bare ticket is accepted too**, which is what makes this the one reader for
/// a paste: what arrives from a chat window is a URL, what arrives from someone
/// who trimmed it is the ticket alone, and a Join that took only the first would
/// refuse the shorter thing for no reason a person could see.
///
/// The whitespace goes both times: a paste carries the newline the message ended
/// on, and a URL wrapped across two lines carries one in the middle of nothing.
/// The case does **not** — a ticket's alphabet is case-sensitive, and lowercasing
/// a link is the one transformation that makes it silently undecodable.
///
/// `None` where there is nothing left to try, so the caller has one thing to
/// report rather than two.
pub fn ticket_in(link: &str) -> Option<&str> {
    let text = link.trim();
    let ticket = match text.split_once('#') {
        Some((_, fragment)) => fragment,
        None => text,
    }
    .trim();
    (!ticket.is_empty()).then_some(ticket)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_link_is_the_web_client_and_the_ticket() {
        let link = invite_link("starkAo3LrQ");
        assert!(link.starts_with(WEB_APP), "{link} does not name the client");
        assert_eq!(ticket_in(&link), Some("starkAo3LrQ"));
    }

    /// The example a person actually pastes, whole.
    #[test]
    fn a_pasted_link_gives_up_its_ticket() {
        let link = "https://cbbowen.github.io/stark-vibed/#starkAo3LrQrCUBiA4R1YshoVBJcMfmerFqMWi2h3irCDYxv7UcSoiHdg0CCmYVCGQTBNTGpSsJoGti1uLI1dwNie_L6oR6MMGCpR2lqMmz7NW7v3-D7bNJfk0zr6_5XT8OU1m5uHh-fARhQj6Lqi1TA2tAlX5UAdivwUJBaIKgsgEmkE-OK17Ure-XXP33rZ2LolYt5OHfOxKOxfSmBdIw";
        assert_eq!(ticket_in(link), Some(&link[link.find('#').unwrap() + 1..]));
    }

    /// A ticket someone trimmed the URL off is still a ticket.
    #[test]
    fn a_bare_ticket_is_accepted() {
        assert_eq!(ticket_in("starkAo3LrQ"), Some("starkAo3LrQ"));
    }

    /// What a paste actually carries around the link.
    #[test]
    fn whitespace_around_a_paste_is_not_part_of_the_ticket() {
        assert_eq!(ticket_in("  \n#starkAo3LrQ\n "), Some("starkAo3LrQ"));
        assert_eq!(ticket_in("\tstarkAo3LrQ  "), Some("starkAo3LrQ"));
    }

    /// The fragment is everything after the **first** `#`, so a query string
    /// before it is cut and a `#` after it — which no ticket contains — is kept
    /// rather than becoming a second boundary.
    #[test]
    fn only_the_first_hash_is_a_boundary() {
        assert_eq!(ticket_in("https://x.test/?a=b#stark123"), Some("stark123"));
        assert_eq!(ticket_in("#stark#123"), Some("stark#123"));
    }

    /// Nothing to try is answered once, here, rather than by a parse failure the
    /// caller would have to phrase differently.
    #[test]
    fn there_is_nothing_in_an_empty_paste() {
        assert_eq!(ticket_in(""), None);
        assert_eq!(ticket_in("   "), None);
        assert_eq!(ticket_in("https://cbbowen.github.io/stark-vibed/#"), None);
    }

    /// The case survives. A ticket is base64url and case-carrying, so a reader
    /// that normalized would hand `stark-net` a link that cannot decode — and the
    /// failure would arrive as "bad ticket" with the paste looking perfect.
    #[test]
    fn the_case_of_a_ticket_is_left_alone() {
        assert_eq!(ticket_in("#starkAo3LrQ"), Some("starkAo3LrQ"));
    }
}
