//! The two halves of `crate::platform` declare one surface (§11).
//!
//! `src/platform/web.rs` is compiled only for wasm32 and `src/platform/stub.rs` only off
//! it, so no one build sees both: the host build checks every call site against the stub,
//! the wasm build against the browser. A stub that drifts from its web twin compiles on
//! each side, until a call site written against one meets the other in the build nobody
//! runs tests in. So this reads both files and compares what they declare.

use std::collections::BTreeMap;
use std::path::Path;

/// Each public item keyed `kind name`, an inherent `impl`'s through its type (`fn
/// pick_file`, `fn KeyEvent::key`, `struct Canvas`), valued with its signature's tokens
/// after [`normalize`]: everything a fn declares after its name, and a const's type.
/// Empty for a type, whose fields are the one thing the two halves are meant to disagree
/// on.
type Items = BTreeMap<String, Vec<String>>;

#[test]
fn web_and_stub_declare_the_same_names() {
    let (web, stub) = (declared("web.rs"), declared("stub.rs"));
    let only = |a: &Items, b: &Items| {
        a.keys()
            .filter(|name| !b.contains_key(*name))
            .cloned()
            .collect::<Vec<_>>()
    };
    let (only_web, only_stub) = (only(&web, &stub), only(&stub, &web));
    assert!(
        only_web.is_empty() && only_stub.is_empty(),
        "platform/web.rs and platform/stub.rs must declare the same public items\n  \
         only in web.rs:  {only_web:?}\n  only in stub.rs: {only_stub:?}"
    );
}

/// Parameter names included, since a stub's names are the ones the host build's docs
/// show. What is not part of a signature is ignored: layout, a trailing comma, a `mut`
/// binding, and the `_` a stub puts on a parameter it does not read.
#[test]
fn web_and_stub_agree_on_every_signature() {
    let (web, stub) = (declared("web.rs"), declared("stub.rs"));
    let differing: Vec<String> = web
        .iter()
        .filter_map(|(name, ours)| {
            let theirs = stub.get(name)?;
            (ours != theirs).then(|| {
                format!(
                    "{name}\n  web.rs:  {}\n  stub.rs: {}",
                    join(ours),
                    join(theirs)
                )
            })
        })
        .collect();
    assert!(
        differing.is_empty(),
        "a stub's signature differs from its web twin's:\n{}",
        differing.join("\n")
    );
}

/// The scanner, on everything it has to read past. A scanner that lost its place would
/// find fewer items or none, and the two tests above would agree with each other about
/// nothing.
#[test]
fn the_scanner_reads_signatures_through_what_surrounds_them() {
    let source = r##"
//! `pub fn in_a_line_comment() {` is prose.
use super::{A, B};

/* pub fn in_a_block_comment() { /* nested } */ */
const BRACE: char = '{';
const ESCAPED: char = '\'';
const BYTE: u8 = b'{';
#[derive(Clone)]
pub struct Tuple(inner::Thing);
pub(crate) struct Restricted;
pub const LIMIT: u32 = 4;

pub fn multi_line<'a>(
    mut _first: &'a str,
    second: impl FnMut(Vec<u8>) -> bool + 'static,
) -> Result<(), String>
where
    'a: 'static,
{
    let s = "pub fn in_a_string() { \" {";
    let r = r#"pub fn in_a_raw_string() {"#;
    fn nested() {}
}

fn private(f: impl Fn()) -> impl Iterator<Item = u8> {
    pub fn inside_a_body() {}
}

pub fn digest(bytes: [u8; 4], mut r#type: u8) -> [u8; 4] {
    bytes
}

pub fn each<F>(f: F)
where
    for<'a> F: Fn(&'a str),
{
}

pub const fn limit() -> u32 {
    LIMIT
}

impl Tuple {
    pub const SIZE: usize = 1;
    pub fn method(&mut self, _x: u8) {}
    fn private_method(&self) {}
}

pub struct Holder<T>(T);

impl<T> Holder<T>
where
    for<'a> T: Fn(&'a u8),
{
    pub const SIZE: [u8; 4] = [0; 4];
    pub fn call(&self) {}
}

impl Clone for Restricted {
    fn clone(&self) -> Self {
        Restricted
    }
}

pub async fn after_the_impls() {}
"##;
    let found: Vec<(String, String)> = items(source)
        .into_iter()
        .map(|(name, signature)| (name, join(&signature)))
        .collect();
    let expected = [
        ("const Holder<T>::SIZE", ": [u8; 4]"),
        ("const LIMIT", ": u32"),
        ("const Tuple::SIZE", ": usize"),
        ("fn Holder<T>::call", "fn(&self)"),
        ("fn Tuple::method", "fn(&mut self, x: u8)"),
        ("fn after_the_impls", "async fn()"),
        ("fn digest", "fn(bytes: [u8; 4], r#type: u8) -> [u8; 4]"),
        ("fn each", "fn<F>(f: F) where for<'a> F: Fn(&'a str)"),
        ("fn limit", "const fn() -> u32"),
        (
            "fn multi_line",
            "fn<'a>(first: &'a str, second: impl FnMut(Vec<u8>) -> bool + 'static) \
             -> Result<(), String> where 'a: 'static",
        ),
        ("struct Holder", ""),
        ("struct Restricted", ""),
        ("struct Tuple", ""),
    ]
    .map(|(name, signature)| (name.to_string(), signature.to_string()));
    assert_eq!(found, expected, "the fixture's public items, read back");
}

fn declared(file: &str) -> Items {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/platform")
        .join(file);
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} does not read: {e}", path.display()));
    let found = items(&source);
    assert!(
        !found.is_empty(),
        "no public item was found in {}, which is a green test that read nothing",
        path.display()
    );
    found
}

/// The public items `source` declares at its top level and in its inherent `impl`s.
fn items(source: &str) -> Items {
    let tokens = tokens(source);
    let mut found = Items::new();
    let mut depth = 0usize;
    // The inherent `impl` whose body is open: its type, and the depth its items sit at.
    let mut inherent: Option<(String, usize)> = None;
    // Set by a `pub` to where the item's qualifiers (`async`, `const`, …) begin.
    let mut public: Option<usize> = None;
    let mut i = 0;
    while let Some(token) = tokens.get(i) {
        let at_item_level = depth == 0 || inherent.as_ref().is_some_and(|(_, d)| *d == depth);
        let next = tokens.get(i + 1).map(String::as_str);
        match token.as_str() {
            "{" => depth += 1,
            "}" => {
                depth -= 1;
                if inherent.as_ref().is_some_and(|(_, d)| *d > depth) {
                    inherent = None;
                }
                public = None;
            }
            ";" => public = None,
            "pub" if at_item_level => {
                if next == Some("(") {
                    i = closing(&tokens, i + 1);
                }
                public = Some(i + 1);
            }
            // Every fn at item level is read to its body, public or not, so no `impl
            // Trait` in its signature can be taken for an `impl` block.
            "fn" if at_item_level => {
                let end = position(&tokens, i, &["{", ";"]);
                if let (Some(start), Some(name)) = (public.take(), next) {
                    let mut signature = tokens[start..=i].to_vec();
                    signature.extend_from_slice(&tokens[(i + 2).min(end)..end]);
                    found.insert(key("fn", inherent.as_ref(), name), normalize(&signature));
                }
                i = end;
                continue;
            }
            "impl" if depth == 0 => {
                let open = position(&tokens, i, &["{"]);
                let header = &tokens[i + 1..open];
                inherent = (!is_trait_impl(header)).then(|| (join(self_type(header)), depth + 1));
                public = None;
                i = open;
                continue;
            }
            kind @ ("struct" | "enum" | "union" | "type" | "trait") if at_item_level => {
                if let (Some(_), Some(name)) = (public.take(), next) {
                    found.insert(key(kind, inherent.as_ref(), name), Vec::new());
                }
            }
            // `const` and `static` items — not a `const fn`, whose `const` is a qualifier.
            kind @ ("const" | "static")
                if at_item_level
                    && public.is_some()
                    && !matches!(next, Some("fn" | "unsafe" | "async" | "extern")) =>
            {
                public = None;
                let name_at = if next == Some("mut") { i + 2 } else { i + 1 };
                let end = position(&tokens, i, &["=", ";"]);
                if let Some(name) = tokens.get(name_at) {
                    let ty = &tokens[(name_at + 1).min(end)..end];
                    found.insert(key(kind, inherent.as_ref(), name), normalize(ty));
                }
            }
            _ => {}
        }
        i += 1;
    }
    found
}

/// An item's key in [`Items`]: `kind name`, or `kind Type::name` inside an inherent `impl`,
/// so two types' items of one name stay two items.
fn key(kind: &str, inherent: Option<&(String, usize)>, name: &str) -> String {
    match inherent {
        Some((ty, _)) => format!("{kind} {ty}::{name}"),
        None => format!("{kind} {name}"),
    }
}

/// Drop from a signature what is not part of one. See
/// [`web_and_stub_agree_on_every_signature`].
fn normalize(signature: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(signature.len());
    for (i, token) in signature.iter().enumerate() {
        let next = signature.get(i + 1).map(String::as_str);
        let after = signature.get(i + 2).map(String::as_str);
        // A parameter's name, less the `_` a stub gives one it does not read.
        let unread = token
            .strip_prefix('_')
            .filter(|name| !name.is_empty() && next == Some(":"));
        match token.as_str() {
            "mut" if next.is_some_and(is_word) && after == Some(":") => {}
            "," if matches!(next, None | Some(")" | ">" | "]")) => {}
            _ => out.push(unread.unwrap_or(token).to_string()),
        }
    }
    out
}

/// Rust source as tokens, with comments gone and every string and character literal
/// one token, so no brace inside either can move the depth. `::` and `->` are one
/// token each, so a lone `:` is always a binding's and a lone `>` always a bracket.
fn tokens(source: &str) -> Vec<String> {
    let chars: Vec<char> = source.chars().collect();
    let word = |c: char| c.is_alphanumeric() || c == '_';
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(&c) = chars.get(i) {
        let next = chars.get(i + 1).copied();
        let (end, keep) = if c.is_whitespace() {
            (i + 1, false)
        } else if c == '/' && next == Some('/') {
            let line_end = chars[i..].iter().position(|&c| c == '\n');
            (line_end.map_or(chars.len(), |n| i + n), false)
        } else if c == '/' && next == Some('*') {
            (block_comment_end(&chars, i), false)
        } else if let Some(end) = literal_end(&chars, i) {
            (end, true)
        } else if word(c) || (c == '\'' && next.is_some_and(word)) {
            // A raw identifier is one word, or `r#impl` would be read as an `impl`.
            let raw = c == 'r' && next == Some('#') && chars.get(i + 2).copied().is_some_and(word);
            let from = if raw { i + 2 } else { i + 1 };
            let rest = chars[from..].iter().take_while(|&&c| word(c)).count();
            (from + rest, true)
        } else if matches!((c, next), (':', Some(':')) | ('-', Some('>'))) {
            (i + 2, true)
        } else {
            (i + 1, true)
        };
        if keep {
            out.push(chars[i..end].iter().collect());
        }
        i = end;
    }
    out
}

/// Past the end of the (possibly nested) block comment opening at `i`.
fn block_comment_end(chars: &[char], mut i: usize) -> usize {
    let mut depth = 0usize;
    while i < chars.len() {
        match (chars[i], chars.get(i + 1)) {
            ('/', Some('*')) => {
                depth += 1;
                i += 2;
            }
            ('*', Some('/')) => {
                depth -= 1;
                i += 2;
                if depth == 0 {
                    return i;
                }
            }
            _ => i += 1,
        }
    }
    chars.len()
}

/// Past the end of the string or character literal starting at `i`, or `None` if none
/// does — which, at a `'`, means a lifetime.
fn literal_end(chars: &[char], i: usize) -> Option<usize> {
    let at = |n: usize| chars.get(i + n).copied();
    // `b"…"`, `r#"…"#`, `br"…"`: a prefix, then the hashes of a raw string, then a quote.
    let prefix = match (at(0)?, at(1)) {
        ('b', Some('r')) => 2,
        ('b' | 'r', _) => 1,
        _ => 0,
    };
    let raw = prefix > 0 && at(prefix - 1) == Some('r');
    let hashes = if raw {
        chars[i + prefix..]
            .iter()
            .take_while(|&&c| c == '#')
            .count()
    } else {
        0
    };
    let quote = i + prefix + hashes;
    match chars.get(quote)? {
        '"' if raw => {
            let closer: Vec<char> = std::iter::once('"')
                .chain(std::iter::repeat_n('#', hashes))
                .collect();
            (quote + 1..chars.len())
                .find(|&j| chars[j..].starts_with(&closer))
                .map(|j| j + closer.len())
        }
        '"' => {
            let mut j = quote + 1;
            while let Some(&c) = chars.get(j) {
                match c {
                    '\\' => j += 2,
                    '"' => return Some(j + 1),
                    _ => j += 1,
                }
            }
            None
        }
        '\'' if !raw => match chars.get(quote + 1)? {
            '\\' => (quote + 3..chars.len())
                .find(|&j| chars[j] == '\'')
                .map(|j| j + 1),
            _ if chars.get(quote + 2) == Some(&'\'') => Some(quote + 3),
            _ => None,
        },
        _ => None,
    }
}

/// The index of the first of `stops` at or after `from` that no bracket opened since
/// encloses, or the end — so neither the `;` of `[u8; 4]` nor the `=` of `Item = u8`
/// ends an item.
///
/// `<` is a bracket in a type and an operator in an expression. A signature holds an
/// expression only in a `{ … }` const block or an array's length, so there it opens
/// nothing.
fn position(tokens: &[String], from: usize, stops: &[&str]) -> usize {
    // Each open bracket's closer, and whether what it encloses is an expression.
    let mut open: Vec<(&str, bool)> = Vec::new();
    for (i, token) in tokens.iter().enumerate().skip(from) {
        let token = token.as_str();
        let in_expression = open.last().is_some_and(|&(_, expression)| expression);
        match token {
            _ if open.is_empty() && stops.contains(&token) => return i,
            "(" => open.push((")", in_expression)),
            "[" => open.push(("]", in_expression)),
            "<" if !in_expression => open.push((">", false)),
            "{" => open.push(("}", true)),
            // An array's length follows its `;`.
            ";" => {
                if let Some(("]", expression)) = open.last_mut() {
                    *expression = true;
                }
            }
            _ if open.last().is_some_and(|&(closer, _)| closer == token) => {
                open.pop();
            }
            _ => {}
        }
    }
    tokens.len()
}

/// The `)` matching the `(` at `open`, or the end.
fn closing(tokens: &[String], open: usize) -> usize {
    let mut depth = 0usize;
    for (i, token) in tokens.iter().enumerate().skip(open) {
        match token.as_str() {
            "(" => depth += 1,
            ")" => {
                depth -= 1;
                if depth == 0 {
                    return i;
                }
            }
            _ => {}
        }
    }
    tokens.len()
}

/// Whether an `impl` header names a trait: it holds a `for` that is not a higher-ranked
/// binder (`for<'a>`), which a where clause on either kind of `impl` may hold.
fn is_trait_impl(header: &[String]) -> bool {
    header.iter().enumerate().any(|(i, token)| {
        let binder = header.get(i + 1).is_some_and(|t| t == "<")
            && header
                .get(i + 2)
                .is_some_and(|t| t.starts_with('\'') || t == ">");
        token == "for" && !binder
    })
}

/// The type an inherent `impl` header is for, without the `<…>` an `impl<T>` opens with
/// or the where clause it may end with.
fn self_type(header: &[String]) -> &[String] {
    let start = if header.first().is_some_and(|t| t == "<") {
        (position(header, 1, &[">"]) + 1).min(header.len())
    } else {
        0
    };
    let ty = &header[start..];
    &ty[..position(ty, 0, &["where"])]
}

/// Tokens back to text, spaced the way rustfmt spaces a signature.
fn join(tokens: &[String]) -> String {
    let mut out = String::new();
    let mut prev: Option<&str> = None;
    for token in tokens {
        if prev.is_some_and(|prev| spaced(prev, token)) {
            out.push(' ');
        }
        out.push_str(token);
        prev = Some(token);
    }
    out
}

fn spaced(prev: &str, token: &str) -> bool {
    ((is_word(prev) || matches!(prev, ")" | ">")) && is_word(token))
        || matches!(prev, "," | ":" | ";" | "->" | "+" | "=")
        || matches!(token, "->" | "+" | "=")
}

/// An identifier, keyword, number, lifetime or literal.
fn is_word(token: &str) -> bool {
    token.starts_with(|c: char| c.is_alphanumeric() || matches!(c, '_' | '\'' | '"'))
}
