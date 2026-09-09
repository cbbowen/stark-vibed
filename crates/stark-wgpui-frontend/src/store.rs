//! Where this frontend's records go: two directories (§11.2, §25.6).
//!
//! The web app has `localStorage` for its rows and IndexedDB for its bytes, and the
//! split is forced there — the first is text, a few megabytes of it shared across
//! every record the origin keeps. Natively neither constraint exists, and the split
//! is kept anyway, because it is the one the format already draws: a row is a file
//! under the config directory, a blob is a file under the cache directory, and what
//! that buys is the same thing it buys in a browser — a user clearing caches loses
//! thumbnails and imported bytes, not their settings.
//!
//! One file per record rather than one file with every record in it. A record is
//! written whole on every change ([`Backend::set`]), so separate files mean a preset
//! save cannot corrupt the shortcuts, and a file that goes bad costs its own record
//! and reads as "nothing stored" — which is the failure the format is already built
//! around. The containment stops at the file, though: a record torn in half is a
//! record lost, so a write is staged beside its target and renamed over it rather
//! than truncated in place ([`Files::write`]).
//!
//! A key is a filename. Every key is `stark.`-prefixed and a blob's is
//! `stark.shapes/<hex>`, so the `/` becomes a directory and the layout on disk is the
//! namespacing the keys already had.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use stark_ui::storage::{Backend, Stored};

/// Two directories, resolved once at startup.
pub struct Files {
    /// Rows: settings, libraries, the identity — what a user would be sorry to lose.
    config: PathBuf,
    /// Bytes: imported shape and substrate images, keyed by content id. Rebuildable
    /// in principle — the id names the field — so this is the half that belongs in a
    /// cache directory the OS may reclaim.
    cache: PathBuf,
}

impl Files {
    /// Resolve the two directories, creating them if this is a first run.
    ///
    /// `None` when the platform will not say where they are, which is the same case
    /// as a browser with storage disabled: the app runs and forgets everything
    /// (`stark_ui::storage`'s "failure is silence").
    ///
    /// Hand-rolled rather than through `directories`/`dirs`: what is wanted is two
    /// paths from two environment variables, and the crates that answer that question
    /// answer eleven others as well.
    pub fn resolve() -> Option<Self> {
        let (config, cache) = platform_dirs()?;
        let (config, cache) = (config.join("stark"), cache.join("stark"));
        // A directory that cannot be made is reported by the first write, not here:
        // this runs before anything has asked for a record, and refusing to start
        // over a store is exactly the trade `storage` declines to make.
        let _ = std::fs::create_dir_all(&config);
        let _ = std::fs::create_dir_all(&cache);
        Some(Self { config, cache })
    }

    /// Where a key's file is. Blob keys carry a `/`, which becomes a directory.
    ///
    /// A key is never a path a caller wrote: every one comes from `Store::named` or
    /// from a content id's hex, so there is nothing here to escape. The guard is that
    /// the whole vocabulary is a closed enum plus a 64-character hash, one level up.
    fn path(&self, key: &str) -> PathBuf {
        let root = if key.contains('/') {
            &self.cache
        } else {
            &self.config
        };
        root.join(key)
    }

    /// Write `bytes` to `key`'s file, making its parent if a blob record is new.
    ///
    /// **A scratch file, flushed, then renamed over the target.** `fs::write`
    /// truncates and then writes, so a kill or a power loss between the two leaves
    /// half a record — and half a record is not half a library, it is none: `load`
    /// reads it as damage and answers "nothing stored", which costs every preset, or
    /// every shortcut, at once. `save_list` runs on every library change, so that
    /// window is entered often. `localStorage.setItem` is atomic, so this is also
    /// what stops the two backends promising different things with only one of them
    /// written down.
    ///
    /// **A failure leaves the previous record whole**, which is the property the
    /// whole arrangement is for: nothing touches `path` until bytes that are all
    /// there are renamed onto it. There is deliberately no remove-then-retry —
    /// `fs::rename` replaces an existing destination on Windows as well as on POSIX
    /// (`MOVEFILE_REPLACE_EXISTING`), so a rename that fails here failed for a reason
    /// unlinking the target would not fix — a scanner holding the file open, most
    /// likely — and removing it first would answer a write this process could not
    /// finish by deleting the copy the user still has.
    fn write(&self, key: &str, bytes: &[u8]) -> bool {
        let path = self.path(key);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let Some(temp) = scratch(&path) else {
            return false;
        };
        if !flushed(&temp, bytes) || std::fs::rename(&temp, &path).is_err() {
            let _ = std::fs::remove_file(&temp);
            return false;
        }
        true
    }
}

/// Write `bytes` to `path` and get them onto the disk before returning.
///
/// The flush is what makes this about **power loss** and not only about a kill: the
/// rename is ordered after the data, so the name can never come to point at a file
/// whose contents never landed. What is still the filesystem's business is whether
/// the rename itself survives — losing it costs the new value, which is the failure
/// this whole path is built to leave behind.
fn flushed(path: &std::path::Path, bytes: &[u8]) -> bool {
    use std::io::Write;
    let Ok(mut file) = std::fs::File::create(path) else {
        return false;
    };
    file.write_all(bytes).is_ok() && file.sync_all().is_ok()
}

/// Where [`Files::write`] stages a record before renaming it into place.
///
/// Beside the target, because a rename is only atomic within one filesystem and the
/// config and cache directories need not be on the same one as the OS temp dir.
///
/// Named per process *and* per call: `path.with_extension("tmp")` would map
/// `stark.prefs` and `stark.presets` onto one scratch file, and two threads saving one
/// record would interleave their bytes into it and rename the result into place.
///
/// A crash between the write and the rename strands one of these. Nothing reads it —
/// a key is a whole filename, never a glob — so what it costs is a few bytes in the
/// config directory, which is the same thing a stranded blob costs and is the side of
/// the trade this ordering deliberately takes.
fn scratch(path: &std::path::Path) -> Option<PathBuf> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let name = path.file_name()?.to_string_lossy().into_owned();
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    Some(path.with_file_name(format!("{name}.{}-{n}.tmp", std::process::id())))
}

impl Backend for Files {
    fn get(&self, key: &str) -> Option<String> {
        std::fs::read_to_string(self.path(key)).ok()
    }

    fn set(&self, key: &str, value: &str) -> bool {
        self.write(key, value.as_bytes())
    }

    fn remove(&self, key: &str) {
        let _ = std::fs::remove_file(self.path(key));
    }

    fn blob_get_many<'a>(&'a self, keys: &'a [String]) -> Stored<'a, Vec<Option<Vec<u8>>>> {
        // Ready rather than spawned: the reads are `std::fs`, which is what a native
        // blob store *is*. The signature is async because the web's answer has to be
        // — IndexedDB is a promise — and a future that is already finished costs a
        // poll. Trading that for a thread pool would be paying for the browser's
        // constraint on a platform that does not have it.
        Box::pin(std::future::ready(
            keys.iter()
                .map(|k| std::fs::read(self.path(k)).ok())
                .collect(),
        ))
    }

    fn blob_put<'a>(&'a self, key: &'a str, bytes: &'a [u8]) -> Stored<'a, bool> {
        Box::pin(std::future::ready(self.write(key, bytes)))
    }

    fn blob_delete<'a>(&'a self, key: &'a str) -> Stored<'a, ()> {
        let _ = std::fs::remove_file(self.path(key));
        Box::pin(std::future::ready(()))
    }
}

/// This platform's config and cache directories, by its own convention.
///
/// Windows keeps both under `%APPDATA%`/`%LOCALAPPDATA%`; the XDG platforms split
/// them and fall back to `~/.config` and `~/.cache`; macOS puts both under
/// `~/Library`. Nothing here is Stark-specific — [`Files::resolve`] adds the one
/// directory name.
fn platform_dirs() -> Option<(PathBuf, PathBuf)> {
    #[cfg(target_os = "windows")]
    {
        let roaming = std::env::var_os("APPDATA").map(PathBuf::from)?;
        // Local, not roaming, for the cache half: it is rebuildable bytes, and
        // roaming them across a domain's machines is bandwidth for nothing.
        let local = std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|| roaming.clone());
        Some((roaming, local.join("cache")))
    }
    #[cfg(target_os = "macos")]
    {
        let home = std::env::var_os("HOME").map(PathBuf::from)?;
        Some((
            home.join("Library/Application Support"),
            home.join("Library/Caches"),
        ))
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        let home = std::env::var_os("HOME").map(PathBuf::from);
        let xdg = |var: &str, fallback: &str| -> Option<PathBuf> {
            std::env::var_os(var)
                .map(PathBuf::from)
                .filter(|p| p.is_absolute())
                .or_else(|| home.as_ref().map(|h: &PathBuf| h.join(fallback)))
        };
        Some((
            xdg("XDG_CONFIG_HOME", ".config")?,
            xdg("XDG_CACHE_HOME", ".cache")?,
        ))
    }
}

/// Whether `path` is under `root` — the property [`Files::path`] relies on and does
/// not check, stated here so a test can.
#[cfg(test)]
fn contained(root: &std::path::Path, path: &std::path::Path) -> bool {
    path.starts_with(root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use strum::VariantArray;

    fn files(dir: &std::path::Path) -> Files {
        Files {
            config: dir.join("config"),
            cache: dir.join("cache"),
        }
    }

    /// A row goes to the config directory and a blob to the cache one, decided by the
    /// `/` a blob key carries — which is the namespacing `Store::named` already does,
    /// read as a path.
    #[test]
    fn a_row_and_a_blob_land_in_different_directories() {
        let f = files(std::path::Path::new("/tmp/x"));
        let row = f.path("stark.prefs");
        let blob = f.path("stark.shapes/00ff");
        assert!(contained(&f.config, &row), "a row is a setting");
        assert!(contained(&f.cache, &blob), "a blob is rebuildable");
        assert!(!contained(&f.config, &blob));
    }

    /// A directory that this test owns and takes away with it, so a run leaves the
    /// machine as it found it whether it passed or not.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("stark-store-{}-{name}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("a scratch directory");
            Self(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A record written over an existing one reads back as the new one, and the
    /// staging file it went through is gone.
    ///
    /// The half a `fs::write` could not promise: the old bytes are whole until the
    /// rename, so a kill mid-write costs the *new* value rather than the record.
    /// What a test can see of that is the two properties below — a leaked `.tmp`
    /// would mean the rename never happened, and a short read would mean it happened
    /// against a truncated file.
    #[test]
    fn a_rewritten_record_replaces_the_old_one_and_leaves_no_scratch() {
        let dir = Scratch::new("rewrite");
        let f = files(&dir.0);
        assert!(f.set("stark.prefs", "{\"tips\":true}"));
        assert!(f.set("stark.prefs", "{\"tips\":false}"));
        assert_eq!(f.get("stark.prefs").as_deref(), Some("{\"tips\":false}"));
        let strays: Vec<_> = std::fs::read_dir(&f.config)
            .expect("the config directory")
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".tmp"))
            .collect();
        assert!(
            strays.is_empty(),
            "a staged write was left behind: {strays:?}"
        );
    }

    /// **A write that cannot finish leaves the record it was replacing whole.**
    ///
    /// The property the staging file is for, and the only half of it a test can see
    /// without killing a process: nothing touches the target until bytes that are all
    /// there are renamed onto it, so a failure costs the *new* value. The write is
    /// made to fail by putting a directory where the record's file goes — the same
    /// answer `set` gives for a full disk or a store it may not write.
    #[test]
    fn a_write_that_fails_leaves_the_stored_record_whole() {
        let dir = Scratch::new("failure");
        let f = files(&dir.0);
        assert!(f.set("stark.prefs", "settings"));
        let staged = f.path("stark.presets");
        std::fs::create_dir_all(&staged).expect("something in the record's way");
        assert!(!f.set("stark.presets", "brushes"), "the write cannot land");
        assert_eq!(
            f.get("stark.prefs").as_deref(),
            Some("settings"),
            "and the record beside it is untouched",
        );
    }

    /// Two records whose keys share a stem stage into different files.
    ///
    /// `path.with_extension("tmp")` — the obvious spelling — maps `stark.prefs` and
    /// `stark.presets` onto one `stark.tmp`, and a rename would then land the wrong
    /// bytes under whichever key finished second. Asserted as "each staging name
    /// carries its own key", which is what that spelling loses; two names merely
    /// *differing* is what the per-call counter gives whatever the key.
    #[test]
    fn records_that_share_a_stem_do_not_share_a_staging_file() {
        let dir = Scratch::new("stems");
        let f = files(&dir.0);
        assert!(f.set("stark.prefs", "settings"));
        assert!(f.set("stark.presets", "brushes"));
        assert_eq!(f.get("stark.prefs").as_deref(), Some("settings"));
        assert_eq!(f.get("stark.presets").as_deref(), Some("brushes"));
        for key in ["stark.prefs", "stark.presets"] {
            let staged = scratch(&f.path(key)).expect("a staging name");
            let name = staged.file_name().expect("a file name").to_string_lossy();
            assert!(
                name.starts_with(&format!("{key}.")),
                "{name} does not say which record it is staging",
            );
        }
    }

    /// A blob record's directory is made on the way in, staging file and all — the
    /// first import writes into a directory that does not exist yet.
    #[test]
    fn a_blobs_directory_is_made_before_it_is_staged_into() {
        let dir = Scratch::new("blobs");
        let f = files(&dir.0);
        assert!(f.write("stark.shapes/00ff", b"png"));
        assert_eq!(
            std::fs::read(f.path("stark.shapes/00ff")).ok(),
            Some(b"png".to_vec())
        );
    }

    /// Every key the registry can produce stays inside the directory it was sent to.
    ///
    /// Not a check against a caller's hostile input — there is no such caller, since
    /// a key is a closed enum's string or a content id's hex — but against the
    /// registry *growing* one: a row whose key held a `..` would write outside the
    /// store, and the enum is edited far from here.
    #[test]
    fn no_key_escapes_its_directory() {
        let f = files(std::path::Path::new("/tmp/x"));
        for &store in stark_ui::storage::Store::VARIANTS {
            let (key, _) = store.named();
            let path = f.path(key);
            assert!(
                contained(&f.config, &path),
                "{key} left the store's directory",
            );
        }
    }
}
