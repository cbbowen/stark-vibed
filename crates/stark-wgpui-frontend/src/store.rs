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
//! around.
//!
//! A key is a filename. Every key is `stark.`-prefixed and a blob's is
//! `stark.shapes/<hex>`, so the `/` becomes a directory and the layout on disk is the
//! namespacing the keys already had.

use std::path::PathBuf;

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
    fn write(&self, key: &str, bytes: &[u8]) -> bool {
        let path = self.path(key);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(path, bytes).is_ok()
    }
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

    /// Every key the registry can produce stays inside the directory it was sent to.
    ///
    /// Not a check against a caller's hostile input — there is no such caller, since
    /// a key is a closed enum's string or a content id's hex — but against the
    /// registry *growing* one: a row whose key held a `..` would write outside the
    /// store, and the enum is edited far from here.
    #[test]
    fn no_key_escapes_its_directory() {
        let f = files(std::path::Path::new("/tmp/x"));
        for store in stark_ui::storage::Store::ALL {
            let (key, _) = store.named();
            let path = f.path(key);
            assert!(
                contained(&f.config, &path),
                "{key} left the store's directory",
            );
        }
    }
}
