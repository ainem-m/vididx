use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use vidx_core::{SceneChange, VidxError};

use crate::dhash::dhash_from_path;

const PERIODIC_INTERVAL: f64 = 5.0;
const DHASH_DISTANCE_THRESHOLD: u32 = 10;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedFrame {
    pub timestamp_sec: f64,
    pub image_path: String,
    pub dhash: u64,
}

/// Extract frames at periodic intervals and scene changes, deduplicating by dHash.
pub async fn analyze_frames(
    frame_dir: &str,
    duration_sec: f64,
    scene_changes: &[SceneChange],
) -> Result<Vec<ExtractedFrame>, VidxError> {
    // Generate target timestamps
    let mut timestamps = generate_timestamps(duration_sec, scene_changes);
    timestamps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    timestamps.dedup_by(|a, b| (*a - *b).abs() < 0.1);

    // Load frames and compute hashes
    let mut frames_with_hashes = Vec::new();
    for (idx, &ts) in timestamps.iter().enumerate() {
        let frame_path = format!("{}/frame_{:03}.jpg", frame_dir, idx);
        let frame_path_buf = PathBuf::from(&frame_path);

        if frame_path_buf.exists() {
            match dhash_from_path(&frame_path_buf) {
                Ok(hash) => {
                    frames_with_hashes.push((ts, hash, frame_path));
                }
                Err(_) => continue,
            }
        }
    }

    // Deduplicate by dHash distance
    let mut deduplicated = Vec::new();
    let mut seen_hashes = Vec::new();

    for (ts, hash, path) in frames_with_hashes {
        let is_duplicate = seen_hashes
            .iter()
            .any(|&h| hamming_distance(hash, h) <= DHASH_DISTANCE_THRESHOLD);

        if !is_duplicate {
            seen_hashes.push(hash);
            deduplicated.push(ExtractedFrame {
                timestamp_sec: ts,
                image_path: path,
                dhash: hash,
            });
        }
    }

    Ok(deduplicated)
}

fn generate_timestamps(duration_sec: f64, scene_changes: &[SceneChange]) -> Vec<f64> {
    let mut timestamps = Vec::new();

    // Add periodic frames
    let mut ts = 0.0;
    while ts < duration_sec {
        timestamps.push(ts);
        ts += PERIODIC_INTERVAL;
    }

    // Add scene change points
    for sc in scene_changes {
        timestamps.push(sc.at_sec);
    }

    timestamps
}

fn hamming_distance(h1: u64, h2: u64) -> u32 {
    (h1 ^ h2).count_ones()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_timestamps_periodic() {
        let timestamps = generate_timestamps(15.0, &[]);
        assert!(timestamps.contains(&0.0));
        assert!(timestamps.contains(&5.0));
        assert!(timestamps.contains(&10.0));
    }

    #[test]
    fn test_generate_timestamps_with_scene_changes() {
        let scene_changes = vec![
            SceneChange {
                at_sec: 7.5,
                score: 0.8,
            },
            SceneChange {
                at_sec: 12.5,
                score: 0.9,
            },
        ];

        let timestamps = generate_timestamps(20.0, &scene_changes);
        assert!(timestamps.contains(&7.5));
        assert!(timestamps.contains(&12.5));
    }

    #[test]
    fn test_hamming_distance_same() {
        let dist = hamming_distance(0b1010101010101010, 0b1010101010101010);
        assert_eq!(dist, 0);
    }

    #[test]
    fn test_hamming_distance_different() {
        let dist = hamming_distance(0b1111111111111111, 0b0000000000000000);
        assert_eq!(dist, 16);
    }

    #[test]
    fn test_hamming_distance_partial() {
        let dist = hamming_distance(0b1111000000000000, 0b0000111111111111);
        assert_eq!(dist, 16);
    }
}
