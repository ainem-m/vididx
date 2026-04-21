use image::{DynamicImage, ImageReader};
use std::io::Cursor;
use std::path::Path;

use vididx_core::VididxError;

const DHASH_SIZE: u32 = 8;

/// Compute dHash (difference hash) of an image.
pub fn dhash_from_path(path: &Path) -> Result<u64, VididxError> {
    let img = ImageReader::open(path)
        .map_err(|e| VididxError::Vision(format!("Failed to open image: {}", e)))?
        .decode()
        .map_err(|e| VididxError::Vision(format!("Failed to decode image: {}", e)))?;

    Ok(dhash_from_image(&img))
}

/// Compute dHash from image bytes.
pub fn dhash_from_bytes(data: &[u8]) -> Result<u64, VididxError> {
    let reader = ImageReader::new(Cursor::new(data));
    let img = reader
        .decode()
        .map_err(|e| VididxError::Vision(format!("Failed to decode image: {}", e)))?;

    Ok(dhash_from_image(&img))
}

/// Compute dHash from DynamicImage.
/// dHash is a 64-bit perceptual hash based on pixel differences.
pub fn dhash_from_image(img: &DynamicImage) -> u64 {
    // Resize to (DHASH_SIZE+1) x DHASH_SIZE for computing differences
    let resized = img.resize_exact(
        DHASH_SIZE + 1,
        DHASH_SIZE,
        image::imageops::FilterType::Lanczos3,
    );

    let gray = resized.to_luma8();
    let mut hash: u64 = 0;

    // Compute hash by comparing adjacent pixels
    for y in 0..DHASH_SIZE {
        for x in 0..DHASH_SIZE {
            let left = gray.get_pixel(x, y)[0] as u32;
            let right = gray.get_pixel(x + 1, y)[0] as u32;

            // Set bit if left > right
            if left > right {
                hash |= 1 << (y * DHASH_SIZE + x);
            }
        }
    }

    hash
}

/// Calculate Hamming distance between two dHashes.
pub fn hamming_distance(hash1: u64, hash2: u64) -> usize {
    (hash1 ^ hash2).count_ones() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, GrayImage, Luma};

    #[test]
    fn test_hamming_distance_same() {
        let hash = 0b1010101010101010u64;
        assert_eq!(hamming_distance(hash, hash), 0);
    }

    #[test]
    fn test_hamming_distance_opposite() {
        let hash1 = 0b1111111111111111u64;
        let hash2 = 0b0000000000000000u64;
        assert_eq!(hamming_distance(hash1, hash2), 16);
    }

    #[test]
    fn test_hamming_distance_partial() {
        let hash1 = 0b1100110011001100u64;
        let hash2 = 0b1100110000110011u64;
        // XOR: 0b0000000011111111 = 8 bits set
        assert_eq!(hamming_distance(hash1, hash2), 8);
    }

    #[test]
    fn test_dhash_from_image_handles_tall_image() {
        let image = GrayImage::from_fn(2, 32, |x, y| {
            Luma([((x * 16 + (y % 16)) as u8).saturating_mul(4)])
        });

        let _ = dhash_from_image(&DynamicImage::ImageLuma8(image));
    }
}
