use reqwest::blocking::Client;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use url::Url;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WixSegmentPattern {
    pub(crate) original_url: Url,
    pub(crate) segment_index: usize,
    prefix: String,
    suffix: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WixResolvedSource {
    pub(crate) source_ref: String,
    pub(crate) pattern: WixSegmentPattern,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WixDownloadPlan {
    pub(crate) source_ref: String,
    pub(crate) video_id: String,
    pub(crate) segments_dir: PathBuf,
    pub(crate) output_mp4: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WixDownloadResult {
    pub(crate) source_ref: String,
    pub(crate) output_mp4: PathBuf,
    pub(crate) segments_dir: PathBuf,
    pub(crate) segment_count: usize,
}

pub(crate) trait WixSourceResolver {
    fn resolve(&self, input: &str) -> Result<WixResolvedSource, String>;
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct WixDirectSegmentResolver;

impl WixSourceResolver for WixDirectSegmentResolver {
    fn resolve(&self, input: &str) -> Result<WixResolvedSource, String> {
        let pattern = WixSegmentPattern::parse(input)?;
        Ok(WixResolvedSource {
            source_ref: input.to_string(),
            pattern,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WixDownloadOptions {
    pub(crate) max_segments: usize,
    pub(crate) stop_after_misses: usize,
}

impl Default for WixDownloadOptions {
    fn default() -> Self {
        Self {
            max_segments: 300,
            stop_after_misses: 3,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct WixDownloader {
    client: Client,
}

impl WixDownloader {
    pub(crate) fn new(client: Client) -> Self {
        Self { client }
    }

    pub(crate) fn download(
        &self,
        resolved: &WixResolvedSource,
        plan: &WixDownloadPlan,
        options: &WixDownloadOptions,
    ) -> Result<WixDownloadResult, String> {
        if options.max_segments == 0 {
            return Err("max_segments must be greater than zero".to_string());
        }
        if options.stop_after_misses == 0 {
            return Err("stop_after_misses must be greater than zero".to_string());
        }

        fs::create_dir_all(&plan.segments_dir).map_err(|err| err.to_string())?;
        let segment_indexes =
            self.download_segments(&resolved.pattern, &plan.segments_dir, options)?;

        if segment_indexes.is_empty() {
            return Err("No downloadable Wix segments were found".to_string());
        }

        let concat_path = plan.segments_dir.join("concat.txt");
        write_concat_file(&concat_path, &plan.segments_dir, &segment_indexes)?;
        remux_segments_to_mp4(&concat_path, &plan.output_mp4)?;

        Ok(WixDownloadResult {
            source_ref: resolved.source_ref.clone(),
            output_mp4: plan.output_mp4.clone(),
            segments_dir: plan.segments_dir.clone(),
            segment_count: segment_indexes.len(),
        })
    }

    fn download_segments(
        &self,
        pattern: &WixSegmentPattern,
        segments_dir: &Path,
        options: &WixDownloadOptions,
    ) -> Result<Vec<usize>, String> {
        let mut indexes = Vec::new();
        let mut misses = 0usize;

        for index in 1..=options.max_segments {
            let destination = segments_dir.join(format!("seg-{index}.ts"));
            if should_reuse_existing_segment(&destination) {
                indexes.push(index);
                misses = 0;
                continue;
            }

            match self.download_one(pattern.segment_url(index), destination) {
                Ok(true) => {
                    indexes.push(index);
                    misses = 0;
                }
                Ok(false) => {
                    misses += 1;
                    if !indexes.is_empty() && misses >= options.stop_after_misses {
                        break;
                    }
                }
                Err(err) => return Err(err),
            }
        }

        Ok(indexes)
    }

    fn download_one(&self, url: Url, destination: PathBuf) -> Result<bool, String> {
        let response = self
            .client
            .get(url.as_str())
            .send()
            .map_err(|err| format!("Request failed for {}: {}", url, err))?;

        if response.status().as_u16() == 404 || response.status().as_u16() == 400 {
            return Ok(false);
        }

        let response = response
            .error_for_status()
            .map_err(|err| format!("Wix segment download failed for {}: {}", url, err))?;
        let bytes = response.bytes().map_err(|err| err.to_string())?;
        let mut file = fs::File::create(destination).map_err(|err| err.to_string())?;
        file.write_all(&bytes).map_err(|err| err.to_string())?;
        Ok(true)
    }
}

fn should_reuse_existing_segment(path: &Path) -> bool {
    path.metadata()
        .map(|metadata| metadata.is_file() && metadata.len() > 0)
        .unwrap_or(false)
}

impl WixSegmentPattern {
    pub(crate) fn parse(input: &str) -> Result<Self, String> {
        let url = Url::parse(input).map_err(|err| format!("Invalid URL: {}", err))?;
        let last_segment = url
            .path_segments()
            .and_then(|mut segments| segments.next_back())
            .ok_or_else(|| "Wix segment URL must include a filename".to_string())?;

        let (prefix, segment_index, suffix) = parse_segment_filename(last_segment)?;
        Ok(Self {
            original_url: url,
            segment_index,
            prefix,
            suffix,
        })
    }

    pub(crate) fn segment_url(&self, index: usize) -> Url {
        let mut url = self.original_url.clone();
        let mut path = url.path().to_string();
        let current_name = format!("{}{}{}", self.prefix, self.segment_index, self.suffix);
        let replacement = format!("{}{}{}", self.prefix, index, self.suffix);
        path = path.replacen(&current_name, &replacement, 1);
        url.set_path(&path);
        url
    }

    pub(crate) fn inferred_video_id(&self) -> Option<String> {
        let segments = self.original_url.path_segments()?;
        let all = segments.collect::<Vec<_>>();
        let video_pos = all.iter().position(|segment| *segment == "video")?;
        let raw = all.get(video_pos + 1)?;
        Some(sanitize_video_id(raw))
    }
}

pub(crate) fn build_download_plan(
    resolved: &WixResolvedSource,
    base_dir: &Path,
    explicit_video_id: Option<&str>,
) -> WixDownloadPlan {
    let video_id = explicit_video_id
        .map(sanitize_video_id)
        .filter(|value| !value.is_empty())
        .or_else(|| resolved.pattern.inferred_video_id())
        .unwrap_or_else(|| "wix-video".to_string());
    let root_dir = base_dir.join(&video_id);

    WixDownloadPlan {
        source_ref: resolved.source_ref.clone(),
        video_id,
        segments_dir: root_dir.join("segments"),
        output_mp4: root_dir.join("source.mp4"),
    }
}

fn parse_segment_filename(filename: &str) -> Result<(String, usize, String), String> {
    let marker = "seg-";
    let marker_index = filename
        .find(marker)
        .ok_or_else(|| "URL does not look like a Wix segment URL".to_string())?;
    let digits_start = marker_index + marker.len();
    let digits = filename[digits_start..]
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();

    if digits.is_empty() {
        return Err("Wix segment URL is missing its numeric segment index".to_string());
    }

    let prefix = filename[..digits_start].to_string();
    let suffix = filename[(digits_start + digits.len())..].to_string();
    let segment_index = digits
        .parse::<usize>()
        .map_err(|err| format!("Invalid segment index: {}", err))?;

    Ok((prefix, segment_index, suffix))
}

fn write_concat_file(
    concat_path: &Path,
    segments_dir: &Path,
    segment_indexes: &[usize],
) -> Result<(), String> {
    let mut file = fs::File::create(concat_path).map_err(|err| err.to_string())?;
    for index in segment_indexes {
        let path = segments_dir
            .join(format!("seg-{index}.ts"))
            .canonicalize()
            .map_err(|err| err.to_string())?;
        writeln!(file, "file '{}'", path.display()).map_err(|err| err.to_string())?;
    }
    Ok(())
}

fn remux_segments_to_mp4(concat_path: &Path, output_mp4: &Path) -> Result<(), String> {
    let status = Command::new("ffmpeg")
        .arg("-y")
        .arg("-f")
        .arg("concat")
        .arg("-safe")
        .arg("0")
        .arg("-i")
        .arg(concat_path)
        .arg("-c")
        .arg("copy")
        .arg(output_mp4)
        .status()
        .map_err(|err| format!("Failed to launch ffmpeg: {}", err))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!("ffmpeg exited with status {}", status))
    }
}

fn sanitize_video_id(input: &str) -> String {
    let sanitized = input
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string();

    if sanitized.is_empty() {
        "wix-video".to_string()
    } else {
        sanitized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wix_segment_pattern_parse_extracts_segment_template() {
        let pattern = WixSegmentPattern::parse(
            "https://repackager.wixmp.com/video.wixstatic.com/video/abc123/2160p/mp4/file.mp4/seg-5-v1-a1.ts?token=demo",
        )
        .unwrap();

        assert_eq!(pattern.segment_index, 5);
        assert_eq!(
            pattern.segment_url(12).as_str(),
            "https://repackager.wixmp.com/video.wixstatic.com/video/abc123/2160p/mp4/file.mp4/seg-12-v1-a1.ts?token=demo"
        );
    }

    #[test]
    fn test_wix_segment_pattern_parse_rejects_non_segment_url() {
        let error = WixSegmentPattern::parse("https://example.com/video/file.mp4").unwrap_err();

        assert_eq!(error, "URL does not look like a Wix segment URL");
    }

    #[test]
    fn test_build_download_plan_prefers_explicit_video_id() {
        let resolved = WixDirectSegmentResolver
            .resolve(
                "https://repackager.wixmp.com/video.wixstatic.com/video/abc123/2160p/mp4/file.mp4/seg-5-v1-a1.ts?token=demo",
            )
            .unwrap();

        let plan = build_download_plan(&resolved, Path::new("testdata"), Some("custom name"));

        assert_eq!(plan.video_id, "custom_name");
        assert_eq!(
            plan.output_mp4,
            PathBuf::from("testdata/custom_name/source.mp4")
        );
        assert_eq!(
            plan.segments_dir,
            PathBuf::from("testdata/custom_name/segments")
        );
    }

    #[test]
    fn test_build_download_plan_falls_back_to_wix_video_id() {
        let resolved = WixDirectSegmentResolver
            .resolve(
                "https://repackager.wixmp.com/video.wixstatic.com/video/8e4fff_02d8693c6cc94172b97e6ca67147cede/2160p/mp4/file.mp4/seg-5-v1-a1.ts?token=demo",
            )
            .unwrap();

        let plan = build_download_plan(&resolved, Path::new("testdata"), None);

        assert_eq!(plan.video_id, "8e4fff_02d8693c6cc94172b97e6ca67147cede");
    }

    #[test]
    fn test_wix_direct_segment_resolver_returns_source_ref() {
        let input = "https://repackager.wixmp.com/video.wixstatic.com/video/abc123/2160p/mp4/file.mp4/seg-5-v1-a1.ts?token=demo";

        let resolved = WixDirectSegmentResolver.resolve(input).unwrap();

        assert_eq!(resolved.source_ref, input);
        assert_eq!(resolved.pattern.segment_index, 5);
    }

    #[test]
    fn test_write_concat_file_writes_absolute_paths() {
        let base = std::env::temp_dir().join(format!(
            "vididx-wix-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let segments_dir = base.join("segments");
        fs::create_dir_all(&segments_dir).unwrap();
        fs::write(segments_dir.join("seg-1.ts"), b"demo").unwrap();
        let concat_path = segments_dir.join("concat.txt");

        write_concat_file(&concat_path, &segments_dir, &[1]).unwrap();

        let content = fs::read_to_string(&concat_path).unwrap();
        let expected = segments_dir.join("seg-1.ts").canonicalize().unwrap();
        assert!(content.contains(&expected.display().to_string()));

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn test_should_reuse_existing_segment_true_for_non_empty_file() {
        let base = std::env::temp_dir().join(format!(
            "vididx-wix-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&base).unwrap();
        let segment = base.join("seg-1.ts");
        fs::write(&segment, b"demo").unwrap();

        assert!(should_reuse_existing_segment(&segment));

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn test_should_reuse_existing_segment_false_for_empty_file() {
        let base = std::env::temp_dir().join(format!(
            "vididx-wix-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&base).unwrap();
        let segment = base.join("seg-1.ts");
        fs::write(&segment, b"").unwrap();

        assert!(!should_reuse_existing_segment(&segment));

        fs::remove_dir_all(base).unwrap();
    }
}
