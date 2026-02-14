//! Smart change detection with git integration and context analysis

use crate::errors::{AutomationError, Result};
use crate::{Change, ChangeAnalysis, ChangeType, ContextAnalysis};
use chrono::{DateTime, Utc};
use git2::{Repository, DiffOptions, DiffFindOptions};
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info, warn};

/// Advanced change tracker with git integration
pub struct ChangeTracker {
    repo: Option<Repository>,
    change_history: Vec<Change>,
    context_cache: HashMap<String, ContextAnalysis>,
}

impl ChangeTracker {
    pub fn new() -> Self {
        // Try to open git repository
        let repo = Repository::discover(".").ok();
        
        if repo.is_some() {
            info!("Git repository detected - change tracking enabled");
        } else {
            warn!("No git repository found - falling back to file system tracking");
        }
        
        Self {
            repo,
            change_history: Vec::new(),
            context_cache: HashMap::new(),
        }
    }
    
    /// Analyze changes in a path with full context
    pub async fn analyze_changes(&self, path: &str) -> Result<ChangeAnalysis> {
        info!("Analyzing changes for: {}", path);
        
        let mut changes = Vec::new();
        let mut total_lines_changed = 0;
        let mut files_affected = Vec::new();
        
        // Check if path exists
        if !Path::new(path).exists() {
            return Err(AutomationError::FileNotFound(path.to_string()));
        }
        
        // Get git diff if repository exists
        if let Some(ref repo) = self.repo {
            match self.get_git_diff(repo, path) {
                Ok(git_changes) => {
                    changes.extend(git_changes);
                }
                Err(e) => {
                    warn!("Git diff failed, using filesystem: {}", e);
                    changes.extend(self.get_filesystem_changes(path)?);
                }
            }
        } else {
            // No git repo, use filesystem
            changes.extend(self.get_filesystem_changes(path)?);
        }
        
        // Calculate statistics
        for change in &changes {
            total_lines_changed += change.diff.lines().count();
            if !files_affected.contains(&change.file_path) {
                files_affected.push(change.file_path.clone());
            }
        }
        
        // Analyze context for each change
        let context_analyses: Vec<ContextAnalysis> = changes
            .iter()
            .map(|c| self.analyze_context(c))
            .collect();
        
        // Detect patterns
        let patterns = self.detect_patterns(&changes);
        
        // Assess impact
        let impact = self.assess_impact(&changes, &context_analyses);
        
        Ok(ChangeAnalysis {
            changes,
            total_lines_changed,
            files_affected,
            context_analyses,
            patterns,
            impact,
            timestamp: Utc::now(),
        })
    }
    
    /// Get changes from git diff
    fn get_git_diff(&self, repo: &Repository, path: &str) -> Result<Vec<Change>> {
        use std::cell::RefCell;
        
        let changes = RefCell::new(Vec::new());
        
        // Get HEAD commit
        let head = repo.head()?;
        let head_commit = head.peel_to_commit()?;
        let head_tree = head_commit.tree()?;
        
        // Get working directory status
        let mut diff_opts = DiffOptions::new();
        diff_opts.pathspec(path);
        
        let diff = repo.diff_tree_to_workdir_with_index(
            Some(&head_tree),
            Some(&mut diff_opts),
        )?;
        
        // First pass: collect file changes
        let mut file_changes: Vec<(String, ChangeType)> = Vec::new();
        diff.foreach(
            &mut |delta, _| {
                let file_path = delta.new_file().path()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                
                let change_type = match delta.status() {
                    git2::Delta::Added => ChangeType::Added,
                    git2::Delta::Deleted => ChangeType::Deleted,
                    git2::Delta::Modified => ChangeType::Modified,
                    git2::Delta::Renamed => ChangeType::Renamed,
                    _ => ChangeType::Modified,
                };
                
                file_changes.push((file_path, change_type));
                true
            },
            None,
            None,
            None,
        )?;
        
        // Second pass: collect diff content
        use std::cell::Cell;
        let current_idx = Cell::new(0);
        let diff_contents = RefCell::new(vec![String::new(); file_changes.len()]);
        
        diff.foreach(
            &mut |_, _| {
                current_idx.set(current_idx.get() + 1);
                true
            },
            None,
            None,
            Some(&mut |_delta, _hunk, line| {
                let idx = current_idx.get();
                if idx > 0 && idx <= diff_contents.borrow().len() {
                    let content = std::str::from_utf8(line.content())
                        .unwrap_or("binary content")
                        .to_string();
                    
                    let prefix = match line.origin() {
                        '+' => "+",
                        '-' => "-",
                        _ => "",
                    };
                    
                    diff_contents.borrow_mut()[idx - 1].push_str(&format!("{}{}", prefix, content));
                }
                true
            }),
        )?;
        
        let diff_contents = diff_contents.into_inner();
        
        // Build final changes
        for ((file_path, change_type), diff) in file_changes.into_iter().zip(diff_contents) {
            changes.borrow_mut().push(Change {
                file_path: file_path.clone(),
                change_type,
                diff,
                old_content: None,
                new_content: None,
                timestamp: Utc::now(),
                author: self.get_last_author(repo, &file_path).unwrap_or_default(),
                commit_message: self.get_last_commit_message(repo, &file_path).unwrap_or_default(),
            });
        }
        
        Ok(changes.into_inner())
    }
    
    /// Get changes from filesystem (fallback)
    fn get_filesystem_changes(&self, path: &str) -> Result<Vec<Change>> {
        let mut changes = Vec::new();
        let path = Path::new(path);
        
        if path.is_file() {
            // Single file
            let metadata = std::fs::metadata(path)?;
            let modified: DateTime<Utc> = metadata.modified()?.into();
            
            changes.push(Change {
                file_path: path.to_string_lossy().to_string(),
                change_type: ChangeType::Modified,
                diff: "File modified (no git history)".to_string(),
                old_content: None,
                new_content: None,
                timestamp: modified,
                author: "unknown".to_string(),
                commit_message: "N/A".to_string(),
            });
        } else if path.is_dir() {
            // Directory - scan for changes
            for entry in walkdir::WalkDir::new(path)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                if entry.file_type().is_file() {
                    let metadata = entry.metadata()?;
                    let modified: DateTime<Utc> = metadata.modified()?.into();
                    
                    changes.push(Change {
                        file_path: entry.path().to_string_lossy().to_string(),
                        change_type: ChangeType::Modified,
                        diff: "File modified (no git history)".to_string(),
                        old_content: None,
                        new_content: None,
                        timestamp: modified,
                        author: "unknown".to_string(),
                        commit_message: "N/A".to_string(),
                    });
                }
            }
        }
        
        Ok(changes)
    }
    
    /// Analyze context around a change
    fn analyze_context(&self, change: &Change) -> ContextAnalysis {
        let mut context = ContextAnalysis::default();
        
        // Read file to get surrounding context
        if let Ok(content) = std::fs::read_to_string(&change.file_path) {
            let lines: Vec<&str> = content.lines().collect();
            
            // Find context around the change
            if let Some(diff_start) = change.diff.lines().next() {
                // Try to find the changed line in the file
                for (idx, line) in lines.iter().enumerate() {
                    if diff_start.contains(line) {
                        // Get 5 lines before and after
                        let start = idx.saturating_sub(5);
                        let end = (idx + 5).min(lines.len());
                        
                        context.surrounding_lines = lines[start..end]
                            .iter()
                            .map(|s| s.to_string())
                            .collect();
                        
                        context.line_number = Some(idx + 1);
                        break;
                    }
                }
            }
            
            // Analyze code structure
            context.code_structure = self.analyze_code_structure(&content);
            
            // Identify dependencies
            context.dependencies = self.identify_dependencies(&content, &change.file_path);
        }
        
        // Determine "why" the change was made (best effort)
        context.why_changed = self.infer_change_reason(change);
        
        context
    }
    
    /// Analyze code structure
    fn analyze_code_structure(&self, content: &str) -> HashMap<String, Vec<String>> {
        let mut structure = HashMap::new();
        
        // Extract functions
        let functions: Vec<String> = content
            .lines()
            .filter(|l| l.trim().starts_with("fn ") || l.trim().starts_with("def ") || l.trim().starts_with("function "))
            .map(|l| l.trim().to_string())
            .collect();
        
        if !functions.is_empty() {
            structure.insert("functions".to_string(), functions);
        }
        
        // Extract classes/structs
        let classes: Vec<String> = content
            .lines()
            .filter(|l| {
                let trimmed = l.trim();
                trimmed.starts_with("class ") || 
                trimmed.starts_with("struct ") || 
                trimmed.starts_with("impl ")
            })
            .map(|l| l.trim().to_string())
            .collect();
        
        if !classes.is_empty() {
            structure.insert("classes_structs".to_string(), classes);
        }
        
        // Extract imports
        let imports: Vec<String> = content
            .lines()
            .filter(|l| {
                let trimmed = l.trim();
                trimmed.starts_with("import ") || 
                trimmed.starts_with("use ") || 
                trimmed.starts_with("from ") ||
                trimmed.starts_with("#include ")
            })
            .map(|l| l.trim().to_string())
            .collect();
        
        if !imports.is_empty() {
            structure.insert("imports".to_string(), imports);
        }
        
        structure
    }
    
    /// Identify dependencies from imports
    fn identify_dependencies(&self, content: &str, file_path: &str) -> Vec<String> {
        let mut deps = Vec::new();
        
        for line in content.lines() {
            let trimmed = line.trim();
            
            // Python/Rust imports
            if trimmed.starts_with("use ") || trimmed.starts_with("import ") {
                deps.push(trimmed.to_string());
            }
            
            // Rust crate dependencies
            if trimmed.starts_with("use crate::") || trimmed.starts_with("use super::") {
                deps.push(trimmed.to_string());
            }
        }
        
        deps
    }
    
    /// Infer why a change was made (best effort)
    fn infer_change_reason(&self, change: &Change) -> Option<String> {
        let mut reasons = Vec::new();
        
        // Check commit message
        if !change.commit_message.is_empty() && change.commit_message != "N/A" {
            reasons.push(format!("Commit message: {}", change.commit_message));
        }
        
        // Check change type
        match change.change_type {
            ChangeType::Added => reasons.push("New functionality added".to_string()),
            ChangeType::Deleted => reasons.push("Code removed (refactoring/cleanup)".to_string()),
            ChangeType::Modified => {
                // Analyze diff to infer reason
                if change.diff.contains("fix") || change.diff.contains("bug") {
                    reasons.push("Bug fix".to_string());
                }
                if change.diff.contains("refactor") || change.diff.contains("clean") {
                    reasons.push("Refactoring".to_string());
                }
                if change.diff.contains("add") || change.diff.contains("new") {
                    reasons.push("Feature addition".to_string());
                }
            }
            _ => {}
        }
        
        if reasons.is_empty() {
            None
        } else {
            Some(reasons.join("; "))
        }
    }
    
    /// Detect patterns across multiple changes
    fn detect_patterns(&self, changes: &[Change]) -> Vec<String> {
        let mut patterns = Vec::new();
        
        // Pattern 1: Multiple files with similar changes
        let mut file_extensions: HashMap<String, usize> = HashMap::new();
        for change in changes {
            if let Some(ext) = Path::new(&change.file_path).extension() {
                let ext = ext.to_string_lossy().to_string();
                *file_extensions.entry(ext).or_insert(0) += 1;
            }
        }
        
        for (ext, count) in file_extensions {
            if count > 3 {
                patterns.push(format!("Bulk changes to .{} files ({} files)", ext, count));
            }
        }
        
        // Pattern 2: Same author multiple changes
        let mut authors: HashMap<String, usize> = HashMap::new();
        for change in changes {
            *authors.entry(change.author.clone()).or_insert(0) += 1;
        }
        
        for (author, count) in authors {
            if count > 5 && author != "unknown" {
                patterns.push(format!("{} made {} changes in this batch", author, count));
            }
        }
        
        // Pattern 3: Time clustering
        if changes.len() > 10 {
            patterns.push("Large batch of changes detected".to_string());
        }
        
        patterns
    }
    
    /// Assess impact of changes
    fn assess_impact(&self, changes: &[Change], contexts: &[ContextAnalysis]) -> crate::ImpactAssessment {
        let mut impact = crate::ImpactAssessment::default();
        
        // Calculate scope
        impact.scope = changes.len() as u32;
        
        // Check for critical files
        let critical_patterns = ["config", "main", "lib", "core", "security", "auth"];
        for change in changes {
            let file_lower = change.file_path.to_lowercase();
            for pattern in &critical_patterns {
                if file_lower.contains(pattern) {
                    impact.critical_files_affected.push(change.file_path.clone());
                    impact.risk_level = impact.risk_level.max(0.7);
                    break;
                }
            }
        }
        
        // Check for breaking changes
        for context in contexts {
            if !context.dependencies.is_empty() {
                impact.potential_breaking_changes = true;
                impact.risk_level = impact.risk_level.max(0.5);
            }
        }
        
        // Calculate complexity
        let total_lines: usize = changes.iter()
            .map(|c| c.diff.lines().count())
            .sum();
        impact.complexity_score = (total_lines as f64 / 100.0).min(1.0);
        
        impact
    }
    
    /// Get last author of a file from git
    fn get_last_author(&self, repo: &Repository, file_path: &str) -> Result<String> {
        let blame = repo.blame_file(Path::new(file_path), None)?;
        
        if let Some(hunk) = blame.iter().last() {
            if let Some(sig) = hunk.final_signature().name() {
                return Ok(sig.to_string());
            }
        }
        
        Ok("unknown".to_string())
    }
    
    /// Get last commit message for a file
    fn get_last_commit_message(&self, repo: &Repository, file_path: &str) -> Result<String> {
        let mut revwalk = repo.revwalk()?;
        revwalk.push_head()?;
        
        for oid in revwalk {
            let oid = oid?;
            let commit = repo.find_commit(oid)?;
            let message = commit.message().unwrap_or("").to_string();
            
            // Check if this commit touched our file
            let tree = commit.tree()?;
            if tree.get_path(Path::new(file_path)).is_ok() {
                return Ok(message.lines().next().unwrap_or("").to_string());
            }
        }
        
        Ok("N/A".to_string())
    }
    
    /// Track a new change
    pub fn track_change(&mut self, change: Change) {
        self.change_history.push(change);
        
        // Keep only last 1000 changes to prevent memory bloat
        if self.change_history.len() > 1000 {
            self.change_history.remove(0);
        }
    }
    
    /// Get change history
    pub fn get_history(&self) -> &[Change] {
        &self.change_history
    }
    
    /// Clear old changes
    pub fn clear_history(&mut self) {
        self.change_history.clear();
        self.context_cache.clear();
    }
}

impl Default for ChangeTracker {
    fn default() -> Self {
        Self::new()
    }
}
