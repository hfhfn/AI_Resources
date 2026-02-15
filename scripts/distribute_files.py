#!/usr/bin/env python3
"""
Dual-Storage Distribution Script v4.1
- Scans project for all files.
- Uploads large files (>50MB) to HuggingFace.
- Implement retry logic for network stability.
- Generates a full manifest (including small files) to avoid GitHub API rate limits.
- Sync Deletion: Cleans up redundant files on HF.
- v4.1: Smart timestamps, automatic .gitignore deletion, 404 handling.
"""
import os
import sys
import json
import logging
import time
import subprocess
from urllib.parse import quote
from pathlib import Path
from datetime import datetime
from functools import wraps

# --- Configuration ---
SIZE_THRESHOLD = 50 * 1024 * 1024  # 50MB
HF_REPO_ID = "hfhfn/AI_Resources"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Exclude directories
EXCLUDE_DIRS = {'.git', '.idea', '.vscode', 'venv', 'node_modules', '__pycache__', '.serena', '.github'}

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'distribute.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def retry(exceptions, tries=3, delay=2, backoff=2):
    """Retry decorator with exponential backoff."""
    def decorator(f):
      @wraps(f)
      def wrapper(*args, **kwargs):
        mtries, mdelay = tries, delay
        while mtries > 1:
          try:
            return f(*args, **kwargs)
          except exceptions as e:
            logger.warning(f"{str(e)}, Retrying in {mdelay} seconds...")
            time.sleep(mdelay)
            mtries -= 1
            mdelay *= backoff
        return f(*args, **kwargs)
      return wrapper
    return decorator

def get_file_info(path):
    stats = path.stat()
    return {
        "size": stats.st_size,
        "mtime": stats.st_mtime
    }

def run_git_cmd(args):
    try:
        subprocess.run(['git'] + args, cwd=PROJECT_ROOT, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        logger.debug(f"Git command failed: {args} - {e}")


def escape_gitignore_path(path):
    """Escape special gitignore glob characters in a file path."""
    result = path.replace('\\', '\\\\')
    for ch in ('*', '?', '[', ']'):
        result = result.replace(ch, '\\' + ch)
    # Escape leading # or !
    if result.startswith('#'):
        result = '\\#' + result[1:]
    elif result.startswith('!'):
        result = '\\!' + result[1:]
    # Escape trailing spaces
    if result.endswith(' '):
        stripped = result.rstrip(' ')
        trailing_count = len(result) - len(stripped)
        result = stripped + ('\\ ' * trailing_count)
    return result


def unescape_gitignore_rule(rule):
    """Remove backslash escaping from a gitignore rule to recover the literal path."""
    result = []
    i = 0
    while i < len(rule):
        if rule[i] == '\\' and i + 1 < len(rule) and rule[i + 1] in ('\\', '*', '?', '[', ']', '#', '!', ' '):
            result.append(rule[i + 1])
            i += 2
        else:
            result.append(rule[i])
            i += 1
    return ''.join(result)


def is_pattern_rule(rule):
    """Returns True if the rule contains unescaped wildcards (* or ?)."""
    i = 0
    while i < len(rule):
        if rule[i] == '\\' and i + 1 < len(rule):
            i += 2  # Skip escaped character
        elif rule[i] in ('*', '?'):
            return True
        else:
            i += 1
    return False


def match_path_against_rule(path, rule):
    """Check if a path matches a gitignore rule (literal or pattern).

    For literal rules: unescape and compare with ==.
    For pattern rules: split on /, compare segment by segment, * matches any single segment.
    """
    if not is_pattern_rule(rule):
        return path == unescape_gitignore_rule(rule)
    # Pattern rule: split on / and compare segment by segment
    path_parts = path.split('/')
    rule_parts = rule.split('/')
    if len(path_parts) != len(rule_parts):
        return False
    for pp, rp in zip(path_parts, rule_parts):
        if rp == '*':
            continue  # Wildcard matches any single segment
        if pp != unescape_gitignore_rule(rp):
            return False
    return True


def match_path_against_rules(path, rules):
    """Returns True if the path matches any rule in the list."""
    return any(match_path_against_rule(path, rule) for rule in rules)


def optimize_gitignore_rules(file_paths):
    """Optimize a list of file paths into gitignore rules with wildcards where possible.

    Groups paths that differ in exactly one directory segment (3+ paths required)
    into * wildcard patterns. Remaining paths are individually escaped.
    """
    from collections import defaultdict

    if not file_paths:
        return []

    rules = []
    used = set()

    # Group by segment count
    by_seg_count = defaultdict(list)
    for p in file_paths:
        segs = p.split('/')
        by_seg_count[len(segs)].append((p, segs))

    for seg_len, items in by_seg_count.items():
        if len(items) < 3:
            continue
        # Try each position as the variable one
        for var_pos in range(seg_len):
            groups = defaultdict(list)
            for p, segs in items:
                if p in used:
                    continue
                key = tuple(segs[:var_pos] + segs[var_pos + 1:])
                groups[key].append(p)

            for key, group in groups.items():
                if len(group) >= 3:
                    # Create wildcard pattern
                    template = list(group[0].split('/'))
                    template[var_pos] = '*'
                    rules.append('/'.join(template))
                    used.update(group)

    # Remaining paths: escape individually
    for p in sorted(file_paths):
        if p not in used:
            rules.append(escape_gitignore_path(p))

    return rules


def _delete_file_from_hf(rel_path):
    """Delete a file from HuggingFace, handling 404 gracefully."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        logger.info(f"    Deleting from HuggingFace: {rel_path}")
        api.delete_file(path_in_repo=rel_path, repo_id=HF_REPO_ID, repo_type="dataset",
                       commit_message=f"Auto delete: {os.path.basename(rel_path)}")
        logger.info(f"    [OK] Deleted from HuggingFace: {rel_path}")
    except Exception as e:
        error_str = str(e)
        if "404" in error_str or "not exist" in error_str.lower():
            logger.info(f"    [OK] File already deleted from HuggingFace: {rel_path}")
        else:
            logger.warning(f"    [WARNING] Could not delete {rel_path} from HF: {e}")

def scan_files():
    large_files = []
    small_files = []
    logger.info(f"Scanning files (Threshold: {SIZE_THRESHOLD/1024/1024:.0f}MB)...")

    for path in PROJECT_ROOT.rglob('*'):
        if not path.is_file(): continue
        parts = path.relative_to(PROJECT_ROOT).parts

        # Strict filtering logic
        if any(p.startswith('.') for p in parts): continue # Hidden files/folders
        if any(ex in parts for ex in EXCLUDE_DIRS): continue # Exclude directories
        if 'scripts' in parts: continue # Explicitly exclude scripts folder
        if path.parent == PROJECT_ROOT and path.name in ['index.html', 'README.md', 'setup.bat', 'setup.sh', 'distribute.log', '.gitignore', '.gitattributes', '.nojekyll']: continue

        try:
            info = get_file_info(path)
            if info["size"] >= SIZE_THRESHOLD:
                large_files.append(path)
            else:
                small_files.append(path)
        except OSError: pass

    return large_files, small_files

@retry(Exception, tries=3, delay=5)
def upload_file_to_hf(api, file_path, rel_path):
    logger.info(f"   > Uploading: {rel_path} ({get_file_info(file_path)['size']/1024/1024:.1f} MB)")
    api.upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=rel_path,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=f"Upload large file: {os.path.basename(rel_path)}"
    )

def upload_to_hf(files):
    if not files: return
    logger.info(f"Uploading {len(files)} files to HuggingFace ({HF_REPO_ID})...")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        logger.info(f"Logged in as: {user['name']}")

        for file_path in files:
            rel_path = file_path.relative_to(PROJECT_ROOT).as_posix()
            upload_file_to_hf(api, file_path, rel_path)

        logger.info("[OK] HF Upload complete")
        return True
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Upload flow failed: {str(e)}")
        return False

def read_gitignore_managed_rules():
    """Read the auto-managed HF rules from .gitignore (may include patterns and escaped paths)."""
    gitignore_path = PROJECT_ROOT / '.gitignore'
    managed = []
    if gitignore_path.exists():
        header = "# [Auto] Large files managed by HuggingFace\n"
        in_auto = False
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line == header:
                    in_auto = True
                    continue
                if in_auto:
                    if line.strip() == "":
                        in_auto = False
                        continue
                    managed.append(line.rstrip())
    return managed

def sync_hf_deletions(local_large_files):
    logger.info(f"Checking for redundant files on HuggingFace ({HF_REPO_ID})...")
    deleted_files = []
    try:
        from huggingface_hub import HfApi, list_repo_files
        api = HfApi()
        remote_files = list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset")
        local_rel_paths = {f.relative_to(PROJECT_ROOT).as_posix() for f in local_large_files}

        # Protect files that are listed in .gitignore auto section
        # (they may not exist locally in CI/fresh-clone, but are still managed)
        managed_rules = read_gitignore_managed_rules()

        to_delete = [rf for rf in remote_files
                     if rf not in local_rel_paths
                     and not match_path_against_rules(rf, managed_rules)
                     and not rf.endswith(('.gitattributes', 'README.md', '.gitignore'))]

        if to_delete:
            logger.info(f"Found {len(to_delete)} redundant files. Deleting...")
            for rf in to_delete:
                logger.info(f"   - Deleting: {rf}")
                api.delete_file(path_in_repo=rf, repo_id=HF_REPO_ID, repo_type="dataset", commit_message=f"Sync delete: {os.path.basename(rf)}")
                deleted_files.append(Path(rf))
            logger.info(f"[OK] Sync deletion complete (Removed {len(to_delete)} files)")
        else:
            logger.info("No redundant files found.")
    except Exception as e:
        logger.warning(f"Sync deletion failed: {str(e)}")

    return deleted_files

def update_gitignore_and_git(large_files, hf_files_to_delete):
    logger.info("Processing Git tracking & .gitignore...")
    gitignore_path = PROJECT_ROOT / '.gitignore'

    # Read existing content
    lines = []
    existing_auto_rules = []  # Raw rule strings (may be patterns or escaped paths)

    if gitignore_path.exists():
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

    # Extract existing auto-generated rules
    header = "# [Auto] Large files managed by HuggingFace\n"
    in_auto_section = False

    for line in lines:
        if line == header:
            in_auto_section = True
            continue
        if in_auto_section:
            if line.strip() == "":
                in_auto_section = False
                continue
            existing_auto_rules.append(line.rstrip())

    # Remove old auto-generated section
    new_content = []
    skip = False

    for line in lines:
        if line == header:
            skip = True
            continue
        if skip:
            if line.strip() == "":
                skip = False
                continue
            else:
                continue

        new_content.append(line)

    # Remove trailing empty lines
    while new_content and new_content[-1].strip() == "":
        new_content.pop()

    # Git rm --cached for large files
    for f in large_files:
        run_git_cmd(['rm', '--cached', str(f)])

    # Collect current large file relative paths
    current_large_paths = set()
    for f in large_files:
        rel = f.relative_to(PROJECT_ROOT).as_posix()
        current_large_paths.add(rel)

    # Determine rules to remove
    rules_to_remove = set()
    skip_deletion = (not large_files) and (len(existing_auto_rules) > 0)

    if skip_deletion:
        logger.info("    No local large files found but gitignore has rules — skipping deletion (CI/fresh-clone detected)")

    for rule in existing_auto_rules:
        if is_pattern_rule(rule):
            # Pattern rule: keep if any current large file matches
            if not any(match_path_against_rule(p, rule) for p in current_large_paths):
                if skip_deletion:
                    logger.info(f"    Keeping rule (protected): {rule}")
                    continue
                rules_to_remove.add(rule)
                logger.info(f"    Removing pattern rule (no matching files): {rule}")
        else:
            # Literal (escaped) rule: unescape and check file existence
            literal_path = unescape_gitignore_rule(rule)
            file_path = PROJECT_ROOT / literal_path
            if not file_path.exists():
                if skip_deletion:
                    logger.info(f"    Keeping rule (protected): {rule}")
                    continue
                rules_to_remove.add(rule)
                logger.info(f"    Removing rule for deleted local file: {literal_path}")
                _delete_file_from_hf(literal_path)

    # Generate new optimized rules from current large files
    new_rules = optimize_gitignore_rules(list(current_large_paths))

    # Merge: keep existing rules that weren't removed, add new rules
    kept_rules = [r for r in existing_auto_rules if r not in rules_to_remove]
    all_rules_set = set(kept_rules)
    for r in new_rules:
        all_rules_set.add(r)
    all_rules = sorted(all_rules_set)

    # Write updated gitignore
    with open(gitignore_path, 'w', encoding='utf-8') as f:
        # Write existing content
        f.writelines(new_content)

        # Add separator and auto section if there are any rules
        if all_rules:
            # Ensure blank line before auto section
            if new_content and new_content[-1].strip() != "":
                f.write("\n")
            f.write("\n" + header)
            for rule in sorted(all_rules):
                f.write(f"{rule}\n")

    removed_count = len(rules_to_remove)
    logger.info(f"Updated .gitignore with {len(all_rules)} rules (new: {len(new_rules)}, removed: {removed_count})")

def generate_manifest(large_files, small_files):
    logger.info("Generating full manifest (data/file_manifest.json)...")
    manifest = {
        "hf_repo_id": HF_REPO_ID,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "files": []
    }

    # Helper to add file to manifest
    def add_to_manifest(file_list, is_hf):
        entries = []
        for f in file_list:
            rel = f.relative_to(PROJECT_ROOT).as_posix()
            # URL encode the path for robustness with Chinese characters
            quoted_rel = quote(rel)
            info = get_file_info(f)

            if is_hf:
                url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{quoted_rel}?download=true"
            else:
                url = f"https://raw.githubusercontent.com/hfhfn/AI_Resources/main/{quoted_rel}"

            entries.append({
                "name": f.name,
                "path": rel,
                "extension": f.suffix.lower().lstrip('.'),
                "size_mb": round(info["size"] / (1024 * 1024), 2),
                "url": url,
                "is_hf": is_hf,
                "last_modified": datetime.fromtimestamp(info["mtime"]).strftime("%Y-%m-%d %H:%M:%S")
            })

        # Sort by path for consistent ordering
        return sorted(entries, key=lambda x: x["path"])

    # Add HF files first, then small files, both sorted
    manifest["files"].extend(add_to_manifest(large_files, True))
    manifest["files"].extend(add_to_manifest(small_files, False))

    manifest_dir = PROJECT_ROOT / 'data'
    manifest_dir.mkdir(exist_ok=True)
    manifest_path = manifest_dir / 'file_manifest.json'

    # Read managed rules from .gitignore (may include patterns and escaped paths)
    managed_rules = read_gitignore_managed_rules()

    # Check if manifest exists and try to preserve HuggingFace files that are still managed
    old_hf_files = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                old_manifest = json.load(f)

            # Collect old HuggingFace files that are STILL MANAGED in .gitignore
            for file_entry in old_manifest.get('files', []):
                if file_entry.get('is_hf'):
                    file_path = file_entry['path']
                    # Only preserve if the file matches a current .gitignore rule
                    if match_path_against_rules(file_path, managed_rules):
                        old_hf_files[file_path] = file_entry
                    else:
                        logger.info(f"    Removing unmanaged HF file from manifest: {file_path}")
        except Exception as e:
            logger.debug(f"Could not read old manifest: {e}")

    # Get paths of newly scanned HuggingFace files
    new_hf_paths = {f.relative_to(PROJECT_ROOT).as_posix() for f in large_files}

    # Add back HuggingFace files that were in the old manifest and are still managed
    for old_path, old_entry in old_hf_files.items():
        if old_path not in new_hf_paths:
            logger.info(f"    Preserving managed HF file: {old_path}")
            manifest["files"].insert(0, old_entry)

    # Handle managed files that exist in .gitignore but not in local or old manifest
    # Skip pattern rules (can't restore individual entries from patterns)
    for rule in managed_rules:
        if is_pattern_rule(rule):
            continue
        managed_file = unescape_gitignore_rule(rule)
        if managed_file not in new_hf_paths and managed_file not in old_hf_files:
            # This file is managed but not in manifest - try to get info from HuggingFace
            logger.info(f"    Restoring managed HF file entry: {managed_file}")
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                repo_info = api.repo_info(repo_id=HF_REPO_ID, repo_type="dataset")
                # Create entry with placeholder info
                quoted_rel = quote(managed_file)
                manifest["files"].insert(0, {
                    "name": os.path.basename(managed_file),
                    "path": managed_file,
                    "extension": os.path.splitext(managed_file)[1].lower().lstrip('.'),
                    "size_mb": 0,
                    "url": f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{quoted_rel}?download=true",
                    "is_hf": True,
                    "last_modified": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            except Exception as e:
                logger.debug(f"Could not restore HF file {managed_file}: {e}")

    # Check if manifest content would be the same (except updated_at)
    # BUT: if .gitignore rules changed (especially decreased), always update timestamp
    preserve_timestamp = False
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                old_manifest = json.load(f)

            # Get old .gitignore rules
            old_gitignore_path = PROJECT_ROOT / '.gitignore'
            old_managed_hf_files = set()

            # We can't actually get the old .gitignore easily, but we can check
            # if the number of managed files changed significantly
            old_hf_count = len([f for f in old_manifest.get('files', []) if f.get('is_hf')])
            new_hf_count = len([f for f in manifest['files'] if f.get('is_hf')])

            # If HF file count decreased, it means user deleted rules - always update timestamp
            if new_hf_count < old_hf_count:
                logger.info(f"    (HF files decreased: {old_hf_count} → {new_hf_count}, updating timestamp)")
                preserve_timestamp = False
            else:
                # Only preserve timestamp if everything matches exactly
                old_files = old_manifest.get('files', [])
                new_files = manifest['files']

                if len(old_files) == len(new_files):
                    # Compare file entries - must be in same order and identical
                    all_same = True
                    for old_file, new_file in zip(old_files, new_files):
                        if old_file != new_file:
                            all_same = False
                            break

                    if all_same:
                        # Preserve the old timestamp if content hasn't changed
                        preserve_timestamp = True
                        manifest['updated_at'] = old_manifest['updated_at']
                        logger.info(f"    (Timestamp preserved - no content changes)")
        except Exception as e:
            logger.debug(f"Could not compare manifests: {e}")

    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(f"[OK] Manifest generated with {len(manifest['files'])} total files")

def main():
    start_time = time.time()
    try:
        large, small = scan_files()
        logger.info(f"Stats: {len(large)} large files, {len(small)} small files")

        upload_to_hf(large)
        deleted_files = sync_hf_deletions(large)
        update_gitignore_and_git(large, deleted_files)
        generate_manifest(large, small)

        elapsed = time.time() - start_time
        logger.info(f"\n[OK] All steps complete in {elapsed:.1f}s! Ready for git push.")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
