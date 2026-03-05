"""Task image path resolution.

Resolves Label Studio image URLs to local file paths using local upload lookup,
URL normalization, and fallback API download resolver.
"""

import logging
import os
from urllib.parse import urljoin, urlparse, unquote


class PathResolver:
    """Infrastructure adapter for resolving image paths for inference."""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def _resolve_upload_local_file(self, image_url: str, task_id=None):
        """Resolve `/data/upload/...` URL to an existing local media file path."""
        if not isinstance(image_url, str) or not image_url:
            return None

        parsed = urlparse(image_url)
        url_path = parsed.path if parsed.scheme else image_url
        upload_prefix = '/data/upload/'
        if not url_path.startswith(upload_prefix):
            return None

        decoded_upload_path = unquote(url_path[len(upload_prefix):])
        relative_upload_path = os.path.normpath(decoded_upload_path.replace('/', os.sep)).lstrip('\\/')

        candidate_base_dirs = []
        env_base_dir = os.getenv('LABEL_STUDIO_BASE_DATA_DIR')
        if env_base_dir:
            candidate_base_dirs.append(env_base_dir)

        local_appdata = os.getenv('LOCALAPPDATA')
        if local_appdata:
            candidate_base_dirs.append(os.path.join(local_appdata, 'label-studio', 'label-studio'))

        user_profile = os.path.expanduser('~')
        candidate_base_dirs.append(os.path.join(user_profile, '.local', 'share', 'label-studio'))

        for base_dir in candidate_base_dirs:
            candidate_path = os.path.join(base_dir, 'media', 'upload', relative_upload_path)
            if os.path.exists(candidate_path):
                self.logger.info(
                    "Resolved upload file from local filesystem for task_id=%s: %s",
                    task_id,
                    candidate_path
                )
                return candidate_path

        self.logger.debug(
            "Upload file not found in local candidate directories for task_id=%s: %s",
            task_id,
            url_path
        )
        return None

    def resolve(self, image_url: str, task_id, get_local_path):
        """Resolve image URL to local path, with Label Studio API fallback."""
        local_upload_path = self._resolve_upload_local_file(image_url, task_id=task_id)
        if local_upload_path:
            return local_upload_path

        ls_host = (
            os.getenv('LABEL_STUDIO_URL')
            or os.getenv('LABEL_STUDIO_HOST')
            or os.getenv('LABEL_STUDIO_HOSTNAME')
        )
        normalized_url = image_url
        if isinstance(image_url, str) and image_url.startswith('/') and ls_host:
            normalized_url = urljoin(ls_host.rstrip('/') + '/', image_url.lstrip('/'))
            self.logger.info(
                "Normalized relative image url for task_id=%s: %s -> %s",
                task_id,
                image_url,
                normalized_url
            )

        try:
            return get_local_path(normalized_url, task_id=task_id)
        except Exception as exc:
            ls_token = os.getenv('LABEL_STUDIO_API_KEY') or os.getenv('LABEL_STUDIO_ACCESS_TOKEN')
            if not ls_token:
                self.logger.warning(
                    "LABEL_STUDIO_API_KEY/LABEL_STUDIO_ACCESS_TOKEN is not set, API download may fail with 401"
                )
            if not ls_host:
                if isinstance(image_url, str) and image_url.startswith('/'):
                    self.logger.error(
                        "Relative image url received but LABEL_STUDIO_URL/LABEL_STUDIO_HOSTNAME is not set"
                    )
                raise exc

            self.logger.warning(
                "get_local_path failed for task_id=%s, retry with ls_host=%s", task_id, ls_host
            )
            return get_local_path(
                normalized_url,
                task_id=task_id,
                ls_host=ls_host,
                ls_access_token=ls_token
            )
