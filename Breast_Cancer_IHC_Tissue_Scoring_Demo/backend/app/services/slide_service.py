"""Upload policy for slides."""

from app.core.config import settings
from app.schemas.slide import UploadCapability


class SlideService:
    """Reports what the upload endpoint will accept."""

    def upload_capability(self) -> UploadCapability:
        """Upload is live: step 1 reads whatever is uploaded, for real."""
        return UploadCapability(
            enabled=True,
            reason=(
                "Upload a whole-slide image to run steps 1 and 2 against it - reading the "
                "pyramid, then GrandQC quality control. The remaining steps are documented "
                "but not implemented yet."
            ),
            accepted_formats=sorted(settings.allowed_slide_ext),
            max_file_size_mb=settings.max_upload_bytes // (1024 * 1024),
            chunk_size_bytes=settings.upload_chunk_size,
        )


slide_service = SlideService()
