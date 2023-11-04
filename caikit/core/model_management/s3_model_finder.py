# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The S3Model locates models stored in a s3 storage that contain a caikit-native
config.yml file.

Configuration for LocalModelFinder lives under the config as follows:

model_management:
    finders:
        <finder name>:
            type: S3
            config:
                # Main endpoint for s3 storage
                endpoint:
                # Access key for S3 storage
                access_key:
                # Secret key for S3 storage
                secret_key:
                # Download directory for s3 objects
                download_path: TemporaryDirectory()
                # Timeout for S3 connections
                timeout: 60
                # Retry count for the S3 connection
                retry_count: 5

"""
# Standard
from typing import Optional
import os
import tempfile
from pathlib import Path
import zipfile
import tarfile

# First Party
import aconfig
import alog

# Local
from ..exceptions.caikit_core_exception import CaikitCoreException, CaikitCoreStatusCode
from ..modules import ModuleConfig
from .model_finder_base import ModelFinderBase
from ..toolkit.s3_client import  S3Client

log = alog.use_channel("S3FIND")


class S3ModelFinder(ModelFinderBase):
    name = "S3"

    def __init__(self, config: aconfig.Config, instance_name: str):
        """Initialize with an optional path prefix"""
        self._instance_name = instance_name
        self._s3_client = S3Client(config.endpoint, config.access_key, config.secret_key, config.get("timeout", 60), config.get("retry_count", 5))
        if config.get("download_path"):
            self._download_path = Path(config.get("download_path"))
        else:
            self._download_directory = tempfile.TemporaryDirectory()
            self._download_path = Path(self._download_directory.name)

    def __del__(self):
        """Cleanup the temporary directory if one was used"""
        if hasattr(self, "_download_directory"):
            self._download_directory.cleanup()

    def find_model(
        self,
        model_path: str,
        prefix: Optional[str]="s3",
        **__,
    ) -> Optional[ModuleConfig]:
        """Find a model at the s3 bucket/key or with a predefined prefix

        Args:
            model_path: str
                The s3 model in either "bucket/key" or "prefix://bucket/key" form
            prefix: Optional[str]=s3
                Optional uri prefix for s3 path
        """

        bucket_name, object_key = model_path.replace(f"{prefix}:/","").split("/",1)

        log.debug("Fetching object %s in bucket %s", object_key, bucket_name)

        # Store object by hash to ensure models aren't downloaded multiple times
        object_hash =self._s3_client.get_object_hash(bucket_name, object_key)
        if not object_hash:
            raise CaikitCoreException(CaikitCoreStatusCode.NOT_FOUND, f"Object not found in bucket {bucket_name} at key {object_key}")

        # Check if object has been downloaded before
        destination_path = self._download_path.joinpath(object_hash)
        destination_path_str = str(destination_path.absolute())
        if destination_path.exists():
            if destination_path.is_dir():
                log.debug("<COR40579825>", "Destination path %s already exists. Loading config", destination_path_str)
                return ModuleConfig.load(destination_path_str)
            else:
                raise CaikitCoreException(CaikitCoreStatusCode.UNKNOWN, f"Destination path {destination_path_str} already exists and is not a directory")

        # Download object to archive
        archive_path = self._download_path.joinpath(f"{object_hash}.compressed")
        archive_path_str = str(archive_path)
        log.debug(f"Downloading model archive to %s",archive_path_str)
        with archive_path.open("wb+") as archive_file:
            self._s3_client.download_object(bucket_name, object_key, archive_file)

        # Extract zip or tar file to destination path
        if zipfile.is_zipfile(archive_path_str):
            with zipfile.ZipFile(archive_path_str, 'r') as archive:
                log.debug("<COR39956212>","Extracting zip archive to %s",destination_path_str)
                archive.extractall(destination_path_str)
        elif tarfile.is_tarfile(archive_path_str):
            with tarfile.TarFile(archive_path_str, "r") as archive:
                log.debug("<COR96230654>","Extracting tar archive to %s",destination_path_str)
                archive.extractall(destination_path_str)
        else:
            raise CaikitCoreException(CaikitCoreStatusCode.UNKNOWN, "Unknown archive type. Downloaded file is not a tar or zip")

        return ModuleConfig.load(destination_path_str)
