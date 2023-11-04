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

# Standard
import io
import os
import sys
import threading
from http import HTTPStatus
from typing import Union, IO, Optional, List

# Third Party
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.client import Config as BotoClientConfig
from botocore.exceptions import ClientError, ConnectionError
from grpc import StatusCode

# First party
import alog

# Local
from caikit.core.exceptions.caikit_core_exception import CaikitCoreException, CaikitCoreStatusCode

log = alog.use_channel('S3-CLIENT')


class S3Client:
    """S3Client connects to an S3 storage instance and can perform the following operations.

    - Verify that a connection to S3 can be established. (performed on initialization)
    - Download an artifact from S3
    - Upload an artifact to S3
    - Get the size of an artifact in S3
    - Get the hash of an artifact in S3
    - List objects in a bucket
    - Check if a file exists
    """

    def __init__(self, s3_endpoint:str, s3_access_key:str,
                 s3_secret_key:str, timeout:Optional[float]=60, retry_count:Optional[int]=5, block_size:Optional[int]=25600, thread_count:Optional[int]=10):
        """Initialize an S3 Client object and ensure connection is active

        Args:
            s3_endpoint: str
                The endpoint for the s3 object store
            s3_access_key: str
                The access key for the server
            s3_secret_key: str
                The secret key for the server
            timeout: Optional[float]
                The default timeout for both read and connection
            retry_count: Optional[int]
                The number of times to retry a connection
            block_size: Optional[int]
                The chunk size to download multi-part files with
            thread_count: Optional[int]
                The number of threads that will be used for downloading and uploading

        """
        self.s3 = boto3.resource('s3',
                                 config=BotoClientConfig(read_timeout=timeout,
                                                         connect_timeout=timeout,
                                                         retries={
                                                             'max_attempts': retry_count}),
                                 endpoint_url=s3_endpoint,
                                 aws_access_key_id=s3_access_key,
                                 aws_secret_access_key=s3_secret_key)
        self.client = self.s3.meta.client

        # save a boto3 transfer config for multipart uploads
        self.multipart_transfer_config = TransferConfig(multipart_threshold=block_size,
                                                        max_concurrency=thread_count,
                                                        multipart_chunksize=block_size)

        # Check connectivity by checking to see if we can list buckets.
        try:
            self.client.list_buckets()
            log.info("<COR24924905I>", "S3Client initialization complete for endpoint %s" % s3_endpoint)
        except ClientError as e:
            message = "Unable to access S3 [Check your credentials]! Reason: %s" % (repr(e))
            log.error("<COR24924905I>", message)
            raise CaikitCoreException(CaikitCoreStatusCode.UNAUTHORIZED, message) from e
        except ConnectionError as e:
            message = repr(e)
            log.error("<COR52009487E>", "Unable to connect to S3 endpoint to fetch buckets" \
                         + " because: %s", message)
            raise CaikitCoreException(CaikitCoreStatusCode.CONNECTION_ERROR, message) from e
        except Exception as e:
            message = repr(e)
            log.error("<COR52492827E>", "Encountered unexpected error when initializing the" \
                         + " S3 client: %s", message)
            raise CaikitCoreException(CaikitCoreStatusCode.UNKNOWN, message) from e

    def download_object(self, s3_bucket_name: str, obj_key: str, file_like_obj: IO[bytes]):
        """Download an artifact into an existing file_like object

        Args:
            s3_bucket_name: str
                The name of the S3 bucket that the artifact is in
            obj_key: str
                The key of the artifact to download. This is the {key} part of s3://{bucket}/{key}
            file_like_obj: IO[bytes]
                The file object to write the s3 file into
                            """
        bucket = self._get_boto_bucket(s3_bucket_name)
        log.debug("<COR51313422>", "Downloading {0}/{1} to memory".format(bucket, obj_key))
        try:
            bucket.download_fileobj(Key=obj_key,
                                    Fileobj=file_like_obj,
                                    Callback=self.__ProgressPercentage(s3_client=self.client,
                                                                       bucket_name=bucket.name, source=obj_key,
                                                                       destination=repr(file_like_obj), action="Downloading"))
        except ClientError as e:
            if 'ResponseMetadata' in e.response.keys() and e.response['ResponseMetadata']['HTTPStatusCode'] == \
                    HTTPStatus.NOT_FOUND:
                message = "Object '%s'' not found in Bucket %s!" % (obj_key, s3_bucket_name)
                log.warning("<COR50426407E>", message)
                raise CaikitCoreException(CaikitCoreStatusCode.NOT_FOUND, message) from e
            else:
                message = "S3 Download failed unexpectedly! Reason: %s" % (repr(e))
                log.error("<COR74229107E>", message)
                raise CaikitCoreException(CaikitCoreStatusCode.UNKNOWN, message) from e
        except ConnectionError as e:
            message = str(e)
            log.error("<COR84713311E>", message)
            raise CaikitCoreException(CaikitCoreStatusCode.CONNECTION_ERROR, message) from e


    def upload_object(self, artifact: Union[str, io.BytesIO], s3_bucket_name: str, obj_key: str):
        """Upload an object (zip file) to S3

        Args:
            artifact: Union[str, io.BytesIO]
                Either a path on disk of the artifact to upload, or the artifact itself
            s3_bucket_name: str
                The name of the S3 bucket to upload the artifact to
            obj_key: str
                The key to upload the artifact as. This is the {key} part of s3://{bucket}/{key}
        """
        bucket = self._get_boto_bucket(s3_bucket_name)

        try:
            # Check if artifact is an io.IOBase
            if hasattr(artifact, "read"):
                artifact.seek(0)
                bucket.upload_fileobj(artifact,
                                      obj_key,
                                      Config=self.multipart_transfer_config,
                                      Callback=self.__ProgressPercentage(s3_client=self.client,
                                                                         bucket_name=bucket.name,
                                                                         source=artifact, destination=obj_key,
                                                                         source_size=sys.getsizeof(artifact)))
            else:
                # It's probably a filename
                bucket.upload_file(artifact,
                                   obj_key,
                                   Config=self.multipart_transfer_config,
                                   Callback=self.__ProgressPercentage(s3_client=self.client,
                                                                      bucket_name=bucket.name,
                                                                      source=artifact, destination=obj_key))
        except ClientError as e:
            message = "Error while uploading artifact archive to S3 bucket: %s" % (str(e.response))
            log.error("<COR86753092E>", message)
            raise CaikitCoreException(CaikitCoreStatusCode.UNKNOWN, message) from e
        except ConnectionError as e:
            message = "Unable to connect to S3! Could not upload archive: %s. Reason: %s" % (artifact, repr(e))
            log.error("<COR86753093E>", message)
            raise CaikitCoreException(CaikitCoreStatusCode.CONNECTION_ERROR, message) from e


    def get_object_data(self, s3_bucket_name: str, obj_key: str) -> bytes:
        """Download any artifact from S3 into the client's memory.

        Args:
            s3_bucket_name: str
                The name of the S3 bucket that the artifact is in
            obj_key: str
                The key of the artifact to download. This is the {key} part of s3://{bucket}/{key}
        Returns:
            Downloaded model: io.BytesIO
                A file-like object with the downloaded bytes
         """

        byte_buffer = io.BytesIO()
        self.download_object(s3_bucket_name, obj_key, byte_buffer)
        return byte_buffer.getvalue()

    def get_object_size(self, s3_bucket_name:str, obj_key:str)->Optional[int]:
        """Given a key in a bucket, return an estimated size of the artifact without downloading it, if it exists.
            This should be an inexpensive call.
        Args:
            s3_bucket_name: str
                The name of the S3 bucket that the artifact is in
            obj_key: str
                The key of the artifact to get the size of. This is the {key} part of s3://{bucket}/{key}
        Returns:
            Artifact size: Optional[int]
             The estimated size of the artifact in bytes or None if the object doesn't exit
        """
        bucket = self._get_boto_bucket(s3_bucket_name)
        try:
            return bucket.Object(obj_key).content_length
        except ClientError:
            message = "Unable to check COS size as '%s' not found in bucket %s" % (obj_key, s3_bucket_name)
            log.info("<COR13811412E>", message)
            return None
        except ConnectionError as e:
            message = "Unable to connect to S3! Could not estimate size of object: %s. Reason: %s" % (
                obj_key, repr(e))
            log.error("<COR88881999E>", message)
            raise CaikitCoreException(CaikitCoreStatusCode.CONNECTION_ERROR, message) from e

    def get_object_hash(self, s3_bucket_name:str, obj_key:str)->Optional[str]:
        """Given a path in s3 return the objects e-tag or hash.

        Args:
            s3_bucket_name: str
                The name of the S3 bucket that the artifact is in
            obj_key:  str
                The key of the artifact to check the existence of. This is the {key} part of s3://{bucket}/{key}

        Returns
            object_etag: str
                The object's etag aka it's unique hash
        """
        bucket = self._get_boto_bucket(s3_bucket_name)

        try:
            return bucket.Object(obj_key).e_tag.replace('"','')
        except ClientError:
            message = "Unable to get s3 etag for '%s' not found in bucket %s" % (obj_key, s3_bucket_name)
            log.info("<COR13811412E>", message)
            return None
        except ConnectionError as e:
            message = "Unable to connect to S3! Could not estimate size of object: %s. Reason: %s" % (
                obj_key, repr(e))
            log.error("<COR88881999E>", message)
            raise CaikitCoreException(CaikitCoreStatusCode.CONNECTION_ERROR, message) from e


    def exists(self, s3_bucket_name:str, obj_key:str)->bool:
        """Given a path in s3, check if a file already exists there or not

        Args:
            s3_bucket_name: str
                The name of the S3 bucket that the artifact is in
            obj_key:  str
                The key of the artifact to check the existence of. This is the {key} part of s3://{bucket}/{key}

        Returns:
            file_exists: bool
             True iff the object exists in the bucket
        """
        bucket = self._get_boto_bucket(s3_bucket_name)
        objs = list(bucket.objects.filter(Prefix=obj_key))
        return len(objs) > 0 and objs[0].key == obj_key

    def list_objects_by_prefix(self, bucket_name: str, prefix: Optional[str] = "")->List[str]:
        """
        Args:
            bucket_name: str
                name of bucket to list objects from
            prefix: Optional[str]
                S3 prefix to list objects under

        Returns:
            list_of_objects: List[str]
                Keys of all objects found in the bucket that match the prefix
        """
        objs_found = []
        is_truncated = True
        next_key_marker = ""

        try:
            while is_truncated:
                objs_fetched = self.client.list_objects(Bucket=bucket_name, Marker=next_key_marker, Prefix=prefix)

                # If we got nothing back, return whatever we have
                if objs_fetched.get("Contents") is None:
                    return objs_found

                objs_found.extend(objs_fetched.get("Contents"))
                is_truncated = objs_fetched.get("IsTruncated")
                if is_truncated:
                    next_key_marker = objs_fetched.get("NextMarker")

        except ClientError as e:
            message = "Error while listing artifacts in S3 bucket: %s" % (str(e.response))
            log.error("<COR86743092E>", message)
            raise CaikitCoreException(CaikitCoreStatusCode.UNKNOWN, message) from e
        except ConnectionError as e:
            message = "Unable to connect to S3! Could not list items in bucket %s. Reason: %s" % (bucket_name, repr(e))
            log.error("<COR86743093E>", message)
            raise CaikitCoreException(CaikitCoreStatusCode.CONNECTION_ERROR, message) from e

        file_objs = list(filter(lambda x: x.get('Size') > 0, objs_found))
        return [x.get('Key') for x in file_objs]

    def _get_boto_bucket(self, bucket_name: str)->"Bucket":
        """Get the S3 bucket object from the S3 instance (if the bucket exists and we have
        permission to access it).

        Args:
            bucket_name: str
                String name of the bucket

        Returns:
            boto3_bucket: boto3.resources.factory.s3.Bucket
                A Boto3 Bucket object connected to S3
        """
        # Check to make sure that the bucket exists and that we have access to it
        try:
            self.client.head_bucket(Bucket=bucket_name)
        except ClientError as e:
            if 'ResponseMetadata' in e.response.keys() and e.response['ResponseMetadata'][
                'HTTPStatusCode'] == HTTPStatus.NOT_FOUND:
                message = "Unable to access bucket: %s" % (bucket_name)
                log.error("<COR82613912E>", message)
                raise CaikitCoreException(CaikitCoreStatusCode.INVALID_ARGUMENT, message) from e
            else:
                message = "Unable to get bucket head! Reason: %s" % (repr(e))
                log.error("<COR90465038E>", message)
                raise CaikitCoreException(CaikitCoreStatusCode.UNKNOWN, message) from e
        except ConnectionError as e:
            message = "Unable to connect to S3, so unable to get bucket %s. Reason: %s" % (bucket_name, repr(e))
            log.error("<COR82782912E>", message)
            raise CaikitCoreException(CaikitCoreStatusCode.CONNECTION_ERROR, message) from e
        return self.s3.Bucket(bucket_name)


    # Private class for upload percentage reporting
    class __ProgressPercentage:
        def __init__(self, s3_client, bucket_name, source, destination, action="Uploading", source_size=0):
            self._last_printed = -100
            self._source = source
            self._destination = destination
            if action == "Uploading":
                if isinstance(source, str) and os.path.exists(source):
                    self._size = float(os.path.getsize(source))
                else:
                    self._size = source_size
            else:
                self._size = s3_client.head_object(Bucket=bucket_name, Key=source).get("ContentLength")
            self._seen_so_far = 0
            self._lock = threading.Lock()
            self._action = action

        def __call__(self, bytes_amount):
            with self._lock:
                self._seen_so_far += bytes_amount
                percentage = (self._seen_so_far / self._size) * 100
                if (percentage > 2 + self._last_printed) or percentage == 100:
                    message = "{}: {} to {}  {} / {}  ({:04.2f}%%)".format(
                        self._action, self._source, self._destination, self._seen_so_far,
                        self._size, percentage)
                    log.debug("<COR12543243I>", message)
                    self._last_printed = percentage
