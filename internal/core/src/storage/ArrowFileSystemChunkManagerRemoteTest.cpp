// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/EasyAssert.h"
#include "storage/ArrowFileSystemChunkManager.h"
#include "storage/Types.h"
#include "storage/Util.h"

using namespace milvus::storage;

// Real object storage coverage. Explicit configuration must fail on connection
// or startup-probe errors instead of silently skipping a broken FS backend.
TEST(ArrowFileSystemChunkManagerRemote, MinioCRUD) {
    const auto* enabled = std::getenv("MILVUS_TEST_REMOTE_FS");
    if (enabled == nullptr || std::string(enabled) != "1") {
        GTEST_SKIP() << "set MILVUS_TEST_REMOTE_FS=1 to run against MinIO";
    }
    const char* address = std::getenv("MINIO_ADDRESS");
    ASSERT_NE(address, nullptr);
    ASSERT_FALSE(std::string(address).empty());

    StorageConfig config;
    config.storage_type = "remote";
    config.cloud_provider = "aws";
    config.address = address;
    config.bucket_name = "a-bucket";
    config.access_key_id = "minioadmin";
    config.access_key_value = "minioadmin";
    config.useSSL = false;
    config.useIAM = false;
    config.root_path = "files";

    // A missing bucket and rejected credentials must fail at construction,
    // even though an empty probe prefix in a valid bucket is accepted.
    auto invalid = config;
    invalid.bucket_name = "missing-fs-chunk-" + std::to_string(::getpid());
    try {
        CreateChunkManager(invalid);
        FAIL() << "missing bucket must fail the startup probe";
    } catch (const milvus::SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), milvus::ErrorCode::ObjectNotExist);
    }
    invalid = config;
    invalid.access_key_id = "invalid-fs-chunk-key";
    try {
        CreateChunkManager(invalid);
        FAIL() << "invalid credentials must fail the startup probe";
    } catch (const milvus::SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), milvus::ErrorCode::StorageError);
    }

    auto cm_ptr = CreateChunkManager(config);
    ASSERT_NE(cm_ptr, nullptr);
    auto& cm = *cm_ptr;
    EXPECT_EQ(cm.GetName(), "ArrowFileSystemChunkManager");
    EXPECT_EQ(cm.GetBucketName(), "a-bucket");

    // pid-scoped prefix so concurrent/repeat runs don't collide; the Arrow
    // filesystem is rooted at the bucket, so this same key addresses the object
    // for every operation.
    std::string prefix = "arrow_fs_cm_ut/" + std::to_string(::getpid()) + "/";
    std::string path = prefix + "insert_log/1/2/3";
    uint8_t data[5] = {0x17, 0x32, 0x00, 0x34, 0x23};

    auto compatibility_config = config;
    compatibility_config.storage_type = "minio";
    compatibility_config.cloud_provider = "";
    const auto compatibility_cm = CreateChunkManager(compatibility_config);
    EXPECT_EQ(compatibility_cm->GetName(), "ArrowFileSystemChunkManager");

    const auto empty_path = prefix + "empty";
    cm.Write(empty_path, data, 0);
    EXPECT_TRUE(cm.Exist(empty_path));
    EXPECT_EQ(cm.Size(empty_path), 0);
    cm.Remove(empty_path);
    EXPECT_FALSE(cm.Exist(prefix.substr(0, prefix.size() - 1)));

    // clean any leftovers from a previous crashed run under this pid
    for (const auto& p : cm.ListWithPrefix(prefix)) {
        cm.Remove(p);
    }

    // Write, Exist, Size, and Read all use the same filesystem.
    EXPECT_FALSE(cm.Exist(path));
    cm.Write(path, data, sizeof(data));
    EXPECT_TRUE(cm.Exist(path));
    EXPECT_EQ(cm.Size(path), sizeof(data));

    // Read (Arrow data plane) round-trips the exact bytes
    uint8_t readdata[20] = {0};
    EXPECT_EQ(cm.Read(path, readdata, sizeof(data)), sizeof(data));
    EXPECT_EQ(std::memcmp(data, readdata, sizeof(data)), 0);

    // overwrite: multipart finalize replaces the object, Size reflects the new
    // length (not appended)
    uint8_t data2[3] = {0x1, 0x2, 0x3};
    cm.Write(path, data2, sizeof(data2));
    EXPECT_EQ(cm.Size(path), sizeof(data2));

    // Raw-prefix listing returns complete keys. A second
    // object whose key extends the first (".../3" vs ".../30") pins the raw
    // byte-prefix semantics: a prefix ending mid-segment matches both, which a
    // directory-style FileSelector would miss (buqian review #1).
    std::string path2 = prefix + "insert_log/1/2/30";
    cm.Write(path2, data2, sizeof(data2));

    auto listed = cm.ListWithPrefix(prefix);
    std::sort(listed.begin(), listed.end());
    ASSERT_EQ(listed.size(), 2);
    EXPECT_EQ(listed[0], path);
    EXPECT_EQ(listed[1], path2);

    auto listed_mid = cm.ListWithPrefix(path);  // prefix ends mid-segment
    std::sort(listed_mid.begin(), listed_mid.end());
    ASSERT_EQ(listed_mid.size(), 2);
    EXPECT_EQ(listed_mid[0], path);
    EXPECT_EQ(listed_mid[1], path2);

    // not-found classification must survive the S3 error path
    // (ExtendStatusDetail AwsErrorNotFound -> ObjectNotExist)
    try {
        cm.Size(prefix + "not/exist");
        FAIL() << "Size on missing object should throw";
    } catch (const milvus::SegcoreError& e) {
        EXPECT_EQ(e.get_error_code(), milvus::ErrorCode::ObjectNotExist);
    }
    try {
        cm.Read(prefix + "not/exist", readdata, sizeof(readdata));
        FAIL() << "Read on missing object should throw";
    } catch (const milvus::SegcoreError& e) {
        EXPECT_EQ(e.get_error_code(), milvus::ErrorCode::ObjectNotExist);
    }
    // Remove of a missing object is swallowed (idempotent), matching legacy
    EXPECT_NO_THROW(cm.Remove(prefix + "not/exist"));

    // Remove remains idempotent through the FS adapter.
    cm.Remove(path);
    EXPECT_FALSE(cm.Exist(path));
    cm.Remove(path2);
    EXPECT_EQ(cm.ListWithPrefix(prefix).size(), 0);
}
