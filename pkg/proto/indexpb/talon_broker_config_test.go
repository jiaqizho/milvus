// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package indexpb_test

import (
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/pkg/v3/proto/clusteringpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexcgopb"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
)

func TestTalonBrokerStorageConfigWireCompatibility(t *testing.T) {
	for _, endpoint := range []string{"", "http://broker:9000/credentials"} {
		input := &indexpb.StorageConfig{UseIAM: true, TalonCredentialEndpoint: endpoint}
		data, err := proto.Marshal(input)
		require.NoError(t, err)
		for _, target := range []interface {
			proto.Message
			GetUseIAM() bool
			GetTalonCredentialEndpoint() string
		}{&indexcgopb.StorageConfig{}, &clusteringpb.StorageConfig{}} {
			require.NoError(t, proto.Unmarshal(data, target))
			require.True(t, target.GetUseIAM())
			require.Equal(t, endpoint, target.GetTalonCredentialEndpoint())
		}
	}
}
