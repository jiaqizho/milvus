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

//go:build test

package packed

import (
	"context"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func TestAsyncManifestWorkersGrowOnDemand(t *testing.T) {
	for _, limit := range []int{3, 1024} {
		t.Run(strconv.Itoa(limit), func(t *testing.T) {
			io := NewManifestIOContext(limit)
			defer io.Close()
			workerCount := func() int {
				io.taskMu.Lock()
				defer io.taskMu.Unlock()
				return io.workerCount
			}
			idle := func() bool {
				io.taskMu.Lock()
				defer io.taskMu.Unlock()
				return io.pendingTasks == 0
			}
			// Sequential demand reuses one worker, even with a large upper bound.
			for range 2 {
				release := make(chan struct{})
				var unblock sync.Once
				defer unblock.Do(func() { close(release) })
				entered, err := testQueueManifestBlock(io, release)
				require.NoError(t, err)
				select {
				case <-entered:
				case <-time.After(5 * time.Second):
					t.Fatal("worker did not start")
				}
				require.Equal(t, 1, workerCount())
				unblock.Do(func() { close(release) })
				require.Eventually(t, idle, time.Second, time.Millisecond)
			}
			// Three blocked native tasks must all progress without exceeding demand.
			release := make(chan struct{})
			var unblock sync.Once
			defer unblock.Do(func() { close(release) })
			for i := 1; i <= 3; i++ {
				entered, err := testQueueManifestBlock(io, release)
				require.NoError(t, err)
				select {
				case <-entered:
				case <-time.After(5 * time.Second):
					t.Fatal("workers did not grow with demand")
				}
				require.Equal(t, i, workerCount())
			}
			if limit == 3 {
				entered, err := testQueueManifestBlock(io, release)
				require.NoError(t, err)
				require.Equal(t, 3, workerCount())
				select {
				case <-entered:
					t.Fatal("worker count exceeded the limit")
				default:
				}
			}
			closed := make(chan struct{})
			go func() { io.Close(); close(closed) }()
			select {
			case <-closed:
				t.Fatal("Close returned before running tasks finished")
			case <-time.After(20 * time.Millisecond):
			}
			unblock.Do(func() { close(release) })
			select {
			case <-closed:
			case <-time.After(5 * time.Second):
				t.Fatal("Close did not join workers")
			}
			require.True(t, idle())
		})
	}
}

func TestAsyncManifestAdmissionCancellation(t *testing.T) {
	io := NewManifestIOContext(1)
	defer io.Close()
	io.slots <- struct{}{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	require.ErrorIs(t, io.acquire(ctx), context.Canceled)
	<-io.slots
}

func TestAsyncManifestCloseWakesAdmissionWaiters(t *testing.T) {
	io := NewManifestIOContext(1)
	require.NoError(t, io.acquire(context.Background()))
	var release sync.Once
	defer io.Close()
	defer release.Do(io.release)
	waiting := make(chan error, 1)
	go func() { waiting <- io.acquire(context.Background()) }()
	closed := make(chan struct{})
	go func() { io.Close(); close(closed) }()
	select {
	case err := <-waiting:
		require.ErrorIs(t, err, merr.ErrServiceUnavailable)
	case <-time.After(time.Second):
		t.Fatal("Close did not wake queued admission")
	}
	release.Do(io.release)
	select {
	case <-closed:
	case <-time.After(time.Second):
		t.Fatal("Close did not finish after callers drained")
	}
}
