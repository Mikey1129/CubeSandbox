// Copyright (c) 2026 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"

	"github.com/tencentcloud/CubeSandbox/cube-lifecycle-manager/internal/lifecycle"
	"github.com/tencentcloud/CubeSandbox/cube-lifecycle-manager/internal/proxypush"
	"github.com/tencentcloud/CubeSandbox/cube-lifecycle-manager/internal/redisstream"
	"github.com/tencentcloud/CubeSandbox/cube-lifecycle-manager/internal/registry"
	"github.com/tencentcloud/CubeSandbox/cube-lifecycle-manager/internal/statesync"
)

type standbyStatus struct{}

func (standbyStatus) IsLeader() bool { return false }
func (standbyStatus) Enabled() bool  { return true }

func TestCatchUpStreamToDrainsPromotionHighWater(t *testing.T) {
	server := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: server.Addr()})
	t.Cleanup(func() { _ = rdb.Close() })
	ctx := context.Background()

	addEvent := func(id, sandboxID string) {
		t.Helper()
		payload, err := json.Marshal(lifecycle.SandboxLifecycleMeta{SandboxID: sandboxID})
		if err != nil {
			t.Fatal(err)
		}
		if err := rdb.XAdd(ctx, &redis.XAddArgs{
			Stream: lifecycle.EventStreamKey,
			ID:     id,
			Values: map[string]interface{}{
				lifecycle.FieldOp:        lifecycle.OpCreate,
				lifecycle.FieldSandboxID: sandboxID,
				lifecycle.FieldPayload:   string(payload),
				lifecycle.FieldTimestamp: time.Now().UnixMilli(),
			},
		}).Err(); err != nil {
			t.Fatal(err)
		}
	}
	addEvent("1-0", "already-applied")
	addEvent("2-0", "must-catch-up")

	stream := redisstream.New(rdb, zap.NewNop())
	reg := registry.New()
	reg.Upsert(lifecycle.SandboxLifecycleMeta{SandboxID: "already-applied"})
	progress := newStreamProgress("1-0")
	deps := statesync.Deps{Registry: reg, Leader: standbyStatus{}, Log: zap.NewNop()}
	push := proxypush.New(nil, "", time.Second, zap.NewNop())

	if err := catchUpStreamTo(
		ctx, "2-0", stream, push, reg, deps, progress, zap.NewNop(),
	); err != nil {
		t.Fatal(err)
	}
	if got := progress.Cursor(); got != "2-0" {
		t.Fatalf("cursor = %q, want 2-0", got)
	}
	if reg.Get("must-catch-up") == nil {
		t.Fatal("promotion catch-up did not apply high-water event")
	}
}

func TestStreamProgressRejectsPrefetchedOlderBatch(t *testing.T) {
	progress := newStreamProgress("100-0")
	progress.Advance("200-0") // promotion catch-up
	if progress.ShouldApply("150-0") {
		t.Fatal("prefetched event older than promotion high-water was accepted")
	}
	if !progress.ShouldApply("201-0") {
		t.Fatal("new event after promotion high-water was rejected")
	}
}

func TestResolvePromotionStatePrefersSharedRedis(t *testing.T) {
	server := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: server.Addr()})
	t.Cleanup(func() { _ = rdb.Close() })
	stream := redisstream.New(rdb, zap.NewNop())
	ctx := context.Background()
	entry := registry.Entry{
		Meta:         lifecycle.SandboxLifecycleMeta{SandboxID: "sbx"},
		RuntimeState: lifecycle.StatePaused,
	}

	if err := stream.SetState(ctx, "sbx", lifecycle.StateRunning, time.Minute); err != nil {
		t.Fatal(err)
	}
	state, err := resolvePromotionState(ctx, stream, entry)
	if err != nil || state != lifecycle.StateRunning {
		t.Fatalf("resolvePromotionState() = (%q, %v), want running", state, err)
	}

	if err := stream.ClearState(ctx, "sbx"); err != nil {
		t.Fatal(err)
	}
	state, err = resolvePromotionState(ctx, stream, entry)
	if err != nil || state != lifecycle.StatePaused {
		t.Fatalf("fallback resolvePromotionState() = (%q, %v), want paused", state, err)
	}
}
