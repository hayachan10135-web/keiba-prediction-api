-- ====================================================================
-- 【全テーブル一括初期化・修復DDL (Idempotent: 冪等)】
-- すでに存在するテーブルやデータは破壊せず、未作成のテーブル・インデックスのみを安全に自動構築します
-- ====================================================================

-- 1. レーススケジュール管理テーブル
CREATE TABLE IF NOT EXISTS keiba_scheduled_races (
    race_id VARCHAR(30) PRIMARY KEY,
    race_date VARCHAR(20) NOT NULL,
    venue_name VARCHAR(20) NOT NULL,
    race_no INTEGER NOT NULL,
    post_time VARCHAR(20),
    post_datetime TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'scheduled',
    is_settled INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sched_races_date ON keiba_scheduled_races (race_date, venue_name, race_no);

-- 2. 予想データキャッシュテーブル (JSON)
CREATE TABLE IF NOT EXISTS keiba_predictions_cache (
    race_id VARCHAR(30) PRIMARY KEY,
    race_date VARCHAR(20) NOT NULL,
    data_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_pred_cache_date ON keiba_predictions_cache (race_date);

-- 3. レース確定着順・払戻結果キャッシュテーブル (JSON)
CREATE TABLE IF NOT EXISTS keiba_race_results_cache (
    race_id VARCHAR(30) PRIMARY KEY,
    result_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. 成績確定・収支台帳テーブル
CREATE TABLE IF NOT EXISTS keiba_race_settlements (
    race_id VARCHAR(30) PRIMARY KEY,
    race_date VARCHAR(20) NOT NULL,
    race_name VARCHAR(100),
    venue_name VARCHAR(20),
    race_no INTEGER,
    strategy_type VARCHAR(50) DEFAULT 'skip',
    target_umaban INTEGER,
    horse_name VARCHAR(50),
    grade VARCHAR(10),
    is_pass BOOLEAN DEFAULT TRUE,
    total_bet INTEGER DEFAULT 0,
    total_payout INTEGER DEFAULT 0,
    actual_order INTEGER,
    is_hit BOOLEAN DEFAULT FALSE,
    hit_type VARCHAR(30) DEFAULT '見送り',
    tansho_bet INTEGER DEFAULT 0,
    fukusho_bet INTEGER DEFAULT 0,
    umatan_bet INTEGER DEFAULT 0,
    umaren_bet INTEGER DEFAULT 0,
    trio_bet INTEGER DEFAULT 0,
    tansho_payout INTEGER DEFAULT 0,
    fukusho_payout INTEGER DEFAULT 0,
    umatan_payout INTEGER DEFAULT 0,
    umaren_payout INTEGER DEFAULT 0,
    trio_payout INTEGER DEFAULT 0,
    hypothetical_bet INTEGER DEFAULT 0,
    hypothetical_payout INTEGER DEFAULT 0,
    pass_benefit INTEGER DEFAULT 0,
    bet_breakdown JSONB DEFAULT '{}'::jsonb,
    payout_breakdown JSONB DEFAULT '{}'::jsonb,
    bet_details JSONB DEFAULT '{}'::jsonb,
    payout_details JSONB,
    mark_orders JSONB,
    top3_orders JSONB,
    track_bias JSONB,
    odds_alerts JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_settlements_date ON keiba_race_settlements (race_date);
CREATE INDEX IF NOT EXISTS idx_settlements_hit ON keiba_race_settlements (is_hit);

-- 5. 実馬券発注台帳テーブル (自動投票・手動テスト発注)
CREATE TABLE IF NOT EXISTS keiba_real_bets (
    id BIGSERIAL PRIMARY KEY,
    race_id VARCHAR(30) NOT NULL,
    platform VARCHAR(20) NOT NULL,
    receipt_no VARCHAR(50),
    total_amount INTEGER NOT NULL DEFAULT 0,
    items JSONB NOT NULL DEFAULT '[]'::jsonb,
    status VARCHAR(30) NOT NULL DEFAULT 'success',
    is_manual_bet BOOLEAN NOT NULL DEFAULT FALSE,
    bet_time TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_keiba_real_bets_race_id ON keiba_real_bets (race_id);
CREATE INDEX IF NOT EXISTS idx_keiba_real_bets_manual ON keiba_real_bets (is_manual_bet);
CREATE INDEX IF NOT EXISTS idx_keiba_real_bets_bet_time ON keiba_real_bets (bet_time);
CREATE UNIQUE INDEX IF NOT EXISTS idx_real_bets_auto_attempt_unique
ON keiba_real_bets (race_id)
WHERE is_manual_bet = FALSE;

-- 6. システム運用メタ情報テーブル (設定・バージョン)
CREATE TABLE IF NOT EXISTS keiba_app_meta (
    key VARCHAR(50) PRIMARY KEY,
    value JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
