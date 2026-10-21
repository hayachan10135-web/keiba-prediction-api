-- ====================================================================
-- マイグレーション: 実馬券発注台帳 (keiba_real_bets) テーブル新規作成
-- 目的: 自動投票およびマニュアル手動・実機テスト発注の履歴を安全に永続化する
-- ====================================================================

-- 1. keiba_real_bets テーブルの新規作成
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

-- 2. 検索・集計クエリ最適化のためのインデックス
CREATE INDEX IF NOT EXISTS idx_keiba_real_bets_race_id 
ON keiba_real_bets (race_id);

CREATE INDEX IF NOT EXISTS idx_keiba_real_bets_manual 
ON keiba_real_bets (is_manual_bet);

CREATE INDEX IF NOT EXISTS idx_keiba_real_bets_bet_time 
ON keiba_real_bets (bet_time);

CREATE UNIQUE INDEX IF NOT EXISTS idx_real_bets_auto_attempt_unique
ON keiba_real_bets (race_id)
WHERE is_manual_bet = FALSE;

-- 3. テーブル・カラムコメント
COMMENT ON TABLE keiba_real_bets IS '実馬券発注台帳（自動投票およびマニュアル投票の履歴）';
COMMENT ON COLUMN keiba_real_bets.is_manual_bet IS '手動・実機テスト投票フラグ (TRUEの場合、AI公式収支サマリーから完全除外)';
