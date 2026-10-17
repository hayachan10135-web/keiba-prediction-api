-- ====================================================================
-- マイグレーション: マニュアル（手動・実機テスト）投票機能対応
-- テーブル: keiba_real_bets に is_manual_bet カラムを追加
-- 目的: 手動/実機テスト発注の履歴を保存しつつ、AI収支集計から完全除外する
-- ====================================================================

-- 1. keiba_real_bets テーブルに is_manual_bet カラムを追加 (デフォルト FALSE)
ALTER TABLE IF EXISTS keiba_real_bets
ADD COLUMN IF NOT EXISTS is_manual_bet BOOLEAN DEFAULT FALSE;

-- 2. 検索・集計クエリ最適化のためのインデックスを作成
CREATE INDEX IF NOT EXISTS idx_keiba_real_bets_manual 
ON keiba_real_bets (is_manual_bet);

CREATE INDEX IF NOT EXISTS idx_keiba_real_bets_race_date 
ON keiba_real_bets (bet_time);

-- 3. 既存レコードがある場合、すべて FALSE に更新
UPDATE keiba_real_bets 
SET is_manual_bet = FALSE 
WHERE is_manual_bet IS NULL;

-- 完了メッセージ
COMMENT ON COLUMN keiba_real_bets.is_manual_bet IS '手動・実機テスト投票フラグ (TRUEの場合、AI公式収支サマリーから完全除外)';

