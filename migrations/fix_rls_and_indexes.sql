-- ====================================================================
-- 【Supabase 完全整合化スクリプト】
-- 1. keiba_real_bets の RLS ポリシー作成 (外部アクセス拒絶の解除)
-- 2. 実態カラム構成に100%合致した最適化インデックスの作成
-- ====================================================================

-- --------------------------------------------------------------------
-- ① keiba_real_bets の RLS ポリシー設定 (他テーブルと同等の全許可)
-- --------------------------------------------------------------------
DROP POLICY IF EXISTS "Allow all operations on keiba_real_bets" ON keiba_real_bets;
CREATE POLICY "Allow all operations on keiba_real_bets" 
ON keiba_real_bets 
FOR ALL 
USING (true) 
WITH CHECK (true);

-- --------------------------------------------------------------------
-- ② 実際のカラム構成に基づく最適化インデックスの作成
-- --------------------------------------------------------------------

-- 1. keiba_scheduled_races (実態: race_date, is_settled, is_predicted, post_datetime)
CREATE INDEX IF NOT EXISTS idx_sched_races_date_status 
ON keiba_scheduled_races (race_date, is_settled, is_predicted);

CREATE INDEX IF NOT EXISTS idx_sched_races_post_datetime 
ON keiba_scheduled_races (post_datetime);

-- 2. keiba_predictions_cache (実態: race_date)
CREATE INDEX IF NOT EXISTS idx_pred_cache_race_date 
ON keiba_predictions_cache (race_date);

-- 3. keiba_race_results_cache (実態: race_date)
CREATE INDEX IF NOT EXISTS idx_results_cache_race_date 
ON keiba_race_results_cache (race_date);

-- 4. keiba_race_settlements (実態: race_date, is_pass, is_hit)
CREATE INDEX IF NOT EXISTS idx_settlements_date_pass_hit 
ON keiba_race_settlements (race_date, is_pass, is_hit);

-- 5. keiba_real_bets (実態: race_id, is_manual_bet, bet_time)
CREATE INDEX IF NOT EXISTS idx_real_bets_race_id 
ON keiba_real_bets (race_id);

CREATE INDEX IF NOT EXISTS idx_real_bets_manual_filter 
ON keiba_real_bets (is_manual_bet, bet_time);

-- 完了確認メッセージ
SELECT 'RLSポリシーおよび全インデックスの整合化が正常に完了しました！' AS result;
