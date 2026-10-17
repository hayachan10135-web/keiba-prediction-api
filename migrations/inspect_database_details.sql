-- ====================================================================
-- 【Supabase 完全詳細調査スクリプト】
-- 1. 全カラム名 ＆ データ型
-- 2. 全インデックス ＆ インデックス対象カラム
-- 3. RLS (Row Level Security: 行単位セキュリティ) の有効状態
-- ====================================================================

-- --------------------------------------------------------------------
-- ① 全テーブルの全カラム構成（名前・型・デフォルト・Nullable）
-- --------------------------------------------------------------------
SELECT 
    table_name AS "テーブル名",
    ordinal_position AS "列順",
    column_name AS "カラム名",
    data_type AS "データ型",
    is_nullable AS "NULL許可",
    column_default AS "デフォルト値"
FROM information_schema.columns
WHERE table_schema = 'public' 
  AND table_name IN (
      'keiba_scheduled_races',
      'keiba_predictions_cache',
      'keiba_race_results_cache',
      'keiba_race_settlements',
      'keiba_real_bets',
      'keiba_app_meta'
  )
ORDER BY table_name, ordinal_position;

-- --------------------------------------------------------------------
-- ② 全テーブルのインデックス一覧（インデックス名・対象カラム・ユニーク制約）
-- --------------------------------------------------------------------
SELECT
    tablename AS "テーブル名",
    indexname AS "インデックス名",
    indexdef AS "インデックス定義 (対象カラム)"
FROM pg_indexes
WHERE schemaname = 'public'
  AND tablename IN (
      'keiba_scheduled_races',
      'keiba_predictions_cache',
      'keiba_race_results_cache',
      'keiba_race_settlements',
      'keiba_real_bets',
      'keiba_app_meta'
  )
ORDER BY tablename, indexname;

-- --------------------------------------------------------------------
-- ③ 全テーブルの RLS (Row Level Security) 状態およびポリシー
-- --------------------------------------------------------------------
SELECT
    c.relname AS "テーブル名",
    CASE WHEN c.relrowsecurity THEN '🔒 有効 (Enabled)' ELSE '🔓 無効 (Disabled)' END AS "RLS状態",
    CASE WHEN c.relforcerowsecurity THEN '強制 (Enforced)' ELSE '通常 (Normal)' END AS "RLS強制設定",
    COALESCE(p.polname, 'ポリシーなし (全アクセス許可または全拒否)') AS "ポリシー名",
    p.polcmd AS "許可コマンド (SELECT/INSERT等)"
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_policy p ON p.polrelid = c.oid
WHERE n.nspname = 'public'
  AND c.relname IN (
      'keiba_scheduled_races',
      'keiba_predictions_cache',
      'keiba_race_results_cache',
      'keiba_race_settlements',
      'keiba_real_bets',
      'keiba_app_meta'
  )
ORDER BY c.relname;
