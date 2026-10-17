-- ====================================================================
-- 【スキーマ健全性診断SQL】
-- 本システムが必要とする全6テーブルの存在状態を1発で診断・表示します
-- ====================================================================

WITH required_tables AS (
    SELECT unnest(ARRAY[
        'keiba_scheduled_races',
        'keiba_predictions_cache',
        'keiba_race_results_cache',
        'keiba_race_settlements',
        'keiba_real_bets',
        'keiba_app_meta'
    ]) AS table_name,
    unnest(ARRAY[
        'レース日程・発走時刻・スケジュール管理',
        'AI予想データ (各ステージ推論JSON)',
        '確定着順・公式払戻金データJSON',
        '成績確定台帳 (収支サマリー・的中計算)',
        '実馬券発注台帳 (自動/マニュアル投票)',
        'システム運用メタ設定 (投票モード・残高等)'
    ]) AS description
)
SELECT 
    rt.table_name AS "テーブル名",
    rt.description AS "役割・用途",
    CASE 
        WHEN t.table_name IS NOT NULL THEN '✅ 存在 (OK)'
        ELSE '❌ 未作成 (MISSING)'
    END AS "ステータス",
    COALESCE(c.column_count, 0) AS "カラム数",
    COALESCE(idx.index_count, 0) AS "インデックス数"
FROM required_tables rt
LEFT JOIN information_schema.tables t 
    ON t.table_schema = 'public' AND t.table_name = rt.table_name
LEFT JOIN (
    SELECT table_name, count(*) AS column_count 
    FROM information_schema.columns 
    WHERE table_schema = 'public' 
    GROUP BY table_name
) c ON c.table_name = rt.table_name
LEFT JOIN (
    SELECT tablename, count(*) AS index_count 
    FROM pg_indexes 
    WHERE schemaname = 'public' 
    GROUP BY tablename
) idx ON idx.tablename = rt.table_name
ORDER BY 
    CASE WHEN t.table_name IS NULL THEN 1 ELSE 2 END,
    rt.table_name;
