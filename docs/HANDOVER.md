# 競馬AI自動投票システム 引き継ぎ書（HANDOVER.md）

本書は、競馬AI予想システムおよび自動投票システム（即PAT / UMACAスマート対応）の現在の実装状況、データベース定義、今後のタスク、および安全運用上の重要注意点をまとめた引き継ぎドキュメントです。
本システムの仕様および詳細設計の最上位の正（SSOT）は、**`REQUIREMENTS_SPECIFICATION.md`**（要件定義書）および **`SYSTEM_ARCHITECTURE.md`**（詳細設計書）です。

---

## 1. プロジェクト全体概要

本システムは、競馬AI予想モデル（TTVR-C: モデルA / モデルB）が算出する期待値・高回収率シグナル（リアルタイムトラックバイアス連動）に基づき、JRAの投票プラットフォーム（**即PAT** または **UMACAスマート**）へ安全に完全自動投票を行う統合システムです。

### 構成アーキテクチャ（完全クラウド自律型）
```
[ 競馬AI予想モデル (TTVR-C) ]
          │
          ▼ 期待値・推奨買い目算出 & トラックバイアス連動判定
[ backend-render (FastAPI) ] ── (Render クラウド常時24時間自律稼働)
     ├─ バックグラウンドスケジューラ (60m馬体重 -> 30m -> 15m -> 5m最終確定)
     ├─ 物理分割アーキテクチャ (app/core, app/strategies, app/betting, app/settlement, app/ml, app/scrapers)
     ├─ 超軽量 即PAT クラウドクライアント (ipat_client.py: pure-requests直接通信)
     ├─ マニュアル実機テスト投票サービス (manual_service.py: FR-BET-07 ★AI収支集計から完全除外)
     ├─ 残高不足監視 & FCMプッシュ通知 (🎫投票完了 / ⚠️エラー / ⚠️残高不足警告)
     └─ Supabase (keiba_app_meta, keiba_predictions_cache, keiba_race_settlements, keiba_real_bets, keiba_scheduled_races)
          │
          ├───────────────────────────────┐
          ▼ (0.3秒 超高速直接発注)          ▼ (FCM プッシュ通知 & 設定同期)
[ JRA 即PAT (g.ipat.jra.go.jp) ]   [ Android アプリ (Kotlin / Compose) ]
※事前手動入金残高内でのみ動作       ├─ ⚙️ 設定・自動投票画面 (SettingsScreen)
                                  ├─ 📱 マニュアル実機テスト投票カード (ManualBetCard)
                                  ├─ 残り投票予定額 & 差引余力リアルタイム表示
                                  └─ レース一覧・詳細・日別サマリー閲覧
```

---

## 2. 現在の実装状況（Status: 2026年10月17日 22:45 現在）

### ① バックエンドの物理モジュール分割状況（⚠️ 段階移行中・main.py をエントリーポイントとして稼働中）
Render本番環境は依然として **`main.py`（約8,400行）をエントリーポイント（`uvicorn main:app`）として稼働**しています。
一部機能のモジュール化（`app/` ディレクトリ）は完了しており、`main.py` からインポートして使用されていますが、**完全な物理分割（main.py のスリム化・ルーター分割）は未完了**です：
- **`app/` に切り出され `main.py` からインポート利用されているモジュール**:
  - `app/core/timezone.py`: JSTタイムゾーン安全パース・ISO変換（`now_jst`, `today_jst_str`, `parse_post_datetime` 等）
  - `app/strategies/arbiter.py`: 戦略調停エンジン（`StrategyArbiter`）
  - `app/betting/factory.py`: 投票クライアント生成ファクトリ（`BettingClientFactory`）
  - `app/betting/manual_service.py`: マニュアル実機テスト投票サービス（`execute_manual_bet`）
- **`app/` 内にコードは存在するが、本番の `main.py` 内に同等コードが残存し直接実行されている機能**:
  - スクレイピング（`netkeiba` からのレース情報・結果取得）
  - AI推論エンジン（LightGBM モデルロード、特徴量計算、オッズ取得、印付け）
  - バックグラウンド定期メンテナンスタスク（発走前自動予想、直前オッズ更新、結果自動確定）
  - 成績確定・サマリー計算（`verify_race_result`, `get_daily_summary`, `get_aggregate_summary` 等）
  - APIエンドポイント群（全ルーティング定義）
- **現状の理由**:
  週末のレース開催時の本番運用（リアルタイム自動投票・AI予想）を最優先で安定稼働させるため、8,000行の一括解体に伴う循環インポートやデプロイ破損リスクを回避し、新設機能（マニュアルテスト投票、投票ファクトリ、タイムゾーン正規化等）から安全に段階的移行を進めている状態です。

### ② 新要件 `FR-BET-07` マニュアル実機テスト投票機能（✅ 実装・実機疎通確認完了）
- **APIエンドポイント**: `POST /betting/manual`
  - ユーザーが指定した競馬場・レース番号・式別・馬番・金額を直接即PAT/UMACAへ発注。
  - **★【厳格な集計除外規定（FR-SET-05）】**:
    発注された馬券レコードは `is_manual_bet=true` で `keiba_real_bets` に永続化され、**AIモデルの実戦成績集計（日次サマリー・累計サマリー・回収率計算）から物理的・恒久的に完全除外**されます。
- **即PAT画面遷移セッションバグの根本解決**:
  - トップメニューから選択入力画面（`kw_150.cgi`）へ遷移する際、従来の全input走査により別ダミーフォームのパラメータ（`g=028`/`029`）で上書きされていた不具合を改修。
  - `emenu` フォームから正規のセッション値（`g=021` 等）を抽出し、`kw_150` ➔ `kw_160` ➔ `kw_170` ➔ `kw_180` の各ステップで hidden input を完全引き継ぎ。
  - **実機検証**: 新潟1R 複勝5番 100円の実機テスト投票が即PATへ正常に執行され、受付完了および残高更新（3,900円 ➔ 3,800円）を確認完了。

### ③ Android アプリ（`android-app`）への機能実装 & 最新APK配置（✅ 完了）
- **設定画面（`SettingsScreen.kt`）に `ManualBetCard` を追加**:
  - 開催場選択（全10場: 東京・中山・京都・阪神・福島・新潟・中京・小倉・札幌・函館）
  - レース番号選択（1〜12R）
  - 式別選択（単勝・複勝・馬連・馬単・ワイド・3連複・3連単）
  - 馬番指定（券種ごとの必要頭数バリデーション付き）
  - 金額入力（100円プリセットボタン付き）
  - 模擬発注（Dry-run）スイッチ（デフォルトONで安全保護）
  - 誤発注防止の最終確認モーダルダイアログ & 二重タップ防止ロック
  - 受付番号・発注金額・口座残高のインライン即時表示
- **最新APKビルド & 自動エクスポート**:
  - `./gradlew assembleDebug` でエラーなくビルド完了。
  - **配置先**: `G:\マイドライブ\keiba_data\keiba_ai.apk`（約21.1MB、2026/10/17 22:26更新）

### ④ Supabase データベースの完全整合化（✅ 完了）
本番Supabaseの6大テーブル定義とRLSポリシー・インデックスを、実態カラム定義と完全整合化：
1. **`keiba_app_meta`**: システム運用メタ設定（投票モード、残高等）
2. **`keiba_predictions_cache`**: AI予想データJSON（`race_id`, `race_date`, `data_json`）
3. **`keiba_race_results_cache`**: 確定着順・払戻金JSON（`race_id`, `race_name`, `race_date`, `result_json`）
4. **`keiba_race_settlements`**: 成績確定台帳（33カラム、全券種別収支、的中フラグ）
5. **`keiba_real_bets`**: 実馬券台帳（`id`, `race_id`, `platform`, `receipt_no`, `total_amount`, `items`, `status`, `is_manual_bet`, `bet_time`）
   - **RLS解放**: `Allow all operations on keiba_real_bets` 設定完了。
6. **`keiba_scheduled_races`**: レース日程（`race_id`, `race_date`, `race_name`, `post_datetime`, `is_predicted`, `is_settled`, `retry_count`）
   - 実態カラムに基づき、`race_date`, `post_datetime`, `is_settled` インデックスを正常構築完了。

### ⑤ 投票完了判定の厳格化 & 偽陽性サプレッションの完全撲滅（✅ 2026/10/17 23:00 完了）
マニュアル実機テスト投票および自動投票において、「正常に完了しました」と表示されるが実際には投票されていなかった問題を根本解決：
1. **確定フォーム（`kw_180.cgi`）送信パラメータの厳密抽出**:
   - `kw_170` のHTMLから `soup.find("form", {"action": "kw_180.cgi"})` を厳密に取得し、そのフォーム内の hidden input（買い目コード・トークン等）のみを抽出し、`s=合計金額`, `tou="投票"` と共に POST。過去画面（`kw_150`, `kw_160`）の余計なパラメータ混入によるセッション切断エラー（「再度ログインしてください」）を完全防止。
2. **投票結果画面（`kw_180.cgi` レスポンス）の厳格判定（Fail-Fast）**:
   - 「【投票結果】」または「ご投票ありがとうございました」が存在し、かつ `受付番号\s*[:：]?\s*(\d{4,})`（4桁数字）が抽出できた場合のみ `status: success` と判定。
   - 受付番号が取れなかった場合やエラー画面の場合に、ダミー文字列（"受付完了"）でフォールバックして success を返すコードを完全に排除。不成立時は必ず画面のエラー文言を抽出して `status: failed` を返却。
3. **実機IPAT・照会メニューでの確認完了**:
   - 実際にJRA即PATサーバーで受付番号 `0007`, `0008`, `0009` が正規発行され、即PAT照会メニュー（`kw_080.cgi` / `kw_100.cgi`）にて投票が完全に成立していることを実証確認済。
   - **アプリ側確認完了**: Androidアプリのマニュアル実機テスト投票カード（`ManualBetCard`）からも即PAT投票が正常に執行・照会されることを確認済。

### ⑥ 実機・実資金テスト執行時の事前確認義務（✅ 運用ルール化完了）
- **AIエージェントによる独断での実機発注の絶対禁止**:
  AIエージェントが開発、デバッグ、テスト、保守作業中に、スクリプト実行やAPI呼び出しを通じて **実際の資金が引き落とされる実機発注（即PAT / UMACAスマートの購入確定・実投票）を執行する際は、必ず事前にユーザーへ明示的な確認（場・R・式別・馬番・金額・理由）を行い、ユーザーの承諾・許可を得てから実行すること**。
- **検証作業の標準**:
  疎通・UI動作・パラメータ検証は `dry_run=True`（模擬発注）を最優先とし、実資金テストを行う前には必ずユーザーへ都度確認を行う。

### ⑦ 予想印と確定結果着順印の二重同期・セルフヒーリング保証（✅ 2026/10/17 23:25 完了）
予想カード（上段）と確定結果カード（下段）において、印（◎、◯等）や本命馬が食い違う問題を根本的に解消：
1. **問題の根本原因**:
   - レース確定直後に `verify_race_result` が実行された後、直前オッズ変動等で `predictions_cache` 側の印（◎昇格等）が更新された際、DB（`keiba_race_settlements`）の `top3_orders` や `target_umaban` が古い予想時点のまま固定化され、更新されない不具合が存在した。
   - Androidアプリ側でも `verificationsCache` に旧結果が存在する場合、プルリフレッシュ時も再フェッチを行わず、カード描画時も画面上部の `horses` を見ずに結果レコード単体の古い印を表示していた。
2. **バックエンドの根本改修 & 自己修復（セルフヒーリング）**:
   - `get_cached_settlement` / `get_daily_summary` において、`pred_cache` の本命馬番（`target_umaban`）と DB の `target_umaban` に不一致が検知された場合、自動で `process_settlement_calculation` を再実行して DB を最新予想基準で完全修復。
   - `extract_top3_orders` および新設 `sync_orders_marks` により、確定着順（`top3_orders`）や印馬着順（`mark_orders`）の印を常に最新の予想データ（`pred_cache`）の印で強制同期して返却。
   - `save_settlement` で `top3_orders` カラムへの永続化を保証。
   - `/admin/heal-settlement-marks` エンドポイントを新設し、DB内の全レコードを一括走査・修復可能に。
3. **Androidアプリ側の二重防御**:
   - `MainScreen.kt`（`RaceSettlementResultCard`）において、画面上に表示されている `horses` の印を最優先で参照（`horses?.firstOrNull { it.umaban == t.umaban }?.mark ?: t.mark`）し、同一画面内の上下の印を100%完全一致化。
   - `MainViewModel.kt` において、発走後のプルリフレッシュ（`refreshPrediction`）時に確定キャッシュをクリアして再取得を保証。レース選択時も確定済みレースであればサイレントに最新確定データをフェッチしてキャッシュを更新。

---

## 3. ディレクトリ構造

```
c:/my-keiba-app-project/
├── REQUIREMENTS_SPECIFICATION.md # 【正】要件定義書 (SSOT)
├── SYSTEM_ARCHITECTURE.md        # 【正】詳細設計書 (SSOT)
├── AGENT_RULES.md                # 開発・運用ルール
├── HANDOVER.md                   # 本引き継ぎ書
├── auto_betting/                 # ローカル検証・Playwrightツール（参考保持）
│   ├── .env                      # 【秘匿】認証情報（git除外）
│   ├── ipat_client.py            # ローカル用Playwright即PATクライアント
│   └── umaca_client.py           # UMACAスマート照会クライアント
├── backend-render/               # FastAPI バックエンド（Render本番）
│   ├── app/
│   │   ├── core/                 # config, database, timezone
│   │   ├── strategies/           # base, strategy_b, strategy_a, arbiter, daily_promotion, etc.
│   │   ├── betting/              # base_client, ipat_client, umaca_client, factory, manual_service
│   │   ├── scrapers/             # netkeiba_client (指数バックオフリトライ)
│   │   ├── settlement/           # payout_verifier (全券種払戻照合)
│   │   └── ml/                   # track_bias (リアルタイムトラックバイアス)
│   ├── main.py                   # APIエントリーポイント & スケジューラ
│   ├── ipat_fast_client.py       # （後方互換用）
│   ├── Dockerfile                # コンテナ設定
│   └── requirements.txt
├── android-app/                  # Android ネイティブアプリ (Jetpack Compose)
│   └── app/src/main/java/com/example/keibapredictionapp/
│       ├── api/KeibaApiService.kt       # API定義 (GET /predict, POST /betting/manual 等)
│       ├── model/RaceModels.kt          # データモデル (ManualBetRequest/Response等)
│       ├── viewmodel/MainViewModel.kt   # 状態管理・マニュアル投票実行
│       └── ui/
│           ├── MainScreen.kt            # レース一覧・詳細・日別サマリー
│           └── SettingsScreen.kt        # 設定・自動投票・マニュアル実機テスト投票
└── baselines/                    # 公式ベースライン（変更厳禁: Rule 3）
```

---

## 4. 今後のタスク・ロードマップ

### 【直近の最優先タスク】
```
【Task 1】週末（10/18日）即PAT自動投票の実働モニタリング ────── ⏳ 待機中
     │
     ▼
【Task 2】UMACAスマートの照会・発注テスト ＆ アプリ選択解禁 ───────── 📋 次期フェーズ
     │
     ▼
【Task 3】1日の最大投資上限ガード（安全機能のさらなる強化） ────────── 📋 次期フェーズ
```

#### 1. 週末（10/18 日）即PAT自動投票の実働モニタリング（最優先）
* **実行環境**: Render クラウド常駐スケジューラ（PC起動・アプリ常駐不要）
* **確認項目**:
  1. 朝8:30のダイジェスト通知受信および残高監視が正常に行われるか。
  2. 実馬券・勝負馬券（`is_real_bet_target == True`）に該当したレースで、発走5分前（Stage 5）に自動投票が実行されるか。
  3. スマホへ `🎫【即PAT投票完了】`（または残高不足時の `⚠️【即PAT投票エラー】`）プッシュ通知が正常に届くか。
  4. JRA公式の即PAT投票履歴・残高引き落としと一致しているか。
  5. レース確定後、アプリ内の日別サマリーへ自動で的中・回収率が反映されるか。

#### 2. UMACAスマートの動作検証 ＆ アプリ選択解禁（次期フェーズ）
* **現状**: 現在は安全のためアプリ画面上で「未テストのため選択不可（グレーアウト）」に制限中。
* **実施内容**:
  1. JRA UMACAスマートサイトへのログイン・残高照会・カート投入テストの実施。
  2. Render 経由での安定通信を確認後、`available_modes` の `enabled: true` に切り替えてアプリ側で選択可能に解禁。

#### 3. 1日の最大投資上限ガード（安全機能の強化）
* **目的**: 連敗時や想定外のレース乱発時でも、1日の最大出費を確実にコントロールする。
* **実施内容**:
  1. 設定画面に「1日の最大投資上限額（例: 5,000円 / 10,000円 / 20,000円 / 上限なし）」の選択設定を追加。
  2. 当日の累計自動投票額が上限に達した場合、残高があっても以後のレースを自動で安全スキップするガードを実装。

---

## 5. 安全運用上の絶対原則（Safety Rules）

> [!CAUTION]
> ### 1. 資金安全の絶対原則（自動追加入金の禁止）
> - **銀行口座からの「自動入金指示」は将来にわたっても絶対に実装・有効化しないでください。**
> - システムは「あらかじめ即PATに入金されている残高の範囲内でのみ動作する」設計を厳守しています。残高不足時はエラーではなく「安全スキップ（何もしない）」として安全停止します。

> [!WARNING]
> ### 2. 認証情報の厳重管理
> - 即PATの加入者番号・暗証番号・P-ARS番号等の重要情報は、Render ダッシュボードの「Environment Variables（環境変数）」で厳重に秘匿管理すること。
> - GitHub 等の公開リポジトリへコミット／プッシュすることは厳禁。

> [!IMPORTANT]
> ### 3. 開発規約（Rule 1〜9）
> - **Rule 1**: `backend-render` を修正した場合は、必ず `git push` を行い、Render 側の自動ビルド・デプロイを完了させること。
> - **Rule 2**: 開発ブランチと本番ブランチの整合性を常に維持すること。
> - **Rule 3**: `baselines/` 配下の公式ベースラインコード・モデルは上書き・変更禁止。
> - **Rule 4**: 未来情報混入防止・ゼロリーク保証ルール（Zero-Leakage Protocol）を徹底すること。
> - **Rule 5**: 個人利用特化・フェイルファスト例外方針（エラーの握りつぶし・サイレントフォールバック禁止）。
> - **Rule 6**: 仕様・詳細設計の正（`REQUIREMENTS_SPECIFICATION.md`, `SYSTEM_ARCHITECTURE.md`）を常に尊守し、手戻りのない変更を行うこと。
> - **Rule 7**: マニュアル投票のレコードは `is_manual_bet=true` で永続化し、AI収支集計から恒久的に除外すること。
> - **Rule 8**: Androidアプリ更新時はAPKをビルドし、`G:\マイドライブ\keiba_data\keiba_ai.apk` への配置を確認すること。
> - **Rule 9**: システムの変更や進展時は、本 `HANDOVER.md` を継続的に同期・更新すること。
