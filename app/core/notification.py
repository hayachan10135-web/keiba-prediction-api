# -*- coding: utf-8 -*-
"""
app/core/notification.py

Firebase Cloud Messaging (FCM) プッシュ通知基盤
"""

import os
import json
from typing import Optional, Dict, Any

firebase_app = None
FIREBASE_KEY_PATH = os.environ.get("FIREBASE_KEY_PATH", "firebase_service_account.json")
FIREBASE_KEY_JSON = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY", "")

try:
    import firebase_admin
    from firebase_admin import credentials, messaging

    if FIREBASE_KEY_JSON:
        try:
            cred_dict = json.loads(FIREBASE_KEY_JSON)
            cred = credentials.Certificate(cred_dict)
            firebase_app = firebase_admin.initialize_app(cred)
            print("[Firebase] 環境変数 FIREBASE_SERVICE_ACCOUNT_KEY から正常に初期化しました。")
        except Exception as fe:
            print(f"[Firebase] JSONパース失敗: {fe}")
    elif os.path.exists(FIREBASE_KEY_PATH):
        cred = credentials.Certificate(FIREBASE_KEY_PATH)
        firebase_app = firebase_admin.initialize_app(cred)
        print(f"[Firebase] ファイル {FIREBASE_KEY_PATH} から正常に初期化しました。")
    else:
        print("[Firebase] サービスアカウント情報未設定（設定完了後にプッシュ通知が有効化されます）。")
except Exception as f_err:
    print(f"[Firebase] 初期化待機: {f_err}")


def send_fcm_notification(title: str, body: str, data: Optional[Dict[str, Any]] = None, topic: str = "keiba_real_bet") -> bool:
    """Firebase Cloud Messaging で指定トピックへプッシュ通知を配信"""
    if not firebase_app:
        print(f"[FCM未設定・通知待機] {title} | {body} (Topic: {topic})")
        return False
    try:
        from firebase_admin import messaging
        str_data = {str(k): str(v) for k, v in (data or {}).items()}
        msg = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body
            ),
            data=str_data,
            topic=topic
        )
        resp = messaging.send(msg)
        print(f"[FCM送信成功] {title} (ID: {resp})")
        return True
    except Exception as e:
        print(f"[FCM送信失敗] {title}: {e}")
        return False
