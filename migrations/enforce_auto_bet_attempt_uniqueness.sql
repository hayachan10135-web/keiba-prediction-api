-- Reserve each race before sending an auto-bet request, including pending or unknown outcomes.
CREATE UNIQUE INDEX IF NOT EXISTS idx_real_bets_auto_attempt_unique
ON keiba_real_bets (race_id)
WHERE is_manual_bet = FALSE;
