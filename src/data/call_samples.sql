WITH query2 AS (
  SELECT
    (expiration_date::date - valid_time::date) AS dte,
    strike / underlying_price as moneyness,
    ask - bid as spread,
    ask,
    bid,
    mark,
    underlying_price,
    delta,
    strike,
    contract_type,
    valid_time
  FROM option_chain_history
  WHERE sample = true
  AND root_symbol = 'SPXW'
  AND strike / underlying_price >= 0.99
  AND contract_type = 'CALL'
  AND (expiration_date::date - valid_time::date) <= 9
)

SELECT
  q2.*,
  vix.close AS vix,
  vix9d.close AS vix9d,
  vvix.close AS vvix,
  skew.close AS skew
FROM query2 q2
LEFT JOIN LATERAL (
  SELECT close, valid_time
  FROM price_history
  WHERE symbol = '$VIX'
  AND valid_time <= q2.valid_time
  ORDER BY valid_time DESC
  LIMIT 1
) vix ON true
LEFT JOIN LATERAL (
  SELECT close, valid_time
  FROM price_history
  WHERE symbol = '$VIX9D'
  AND valid_time <= q2.valid_time
  ORDER BY valid_time DESC
  LIMIT 1
) vix9d ON true
LEFT JOIN LATERAL (
  SELECT close, valid_time
  FROM price_history
  WHERE symbol = '$VVIX'
  AND valid_time <= q2.valid_time
  ORDER BY valid_time DESC
  LIMIT 1
) vvix ON true
LEFT JOIN LATERAL (
  SELECT
      close,
      valid_time
  FROM price_history
  WHERE symbol = '$SKEW'
  AND valid_time <= q2.valid_time
  ORDER BY valid_time DESC
  LIMIT 1
) skew ON true