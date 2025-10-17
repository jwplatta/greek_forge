WITH query2 AS (
  SELECT
    (expiration_date::date - valid_time::date) AS dte,
    CASE
      WHEN contract_type = 'CALL' THEN underlying_price / strike
      WHEN contract_type = 'PUT' THEN strike / underlying_price
    END as moneyness,
    mark,
    underlying_price,
    delta,
    strike,
    contract_type,
    valid_time
  FROM option_chain_history
  WHERE sample = true
  AND root_symbol = 'SPXW'
  AND contract_type = %(contract_type)s
  AND (
    (contract_type = 'CALL' AND underlying_price / strike <= 1.01) OR
    (contract_type = 'PUT' AND strike / underlying_price <= 1.01)
  )
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